import time
import os

from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, AutoTokenizer
from vllm import LLM, SamplingParams
import wandb

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from EI.sft_utils import SFTDataset, LRScheduler
from EI.funcs import load_policy_into_vllm_instance
from sft_for_math.helper_funcs import get_response_log_probs, sft_microbatch_train_step, evaluate_vllm
from sft_for_math.utils import logger


class SFTTrainer():
    def __init__(self, 
                 train_model: PreTrainedModel,
                 vllm_model: LLM,
                 dataset: list[dict[str, str]],
                 val_data_path: str,
                 prompt_temp_path: str,
                 cfg: dict,
                 data_size: int,
                 num_rollouts: int,
                 train_device: str, 
                 ei_step: int):
        self.model = train_model
        self.vllm_model = vllm_model
        self.dataset = dataset
        self.val_data_path = val_data_path
        self.prompt_temp_path = prompt_temp_path
        self.cfg = cfg
        self.data_size = data_size
        self.num_rollouts = num_rollouts
        self.train_device = train_device    
        self.ei_step = ei_step

    def set_optimizer(self):
        named_params = {p_name: p for p_name, p in self.model.named_parameters() if p.requires_grad}
        params_to_decay = [p for _,p in named_params.items() if p.dim() >= 2]
        params_not_to_decay = [p for _,p in named_params.items() if p.dim() < 2]
        param_groups = [
            {"params": params_to_decay, "weight_decay": self.cfg["weight_decay"]},
            {"params": params_not_to_decay, "weight_decay": 0.0}
        ]
        self.optimizer = AdamW(
            param_groups,
            lr=self.cfg["max_lr"],
            betas=self.cfg["betas"],
            eps=self.cfg["eps"],
            weight_decay=self.cfg["weight_decay"]
        )

    def init_dataloader(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B", local_files_only=True)
        train_ds = SFTDataset(self.dataset, tokenizer)
        self.train_dataloader = DataLoader(
            train_ds,
            batch_size=self.cfg["micro_batch_size"],
            shuffle=True,
            num_workers=4, 
            pin_memory=True
        )

    def init_wandb_run(self):
        self.wandb_run = wandb.init(
            entity=self.cfg["wandb_team"],
            project=self.cfg["wandb_project"],
            name=f"ds.{self.data_size}_G.{self.num_rollouts}_eps.{self.cfg['num_epochs']}_eistep.{self.ei_step}",
            config={
                "max_lr": self.cfg["max_lr"],
                "batch_size": self.cfg["micro_batch_size"] * self.cfg["grad_accumulation_steps"]
            }
        )
        
        # Setup wandb metrics
        wandb.define_metric("train_step") # the x‑axis for training
        wandb.define_metric("epoch") # the x‑axis for evaluation

        # everything that starts with train/ is tied to train_step
        wandb.define_metric("train/*", step_metric="train_step")

        # everything that starts with eval/ is tied to eval_step
        wandb.define_metric("eval/*", step_metric="epoch")
    
    def evaluate(self, epoch):
        load_policy_into_vllm_instance(self.model, self.vllm_model)

        sampling_params = SamplingParams(
            temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
        )
        eval_metrics, _ = evaluate_vllm(
            vllm_model=self.vllm_model,
            reward_func=r1_zero_reward_fn,
            data_path=self.val_data_path,
            eval_sampling_params=sampling_params,
            prompt_temp_path=self.prompt_temp_path
        )

        format_acc = eval_metrics["format_accuracy"]
        answer_acc = eval_metrics["answer_accuracy"]
        reward_acc = eval_metrics["reward_accuracy"]
        total_epochs = self.cfg["num_epochs"]
        log_info = f"[Epoch:{epoch+1}/{total_epochs}], format accuracy: {format_acc:.3f}, answer accuracy: {answer_acc:.3f}, reward accuracy: {reward_acc:.3f}"
        logger(log_info)
        if self.cfg["use_wandb"]:
            wandb_log = {
                "epoch": epoch,
                "eval/format_accuracy": format_acc,
                "eval/answer_accuracy": answer_acc,
                "eval/reward_accuracy": reward_acc
            }
            self.wandb_run.log(wandb_log)

    def train_epoch(self, epoch):
        # Define steps
        # train_steps = len(train_dataloader)
        # log_steps = int(train_steps * args.log_step_rate)
        # save_steps = int(train_steps * args.save_step_rate)
        # eval_steps = int(train_steps * args.eval_step_rate)
        log_micro_steps = self.cfg["log_micro_steps"]

        start_time = time.time()

        for micro_step, (inputs, targets, response_mask) in enumerate(self.train_dataloader):
            self.model.train()
            inputs = inputs.to(self.train_device)
            targets = targets.to(self.train_device)
            response_mask = response_mask.to(self.train_device)
            log_probs_res = get_response_log_probs(
                self.model,
                inputs,
                targets,
                return_token_entropy=True
            )
            policy_log_probs = log_probs_res["log_probs"]
            token_entropy = log_probs_res["token_entropy"]

            loss_microstep, _ = sft_microbatch_train_step(
                policy_log_probs,
                response_mask,
                self.cfg["grad_accumulation_steps"]
            )

            cur_micro_step = epoch * len(self.train_dataloader) + micro_step
            
            # One (accumlated) train step
            if (cur_micro_step + 1) % self.cfg["grad_accumulation_steps"] == 0:
                # Set the optimizer lr
                lr = self.lr_scheduler.get_lr(epoch, micro_step + 1)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                # Gradient clipping 
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip_norm"])

                # Optimizer makes step
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Log training performance
            if (cur_micro_step + 1) % log_micro_steps == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                cur_time = time.time()
                spent_time = (cur_time - start_time) // 60
                loss = loss_microstep.item() * self.cfg["grad_accumulation_steps"]
                per_token_entropy = token_entropy.mean().item()
                total_epochs = self.cfg["num_epochs"]
                total_steps = len(self.train_dataloader) * total_epochs
                log_info = f"[Epoch:{epoch+1}/{total_epochs}](Step: {cur_micro_step + 1}/{total_steps}), train_loss: {loss:.4f}, token_entropy: {per_token_entropy:.4f}, spent time: {spent_time}min"
                logger(log_info)
                if self.cfg["use_wandb"]:
                    wandb_log = {
                        "train_step": cur_micro_step,
                        "train/loss": loss,
                        "train/token_entropy": per_token_entropy,
                        "train/lr": lr,
                        "train/spent time (min)": spent_time
                    }
                    self.wandb_run.log(wandb_log)

            # # Save checkpoints
            # os.makedirs(args.save_dir, exist_ok=True)
            # save_path = f"{args.save_dir}/lr_{cfg["max_lr"]}_{step + 1}.pt"
            # if (step + 1) % save_steps == 0 or (step + 1) == train_steps:  
            #     save_checkpoint(model, optimizer, step + 1, save_path)
            #     cur_time = time.time()
            #     spent_time = (cur_time - start_time) // 60
            #     log_info = f"(Step: {step + 1}/{train_steps}), saved checkpoint to {save_path}, spent time: {spent_time}min"
            #     logger(log_info)


            # # Evaluate accuracy
            # if (step + 1) % eval_steps == 0 or (step + 1) == train_steps:
            #     eval_step += 1
            #     evaluate(step)


        # At the end of training epoch, evaluate model performance
        self.evaluate(epoch)

    def train(self):
        # Set up optimizer
        self.set_optimizer()

        # Set up dataloader
        self.init_dataloader()

        # Set up learning rate scheduler
        self.lr_scheduler = LRScheduler(
            max_lr=self.cfg["max_lr"],
            min_lr=self.cfg["max_lr"] * 0.1,
            num_epochs=self.cfg["num_epochs"],
            iters_per_epoch=len(self.train_dataloader),
            warmup_ratio=self.cfg["warmup_ratio"],
            cosine_cycle_ratio=self.cfg["cosine_ratio"]
        )

        #Init wandb
        if self.cfg["use_wandb"]:
            self.init_wandb_run()

        # SFT training epochs
        num_epochs = self.cfg["num_epochs"]
        for epoch in range(num_epochs):
            self.train_epoch(epoch)
            print("----------Finished one SFT epoch---------")

        if self.cfg["use_wandb"]:
            self.wandb_run.finish()
            






