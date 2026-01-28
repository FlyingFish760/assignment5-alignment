import argparse
import time
import os
import yaml

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from jaxtyping import Int, Float
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from sft_for_math.data import SFTDataset, SFTValDataset
from sft_for_math.helper_funcs import sft_microbatch_train_step, get_response_log_probs, learning_rate_schedule
from sft_for_math.utils import logger, save_checkpoint

# def train_step(inputs: Int[Tensor, "b seq_len"],
#                 targets: Int[Tensor, "b seq_len"],
#                 step: int) -> Float[Tensor, ""]:
#     '''
#     One training epoch of the complete given data.

#     inputs: Input token ids;
#     targets: Target token ids;

#     '''
#     model.train()

#     optimizer.zero_grad()
#     # Set the optimizer learning rate
#     lr = learning_rate_schedule(
#         step + 1, 
#         max_lr=model_opt_config["max_lr"],
#         min_lr=model_opt_config["max_lr"] * 0.1,
#         warmup_iters=int(train_steps * training_config["warmup_ratio"]),
#         cosine_cycle_iters=int(train_steps * training_config["cosine_ratio"])
#     )
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr

#     # Forward pass
#     logits = model(inputs)

#     # Compute loss
#     loss = cross_entropy(logits, targets)

#     # Back proporgation (to get gradients)
#     loss.backward()

#     # Optimizer step
#     optimizer.step()

#     return loss

def evaluate():
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_dataloader):
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            total_loss += loss

    return total_loss / (step + 1)

def log_train_performance():
    if (step + 1) % log_steps == 0 or (step + 1) == train_steps:
        lr = optimizer.param_groups[0]["lr"]
        cur_time = time.time()
        spent_time = (cur_time - start_time) // 60
        log_info = f"(Step: {step + 1}/{train_steps}), train_loss: {loss_step:.4f}, lr: {lr:.6f}, spent time: {spent_time}min"
        logger(log_info)
        if wandb_config["use_wandb"]:
            wandb_log = {
                "train_step": step,
                "train/loss": loss_step,
                "train/lr": lr,
                "train/spent time (min)": spent_time
            }
            wandb_run.log(wandb_log)

    loss_step = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model pre-train")

    # Trainer config
    parser.add_argument("--config", type=str, default="./configs/base.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device for training")
    parser.add_argument("--batch_size", type=int, help="Number of samples per batch")
    parser.add_argument("--train_data_path", type=str, help="Path to the training dataset")
    parser.add_argument("--val_data_path", type=str, help="Path to the validation dataset")
    parser.add_argument("--log_step_rate", type=float, default=0.01, help="Rate of train steps to log train loss")
    parser.add_argument("--save_step_rate", type=float, default=0.2, help="Rate of train steps to save checkpoint")
    parser.add_argument("--eval_step_rate", type=float, default=0.1, help="Rate of train steps to evaluate model performance")
    # parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--save_dir", type=str, default="../out", help="Directory to save checkpoints")
    parser.add_argument("--load_path", type=str, default=None, help="Path to load checkpoints")

    args = parser.parse_args()

    # Parse config file
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    model_opt_config = cfg["model_opt_config"]
    data_config = cfg["data"]
    training_config = cfg["training"]
    wandb_config = cfg["wandb_logging"]

    #--------------Set up model, optimizer---------------
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True
    )
    named_params = {p_name: p for p_name, p in model.named_parameters() if p.requires_grad}
    params_to_decay = [p for _,p in named_params.items() if p.dim() >= 2]
    params_not_to_decay = [p for _,p in named_params.items() if p.dim() < 2]
    param_groups = [
        {"params": params_to_decay, "weight_decay": cfg["weight_decay"]},
        {"params": params_not_to_decay, "weight_decay": 0.0}
    ]
    optimizer = AdamW(
        param_groups,
        lr=cfg["max_lr"],
        betas=cfg["betas"],
        eps=cfg["eps"],
        weight_decay=cfg["weight_decay"]
    )

    # # Load checkpoints if needed
    # if args.load_path is not None:
    #     start_step = load_checkpoint(args.load_path, model, optimizer)
    # else:
    #     start_step = 0
    start_step = 0

    #--------------Set up dataloader---------------
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B", local_files_only=True)
    train_ds = SFTDataset(args.train_data_path, tokenizer)
    train_dataloader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True
    )
    val_ds = SFTValDataset(args.val_data_path, tokenizer)
    val_dataloader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False
    )

    #--------------Init wandb---------------
    if wandb_config["use_wandb"]:
        wandb_run = wandb.init(
            entity=wandb_config["wandb_team"],
            project=wandb_config["wandb_project"],
            name=wandb_config["wandb_run"],
            config=wandb_config["config"]
        )

    # Setup wandb metrics
    wandb.define_metric("train_step") # the x‑axis for training
    wandb.define_metric("eval_step") # the x‑axis for evaluation

    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")

    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")
    
    #--------------Training loop---------------
    # Define steps
    train_steps = len(train_dataloader)
    log_steps = int(train_steps * args.log_step_rate)
    save_steps = int(train_steps * args.save_step_rate)
    eval_steps = int(train_steps * args.eval_step_rate)

    start_time = time.time()
    loss_step = 0
    for step, (inputs, targets, reponse_mask) in enumerate(train_dataloader, 
                                            start=start_step):
        # Train 
        model.train()
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        log_probs_res = get_response_log_probs(
            model,
            inputs,
            targets,
            return_token_entropy=True
        )

        policy_log_probs = log_probs_res["log_probs"]
        token_entropy = log_probs_res["token_entropy"]

        loss_microstep, _ = sft_microbatch_train_step(
            policy_log_probs,
            reponse_mask,
            cfg["grad_accumulation_steps"]
        )
        loss_step += loss_microstep

        if (step + 1) % cfg["grad_accumulation_steps"] == 0:
            # Set the optimizer lr
            lr = learning_rate_schedule(
                step + 1, 
                max_lr=cfg["max_lr"],
                min_lr=cfg["max_lr"] * 0.1,
                warmup_iters=int(train_steps * cfg["warmup_ratio"]),
                cosine_cycle_iters=int(train_steps * cfg["cosine_ratio"])
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Optimizer makes step
            optimizer.step()
            optimizer.zero_grad()

            # Log training performance
            log_train_performance()

        # Save checkpoints
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = f"{args.save_dir}/lr_{model_opt_config["max_lr"]}_{step + 1}.pt"
        if (step + 1) % save_steps == 0:  
            save_checkpoint(model, optimizer, step + 1, save_path)
            cur_time = time.time()
            spent_time = (cur_time - start_time) // 60
            log_info = f"(Step: {step + 1}/{train_steps}), saved checkpoint to {save_path}, spent time: {spent_time}min"
            logger(log_info)


        # Evaluate accuracy
        if (step + 1) % eval_steps == 0 or (step + 1) == train_steps:
            val_loss = evaluate()
            cur_time = time.time()
            spent_time = (cur_time - start_time) // 60
            log_info = f"(Step: {step + 1}/{train_steps}), val_loss: {val_loss:.4f}, spent time: {spent_time}min"
            logger(log_info)
            if wandb_config["use_wandb"]:
                wandb_log = {
                    "val loss": val_loss,
                    "spent time (min)": spent_time
                }
                wandb_run.log(wandb_log)

    if wandb_config["use_wandb"]:
        wandb_run.finish()