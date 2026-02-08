import random
import json
from typing import Callable, List, Dict
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn as nn
from vllm import LLM, SamplingParams
from transformers import PreTrainedModel, PreTrainedTokenizer

from GRPO.funcs import compute_group_normalized_rewards, grpo_microbatch_train_step, evaluate_vllm
from sft_for_math.helper_funcs import tokenize_prompt_and_output, get_response_log_probs
from sft_for_math.utils import logger

# class GRPODataset(Dataset):
#     def __init__(self,
#                  prompts: List[str],
#                  responses: List[str], 
#                  tokenizer: PreTrainedTokenizer):
#         super().__init__()
#         tokenized_data = tokenize_prompt_and_output(prompts, responses, tokenizer)
#         self.state_ids = tokenized_data["input_ids"]
#         self.action_ids = tokenized_data["labels"]
#         self.response_masks = tokenized_data["response_mask"]
    
#     def __len__(self):
#         return len(self.state_ids)
    
#     def __getitem__(self, index):
#         return self.state_ids[index], self.action_ids[index], self.response_masks[index]
        

class GRPOTrainer:
    def __init__(self, 
                 policy_model: PreTrainedModel,
                 rollout_model: LLM,
                 reward_func: Callable,
                 question_dataset: List[Dict[str, str]],
                 tokenizer: PreTrainedTokenizer,
                 cfg):
        self.cfg = cfg
        self.policy_model = policy_model
        self.rollout_model = rollout_model
        self.reward_func = reward_func
        self.question_dataset = question_dataset
        self.tokenizer = tokenizer

        # Init full evaluation dataset
        with open(self.cfg.val_data_path, "r", encoding="utf-8") as f:
            self.eval_dataset = json.load(f)

        # Init optimizer
        self.optimizer = AdamW(
            self.policy_model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas
        )

    def load_policy_into_vllm_instance(self, policy: PreTrainedModel, llm: LLM):
        """
        Copied from https://github.com/huggingface/trl/blob/
        22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
        """
        state_dict = policy.state_dict()
        llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(state_dict.items())
    
    def sample_batch_question(self, batch_size: int, question_set: List[Dict[str, str]]) -> List[Dict[str, str]]:
        sampled_inds = random.sample(range(len(question_set)), k=batch_size)
        return [question_set[i] for i in sampled_inds]
        
    def sample_rollouts(
            self,
            model: LLM, 
            sampling_params: SamplingParams, 
            question_set: List[Dict[str, str]],
            prompt_template: str
        ) -> list[dict[str, str]]:
        '''
        For each question in a question dataset, sample rollouts using a model.
        Args:
            model (LLM): a vllm model instance used to run rollouts.
            sampling_params (SamplingParams): sampling parametes.
            qeustion_set (List[Dict[str, str]]): the question set from which rollouts are made.
            prompt_template (str): the prompt template.

        Returns:
            rollouts: (list[dict[str, str]]): a list of rollouts containing prompt, answer, and reponse.
        '''
        # Format questions into prompts
        prompts = []
        for q in question_set:
            problem = q["problem"]
            prompts.append(prompt_template.format(question=problem))

        # Run rollouts
        outputs = model.generate(prompts, sampling_params)
        
        rollouts = []
        for i, output in enumerate(outputs):
            prompt = prompts[i]
            answer = question_set[i]["expected_answer"]
            for out in output.outputs:
                response = out.text
                rollouts.append(
                    {"prompt": prompt,
                    "answer": answer,
                    "response": response}
                )

        return rollouts

    def evaluate_model(self, grpo_step):
        self.load_policy_into_vllm_instance(self.policy_model, self.rollout_model)

        # Sample a batch of evaluation questions
        sub_eval_dataset = self.sample_batch_question(
            batch_size=self.cfg.eval_batch_size, 
            question_set=self.eval_dataset
        )

        # Evaluate model on the batch of evaluation questions
        sampling_params = SamplingParams(
            temperature=self.cfg.sampling_temperature, 
            max_tokens=self.cfg.sampling_max_tokens, 
            stop=self.cfg.sampling_stop, 
            include_stop_str_in_output=self.cfg.include_stop_str_in_output
        )
        prompt_template = Path(self.cfg.prompt_temp_path).read_text(encoding="utf-8")
        eval_metrics, _ = evaluate_vllm(
            vllm_model=self.rollout_model,
            reward_func=self.reward_func,
            dataset=sub_eval_dataset,
            eval_sampling_params=sampling_params,
            prompt_template=prompt_template
        )

        format_acc = eval_metrics["format_accuracy"]
        answer_acc = eval_metrics["answer_accuracy"]
        reward_acc = eval_metrics["reward_accuracy"]
        log_info = f"[GRPO step:{grpo_step+1}/{self.cfg.n_grpo_steps}], format accuracy: {format_acc:.3f}, answer accuracy: {answer_acc:.3f}, reward accuracy: {reward_acc:.3f}"
        logger(log_info)
        # if self.cfg["use_wandb"]:
        #     wandb_log = {
        #         "epoch": epoch,
        #         "eval/format_accuracy": format_acc,
        #         "eval/answer_accuracy": answer_acc,
        #         "eval/reward_accuracy": reward_acc
        #     }
        #     self.wandb_run.log(wandb_log)

    def train_step(self):
        # Sample a batch of questions
        num_sample_questions = self.cfg.rollout_batch_size // self.cfg.group_size
        sample_questions = self.sample_batch_question(
            batch_size=num_sample_questions, 
            question_set=self.question_dataset
        )

        # Set the old policy model (for running rollouts)
        self.load_policy_into_vllm_instance(self.policy_model, self.rollout_model)

        # Run rollouts
        prompt_template = Path(self.cfg.prompt_temp_path).read_text(encoding="utf-8")
        sampling_params = SamplingParams(
            n=self.cfg.group_size,
            temperature=self.cfg.sampling_temperature, 
            min_tokens=self.cfg.sampling_min_tokens,
            max_tokens=self.cfg.sampling_max_tokens, 
            stop=self.cfg.sampling_stop, 
            include_stop_str_in_output=self.cfg.include_stop_str_in_output
        )
        rollouts = self.sample_rollouts(
            self.rollout_model,
            sampling_params,
            sample_questions,
            prompt_template
        )
        rollout_prompts, rollout_responses, rollout_answers = [], [], []
        for r in rollouts:
            rollout_prompts.append(r["prompt"])
            rollout_responses.append(r["response"])
            rollout_answers.append(r["answer"])
        

        # Compute raw reward/ group-normalized advantages
        raw_rewards, advantages, reward_metadata = compute_group_normalized_rewards(
            self.reward_func,
            rollout_responses,
            rollout_answers,
            self.cfg.group_size,
            self.cfg.advantage_eps,
            self.cfg.use_std_normalization
        )

        # #--------------(Train steps) Update policy using policy gradient---------------
        # Tokenize prompts and responses
        tokenized_res = tokenize_prompt_and_output(
            rollout_prompts,
            rollout_responses,
            self.tokenizer
        )
        state_ids = tokenized_res["input_ids"]
        action_ids = tokenized_res["labels"]
        response_masks = tokenized_res["response_mask"]

        # Pre-compute old policy probs
        if self.cfg.loss_type == "grpo_clip":
            old_policy_probs_list = []
            for old_batch_ind in range(0, len(state_ids), self.cfg.micro_batch_size):
                state_ids_batch = state_ids[old_batch_ind: old_batch_ind+self.cfg.micro_batch_size].to(self.cfg.policy_device)
                action_ids_batch = action_ids[old_batch_ind: old_batch_ind+self.cfg.micro_batch_size].to(self.cfg.policy_device)
                with torch.inference_mode():
                    log_prob_res = get_response_log_probs(
                        self.policy_model,
                        state_ids_batch,
                        action_ids_batch,
                        return_token_entropy=False
                    )
                    old_policy_probs_list.append(log_prob_res["log_probs"])
            old_log_policy_probs = torch.cat(old_policy_probs_list, dim=0)
        else:
            old_log_policy_probs = None

        # Policy update steps
        self.policy_model.train()

        grad_accumulation_steps = self.cfg.train_batch_size // self.cfg.micro_batch_size
        n_train_steps_per_epoch = self.cfg.rollout_batch_size // self.cfg.micro_batch_size

        self.optimizer.zero_grad()
        for epoch in range(self.cfg.epochs_per_rollout_batch):
            for step in range(n_train_steps_per_epoch):
                batch_start_ind = step * self.cfg.micro_batch_size
                batch_end_ind = batch_start_ind + self.cfg.micro_batch_size

                state_ids_batch = state_ids[batch_start_ind: batch_end_ind].to(self.cfg.policy_device)
                action_ids_batch = action_ids[batch_start_ind: batch_end_ind].to(self.cfg.policy_device)
                response_masks_batch = response_masks[batch_start_ind: batch_end_ind].to(self.cfg.policy_device)

               # Compute policy_log_probs
                log_prob_res = get_response_log_probs(
                    self.policy_model,
                    input_ids=state_ids_batch,
                    labels=action_ids_batch,
                    return_token_entropy=True
                )
                policy_log_probs_batch = log_prob_res["log_probs"]
                token_entropy = log_prob_res["token_entropy"] 

                #  GRPO micro train step
                if old_log_policy_probs is not None:
                    old_log_probs_batch = old_log_policy_probs[batch_start_ind: batch_end_ind]
                else:
                    old_log_probs_batch = None
                loss, meta_data = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs_batch,
                    response_mask=response_masks_batch,
                    gradient_accumulation_steps=grad_accumulation_steps,
                    loss_type=self.cfg.loss_type,
                    raw_rewards=raw_rewards[batch_start_ind: batch_end_ind],
                    advantages=advantages[batch_start_ind: batch_end_ind],
                    old_log_probs=old_log_probs_batch,
                    cliprange=self.cfg.clip_range
                )

                # Optimizer step
                if (step + 1) % grad_accumulation_steps == 0:
                    # Gradient clipping
                    nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=self.cfg.max_grad_norm)
                    # Makes step
                    self.optimizer.step()
                    self.optimizer.zero_grad()



if __name__ == "__main__":
    grpo_trainer = GRPOTrainer()

    # # Test sample_batch_question()
    # test_ds = [1, 2, 3, 4, 5]
    # batch = grpo_trainer.sample_batch_question(test_ds, batch_size=3)
    # print(batch)

    # # Test sample_rollouts
    # vllm_model = grpo_trainer.init_vllm(model_id="Qwen/Qwen2.5-Math-1.5B",
    #     device="cuda:0",
    #     seed=123,
    #     gpu_memory_utilization=0.8)
    # sampling_parms = SamplingParams(
    #         n=3,
    #         temperature=1.0, 
    #         top_p=1.0, 
    #         max_tokens=1024, 
    #         stop=["</answer>"], 
    #         include_stop_str_in_output=True
    #     )
    # with open("data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b_filtered.jsonl", "r", encoding="utf-8") as f:
    #     question_ds = json.load(f)
    # question_set = grpo_trainer.sample_batch_question(
    #     question_ds,
    #     batch_size=2
    # )
    # prompt_temp = Path("cs336_alignment/prompts/r1_zero.prompt").read_text(encoding="utf-8")
    # rollouts = grpo_trainer.sample_rollouts(
    #     model=vllm_model,
    #     sampling_params=sampling_parms,
    #     question_set=question_set,
    #     prompt_template=prompt_temp
    # )
    # for r in rollouts:
    #     print(r)

    