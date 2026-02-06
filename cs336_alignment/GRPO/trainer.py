import random
import json
from typing import Callable, List, Dict
from pathlib import Path
from urllib import response

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch
from vllm import LLM, SamplingParams
from transformers import PreTrainedModel, PreTrainedTokenizer

from GRPO.funcs import compute_group_normalized_rewards, grpo_microbatch_train_step
from sft_for_math.helper_funcs import tokenize_prompt_and_output, get_response_log_probs

class GRPODataset(Dataset):
    def __init__(self,
                 prompts: List[str],
                 responses: List[str], 
                 tokenizer: PreTrainedTokenizer):
        super().__init__()
        tokenized_data = tokenize_prompt_and_output(prompts, responses, tokenizer)
        self.state_ids = tokenized_data["input_ids"]
        self.action_ids = tokenized_data["labels"]
        self.response_masks = tokenized_data["response_mask"]
    
    def __len__(self):
        return len(self.state_ids)
    
    def __getitem__(self, index):
        return self.state_ids[index], self.action_ids[index], self.response_masks[index]
        

class GRPOTrainer:
    def __init__(self, 
                 policy_model: PreTrainedModel,
                 rollout_model: LLM,
                 reward_func: Callable,
                 question_dataset: List[Dict[str, str]],
                 tokenizer: PreTrainedTokenizer,
                 cfg):
        self.policy_model = policy_model
        self.rollout_model = rollout_model
        self.reward_func = reward_func
        self.question_dataset = question_dataset
        self.tokenizer = tokenizer
        # Init optimizer
        self.optimizer = AdamW(
            self.policy_model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas
        )
        self.cfg = cfg

    def load_policy_into_vllm_instance(self, policy: PreTrainedModel, llm: LLM):
        """
        Copied from https://github.com/huggingface/trl/blob/
        22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
        """
        state_dict = policy.state_dict()
        llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(state_dict.items())
    
    def sample_batch_question(self, batch_size: int) -> List[Dict[str, str]]:
        sampled_inds = random.sample(range(len(self.question_dataset)), k=batch_size)
        return [self.question_dataset[i] for i in sampled_inds]
        
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


    def train_step(self):
        # Sample a batch of questions
        num_sample_questions = self.cfg.rollout_batch_size // self.cfg.group_size
        sample_questions = self.sample_batch_question(batch_size=num_sample_questions)

        # Set the old policy model (for running rollouts)
        self.load_policy_into_vllm_instance(self.policy_model, self.rollout_model)

        # Run rollouts
        prompt_template = Path(self.cfg.prompt_temp_path).read_text(encoding="utf-8")
        sampling_parms = SamplingParams(
            n=self.cfg.group_size,
            temperature=self.cfg.sampling_temperature, 
            min_tokens=self.cfg.sampling_min_tokens,
            max_tokens=self.cfg.sampling_max_tokens, 
            stop=self.cfg.sampling_stop, 
            include_stop_str_in_output=self.cfg.include_stop_str_in_output
        )
        rollouts = self.sample_rollouts(
            self.rollout_model,
            sampling_parms,
            sample_questions,
            prompt_template
        )
        rollout_prompts, rollout_responses, rollout_answers = [], [], []
        for r in rollouts:
            rollout_prompts.append(r["prompt"])
            rollout_responses.append(r["response"])
            rollout_answers.append(r["answer"])
        

        # Compute raw reward/ group-normalized advantages
        raw_reward, advantages, reward_metadata = compute_group_normalized_rewards(
            self.reward_func,
            rollout_responses,
            rollout_answers,
            self.cfg.group_size,
            self.cfg.advantage_eps,
            self.cfg.use_std_normalization
        )

        # Update policy using policy gradient
        dataset = GRPODataset(rollout_prompts, rollout_responses, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.micro_batch_size,
            shuffle=True,
            num_workers=4, 
            pin_memory=True
        )
        grad_accumlation_steps = self.cfg.train_batch_size // self.cfg.micro_batch_size
        for epoch in self.cfg.epochs_per_rollout_batch:
            for step, (state_ids, action_ids, response_masks) in enumerate(dataloader):
               # Compute policy_log_probs
                log_prob_res = get_response_log_probs(
                    self.policy_model,
                    input_ids=state_ids,
                    labels=action_ids,
                    return_token_entropy=True
                )
                policy_log_probs = log_prob_res["log_probs"]
                token_entropy = log_prob_res["token_entropy"] 

                #  GRPO micro train step
                loss, meta_data = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_masks,
                    gradient_accumulation_steps=grad_accumlation_steps,
                    loss_type=self.cfg.loss_type,
                    raw_rewards=
                )


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

    