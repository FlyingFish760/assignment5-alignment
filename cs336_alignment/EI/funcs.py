import random
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import torch
from transformers import PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed


def get_batch_question(dataset: list[dict[str, str]], batch_size: int):
    '''
    Sample a batch of question-answer paris from a dataset

    Args:
        dataset (list[dict[str, str]]): A list of samples(dict) containing a question
          and an answer.
        batch_size (int)

    Returns:
        batch (dict[str, list[str]]): A sampled batch of the dataset
    '''
    sample_inds = random.sample(range(len(dataset)), batch_size)
    questions, answers = [], []
    for i in sample_inds:
        sample = dataset[i]
        questions.append(sample["problem"])
        answers.append(sample["expected_answer"])

    batch = {
        "questions": questions,
        "answers": answers
    }
    return batch

def sample_rollouts(model: LLM, 
                    sampling_params: SamplingParams, 
                    question_set: dict[str, list[str]],
                    prompt_temp_path: str) -> list[dict[str, str]]:
    '''
    For each question in a question dataset, sample rollouts using a model.
    Args:
        model (LLM): a vllm model instance used to run rollouts.
        sampling_params (SamplingParams): sampling parametes.
        qeustion_set (dict[str, list[str]]): the question set from which rollouts are made.
        prompt_temp_path (str): path to the prompt template.

    Returns:
        rollouts: (list[dict[str, str]]): a list of rollouts containing prompt, answer, and reponse.
    '''
    # Format questions into prompts
    prompt_temp = Path(prompt_temp_path).read_text(encoding="utf-8")
    prompts = []
    for question in question_set["questions"]:
        prompts.append(prompt_temp.format(question=question))

    # Run rollouts
    outputs = model.generate(prompts, sampling_params)
    
    answers = question_set["answers"]
    rollouts = []
    for i, output in enumerate(outputs):
        prompt = prompts[i]
        answer = answers[i]
        for out in output.outputs:
            response = out.text
            rollouts.append(
                {"prompt": prompt,
                 "answer": answer,
                 "response": response}
            )

    return rollouts

def get_correct_dataset(dataset: list[dict[str, str]], 
                        reward_func: Callable) -> list[dict[str, str]]:
    '''
    Filter out wrong responses in a dataset using a reward_func.
    
    Args:
        dataset (list[dict[str, str]]): The dataset to be filtered, in which each sample contains a 
            question, a reponse, and an answer.
        reward_func (Callable[str, str]): reward function to make judgements

    Returns:
        correct_dataset (list[dict[str, str]]): Filtered dataset
    '''
    correct_dataset = []
    for sample in dataset:
        response = sample["response"]
        answer = sample["answer"]
        res = reward_func(response, answer)
        if res["reward"] == 1:
            correct_dataset.append(sample)

    return correct_dataset

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
    return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


if __name__ == "__main__":
    pass

    # import json
    # from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    # # Test get_batch_question
    # with open("data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b_filtered.jsonl") as f:
    #     data = json.load(f)

    # batch = get_batch_question(data, batch_size=512)

    # # Test sample_rollouts
    # model = LLM("Qwen/Qwen2.5-Math-1.5B")
    # sampling_params = SamplingParams(
    #     n = 3, 
    #     temperature=1.0, 
    #     top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    # )
    # rollouts = sample_rollouts(
    #     model,
    #     sampling_params,
    #     batch,
    #     "cs336_alignment/prompts/r1_zero.prompt"
    # )

    # # Test get_correct_dataset
    # corrct_dataset = get_correct_dataset(
    #     rollouts,
    #     r1_zero_reward_fn
    # )