from collections.abc import Callable
from curses import meta
from typing import Dict, List, Tuple, Literal, Optional, Iterator
from unittest.mock import patch

import math
import torch
from torch.nn import Parameter
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from einops import repeat

from cs336_alignment.sft_for_math.helper_funcs import masked_normalize

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], Dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized within groups.

    Args:
        reward_fn (Callable[[str, str], dict[str, float]]):
            Scores a rollout response against the ground truth. Should return a dict
            with keys:
                - "reward"
                - "format_reward"
                - "answer_reward"

        rollout_responses (list[str]):
            Rollouts sampled from the policy. Length is:
                rollout_batch_size = n_prompts_per_rollout_batch * group_size

        repeated_ground_truths (list[str]):
            Ground-truth answers corresponding to each rollout response.
            Length equals rollout_batch_size, since each ground truth is repeated
            `group_size` times.

        group_size (int):
            Number of responses per question (i.e., per group).

        advantage_eps (float):
            Small constant added for numerical stability during normalization.

        normalize_by_std (bool):
            If True, normalize rewards by subtracting the group mean and dividing
            by the group standard deviation.
            If False, only subtract the group mean.

    Returns:
        tuple[
            torch.Tensor,  # advantages, shape (rollout_batch_size,)
            torch.Tensor,  # raw_rewards, shape (rollout_batch_size,)
            dict[str, float],  # metadata (e.g., mean/std/min/max of rewards)
        ]
    """
    # Compute raw rewards for each rollout
    rewards_list = []
    for rollout, gt in zip(rollout_responses, repeated_ground_truths):
        res = reward_fn(rollout, gt)
        rewards_list.append(res["reward"])

    # Compute group-normalized rewards (advantages)
    advantages_list = []
    for i in range(0, len(rewards_list), group_size):
        group = rewards_list[i: i+group_size]
        group_mean = sum(group) / len(group)
        group_std = math.sqrt(sum((x - group_mean) ** 2 for x in group) / (group_size - 1))
        for j in range(group_size):
            if normalize_by_std:
                adv = (group[j] - group_mean) / (group_std + advantage_eps)
            else:
                adv = (group[j] - group_mean)
            advantages_list.append(adv)

    rewards = torch.tensor(rewards_list)
    advantages = torch.tensor(advantages_list)

    metadata = {}
    return (advantages, rewards, metadata)

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the policy-gradient loss at every token.
    
    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar 
                                   reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), 
                          log probs for each token.
                          
    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per-token 
        policy-gradient loss.
    """
    seq_len = policy_log_probs.shape[-1]
    repeat_r_advs = repeat(raw_rewards_or_advantages, "b one -> b (seq_len one)", seq_len=seq_len)
    return -repeat_r_advs * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Computes the clipped surrogate loss for GRPO (Group Relative Policy Optimization).

    Args:
        advantages: torch.Tensor of shape (batch_size, 1).
            The per-example relative advantages computed across the group.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length).
            Per-token log probabilities from the current policy being trained.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length).
            Per-token log probabilities from the old (reference) policy.
        cliprange: float.
            The clipping parameter epsilon (e.g., 0.2) to limit the magnitude of 
            the policy update.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: torch.Tensor of shape (batch_size, sequence_length).
                The per-token clipped policy gradient loss.
            metadata: dict[str, torch.Tensor].
                Contains training metrics such as 'clip_fraction' or a mask 
                indicating where the loss was clipped (where the clipped RHS 
                was lower than the unclipped LHS).
    """
    seq_len = policy_log_probs.shape[-1]
    repeated_advs = repeat(advantages, "b one -> b (seq_len one)", seq_len=seq_len)

    prob_ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(prob_ratio, 1 - cliprange, 1 + cliprange)
    left_hand_side = prob_ratio * repeated_advs
    right_hand_side = clipped_ratio * repeated_advs

    loss = -torch.min(left_hand_side, right_hand_side)

    was_clipped = right_hand_side < left_hand_side
    metadata = {"was_clipped": was_clipped}
    
    return (loss, metadata)


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.

    Args:
        policy_log_probs:
            Tensor of shape (batch_size, sequence_length),
            per-token log-probabilities from the policy being trained.
        loss_type:
            One of {"no_baseline", "reinforce_with_baseline", "grpo_clip"}.
        raw_rewards:
            Required if loss_type == "no_baseline";
            tensor of shape (batch_size, 1).
        advantages:
            Required for "reinforce_with_baseline" and "grpo_clip";
            tensor of shape (batch_size, 1).
        old_log_probs:
            Required for "grpo_clip";
            tensor of shape (batch_size, sequence_length).
        cliprange:
            Required for "grpo_clip";
            scalar ε used for clipping.

    Returns:
        A tuple of:
        - loss:
            Tensor of shape (batch_size, sequence_length),
            per-token loss.
        - metadata:
            Dictionary of statistics from the underlying routine
            (e.g., clip fraction for GRPO-Clip).
    """
    # Argument check
    assert loss_type in ["no_baseline", "reinforce_with_baseline", "grpo_clip"], \
        f"Unknown loss type of {loss_type!r}"

    metadata = {}

    if loss_type == "no_baseline":
        if raw_rewards == None:
            raise ValueError("raw_rewards is required for 'no_baseline' loss.")
        loss = compute_naive_policy_gradient_loss(
            raw_rewards,
            policy_log_probs
        )  

    elif loss_type == "reinforce_with_baseline":
        if advantages == None:
            raise ValueError("advantages is required for 'reinforce_with_baseline' loss.")
        loss = compute_naive_policy_gradient_loss(
            advantages,
            policy_log_probs
        )

    else:
        if advantages == None:
            raise ValueError("advantages is required for 'reinforce_with_baseline' loss.")
        if old_log_probs == None:
            raise ValueError("old_log_probs is required for 'reinforce_with_baseline' loss.")
        if cliprange == None:
            raise ValueError("cliprange is required for 'reinforce_with_baseline' loss.")
        
        loss, metadata_clip_loss = compute_grpo_clip_loss(
            advantages,
            policy_log_probs,
            old_log_probs,
            cliprange
        )

        clip_fraction = metadata_clip_loss["was_clipped"].float().mean().item()
        metadata["clip_frac"] = clip_fraction
    

    return (loss, metadata)

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute the mean of `tensor` considering only elements where `mask == 1`.

    Args:
        tensor: torch.Tensor
            The data to be averaged.
        mask: torch.Tensor
            Same shape as `tensor`; positions with 1 are included in the mean.
        dim: int | None
            Dimension over which to average. If None, compute the mean over
            all masked elements.

    Returns:
        torch.Tensor
            The masked mean; shape matches `tensor.mean(dim)` semantics.
    """
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    norm_type: Literal["maksed_mean", "maksed_norm"] = "maksed_mean",
    norm_const: float = 1
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.

    Args:
        policy_log_probs:
            Tensor of shape (batch_size, sequence_length),
            per-token log-probabilities from the policy being trained.
        response_mask:
            Tensor of shape (batch_size, sequence_length),
            1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps:
            Number of microbatches per optimizer step.
        loss_type:
            One of {"no_baseline", "reinforce_with_baseline", "grpo_clip"}.
        raw_rewards:
            Needed when loss_type == "no_baseline";
            shape (batch_size, 1).
        advantages:
            Needed when loss_type != "no_baseline";
            shape (batch_size, 1).
        old_log_probs:
            Required for GRPO-Clip;
            shape (batch_size, sequence_length).
        cliprange:
            Clip parameter ε for GRPO-Clip.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
                this so we can log it.
            metadata: Dict with metadata from the underlying loss call, and any other statistics you
                might want to log.
    """
    assert norm_type in ["maksed_mean", "maksed_norm"], f"norm type {norm_type} must be 'maksed_mean'/ 'maksed_norm"

    metadata = {}

    # Compute loss -> (b, seq_len)
    loss, metadata_policy_loss = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )

    # Apply mask and normalization to the loss -> (b,)
    if norm_type == "maksed_mean":
        loss_normed = masked_mean(loss, response_mask, dim=-1)
    elif norm_type == "maksed_norm":
        loss_normed = masked_normalize(loss, response_mask, normalize_constant=norm_const, dim=-1)

    # Average over batch, take into account gradient accumlation, .backward()
    loss_final = torch.mean(loss_normed) / gradient_accumulation_steps
    loss_final.backward()

    if loss_type == "grpo_clip":
        metadata.update(metadata_policy_loss)

    return (loss_final, metadata)

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
    
def evaluate_vllm(
    vllm_model: LLM,
    reward_func: Callable[[str, str], dict[str, float]],
    dataset: List[Dict[str, str]],
    eval_sampling_params: SamplingParams,
    prompt_template: str
):
    '''
    '''
    # format each example as a string prompt for the language model using the r1_zero prompt
    formatted_prompts = []
    for sample in dataset:
        formatted_p = prompt_template.format(
            question = sample["problem"]
        )
        formatted_prompts.append(formatted_p)

    # generate model outputs for each example
    outputs = vllm_model.generate(formatted_prompts, eval_sampling_params)

    # compute the relevant evaluation metrics
    records = []
    format_score, answer_score, reward_score = 0, 0, 0
    num_samples = len(dataset)
    for i in range(num_samples):
        output = outputs[i]
        ground_truth = dataset[i]["expected_answer"]

        prompt = output.prompt
        generated_text = output.outputs[0].text
        eval_score = reward_func(
            response = generated_text,
            ground_truth = ground_truth
        )
        f_score = eval_score["format_reward"]
        ans_score = eval_score["answer_reward"]
        r_score = eval_score["reward"]

        record = {
            "prompt": prompt,
            "generated_text": generated_text,
            "eval_score": eval_score
        }
        records.append(record)

        format_score += f_score
        answer_score += ans_score
        reward_score += r_score

    # eval_metrics = {
    #     "format_accuracy": format_score / num_samples,
    #     "answer_accuracy": answer_score / num_samples,
    #     "reward_accuracy": reward_score / num_samples
    # }

    eval_metrics = {
        "format_accuracy": format_score,
        "answer_accuracy": answer_score,
        "reward_accuracy": reward_score
    }

    return eval_metrics, records

def get_grad_l2_norm(params: Iterator[Parameter]) -> float:
    total_norm = 0
    for p in params:
        if p.requires_grad:
            p_norm = p.grad.detach().norm(2) 
            total_norm += p_norm.item() ** 2

    return total_norm ** (1 / 2)