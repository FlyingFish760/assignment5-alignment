from collections.abc import Callable
from typing import Dict, List, Tuple, Literal, Optional
from unittest.mock import patch

import math
import torch
from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from einops import repeat

# from cs336_alignment.GRPO.utils import 

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
    rewards = []
    for rollout, gt in zip(rollout_responses, repeated_ground_truths):
        res = reward_fn(rollout, gt)
        rewards.append(res["reward"])

    # Compute group-normalized rewards (advantages)
    advantages = []
    for i in range(0, len(rewards), group_size):
        group = rewards[i: i+group_size]
        group_mean = sum(group) / len(group)
        group_std = math.sqrt(sum((x - group_mean) ** 2 for x in group) / (group_size - 1))
        for j in range(group_size):
            if normalize_by_std:
                adv = (group[j] - group_mean) / (group_std + advantage_eps)
            else:
                adv = (group[j] - group_mean)
            advantages.append(adv)

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

    loss = -torch.min(prob_ratio * repeated_advs, clipped_ratio * repeated_advs)
    metadata = {}
    
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

    if loss_type == "no_baseline":
        if raw_rewards == None:
            raise ValueError("raw_rewards is required for 'no_baseline' loss.")
        loss = compute_naive_policy_gradient_loss(
            raw_rewards,
            policy_log_probs
        )
        metadata = {}

    elif loss_type == "reinforce_with_baseline":
        if advantages == None:
            raise ValueError("advantages is required for 'reinforce_with_baseline' loss.")
        loss = compute_naive_policy_gradient_loss(
            advantages,
            policy_log_probs
        )
        metadata = {}

    else:
        if advantages == None:
            raise ValueError("advantages is required for 'reinforce_with_baseline' loss.")
        if old_log_probs == None:
            raise ValueError("old_log_probs is required for 'reinforce_with_baseline' loss.")
        if cliprange == None:
            raise ValueError("cliprange is required for 'reinforce_with_baseline' loss.")
        loss, metadata = compute_grpo_clip_loss(
            advantages,
            policy_log_probs,
            old_log_probs,
            cliprange
        )

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
    # Compute loss -> (b, seq_len)
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )

    # Compute masked_mean loss to get per-sample loss -> (b,)
    loss_per_sample = masked_mean(loss, response_mask, dim=-1)

    # Average over batch, take into account gradient accumlation, .backward()
    loss_final = torch.mean(loss_per_sample) / gradient_accumulation_steps
    loss_final.backward()

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
    