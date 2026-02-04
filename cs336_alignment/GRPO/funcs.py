from collections.abc import Callable
from typing import Dict, List, Tuple

import math
import torch

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
        group_std = math.sqrt(sum((x - group_mean) ** 2 for x in group) / len(group))
        for j in range(group_size):
            if normalize_by_std:
                adv = (group[j] - group_mean) / (group_std + advantage_eps)
            else:
                adv = (group[j] - group_mean)
            advantages.append(adv)

    metadata = {}
    return (advantages, rewards, metadata)