from dataclasses import dataclass, field
from typing import Tuple, Optional, List
# from collections.abc import 

@dataclass
class DeviceConfig:
    policy_device: str = "cuda:0"
    rollout_device: str = "cuda:1"

@dataclass
class VllmConfig:    
    vllm_seed: int = 42
    gpu_memory_utilization: float = 0.85


@dataclass
class SamplingConfig:
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024
    sampling_stop: List[str] = field(default_factory=lambda: ["</answer>"])
    include_stop_str_in_output: bool = True


@dataclass
class OptimConfig:
    lr: float = 1e-5
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0


@dataclass
class GRPOConfig:
    n_grpo_steps: int = 200
    rollout_batch_size: int = 256  # Number of rollouts per grpo step
    group_size: int = 8

    epochs_per_rollout_batch: int = 1   # On-policy
    train_batch_size: int = 256   # On-policy
    micro_batch_size: int = 2

    advantage_eps: float = 1e-6
    use_std_normalization: bool = True

    loss_type: str = "reinforce_with_baseline"   # Literal["no_baseline","reinforce_with_baseline","grpo_clip"]

    clip_range: float = 0.2

    log_loss_steps: int = 8   # Better to be divisible by the train_steps per epoch


@dataclass
class DataConfig:
    train_data_path: str = "../data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b_filtered.jsonl"
    val_data_path: str = "../data/sft-cs336-assign5-datasets/sft-reason/val.jsonl"
    prompt_temp_path: str = "prompts/r1_zero.prompt"


@dataclass
class EvalConfig:
    eval_batch_size: int = 1024
    eval_grpo_steps: int = 10

@dataclass
class WandbConfig:
    use_wandb: bool = True
    wandb_team: str = "cs336_assign5"
    wandb_project: str = "GRPO"


@dataclass
class ExperimentConfig:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
