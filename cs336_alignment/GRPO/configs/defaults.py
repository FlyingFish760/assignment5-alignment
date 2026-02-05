from dataclasses import dataclass
from pathlib import Path


@dataclass
class PathConfig:
    train_data_path: Path = Path("data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b_filtered.jsonl")

@dataclass
class TrainConfig:
    rollout_batch_size: int = 256