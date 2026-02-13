import json
import random
import argparse
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from GRPO.trainer import GRPOTrainer
from GRPO.funcs import init_vllm
from drgrpo_grader import r1_zero_reward_fn
from omegaconf import OmegaConf
from GRPO.configs.defaults import ExperimentConfig
        


def load_config(cfg_path: str) -> ExperimentConfig:
    # 1. Create structured (typed) base config
    base_cfg = OmegaConf.structured(ExperimentConfig)

    # 2. Load YAML overrides
    yaml_cfg = OmegaConf.load(cfg_path)

    # 3. Merge (YAML overrides dataclass defaults)
    cfg = OmegaConf.merge(base_cfg, yaml_cfg)

    # 4. (Optional but recommended) Freeze config
    OmegaConf.set_readonly(cfg, True)

    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None, help="Path to the config file")
    args = parser.parse_args()

    if args.config_path is not None:
        cfg = load_config(args.config_path)
    else:
        cfg = OmegaConf.structured(ExperimentConfig)
        OmegaConf.set_readonly(cfg, True)
    
    # Parameter sanity check
    train_batch_size = cfg.grpo.train_batch_size
    micro_batch_size = cfg.grpo.micro_batch_size
    rollout_batch_size = cfg.grpo.rollout_batch_size
    group_size = cfg.grpo.group_size
    assert train_batch_size % micro_batch_size == 0, (
        "train_batch_size must be divisible by micro_batch_size"
    )
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )    #n_prompts_per_rollout_batch = rollout_batch_size // group_size
    
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )

    assert rollout_batch_size % train_batch_size == 0, (
        "rollout_batch_size must be divisible by train_batch_size"
    )   # n_optimizer_steps = rollout_batch_size // train_batch_size

    #--------------Init policy model, rollout model and tokenizer---------------
    policy_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True
    )
    policy_model = policy_model.to(cfg.device.policy_device)

    rollout_model = init_vllm(
        model_id="Qwen/Qwen2.5-Math-1.5B",
        device=cfg.device.rollout_device,
        seed=cfg.vllm.vllm_seed,
        gpu_memory_utilization=0.8
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        local_files_only=True
    )

    #--------------Init reward function, question dataset, and validation dataset---------------
    reward_func = r1_zero_reward_fn
    with open(cfg.data.train_data_path, "r", encoding="utf-8") as f:
        question_dataset = json.load(f)

    #--------------Training---------------
    start_time = time.time()
    grpo_trainer = GRPOTrainer(
        policy_model,
        rollout_model,
        reward_func,
        question_dataset,
        tokenizer=tokenizer,
        cfg=cfg,
        start_time=start_time
    )

    start_time = time.time()
    for grpo_step in range(cfg.grpo.n_grpo_steps):
        print(f"---------------- Started {grpo_step+1} GRPO train steps ----------------")
        grpo_trainer.train_step(grpo_step)
        print(f"**************** Finished {grpo_step+1} GRPO train steps ****************")

        if (grpo_step + 1) % cfg.eval.eval_grpo_steps == 0:
            grpo_trainer.evaluate_model(grpo_step)
            print(f"**************** Finished evaluating {grpo_step+1} GRPO train steps ****************")
            print()
