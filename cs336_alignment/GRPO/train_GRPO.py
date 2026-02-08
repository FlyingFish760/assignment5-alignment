import json
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from GRPO.trainer import GRPOTrainer
from GRPO.funcs import init_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
        

if __name__ == "__main__":
    #--------------Init policy model, rollout model and tokenizer---------------
    policy_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True
    )
    policy_model = policy_model.to(cfg.policy_device)

    rollout_model = init_vllm(
        model_id="Qwen/Qwen2.5-Math-1.5B",
        device=cfg.rollout_device,
        seed=cfg.vllm_seed,
        gpu_memory_utilization=0.8
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B", local_files_only=True)

    #--------------Init reward function, question dataset, and validation dataset---------------
    reward_func = r1_zero_reward_fn
    with open(cfg.train_data_path, "r", encoding="utf-8") as f:
        question_dataset = json.load(f)

    #--------------Training---------------
    grpo_trainer = GRPOTrainer(
        policy_model,
        rollout_model,
        reward_func,
        question_dataset,
        tokenizer=tokenizer,
        cfg=cfg
    )

    for grpo_step in range(cfg.n_grpo_steps):
        print(f"---------------- Started {grpo_step+1} GRPO train steps ----------------")
        grpo_trainer.train_step()
        print(f"**************** Finished {grpo_step+1} GRPO train steps ****************")

        if (grpo_step + 1) % cfg.eval_grpo_steps == 0:
            grpo_trainer.evaluate_model(grpo_step)
            print(f"**************** Finished evaluating {grpo_step+1} GRPO train steps ****************")

