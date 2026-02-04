import argparse
import json
import yaml

import torch
from transformers import AutoModelForCausalLM

from EI.funcs import *
from EI.sft import SFTTrainer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument("--data_path", type=str, help="Path to the quesiton dataset")
    parser.add_argument("--val_data_path", type=str, help="Path to the validation dataset")
    parser.add_argument("--prompt_temp_path", type=str, help="Path to the prompt template")
    parser.add_argument("--train_device", type=str, default="cuda:0", help="Device to train the model")
    parser.add_argument("--inf_device", type=str, default="cuda:1", help="Device to run vllm model for inference")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sft_config = cfg["sft_config"]

    #--------------Init policy model and inference model---------------
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True
    )
    model = model.to(args.train_device)
    vllm_model = init_vllm(
        model_id="Qwen/Qwen2.5-Math-1.5B",
        device=args.inf_device,
        seed=cfg["vllm_seed"],
        gpu_memory_utilization=0.8
    )

    #--------------Run expert iterations---------------
    for ei_step in range(cfg["n_ei_steps"]):
        # Load the data
        with open(args.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Sample a batch of questions from the data
        sample_batch = get_batch_question(data, batch_size=cfg["batch_size"])

        # Load the policy into vllm model (for running rollouts)
        load_policy_into_vllm_instance(model, vllm_model)

        # Sample rollouts using the batch of questions
        sampling_parms = SamplingParams(
            n=cfg["n_rollouts"],
            temperature=1.0, 
            top_p=1.0, 
            max_tokens=1024, 
            stop=["</answer>"], 
            include_stop_str_in_output=True
        )
        rollouts = sample_rollouts(
            vllm_model,
            sampling_parms,
            sample_batch,
            args.prompt_temp_path
        )

        # Filter out wrong responses to get a correct dataset
        correct_data = get_correct_dataset(rollouts, r1_zero_reward_fn)
        filtered_size = len(correct_data)
        rollouts_size = len(rollouts)
        correct_ratiio = filtered_size / rollouts_size
        print("----------------------------------")
        print(f"Filtered out {filtered_size}/{rollouts_size} rollouts, the ratio is {correct_ratiio}.")
        print("----------------------------------")
        print()

        # Run SFT on the correct dataset
        sft_trainer = SFTTrainer(
            model,
            vllm_model,
            correct_data,
            args.val_data_path,
            args.prompt_temp_path,
            sft_config,
            data_size=cfg["batch_size"],
            num_rollouts=cfg["n_rollouts"],
            train_device=args.train_device,
            ei_step=ei_step
        )
        sft_trainer.train()
        print("**************************************************************************")
        print(f"**********************Finished {ei_step+1} EI steps**********************")
        print("**************************************************************************")
        print()
