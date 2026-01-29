from collections.abc import Callable
from pathlib import Path
import json
from urllib import response

from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn
from sft_for_math.helper_funcs import evaluate_vllm
    

if __name__ == "__main__":
    data_path = r"../data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b_filtered.jsonl"
    prompt_template_path = r"./prompts/r1_zero.prompt"
    save_path = "zero_shot_performance.jsonl"
    
    # Create a sampling params object. Based on Dr.GRPO: stop when the model completes its answer
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )

    # Create model
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")

    # Evaluate zero-shot performance of the model
    eval_results = evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        data_path,
        sampling_params,
        prompt_template_path
    )

    # serialize the original examples, the model-generated outputs, and their corresponding evaluation 
    # scores to disk
    with open(save_path, "w", encoding='utf-8') as f:
        for res in eval_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")