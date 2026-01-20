from collections.abc import Callable
from pathlib import Path
import json
from urllib import response

from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn

def evaluate_vllm(
    vllm_model: LLM,
    reward_func: Callable[[str, str], dict[str, float]],
    data_path: str,
    eval_sampling_params: SamplingParams,
    prompt_temp_path: str
) -> list[dict]:
    '''
    Evaluatea language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.

    Params
    data_path: example data path to a jsonl file
    '''
    # load the example data
    with open(data_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    # format each example as a string prompt for the language model using the r1_zero prompt
    formatted_prompts = []
    prompt_template = Path(prompt_temp_path).read_text(encoding="utf-8")
    for sample in data:
        formatted_p = prompt_template.format(
            question = sample["problem"]
        )
        formatted_prompts.append(formatted_p)

    # generate model outputs for each example
    outputs = vllm_model.generate(formatted_prompts, eval_sampling_params)

    # compute the relevant evaluation metrics
    records = []
    num_samples = len(data)
    for i in range(num_samples):
        output = outputs[i]
        ground_truth = data[i]["expected_answer"]

        prompt = output.prompt
        generated_text = output.outputs[0].text
        eval_score = reward_func(
            response = generated_text,
            ground_truth = ground_truth
        )
        record = {
            "prompt": prompt,
            "generated_text": generated_text,
            "eval_score": eval_score
        }
        records.append(record)

    return records

    

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