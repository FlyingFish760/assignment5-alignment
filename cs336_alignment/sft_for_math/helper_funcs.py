from typing import Callable
import json
from pathlib import Path
import math
from collections.abc import Iterable

import torch
from torch import Tensor, log_, nn
from jaxtyping import Float
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer):
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens
    and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs (list[str]): List of prompt strings.
        output_strs (list[str]): List of output strings.
        tokenizer (PreTrainedTokenizer): Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor]: Let prompt_and_output_lens be a list containing the lengths of
        the tokenized prompt and output strings. The returned dictionary has:
            - input_ids (torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1)):
              the tokenized prompt+output strings, with the final token sliced off.
            - labels (torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1)):
              shifted input_ids, i.e., the input_ids without the first token.
            - response_mask (torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1)):
              a mask on the response tokens in the labels.
    """
    # Tokenize prompts and outputs, and construct reponse_mask
    tokenized_ids = []
    response_mask = []
    max_prompt_output_len = 0
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_id = tokenizer.encode(prompt, add_special_tokens=False)
        output_id = tokenizer.encode(output, add_special_tokens=False)
        tokenized_id = prompt_id + output_id
        tokenized_ids.append(tokenized_id)

        prompt_output_len = len(tokenized_id)
        if prompt_output_len > max_prompt_output_len:
            max_prompt_output_len = prompt_output_len

        mask = len(prompt_id) * [0] + len(output_id) * [1]
        response_mask.append(mask)

    # Pad tokenized_ids and response_mask
    padded_ids = []
    padded_masks = []
    pad_id = tokenizer.pad_token_id
    for tokenized_id, mask in zip(tokenized_ids, response_mask):
        pad_len = max_prompt_output_len - len(tokenized_id)
        padded_ids.append(tokenized_id + (pad_len * [pad_id]))
        padded_masks.append(mask + (pad_len * [0]))

    # Construct tensors
    encoding = torch.tensor(padded_ids)
    input_ids = encoding[:, :-1]
    labels = encoding[:, 1:]

    response_mask = torch.tensor(padded_masks)[:, 1:]
    
    res = {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }

    return res

def compute_entropy(logits: Float[Tensor, "batch_size sequence_length vocab_size"]) \
    -> Float[Tensor, "batch_size sequence_length"]:
    """
    Compute the per-token entropy of next-token predictions.

    The entropy is computed over the vocabulary dimension for each
    batch element and sequence position.

    Args:
        logits (torch.Tensor):
            A tensor of shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits for next-token prediction.

    Returns:
        torch.Tensor:
            A tensor of shape (batch_size, sequence_length) containing
            the entropy of the next-token prediction at each position.
    """
    # Compute logsumexp of logits
    lg_Z = torch.logsumexp(logits, dim=-1)   # (b, n)

    # Compute log_p
    lg_p = logits - lg_Z.unsqueeze(-1)   # (b, n, vocab_size)

    # Compute p
    p = torch.exp(lg_p)   # (b, n, vocab_size)

    # Compute entropy
    H = lg_Z - torch.sum(p * logits, dim=-1)   # (b, n)

    return H

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: Float[Tensor, "batch_size sequence_length"],
    labels: Float[Tensor, "batch_size sequence_length"],
    return_token_entropy: bool = False,
) -> dict[str, Float[Tensor, "batch_size sequence_length"]]:
    """
    Args:
        model (PreTrainedModel): HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids (torch.Tensor): Tensor of shape (batch_size, sequence_length), concatenated prompt +
            response tokens as produced by your tokenization method.
        labels (torch.Tensor): Tensor of shape (batch_size, sequence_length), labels as produced by
            your tokenization method.
        return_token_entropy (bool): If True, also return per-token entropy by calling compute_entropy.

    Returns:
        dict[str, torch.Tensor]:
            - "log_probs": Tensor of shape (batch_size, sequence_length), conditional log-probabilities
              log p_θ(x_t | x_<t).
            - "token_entropy" (optional): Tensor of shape (batch_size, sequence_length), per-token entropy
              for each position (present only if return_token_entropy=True).
    """
    res = {}

    # Compute next token log-probability
    logits = model(input_ids).logits

    # p = torch.nn.functional.softmax(logits, dim=-1)   # (b, n, vocab_size)
    # p_label = torch.gather(p, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)    # (b, n)
    # log_probs = torch.log(p_label)

    # res["log_probs"] = log_probs
    
    # Calculate log_softmax instead of full softmax + log
    # This is more numerically stable and allows for potential memory optimizations
    log_p = torch.nn.functional.log_softmax(logits, dim=-1)

    # Gather only the log_probs for the specific labels
    log_probs = torch.gather(log_p, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    res["log_probs"] = log_probs

    # Compute per-token entropy (if needed)
    if return_token_entropy:
        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            res["token_entropy"] = token_entropy

    return res

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only elements where mask == 1.

    Args:
        tensor (torch.Tensor): The tensor to sum and normalize.
        mask (torch.Tensor): Same shape as `tensor`; positions with 1 (or True) are included in the sum.
        normalize_constant (float): The constant to divide by for normalization.
        dim (int | None): The dimension to sum along before normalization. If None, sum over all dimensions.

    Returns:
        torch.Tensor: The normalized sum where masked-out elements do not contribute.
    """
    masked_tensor = tensor * mask
    masked_sum = torch.sum(masked_tensor, dim=dim) if dim is not None else torch.sum(masked_tensor)
    masked_norm = masked_sum / normalize_constant
    return masked_norm

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.

    Args:
        policy_log_probs (torch.Tensor):
            Tensor of shape (batch_size, sequence_length) containing per-token
            log-probabilities from the SFT policy being trained.
        response_mask (torch.Tensor):
            Tensor of shape (batch_size, sequence_length), where 1 indicates
            response tokens and 0 indicates prompt or padding tokens.
        gradient_accumulation_steps (int):
            Number of microbatches per optimizer step.
        normalize_constant (float, optional):
            Constant by which to divide the summed loss. Defaults to 1.0.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            - loss (torch.Tensor): Scalar tensor representing the microbatch loss,
              adjusted for gradient accumulation. Returned for logging.
            - metadata (dict[str, torch.Tensor]): Dictionary containing metadata from
              the underlying loss computation and any additional statistics to log.
    """
    # Compute loss (negative log-likelihood) with mask and normlization constant
    masked_sum_probs = masked_normalize(
            policy_log_probs, 
            response_mask, 
            normalize_constant,
            dim=-1
        )   # (b,)
    batch_size = policy_log_probs.shape[0]
    loss = -torch.sum(masked_sum_probs) / batch_size / gradient_accumulation_steps
    loss.backward()
    meta_data = {}

    return (loss, meta_data)

def log_generations(
    data_path: str,
    model: LLM,
    sampling_params: SamplingParams,
    reward_func: Callable
):
    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract question/ reasoning trace/ answer from the data, and format prompts
    questions = []
    reasoning_traces = []
    answers = []
    prompts = []
    prompt_template = Path(r"cs336_alignment/prompts/r1_zero.prompt").read_text(encoding="utf-8")
    for sample in data:
        question = sample["problem"]
        prompts.append(prompt_template.format(question=question))

        questions.append(question)
        reasoning_traces.append(sample["reasoning_trace"])
        answers.append(sample["expected_answer"])

    # Generate response using the model
    outputs = model.generate(prompts, sampling_params)

    # Compute rewards (and log responses)
    rewards = []
    responses = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        answer = answers[i]
        reward = reward_func(response, answer)

        responses.append(response)
        rewards.append(reward)

    # Make log
    records = [
        {
            "question": questions[i],
            "response": responses[i],
            "reasoning_trace": reasoning_traces[i],
            "answer": answers[i],
            "reward": rewards[i]
        }
        for i in range(len(data))
    ]

    return records

def learning_rate_schedule(it: int, 
                           max_lr: float, 
                           min_lr: float, 
                           warmup_iters: int, 
                           cosine_cycle_iters: int) -> float:
    '''
    It: starts from 1
    '''
    if it <= 0:
        raise ValueError(f"Wrong iteration of {it}.")
    if it < warmup_iters:
        lr = it / warmup_iters * max_lr
    elif it <= cosine_cycle_iters:
        lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(((it - warmup_iters) / (cosine_cycle_iters - warmup_iters)) * math.pi))
    else:
        lr = min_lr
    return lr

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
    format_score, answer_score, reward_score = 0, 0, 0
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

    eval_metrics = {
        "format_accuracy": format_score / num_samples,
        "answer_accuracy": answer_score / num_samples,
        "reward_accuracy": reward_score / num_samples
    }

    return eval_metrics, records


if __name__ == "__main__":
    pass

    b, n, vocab_size = 4, 16, 64
    device = "cuda:0"
    # prompt_strs = ["a", "b c"]
    # output_strs = ["fafsfdsfa", "1"]
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

    # res = tokenize_prompt_and_output(
    #     prompt_strs,
    #     output_strs,
    #     tokenizer
    # )

    # print("input_ids:", input_ids)
    # print("labels:", labels)
    # print("response_mask:", response_mask)

    # # Test compute_entropy
    # logits = torch.randn((b, n, v))
    # H = compute_entropy(logits)
    # print(H.shape)

    # # Test get_response_log_probs
    # input_ids = torch.randint(0, vocab_size, (b, n))
    # labels = torch.randint(0, vocab_size, (b, n))
    # model = AutoModelForCausalLM.from_pretrained(
    #     "Qwen/Qwen2.5-Math-1.5B",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     local_files_only=True
    # )
    # res = get_response_log_probs(
    #     model.to(device),
    #     input_ids.to(device),
    #     labels.to(device),
    #     True
    # )
    # print(res.keys())

    # # Test masked_normalize
    # t = torch.tensor([i for i in range(5)])
    # mask = torch.tensor([0, 0, 1, 1, 1])
    # norm_const = 2
    # res = masked_normalize(t, mask, norm_const)
    # print(res)

    # # Test sft_microbatch_train_step
    # log_probs = torch.randn((b, n))
    # mask = torch.randint(0, 2, (b, n))
    # res = sft_microbatch_train_step(
    #     log_probs,
    #     mask,
    #     gradient_accumulation_steps=4
    # )
    # print(res)

    # # Test log_generations
    # model = LLM(model="Qwen/Qwen2.5-Math-1.5B")
    # sampling_params = SamplingParams(
    #     temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    # )
    # data_path = r"./data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b_filtered.jsonl"
    # log = log_generations(
    #     data_path,
    #     model,
    #     sampling_params,
    #     r1_zero_reward_fn
    # )
    # print(log[0])