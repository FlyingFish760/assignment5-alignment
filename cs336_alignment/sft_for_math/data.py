import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

# from helper_funcs import tokenize_prompt_and_output

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

def format_prompt(file_path, prompt_template_path, save_path):
    '''
    Format "problem" in the data file into prompts using a prompt template. 
    Return a new .jsonl file, in which  each example is a JSON element of type {"prompt": str, "response": str} 
    '''
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    prompt_template = Path(prompt_template_path).read_text(encoding="utf-8")
    with open(save_path, "w", encoding="utf-8") as f:
        for sample in data:
            problem = sample["problem"]
            response = sample["reasoning_trace"]
            prompt = prompt_template.format(question=problem)
            item = {"prompt": prompt, "response": response}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer: PreTrainedTokenizer):
        super().__init__()
        tokenized_data = self.tokenize_data(data_path, tokenizer)
        self.input_ids = tokenized_data["input_ids"]
        self.labels = tokenized_data["labels"]
        self.response_masks = tokenized_data["response_mask"]
        
    def tokenize_data(self, data_path, tokenizer):
        prompts = []
        outputs = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                prompts.append(sample["prompt"])
                outputs.append(sample["response"])
        
        tokenized_data = tokenize_prompt_and_output(prompts, outputs, tokenizer)
        return tokenized_data
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.labels[index], self.response_masks[index]




if __name__=="__main__":
    pass

    # format_prompt(
    #     file_path=r"data\sft-cs336-assign5-datasets\sft-reason\sft_gpt-oss-120b_filtered.jsonl",
    #     prompt_template_path=r"cs336_alignment\prompts\r1_zero.prompt",
    #     save_path="data\sft-cs336-assign5-datasets\sft-reason\sft_train.jsonl"
    # )

    # Test SFTDataset
    data_path = "data\sft-cs336-assign5-datasets\sft-reason\sft_train.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(
        r"cs336_alignment\hf_cache\models--Qwen--Qwen2.5-Math-1.5B\snapshots\4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
        local_files_only = True
    )
    sft_ds = SFTDataset(data_path, tokenizer)
    print(sft_ds[0])