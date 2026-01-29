import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

from sft_for_math.helper_funcs import tokenize_prompt_and_output

def format_train_data(file_path, prompt_template_path, save_path):
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

def format_val_data(file_path, prompt_template_path, save_path):
    '''
    Format "problem" in the data file into prompts using a prompt template. 
    Return a new .jsonl file, in which  each example is a JSON element of type {"prompt": str, "answer": str} 
    '''
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    prompt_template = Path(prompt_template_path).read_text(encoding="utf-8")
    with open(save_path, "w", encoding="utf-8") as f:
        for sample in data:
            problem = sample["problem"]
            answer = sample["expected_answer"]
            prompt = prompt_template.format(question=problem)
            item = {"prompt": prompt, "answer": answer}
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

class SFTValDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.prompt_ids, self.answers = self.tokenize_data(data_path, tokenizer)

    def tokenize_data(self, data_path, tokenizer):
        prompts = []
        answers = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                prompts.append(sample["prompt"])
                answers.append(sample["answer"])
        tokenized_ids = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False
        )["input_ids"]
        return tokenized_ids, answers
    
    def __len__(self):
        return len(self.prompt_ids)
    
    def __getitem__(self, index):
        return self.prompt_ids[index], self.answers[index]


if __name__=="__main__":
    pass

    format_train_data(
        file_path=r"data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b_filtered.jsonl",
        prompt_template_path=r"cs336_alignment/prompts/r1_zero.prompt",
        save_path="data/sft-cs336-assign5-datasets/sft-reason/sft_train.jsonl"
    )

    # format_val_data(
    #     file_path=r"data\sft-cs336-assign5-datasets\sft-reason\val.jsonl",
    #     prompt_template_path=r"cs336_alignment\prompts\r1_zero.prompt",
    #     save_path="data\sft-cs336-assign5-datasets\sft-reason\sft_val.jsonl"
    # )


    # # Test SFTDataset
    # data_path = "data\sft-cs336-assign5-datasets\sft-reason\sft_train.jsonl"
    # tokenizer = AutoTokenizer.from_pretrained(
    #     r"cs336_alignment\hf_cache\models--Qwen--Qwen2.5-Math-1.5B\snapshots\4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
    #     local_files_only = True
    # )
    # sft_ds = SFTDataset(data_path, tokenizer)
    # print(sft_ds[0])

    # # Test SFTValDataset
    # data_path = "data\sft-cs336-assign5-datasets\sft-reason\sft_val.jsonl"
    # tokenizer = AutoTokenizer.from_pretrained(
    #     r"cs336_alignment\hf_cache\models--Qwen--Qwen2.5-Math-1.5B\snapshots\4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
    #     local_files_only = True
    # )
    # sft_val_ds = SFTValDataset(data_path, tokenizer)
    # print(sft_val_ds[0])