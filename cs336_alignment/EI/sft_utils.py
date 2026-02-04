import math

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from sft_for_math.helper_funcs import tokenize_prompt_and_output


class SFTDataset(Dataset):
    def __init__(self, data: list[dict[str, str]], tokenizer: PreTrainedTokenizer):
        super().__init__()
        tokenized_data = self.tokenize_data(data, tokenizer)
        self.input_ids = tokenized_data["input_ids"]
        self.labels = tokenized_data["labels"]
        self.response_masks = tokenized_data["response_mask"]
        
    def tokenize_data(self, data, tokenizer):
        prompts = []
        outputs = []
        for sample in data:
            prompts.append(sample["prompt"])
            outputs.append(sample["response"])
        
        tokenized_data = tokenize_prompt_and_output(prompts, outputs, tokenizer)
        return tokenized_data
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.labels[index], self.response_masks[index]
    
class LRScheduler():
    def __init__(self, 
                 max_lr: float,
                 min_lr: float,
                 num_epochs: int, 
                 iters_per_epoch: int,
                 warmup_ratio: float, 
                 cosine_cycle_ratio: float,):
        self.max_lr = max_lr
        self.min_lr = min_lr
        
        self.iters_per_epoch = iters_per_epoch

        total_iters = num_epochs * iters_per_epoch
        self.warmup_iters = int(total_iters * warmup_ratio)
        self.cosine_cycle_iters = int(total_iters * cosine_cycle_ratio)
    
    def get_lr(self, epoch, iter):
        '''
        iter: starts from 1
        '''
        if iter <= 0:
            raise ValueError(f"Wrong itereration of {iter}.")
        
        cur_iter = epoch * self.iters_per_epoch + iter
        if cur_iter < self.warmup_iters:
            lr = cur_iter / self.warmup_iters * self.max_lr
        elif cur_iter <= self.cosine_cycle_iters:
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(((cur_iter - self.warmup_iters) / (self.cosine_cycle_iters - self.warmup_iters)) * math.pi))
        else:
            lr = self.min_lr
        return lr