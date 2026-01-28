import os

import torch
from torch import nn


def logger(info: str):
    print(info)

def save_checkpoint(model: nn.Module | nn.parallel.DistributedDataParallel,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike):
    state_dict = {}
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    opt_state_dict = optimizer.state_dict()
    
    state_dict["model_state_dict"] = model_state_dict
    state_dict["opt_state_dict"] = opt_state_dict
    state_dict["iter"] = iteration
    torch.save(state_dict, out)