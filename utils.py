import operator
from typing import NamedTuple

import torch
import torch.nn as nn

def l2_norm(parameters):
    grad_norm = 0
    for n, p in parameters:
        if "bias" in n:
            continue
        grad_norm += p.norm().cpu()
    return grad_norm

def get_grad_norm(parameters):
    grad_norm = torch.tensor(0.)
    for p in parameters:
        if p.grad is not None:
            grad_norm += p.grad.norm().cpu()
    return grad_norm
