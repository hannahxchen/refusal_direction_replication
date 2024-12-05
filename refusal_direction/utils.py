import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
from einops import rearrange
from torchtyping import TensorType


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def ceildiv(a, b):
    return -(a // -b)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def masked_mean(seq, mask = None, dim = 1, keepdim = False):
    if mask is None:
        return seq.mean(dim = dim)

    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    masked_seq = seq.masked_fill(~mask, 0.)
    numer = masked_seq.sum(dim = dim, keepdim = keepdim)
    denom = mask.sum(dim = dim, keepdim = keepdim)

    masked_mean = numer / denom.clamp(min = 1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.)
    return masked_mean


def kl_div_fn(
    logits_a: TensorType["batch_size", "seq_len", "vocab_size"],
    logits_b: TensorType["batch_size", "seq_len", "vocab_size"],
    mask: Optional[TensorType[int, "batch_size", "seq_len"]] = None,
    epsilon: float=1e-6
) -> TensorType['batch_size']:
    """Compute the KL divergence loss between two tensors of logits."""
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)

    probs_a = logits_a.softmax(dim=-1)
    probs_b = logits_b.softmax(dim=-1)

    kl_divs = torch.sum(probs_a * (torch.log(probs_a + epsilon) - torch.log(probs_b + epsilon)), dim=-1)

    if mask is None:
        return torch.mean(kl_divs, dim=-1)
    else:
        return masked_mean(kl_divs, mask).mean(dim=-1)


def orthogonal_rejection(activation: TensorType[..., -1], unit_direction: TensorType[-1]) -> TensorType[..., -1]:
    ablated_act = activation - (activation @ unit_direction).unsqueeze(-1) * unit_direction
    return ablated_act


def save_to_json_file(filepath: Path, results: List[Dict]):
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)