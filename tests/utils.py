"""
Cutlass bench utils
"""
from typing import Iterable, Tuple

import torch


def to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)


def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def to_bf16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.bfloat16)


def to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert input tensor to float16 format tensor.
    """
    return tensor.to(dtype=torch.float16)


def make_rand_tensors(dtype: torch.dtype, m: int, n: int,
                      k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make two random tensors of shape (m, k) and (n, k) with dtype.
    """
    a = torch.randn((m, k), device='cuda') * 5
    b = torch.randn((n, k), device='cuda').t() * 5

    if dtype == torch.int8:
        return to_int8(a), to_int8(b)
    if dtype == torch.float8_e4m3fn:
        return to_fp8(a), to_fp8(b)

    raise ValueError("unsupported dtype")


def prune_to_2_4(tensor):
    """
    apply pruning to tensor to keep top 2 absolute values in each group of 4
    """
    # Reshape tensor to [N, 4] where N is number of groups of 4
    original_shape = tensor.shape
    reshaped = tensor.reshape(-1, 4)

    # Get indices of top 2 absolute values in each group of 4
    _, indices = torch.topk(torch.abs(reshaped), k=2, dim=1)

    # Create binary mask
    mask = torch.zeros_like(reshaped)
    mask.scatter_(dim=1,
                  index=indices,
                  src=torch.ones_like(indices, dtype=mask.dtype))

    # Apply mask and reshape back
    pruned = reshaped * mask

    # Turn all -0.0 to 0.0
    pruned[pruned == -0.0] = 0.0

    return pruned.reshape(original_shape)


