import argparse
import copy
import itertools
import pickle as pkl
import time
from typing import Callable, Iterable, List, Optional, Tuple
import nvtx

import torch
from utils import make_rand_tensors

from QuantLLM import fp8_blockwise_scaled_mm

def verify(result, ref_result):
    """
    test accuracy
    """
    diff = result - ref_result  
    # print(f"  Difference tensor: {diff}")
    abs_diff = torch.abs(diff)
    max_diff = torch.max(abs_diff)
    mean_diff = torch.mean(abs_diff.float()) 

    rel_diff = (torch.mean(
    torch.abs(result.to(torch.float32) - ref_result.to(torch.float32))) /
                torch.mean(torch.abs(ref_result.to(torch.float32))))
    print(f'. rel diff {rel_diff}')
    print(f"  Max difference: {max_diff}")
    print(f"  Mean difference: {mean_diff}")
 

def native_w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype=torch.float16):
    """This function performs matrix multiplication with block-wise quantization using native torch.

    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.
    """

    A = A.to(torch.float32)
    B = B.to(torch.float32)
    assert A.shape[-1] == B.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (N,)
    A = A.reshape(M, A.shape[-1])
    As = As.reshape(M, As.shape[-1])
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    assert n_tiles == Bs.shape[0]
    assert k_tiles == Bs.shape[1]

    C_shape = (M, N)
    C = torch.zeros(C_shape, dtype=torch.float32, device=A.device)

    A_tiles = [A[:, i * block_k : min((i + 1) * block_k, K)] for i in range(k_tiles)]
    B_tiles = [
        [
            B[
                j * block_n : min((j + 1) * block_n, N),
                i * block_k : min((i + 1) * block_k, K),
            ]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]
    C_tiles = [C[:, j * block_n : min((j + 1) * block_n, N)] for j in range(n_tiles)]
    As_tiles = [As[:, i : i + 1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            a = A_tiles[i]
            b = B_tiles[j][i]
            c = C_tiles[j]
            s = As_tiles[i] * Bs[j][i]
            c[:, :] += torch.matmul(a, b.t()) * s

    C = C.reshape(origin_C_shape).to(output_dtype)
    return C

def run(m, k, n):
    a, b = make_rand_tensors(torch.float8_e4m3fn, m, n, k)
    factor_for_scale = 1e-2
    print("a", a.shape)           # [128, 7168]
    print("b", b.shape)           # [7168, 2048]
    print("stride a", a.stride()) # stride a (7168, 1).  row 
    print("stride b", b.stride()) # stride b (1, 7168)   column

    a_cont = a.contiguous()

    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    block_scale_a = torch.rand((m, k // 128),
                               device="cuda",
                               dtype=torch.float32)
    block_scale_b = torch.rand((k // 128, n // 128),
                               device="cuda",
                               dtype=torch.float32)

    # block_scale_a, block_scale_b = make_rand_tensors(torch.float8_e4m3fn, m, n//128, k//128)

    print("block_scale_a", block_scale_a.shape)
    print("block_scale_b", block_scale_b.shape)
    print("stride scale_a", block_scale_a.stride())
    print("stride scale_b", block_scale_b.stride())

    block_scale_a_M_major = block_scale_a.t().contiguous().t()
    block_scale_b_K_major = block_scale_b.t().contiguous().t()
    bias = torch.zeros((n, ), device="cuda", dtype=torch.bfloat16)
    print("block_scale_a_M_major", block_scale_a_M_major.shape) # ([128, 56]
    print("block_scale_b_K_major", block_scale_b_K_major.shape) # ([56, 16])
    print("stride scale_a", block_scale_a_M_major.stride()) # stride scale_a (1, 128). column
    print("stride scale_b", block_scale_b_K_major.stride()) # stride scale_b (1, 56)   column
    print(m, k, n)
    
    # result_triton = w8a8_block_fp8_matmul(a_cont, b.t(), block_scale_a,
    #                                   block_scale_b.t(), (128, 128))
    result_pytorch = native_w8a8_block_fp8_matmul(a_cont, b.t(), block_scale_a,block_scale_b.t(), (128, 128))

    result_cutlass = fp8_blockwise_scaled_mm(a, b, block_scale_a_M_major, block_scale_b_K_major)
    verify(result_cutlass, result_pytorch)
 

if __name__ == '__main__':
    m = 128
    k = 7168
    n = 1024

    run(m, k, n)

