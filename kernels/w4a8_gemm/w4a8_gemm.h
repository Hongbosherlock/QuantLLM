#pragma once

#include <torch/extension.h>

// W4A8 GEMM: int8 activation × int4 weight → float output
// D = alphaCol * alphaRow * (A @ B)
// A: [M, K] int8, RowMajor
// B: [N, K] int4 packed as uint8, treated as ColumnMajor [K, N]
// alphaCol: [M, 1] float - per-row activation scale
// alphaRow: [1, N] float - per-column weight scale
// D: [M, N] float
torch::Tensor matmul_w4a8(
    const torch::Tensor &A,        // [M, K] int8
    const torch::Tensor &B,        // [N, K/2] uint8 (packed int4)
    const torch::Tensor &alphaCol, // [M, 1] float
    const torch::Tensor &alphaRow  // [1, N] float
);