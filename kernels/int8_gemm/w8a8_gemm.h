#pragma once

#include <torch/extension.h>


// W8A8 GEMM: int8 activation × int8 weight → float output
// D = alphaCol * alphaRow * (A @ B)
// A: [M, K] int8, RowMajor
// B: [N, K] int8, ColumnMajor (stored as [N, K] in row-major view)
// alphaCol: [M, 1] float - per-row activation scale
// alphaRow: [1, N] float - per-column weight scale
// D: [M, N] float
torch::Tensor matmul_w8a8(
    const torch::Tensor &A,        // [M, K] int8
    const torch::Tensor &B,        // [N, K] int8
    const torch::Tensor &alphaCol, // [M, 1] float
    const torch::Tensor &alphaRow  // [1, N] float
);