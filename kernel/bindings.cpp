#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "fp8_blockwise/gemm_fp8.h"
#include <torch/all.h>

torch::Tensor fp8_blockwise_scaled_mm(torch::Tensor a, 
                                torch::Tensor b, 
                                torch::Tensor a_scales, 
                                torch::Tensor b_scales) {   
    auto acc_dtype = torch::kFloat16;
    auto options = torch::TensorOptions().dtype(acc_dtype).device(a.device());
    torch::Tensor out = torch::empty({a.size(0), b.size(1)}, options);

    fp8_blockwise_scaled_mm_sm90(out, a, b, a_scales, b_scales);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cutlass_scaled_mm", &cutlass_scaled_mm, "CUTLASS Scaled Matrix Multiplication");
}

