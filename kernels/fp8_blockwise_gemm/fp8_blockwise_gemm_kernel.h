#include <torch/extension.h>

void fp8_blockwise_scaled_mm_sm90(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
                            torch::Tensor const& a_scales, torch::Tensor const& b_scales);