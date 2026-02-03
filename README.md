# QuantLLM

**Quantization Kernel Library for LLM Inference**

QuantLLM is a high-performance library designed to accelerate large language model (LLM) inference through the use of quantization techniques. Built on top of Cutlass, this library provides optimized quantization kernels that significantly reduce memory usage and computational costs without sacrificing model accuracy.

## Features

- **Efficient Quantization**: Implements state-of-the-art quantization algorithms to minimize model size and enhance inference speed.
- **High Performance**: Optimized for GPU acceleration with Cutlass, ensuring that the quantized kernels are fast and scalable.
- **Support for LLM Inference**: Tailored specifically for large language models, enabling efficient inference on large-scale tasks.

# Supported Quantization Methods

| Quantization Method | Supported |
|---------------------|-----------|
| W8A8-INT8           |   ✔        |
| W4A8-INT8           |   ✔        |
| W8A8-FP8-blockwise  |   ✔       |
| W8A8-FP8            |  ⬜       |
| W8A16               |  ⬜       |
| W4A16               |  ⬜       |


## Dependencies
- CUTLASS
- PyTorch with CUDA 12.4
- NVIDIA-Toolkit 12.4
- CUDA Driver 12.4
- gcc g++ 11.4
- cmake >= 3.28.6

## Installation

To install QuantLLM, clone this repository and build it using CMake:

```bash
git clone --recurse-submodules https://github.com/Hongbosherlock/QuantLLM.git
cd QuantLLM
source env.sh
bash build_cutlass.sh
python setup.py install

