from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

cutlass_path = os.path.join(os.path.dirname(__file__), 'submodules/cutlass')

setup(
    name='cutlass_gemm',
    ext_modules=[
        CUDAExtension(
            name='QuantLLM',
            sources=['kernels/bindings.cpp', 
                     'kernels/fp8_blockwise/gemm_fp8.cu'],
            extra_compile_args={
                'nvcc': [
                    '-DNDEBUG',
                    '-O3', 
                    '-g', 
                    '-lineinfo',
                    '--keep', 
                    '--ptxas-options=--warn-on-local-memory-usage',
                    '--ptxas-options=--warn-on-spills',
                    '--resource-usage',
                    '--source-in-ptx',s
                    '-gencode=arch=compute_90a, code=sm_90a',
                ]
            },
            include_dirs=[
                os.path.join(cutlass_path, 'include'),
                os.path.join(cutlass_path, 'tools/util/include'),
                'kernels/cutlass_extensions',
            ],
            libraries=['cuda'],
            library_dirs=['/usr/local/cuda/lib64'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)