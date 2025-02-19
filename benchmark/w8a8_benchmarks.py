import argparse
import copy
import itertools
import pickle as pkl
import time
from typing import Callable, Iterable, List, Optional, Tuple
import nvtx
import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from utils import make_rand_tensors
# from weight_shapes import WEIGHT_SHAPES

# from vllm import _custom_ops as ops
# from vllm.model_executor.layers.quantization.utils.fp8_utils import (
#     w8a8_block_fp8_matmul)
# from vllm.utils import FlexibleArgumentParser
from QuantLLM import fp8_blockwise_scaled_mm

# DEFAULT_MODELS = list(WEIGHT_SHAPES.keys())
# DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512]
DEFAULT_BATCH_SIZES = [128]

DEFAULT_TP_SIZES = [1]


# bench
def bench_fn(label: str, sub_label: str, description: str, fn: Callable, *args,
             **kwargs) -> TMeasurement:
    min_run_time = 1

    globals = {
        "args": args,
        "kwargs": kwargs,
        "fn": fn,
    }
    return TBenchmark.Timer(
        stmt="fn(*args, **kwargs)",
        globals=globals,
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)


def bench_int8(
        dtype: torch.dtype,
        m: int,
        k: int,
        n: int,
        label: str,
        sub_label: str,
        bench_kernels: Optional[List[str]] = None) -> Iterable[TMeasurement]:
    """Benchmark INT8-based kernels."""
    assert dtype == torch.int8
    a, b = make_rand_tensors(torch.int8, m, n, k)
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    bias = torch.zeros((n, ), device="cuda", dtype=torch.bfloat16)
    azp = torch.zeros((m, ), device="cuda", dtype=torch.int32)
    azp_adj = torch.zeros((n, ), device="cuda", dtype=torch.int32)

    bench_fns = {
        "pytorch_bf16_bf16_bf16_matmul-no-scales":
        lambda: torch.mm(a.to(dtype=torch.bfloat16), b.to(dtype=torch.bfloat16)
                         ),
        "pytorch_fp16_fp16_fp16_matmul-no-scales":
        lambda: torch.mm(a.to(dtype=torch.float16), b.to(dtype=torch.float16)),
        "cutlass_i8_i8_bf16_scaled_mm":
        lambda: ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16),
        "cutlass_i8_i8_bf16_scaled_mm_bias":
        lambda: ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16,
                                      bias),
        "cutlass_i8_i8_bf16_scaled_mm_azp":
        lambda: ops.cutlass_scaled_mm_azp(a, b, scale_a, scale_b, torch.
                                          bfloat16, azp_adj),
        "cutlass_i8_i8_bf16_scaled_mm_azp_bias":
        lambda: ops.cutlass_scaled_mm_azp(a, b, scale_a, scale_b, torch.
                                          bfloat16, azp_adj, None, bias),
        "cutlass_i8_i8_bf16_scaled_mm_azp_pt":
        lambda: ops.cutlass_scaled_mm_azp(a, b, scale_a, scale_b, torch.
                                          bfloat16, azp_adj, azp),
        "cutlass_i8_i8_bf16_scaled_mm_azp_pt_bias":
        lambda: ops.cutlass_scaled_mm_azp(a, b, scale_a, scale_b, torch.
                                          bfloat16, azp_adj, azp, bias),
    }

    timers = []
    for name, fn in bench_fns.items():
        # If bench_kernels is None, run all. Otherwise, run only exact matches.
        if bench_kernels is None or name in bench_kernels:
            print(f"Running {name}")
            timers.append(bench_fn(label, sub_label, name, fn))

    return timers


def bench_fp8(
        dtype: torch.dtype,
        m: int,
        k: int,
        n: int,
        label: str,
        sub_label: str,
        bench_kernels: Optional[List[str]] = None) -> Iterable[TMeasurement]:
    """Benchmark FP8-based kernels."""
    assert dtype == torch.float8_e4m3fn
    a, b = make_rand_tensors(torch.float8_e4m3fn, m, n, k)
    # print("a",a.shape)
    # print("b",b.shape)
    factor_for_scale = 1e-2
    # fp8_info = torch.finfo(torch.float8_e4m3fn)
    # fp8_max, fp8_min = fp8_info.max, fp8_info.min

    # A_fp32 = (torch.rand(m, k, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    # a = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    # B_fp32 = (torch.rand(k, n, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    # b = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    print("a",a.shape)
    print("b",b.shape)
    # M128-k7168-n576
    # 576/128=4.5
    # exit(0)
    a_cont = a.contiguous()
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    block_scale_a = torch.rand((m, k // 128),
                               device="cuda",
                               dtype=torch.float32)
    block_scale_b = torch.rand((k // 128, n // 128),
                               device="cuda",
                               dtype=torch.float32)
    # block_n, block_k = 128,128
    # n_tiles = (n + block_n - 1) // block_n
    # k_tiles = (k + block_k - 1) // block_k

    # block_scale_a = torch.rand(m, k_tiles, dtype=torch.float32, device="cuda") * factor_for_scale
    # block_scale_b = (
    #     torch.rand(k_tiles, n_tiles, dtype=torch.float32, device="cuda")
    #     * factor_for_scale
    # )
    print("block_scale_a",block_scale_a.shape)
    print("block_scale_b",block_scale_b.shape)
    print("stride a",block_scale_a.stride())
    print("stride b",block_scale_b.stride())

    block_scale_a_M_major = block_scale_a.t().contiguous().t()
    block_scale_b_K_major = block_scale_b.t().contiguous().t()
    bias = torch.zeros((n, ), device="cuda", dtype=torch.bfloat16)
    print("block_scale_a_M_major",block_scale_a_M_major.shape)
    print("block_scale_b_K_major",block_scale_b_K_major.shape)
    print("stride a",block_scale_a_M_major.stride())
    print("stride b",block_scale_a_M_major.stride())
    print(m, k, n)
    workspace = torch.zeros(100, dtype=torch.uint8, device="cuda")

    # exit(0)
    bench_fns = {
        "pytorch_bf16_bf16_bf16_matmul-no-scales":
        lambda: torch.mm(a.to(dtype=torch.bfloat16), b.to(dtype=torch.bfloat16)
                         ),
        "pytorch_fp16_fp16_fp16_matmul-no-scales":
        lambda: torch.mm(a.to(dtype=torch.float16), b.to(dtype=torch.float16)),
        "pytorch_fp8_fp8_fp16_scaled_mm":
        lambda: torch._scaled_mm(
            a, b, scale_a, scale_b, out_dtype=torch.float16),
        "pytorch_fp8_fp8_fp16_scaled_mm_fast_accum":
        lambda: torch._scaled_mm(a,
                                 b,
                                 scale_a,
                                 scale_b,
                                 out_dtype=torch.float16,
                                 use_fast_accum=True),
        "pytorch_fp8_fp8_bf16_scaled_mm":
        lambda: torch._scaled_mm(
            a, b, scale_a, scale_b, out_dtype=torch.bfloat16),
        "pytorch_fp8_fp8_bf16_scaled_mm_fast_accum":
        lambda: torch._scaled_mm(a,
                                 b,
                                 scale_a,
                                 scale_b,
                                 out_dtype=torch.bfloat16,
                                 use_fast_accum=True),
        "cutlass_fp8_fp8_bf16_scaled_mm":
        lambda: ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16),
        "cutlass_fp8_fp8_fp16_scaled_mm":
        lambda: ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.float16),
        "cutlass_fp8_fp8_bf16_scaled_mm_bias":
        lambda: ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16,
                                      bias),
        "cutlass_fp8_fp8_fp16_scaled_mm_bias":
        lambda: ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.float16,
                                      bias.to(dtype=torch.float16)),
        "triton_fp8_fp8_fp16_scaled_mm_blockwise":
        lambda: w8a8_block_fp8_matmul(a_cont, b.t(), block_scale_a,
                                      block_scale_b.t(), (128, 128)),
        "cutlass_fp8_fp8_fp16_scaled_mm_blockwise":
        lambda: fp8_blockwise_scaled_mm(a, b, block_scale_a_M_major,
                                      block_scale_b_K_major),
        # lambda: ops.cutlass_scaled_mm(a, b, block_scale_a_M_major,
        #                               block_scale_b_K_major, torch.float16),
    }

    bench_kernels=['cutlass_fp8_fp8_fp16_scaled_mm_blockwise']
    timers = []
    for name, fn in bench_fns.items():
        if bench_kernels is None or name in bench_kernels:
            print(f"Running {name}")
            # with nvtx.annotate(f"{name}-mkn", color="yellow"):
            #     for _ in range(2):
            #         out = fn()
                    # print(out.shape)
            timers.append(bench_fn(label, sub_label, name, fn))
            # result = fn()  # 调用lambda函数，获得矩阵乘法结果


    return timers


def bench(dtype: torch.dtype,
          m: int,
          k: int,
          n: int,
          label: str,
          sub_label: str,
          bench_kernels: Optional[List[str]] = None) -> Iterable[TMeasurement]:
    if dtype == torch.int8:
        return bench_int8(dtype, m, k, n, label, sub_label, bench_kernels)
    if dtype == torch.float8_e4m3fn:
        return bench_fp8(dtype, m, k, n, label, sub_label, bench_kernels)
    raise ValueError("unsupported type")


# runner
def print_timers(timers: Iterable[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()


def run(dtype: torch.dtype,
        MKNs: Iterable[Tuple[int, int, int]],
        bench_kernels: Optional[List[str]] = None) -> Iterable[TMeasurement]:
    results = []
    for m, k, n in MKNs:
        timers = bench(dtype,
                       m,
                       k,
                       n,
                       f"scaled-{dtype}-gemm",
                       f"MKN=({m}x{k}x{n})",
                       bench_kernels=bench_kernels)
        print_timers(timers)
        results.extend(timers)
    return results


def make_output(data: Iterable[TMeasurement],
                MKNs: Iterable[Tuple[int, int, int]],
                base_description: str,
                timestamp=None):
    print(f"== All Results {base_description} ====")
    print_timers(data)

    # pickle all the results
    timestamp = int(time.time()) if timestamp is None else timestamp
    with open(f"{base_description}-{timestamp}.pkl", "wb") as f:
        pkl.dump(data, f)


def run_square_bench(args):
    # data = [
    # "m256-k7168-n1536",
    # "m256-k7168-n1024",
    # "m256-k7168-n2048",
    # ]
    data = [
    "m128-k7168-n1536",
    "m128-k1536-n6144",
    "m128-k7168-n640", #576
    "m128-k4096-n7168",
    "m128-k7168-n9216",
    "m128-k4608-n7168",
    "m128-k7168-n1024",
    "m128-k512-n7168",
    "m256-k7168-n4096",
    "m256-k7168-n2048",
    ]
    # "m256-k7168-n2048",

    MKNs = []
    for item in data:
        parts = item.split("-")
        m = int(parts[0][1:])  # 提取 M 的值
        k = int(parts[1][1:])  # 提取 K 的值
        n = int(parts[2][1:])  # 提取 N 的值
        MKNs.append((m, k, n))
    print(MKNs)
    # dim_sizes = list(
    #     range(args.dim_start, args.dim_end + 1, args.dim_increment))
    # MKNs = list(zip(dim_sizes, dim_sizes, dim_sizes))
    # MKNs = [(128, 128, 128), (192, 192, 192), (256, 256, 256), (320, 320, 320), (384, 384, 384), (448, 448, 448), (512, 512, 512)]
    # print(MKNs)
    # MNK
    # print(MKNs.dtype)
    # exit(0)
    data = run(args.dtype, MKNs, bench_kernels=args.kernels)
    make_output(data, MKNs, f"square_bench-{args.dtype}")


def run_range_bench(args):
    dim_sizes = list(range(args.dim_start, args.dim_end, args.dim_increment))
    n = len(dim_sizes)
    Ms = [args.m_constant] * n if args.m_constant is not None else dim_sizes
    Ks = [args.k_constant] * n if args.k_constant is not None else dim_sizes
    Ns = [args.n_constant] * n if args.n_constant is not None else dim_sizes
    MKNs = list(zip(Ms, Ks, Ns))
    print(MKNs)
    print(MKNs.dtype)
    data = run(args.dtype, MKNs, bench_kernels=args.kernels)
    make_output(data, MKNs, f"range_bench-{args.dtype}")


def run_model_bench(args):
    print("Benchmarking models:")
    for i, model in enumerate(args.models):
        print(f"[{i}]  {model}")

    def model_shapes(model_name: str, tp_size: int) -> List[Tuple[int, int]]:
        KNs = []
        # for KN, tp_split_dim in copy.deepcopy(WEIGHT_SHAPES[model_name]):
        #     KN[tp_split_dim] = KN[tp_split_dim] // tp_size
        #     KNs.append(KN)
        return KNs

    model_bench_data = []
    models_tps = list(itertools.product(args.models, args.tp_sizes))
    for model, tp_size in models_tps:
        Ms = args.batch_sizes
        KNs = model_shapes(model, tp_size)
        MKNs = []
        for m in Ms:
            for k, n in KNs:
                MKNs.append((m, k, n))
        print(MKNs)
        data = run(args.dtype, MKNs, bench_kernels=args.kernels)
        model_bench_data.append(data)

    # Print all results
    for data, model_tp in zip(model_bench_data, models_tps):
        model, tp_size = model_tp
        print(f"== Results {args.dtype} {model}-TP{tp_size} ====")
        print_timers(data)

    timestamp = int(time.time())

    all_data = []
    for d in model_bench_data:
        all_data.extend(d)
    # pickle all data
    with open(f"model_bench-{args.dtype}-{timestamp}.pkl", "wb") as f:
        pkl.dump(all_data, f)


if __name__ == '__main__':

    def to_torch_dtype(dt):
        if dt == "int8":
            return torch.int8
        if dt == "fp8":
            return torch.float8_e4m3fn
        raise ValueError("unsupported dtype")

        # Parser setup with detailed help description
        parser = argparse.ArgumentParser(
            description="""Benchmark Cutlass GEMM.

            To run square GEMMs:
                python3 ./benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 square_bench --dim-start 128 --dim-end 512 --dim-increment 64
            
            To run constant N and K and sweep M:
                python3 ./benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 range_bench --dim-start 128 --dim-end 512 --dim-increment 64 --n-constant 16384 --k-constant 16384
            
            To run dimensions from a model:
                python3 ./benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 model_bench --models meta-llama/Llama-2-7b-hf --batch-sizes 16 --tp-sizes 1
            
            Output:
                - a .pkl file, containing a list of raw torch.benchmark.utils.Measurements for both pytorch and cutlass implementations across different GEMMs.
            """,
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Argument for dtype
        parser.add_argument(
            "--dtype",
            type=to_torch_dtype,
            required=True,
            help="Data type to benchmark. Available options are ['int8', 'fp8']",
        )

        # Argument for specifying kernels
        parser.add_argument(
            "--kernels",
            nargs="+",
            type=str,
            default=None,
            help="Exact names of the kernels to benchmark. If not set, runs all kernels.",
        )

        # Subparsers for different benchmark modes
        subparsers = parser.add_subparsers(dest="cmd")

        # Square Benchmark Configuration
        square_parser = subparsers.add_parser("square_bench")
        square_parser.add_argument("--dim-start", type=int, required=True, help="Start dimension")
        square_parser.add_argument("--dim-end", type=int, required=True, help="End dimension")
        square_parser.add_argument("--dim-increment", type=int, required=True, help="Increment size for dimensions")
        square_parser.set_defaults(func=run_square_bench)

        # Range Benchmark Configuration
        range_parser = subparsers.add_parser("range_bench")
        range_parser.add_argument("--dim-start", type=int, required=True, help="Start dimension")
        range_parser.add_argument("--dim-end", type=int, required=True, help="End dimension")
        range_parser.add_argument("--dim-increment", type=int, required=True, help="Increment size for dimensions")
        range_parser.add_argument("--m-constant", type=int, default=None, help="Constant M value")
        range_parser.add_argument("--n-constant", type=int, default=None, help="Constant N value")
        range_parser.add_argument("--k-constant", type=int, default=None, help="Constant K value")
        range_parser.set_defaults(func=run_range_bench)

        # Model Benchmark Configuration
        model_parser = subparsers.add_parser("model_bench")
        model_parser.add_argument(
            "--tp-sizes",
            nargs="+",
            type=int,
            default=DEFAULT_TP_SIZES,
            help="Tensor parallelism sizes (default: [1, 2, 4])",
        )
        model_parser.add_argument(
            "--batch-sizes",
            nargs="+",
            type=int,
            default=DEFAULT_BATCH_SIZES,
            help="Batch sizes to benchmark (default: [8, 16, 32])",
        )
        model_parser.set_defaults(func=run_model_bench)

        args = parser.parse_args()
        args.func(args)