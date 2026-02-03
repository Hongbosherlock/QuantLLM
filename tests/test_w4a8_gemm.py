#!/usr/bin/env python3
"""
Benchmark script for W4A8 and W8A8 GEMM kernels.

This script tests:
1. Correctness - compare against PyTorch reference
2. Performance - measure throughput in TFLOPS

Usage:
    python benchmark_gemm.py                    # Run all benchmarks
    python benchmark_gemm.py --warmup 10        # Custom warmup iterations
    python benchmark_gemm.py --repeat 100       # Custom benchmark iterations
    python benchmark_gemm.py --sizes "1024,2048,4096"  # Custom M sizes
"""

import argparse
import time
import torch
import numpy as np

try:
    import cutlass_gemm
except ImportError:
    print("Error: cutlass_gemm module not found!")
    print("Please build the extension first:")
    print("  python setup_gemm.py build_ext --inplace")
    print("Or install it:")
    print("  python setup_gemm.py install")
    exit(1)


def quantize_to_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize float tensor to int8 with per-row scaling."""
    # Per-row absmax scaling
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    scale = absmax / 127.0
    x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale


def quantize_to_int4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize float tensor to int4 (packed as uint8) with per-row scaling."""
    # Per-row absmax scaling
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    scale = absmax / 7.0  # int4 range is -8 to 7
    x_int4 = (x / scale).round().clamp(-8, 7).to(torch.int8)
    
    # Pack two int4 values into one uint8
    # x_int4 shape: [N, K], we pack along K dimension
    N, K = x_int4.shape
    assert K % 2 == 0, "K must be even for int4 packing"
    
    x_int4_reshaped = x_int4.view(N, K // 2, 2)
    # Pack: low nibble = first value, high nibble = second value
    low = (x_int4_reshaped[..., 0] & 0x0F).to(torch.uint8)
    high = ((x_int4_reshaped[..., 1] & 0x0F) << 4).to(torch.uint8)
    x_packed = (low | high).to(torch.uint8)
    
    return x_packed, scale


def reference_matmul(A_int: torch.Tensor, B_int: torch.Tensor, 
                     alpha_col: torch.Tensor, alpha_row: torch.Tensor,
                     is_int4: bool = False) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    # Dequantize and compute
    A_float = A_int.float()
    
    if is_int4:
        # Unpack int4
        N, K_packed = B_int.shape
        low = (B_int & 0x0F).to(torch.int8)
        # Sign extend the 4-bit values
        low = torch.where(low > 7, low - 16, low)
        high = ((B_int >> 4) & 0x0F).to(torch.int8)
        high = torch.where(high > 7, high - 16, high)
        B_unpacked = torch.stack([low, high], dim=-1).view(N, K_packed * 2)
        B_float = B_unpacked.float()
    else:
        B_float = B_int.float()
    
    # Compute: D = alphaCol * alphaRow * (A @ B^T)
    result = A_float @ B_float.T
    result = result * alpha_col * alpha_row.T
    return result


def benchmark_kernel(func, args, warmup=10, repeat=100):
    """Benchmark a CUDA kernel."""
    # Warmup
    for _ in range(warmup):
        _ = func(*args)
    torch.cuda.synchronize()
    
    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    
    for i in range(repeat):
        start_events[i].record()
        _ = func(*args)
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return np.mean(times), np.std(times), np.min(times), np.max(times)


def compute_tflops(M, N, K, time_ms, is_int4=False):
    """Compute TFLOPS for GEMM operation."""
    # GEMM has 2*M*N*K FLOPs (multiply-add)
    flops = 2.0 * M * N * K
    tflops = flops / (time_ms * 1e-3) / 1e12
    return tflops


def test_correctness(M, N, K, device='cuda'):
    """Test correctness of W4A8 and W8A8 kernels."""
    print(f"\n{'='*60}")
    print(f"Correctness Test: M={M}, N={N}, K={K}")
    print('='*60)
    
    # Generate random test data
    torch.manual_seed(42)
    A_float = torch.randn(M, K, device=device)
    B_float = torch.randn(N, K, device=device)
    
    # Quantize
    A_int8, alpha_col = quantize_to_int8(A_float)
    B_int8, alpha_row_8 = quantize_to_int8(B_float)
    B_int4, alpha_row_4 = quantize_to_int4(B_float)
    
    # Reshape scales for broadcasting
    alpha_col = alpha_col.view(M, 1).contiguous()
    alpha_row_8 = alpha_row_8.view(1, N).contiguous()
    alpha_row_4 = alpha_row_4.view(1, N).contiguous()
    
    # Test W8A8
    print("\n[W8A8 GEMM]")
    try:
        result_w8a8 = cutlass_gemm.matmul_w8a8(A_int8, B_int8, alpha_col, alpha_row_8)
        ref_w8a8 = reference_matmul(A_int8, B_int8, alpha_col, alpha_row_8, is_int4=False)
        
        # Compare results
        diff_w8a8 = (result_w8a8 - ref_w8a8).abs()
        max_diff = diff_w8a8.max().item()
        mean_diff = diff_w8a8.mean().item()
        rel_diff = (diff_w8a8 / (ref_w8a8.abs() + 1e-6)).mean().item()
        
        print(f"  Max absolute diff: {max_diff:.6f}")
        print(f"  Mean absolute diff: {mean_diff:.6f}")
        print(f"  Mean relative diff: {rel_diff:.6f}")
        print(f"  Result: {'PASS' if max_diff < 1.0 else 'FAIL'}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test W4A8
    print("\n[W4A8 GEMM]")
    try:
        result_w4a8 = cutlass_gemm.matmul_w4a8(A_int8, B_int4, alpha_col, alpha_row_4)
        ref_w4a8 = reference_matmul(A_int8, B_int4, alpha_col, alpha_row_4, is_int4=True)
        
        # Compare results
        diff_w4a8 = (result_w4a8 - ref_w4a8).abs()
        max_diff = diff_w4a8.max().item()
        mean_diff = diff_w4a8.mean().item()
        rel_diff = (diff_w4a8 / (ref_w4a8.abs() + 1e-6)).mean().item()
        
        print(f"  Max absolute diff: {max_diff:.6f}")
        print(f"  Mean absolute diff: {mean_diff:.6f}")
        print(f"  Mean relative diff: {rel_diff:.6f}")
        print(f"  Result: {'PASS' if max_diff < 1.0 else 'FAIL'}")
    except Exception as e:
        print(f"  Error: {e}")


def run_benchmarks(M_sizes, N, K, warmup=10, repeat=100, device='cuda'):
    """Run performance benchmarks for various M sizes."""
    print(f"\n{'='*60}")
    print(f"Performance Benchmark: N={N}, K={K}")
    print(f"Warmup={warmup}, Repeat={repeat}")
    print('='*60)
    
    results = []
    
    print(f"\n{'M':>8} | {'Kernel':>10} | {'Time(ms)':>10} | {'Std(ms)':>8} | {'TFLOPS':>8}")
    print('-' * 60)
    
    for M in M_sizes:
        # Generate test data
        torch.manual_seed(42)
        A_float = torch.randn(M, K, device=device)
        B_float = torch.randn(N, K, device=device)
        
        # Quantize
        A_int8, alpha_col = quantize_to_int8(A_float)
        B_int8, alpha_row_8 = quantize_to_int8(B_float)
        B_int4, alpha_row_4 = quantize_to_int4(B_float)
        
        # Reshape scales
        alpha_col = alpha_col.view(M, 1).contiguous()
        alpha_row_8 = alpha_row_8.view(1, N).contiguous()
        alpha_row_4 = alpha_row_4.view(1, N).contiguous()
        
        # Benchmark W8A8
        try:
            mean_ms, std_ms, min_ms, max_ms = benchmark_kernel(
                cutlass_gemm.matmul_w8a8,
                (A_int8, B_int8, alpha_col, alpha_row_8),
                warmup=warmup, repeat=repeat
            )
            tflops = compute_tflops(M, N, K, mean_ms)
            print(f"{M:>8} | {'W8A8':>10} | {mean_ms:>10.3f} | {std_ms:>8.3f} | {tflops:>8.2f}")
            results.append({'M': M, 'kernel': 'W8A8', 'time_ms': mean_ms, 'tflops': tflops})
        except Exception as e:
            print(f"{M:>8} | {'W8A8':>10} | {'ERROR':>10} | {str(e)[:20]}")
        
        # Benchmark W4A8
        try:
            mean_ms, std_ms, min_ms, max_ms = benchmark_kernel(
                cutlass_gemm.matmul_w4a8,
                (A_int8, B_int4, alpha_col, alpha_row_4),
                warmup=warmup, repeat=repeat
            )
            tflops = compute_tflops(M, N, K, mean_ms, is_int4=True)
            print(f"{M:>8} | {'W4A8':>10} | {mean_ms:>10.3f} | {std_ms:>8.3f} | {tflops:>8.2f}")
            results.append({'M': M, 'kernel': 'W4A8', 'time_ms': mean_ms, 'tflops': tflops})
        except Exception as e:
            print(f"{M:>8} | {'W4A8':>10} | {'ERROR':>10} | {str(e)[:20]}")
        
        # Also benchmark PyTorch FP16 as reference
        try:
            A_fp16 = A_float.half()
            B_fp16 = B_float.half()
            mean_ms, std_ms, _, _ = benchmark_kernel(
                lambda a, b: torch.mm(a, b.T),
                (A_fp16, B_fp16),
                warmup=warmup, repeat=repeat
            )
            tflops = compute_tflops(M, N, K, mean_ms)
            print(f"{M:>8} | {'PyTorch':>10} | {mean_ms:>10.3f} | {std_ms:>8.3f} | {tflops:>8.2f}")
            results.append({'M': M, 'kernel': 'PyTorch_FP16', 'time_ms': mean_ms, 'tflops': tflops})
        except Exception as e:
            print(f"{M:>8} | {'PyTorch':>10} | {'ERROR':>10} | {str(e)[:20]}")
        
        print('-' * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark W4A8 and W8A8 GEMM kernels')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--repeat', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--sizes', type=str, default='1,32,64,128,256,512,1024,2048,4096',
                        help='Comma-separated M sizes')
    parser.add_argument('--N', type=int, default=4096, help='N dimension (output features)')
    parser.add_argument('--K', type=int, default=4096, help='K dimension (hidden size)')
    parser.add_argument('--correctness-only', action='store_true', help='Only run correctness tests')
    parser.add_argument('--benchmark-only', action='store_true', help='Only run performance benchmarks')
    args = parser.parse_args()
    
    # Parse M sizes
    M_sizes = [int(x) for x in args.sizes.split(',')]
    
    # Print GPU info
    print("="*60)
    print("GPU Information")
    print("="*60)
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA not available!")
        return
    
    # Run correctness tests
    if not args.benchmark_only:
        test_correctness(128, args.N, args.K)
    
    # Run benchmarks
    if not args.correctness_only:
        run_benchmarks(M_sizes, args.N, args.K, 
                      warmup=args.warmup, repeat=args.repeat)
    
    print("\nBenchmark complete!")


if __name__ == '__main__':
    main()