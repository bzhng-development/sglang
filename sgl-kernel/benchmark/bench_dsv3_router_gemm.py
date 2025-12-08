import argparse
import os

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

import torch
import torch.nn.functional as F
import triton
import triton.testing
from sgl_kernel import dsv3_router_gemm

from sglang.srt.utils.common import is_sm100_supported

# FlashInfer DSv3 router GEMM (SM100 only, M in [1, 16])
HAS_FLASHINFER = False
if is_sm100_supported():
    from flashinfer.dsv3_ops import mm_M1_16_K7168_N256

    HAS_FLASHINFER = True

# CI environment uses simplified parameters
if IS_CI:
    num_tokens_vals = [1]  # Only test 1 value in CI
    line_vals_bf16 = ["sgl-kernel-256"]  # Only test one implementation in CI
    line_vals_float = ["sgl-kernel-256"]
else:
    num_tokens_vals = [i + 1 for i in range(16)]  # Test 1-16 in full mode
    line_vals_bf16 = ["torch-256", "sgl-kernel-256", "torch-384", "sgl-kernel-384"]
    line_vals_float = [
        "torch-256",
        "sgl-kernel-256",
        "torch-384",
        "sgl-kernel-384",
    ]
    if HAS_FLASHINFER:
        line_vals_float += ["flashinfer-256", "flashinfer-384"]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=num_tokens_vals,
        x_log=False,
        line_arg="impl",
        line_vals=line_vals_bf16,
        line_names=(
            [
                "torch-256",
                "dsv3_router_gemm-256",
                "torch-384",
                "dsv3_router_gemm-384",
            ]
            if not IS_CI
            else ["dsv3_router_gemm-256"]
        ),
        styles=(
            [("blue", "-"), ("orange", "-"), ("green", "-"), ("red", "-")]
            if not IS_CI
            else [("orange", "-")]
        ),
        ylabel="TFLOPs",
        plot_name="input-bf16-output-bf16 dsv3 router gemm throughput",
        args={},
    )
)
def benchmark_bf16_output(num_tokens, impl):
    # M: num_tokens, K: hidden_dim, N: num_experts
    M, K = num_tokens, 7168

    if impl == "torch-256" or impl == "sgl-kernel-256":
        N = 256
    elif impl == "torch-384" or impl == "sgl-kernel-384":
        N = 384
    else:
        raise ValueError(f"Unknown impl: {impl}")

    mat_a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda").contiguous()
    mat_b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda").contiguous()

    quantiles = [0.5, 0.2, 0.8]

    if impl == "torch-256" or impl == "torch-384":

        def runner():
            F.linear(mat_a, mat_b)

    elif impl == "sgl-kernel-256" or impl == "sgl-kernel-384":

        def runner():
            dsv3_router_gemm(mat_a, mat_b, out_dtype=torch.bfloat16)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(runner, quantiles=quantiles)

    def tflops(t_ms):
        flops = 2 * M * K * N
        return flops / (t_ms * 1e-3) / 1e12

    return tflops(ms), tflops(max_ms), tflops(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=num_tokens_vals,
        x_log=False,
        line_arg="impl",
        line_vals=line_vals_float,
        line_names=(
            [
                "torch-256",
                "dsv3_router_gemm-256",
                "torch-384",
                "dsv3_router_gemm-384",
                "flashinfer-256",
                "flashinfer-384",
            ]
            if (not IS_CI and HAS_FLASHINFER)
            else (
                [
                    "torch-256",
                    "dsv3_router_gemm-256",
                    "torch-384",
                    "dsv3_router_gemm-384",
                ]
                if not IS_CI
                else ["dsv3_router_gemm-256"]
            )
        ),
        styles=(
            [
                ("blue", "-"),
                ("orange", "-"),
                ("green", "-"),
                ("red", "-"),
                ("purple", "-"),
                ("brown", "-"),
            ]
            if (not IS_CI and HAS_FLASHINFER)
            else (
                [("blue", "-"), ("orange", "-"), ("green", "-"), ("red", "-")]
                if not IS_CI
                else [("orange", "-")]
            )
        ),
        ylabel="TFLOPs",
        plot_name="input-bf16-output-fp32 dsv3 router gemm throughput",
        args={},
    )
)
def benchmark_float_output(num_tokens, impl):
    # M: num_tokens, K: hidden_dim, N: num_experts
    M, K = num_tokens, 7168

    if impl == "torch-256" or impl == "sgl-kernel-256":
        N = 256
    elif impl == "torch-384" or impl == "sgl-kernel-384":
        N = 384
    else:
        raise ValueError(f"Unknown impl: {impl}")

    mat_a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda").contiguous()
    mat_b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda").contiguous()

    quantiles = [0.5, 0.2, 0.8]

    if impl == "torch-256" or impl == "torch-384":

        def runner():
            F.linear(mat_a, mat_b).to(torch.float32)

    elif impl == "sgl-kernel-256" or impl == "sgl-kernel-384":

        def runner():
            dsv3_router_gemm(mat_a, mat_b, out_dtype=torch.float32)

    elif (impl == "flashinfer-256" or impl == "flashinfer-384") and HAS_FLASHINFER:

        def runner():
            # FlashInfer expects mat_b column-major (K, N) and out float32
            mat_b_col = mat_b.t()
            out = torch.empty((M, N), device="cuda", dtype=torch.float32)
            mm_M1_16_K7168_N256(mat_a, mat_b_col, out, launch_with_pdl=False)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(runner, quantiles=quantiles)

    def tflops(t_ms):
        flops = 2 * M * K * N
        return flops / (t_ms * 1e-3) / 1e12

    return tflops(ms), tflops(max_ms), tflops(min_ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    benchmark_bf16_output.run(print_data=True)
    benchmark_float_output.run(print_data=True)
