import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

import torch
from sgl_kernel import (
    cutlass_fp4_group_mm,
    prepare_moe_input,
    scaled_fp4_experts_quant,
    scaled_fp4_quant,
)

from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams, CutlassMoEType
from sglang.srt.utils import get_device_capability

FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def round_up(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def bench_cutlass_fp4_group_gemm(
    expected_m_per_group: int,
    n: int,
    k: int,
    num_groups: int,
    num_warmup: int,
    num_run: int,
    gemm_type: str = "gemm1",
) -> Tuple[float, int]:
    """Benchmark cutlass_fp4_group_mm for one GEMM type.

    Args:
        expected_m_per_group: Tokens per expert (uniform distribution).
        n: intermediate_size_per_partition (MoE model dimension).
        k: hidden_size (MoE model dimension).
        num_groups: Number of experts.
        gemm_type: "gemm1" (gate-up: M x 2n x k) or "gemm2" (down: M x k x n).

    Returns:
        (average_time_us, total_m)
    """
    device = "cuda"
    dtype = torch.bfloat16
    topk = 1
    total_tokens = expected_m_per_group * num_groups

    # Ensure scaled_fp4_experts_quant can handle total token count
    os.environ["MODELOPT_MAX_TOKENS_PER_EXPERT"] = str(
        max(total_tokens, 65536)
    )

    # Set up CutlassMoEParams
    params = CutlassMoEParams(
        CutlassMoEType.BlockscaledFP4,
        device=device,
        num_experts=num_groups,
        intermediate_size_per_partition=n,
        hidden_size=k,
    )

    # Create uniform topk_ids: each expert gets expected_m_per_group tokens
    topk_ids = torch.arange(num_groups, device=device, dtype=torch.int32)
    topk_ids = topk_ids.repeat_interleave(expected_m_per_group).unsqueeze(1)

    # Prepare MoE routing (computes expert_offsets, problem_sizes, blockscale_offsets)
    a_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device=device)
    c_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device=device)
    prepare_moe_input(
        topk_ids,
        params.expert_offsets,
        params.problem_sizes1,
        params.problem_sizes2,
        a_map,
        c_map,
        num_groups,
        n,
        k,
        params.blockscale_offsets,
    )

    if gemm_type == "gemm1":
        # Gate-up GEMM: C[m, 2n] = A[m, k] * B[2n, k]^T
        act_dim = k
        w_n = 2 * n
        gemm_args = params.to_gemm1_args()
    else:
        # Down GEMM: C[m, k] = A[m, n] * B[k, n]^T
        act_dim = n
        w_n = k
        gemm_args = params.to_gemm2_args()

    # Create and quantize activations
    a = torch.randn(total_tokens, act_dim, device=device, dtype=dtype) / 10
    a_gs = torch.ones(num_groups, device=device, dtype=torch.float32)
    rep_a_fp4, rep_a_blockscale = scaled_fp4_experts_quant(
        a,
        a_gs,
        params.expert_offsets,
        params.blockscale_offsets,
        topk,
        expert_map=a_map,
    )
    del a

    # Create and quantize weights (quantize one expert, copy to all)
    quant_blocksize = 16
    sf_n = round_up(w_n, 128)
    sf_k = round_up(act_dim // quant_blocksize, 4)
    w_fp4 = torch.empty(
        num_groups, w_n, act_dim // 2, device=device, dtype=torch.uint8
    )
    w_blockscale = torch.empty(
        num_groups, sf_n, sf_k, device=device, dtype=torch.float8_e4m3fn
    )
    w_gs = torch.empty(num_groups, device=device, dtype=torch.float32)

    w_single = torch.randn(w_n, act_dim, device=device, dtype=dtype) / 10
    w_amax = torch.abs(w_single).max().to(torch.float32)
    gs_val = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w_amax
    w_fp4_single, w_bs_single = scaled_fp4_quant(w_single, gs_val)
    del w_single

    for expert_idx in range(num_groups):
        w_fp4[expert_idx] = w_fp4_single
        w_blockscale[expert_idx] = w_bs_single
        w_gs[expert_idx] = gs_val
    del w_fp4_single, w_bs_single

    w_alphas = 1.0 / w_gs

    # Warmup
    for _ in range(num_warmup):
        cutlass_fp4_group_mm(
            rep_a_fp4,
            w_fp4,
            rep_a_blockscale,
            w_blockscale,
            w_alphas,
            dtype,
            device,
            gemm_args,
        )
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_run):
        cutlass_fp4_group_mm(
            rep_a_fp4,
            w_fp4,
            rep_a_blockscale,
            w_blockscale,
            w_alphas,
            dtype,
            device,
            gemm_args,
        )
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    avg_us = start_event.elapsed_time(end_event) / num_run * 1000  # us

    return avg_us, total_tokens


@dataclass
class ShapeArg:
    expected_m_per_group: int
    n: int  # intermediate_size_per_partition
    k: int  # hidden_size
    num_groups: int


def benchmark_shapes(
    shape_args: List[ShapeArg],
    num_warmup: int,
    num_run: int,
):
    for shape in shape_args:
        print(
            f"\nBenchmark: M/group={shape.expected_m_per_group}, "
            f"n={shape.n}, k={shape.k}, E={shape.num_groups}"
        )
        for gemm_type in ["gemm1", "gemm2"]:
            if gemm_type == "gemm1":
                gemm_n, gemm_k = 2 * shape.n, shape.k
            else:
                gemm_n, gemm_k = shape.k, shape.n

            total_m = shape.expected_m_per_group * shape.num_groups
            total_flops = 2 * total_m * gemm_n * gemm_k

            try:
                avg_us, _ = bench_cutlass_fp4_group_gemm(
                    shape.expected_m_per_group,
                    shape.n,
                    shape.k,
                    shape.num_groups,
                    num_warmup,
                    num_run,
                    gemm_type,
                )
                tflops = total_flops / (avg_us * 1e6)
                label = "gate-up" if gemm_type == "gemm1" else "down  "
                print(
                    f"  {label} ({total_m}x{gemm_n}x{gemm_k}): "
                    f"{avg_us:.1f} us, {tflops:.2f} TFLOPS"
                )
            except Exception as e:
                label = "gate-up" if gemm_type == "gemm1" else "down  "
                print(f"  {label}: FAILED - {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark cutlass_fp4_group_mm (FP4 grouped GEMM for MoE)"
    )
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-run", type=int, default=100)
    args = parser.parse_args()

    # FP4 grouped GEMM requires sm100+
    major, minor = get_device_capability()
    if major is None or major < 10:
        print("Skipping FP4 group GEMM benchmark")
        if major is not None:
            print(
                f"FP4 operations require sm100+, but found sm{major}{minor}"
            )
        else:
            print("Could not determine device capability")
        return

    if IS_CI:
        shape_args = [
            ShapeArg(expected_m_per_group=128, n=512, k=1024, num_groups=8),
        ]
    else:
        shape_args = [
            # DeepSeek-R1/V3 (hidden=7168, intermediate=2048)
            # TP=8: n=256, k=7168, E=256
            # Prefill, chunk_size=4096
            ShapeArg(expected_m_per_group=128, n=256, k=7168, num_groups=256),
            # Prefill, chunk_size=8192
            ShapeArg(expected_m_per_group=256, n=256, k=7168, num_groups=256),
            # Decode, bs=32
            ShapeArg(expected_m_per_group=1, n=256, k=7168, num_groups=256),
            # EP=8: n=2048, k=7168, E=32
            # Prefill, chunk_size=8192
            ShapeArg(expected_m_per_group=256, n=2048, k=7168, num_groups=32),
            # Decode, bs=128
            ShapeArg(expected_m_per_group=4, n=2048, k=7168, num_groups=32),
            # Qwen3-235B-A22B (hidden=4096, intermediate=1536)
            # TP=4: n=384, k=4096, E=128
            # Prefill, chunk_size=16384
            ShapeArg(expected_m_per_group=1024, n=384, k=4096, num_groups=128),
            # Decode, bs=256
            ShapeArg(expected_m_per_group=16, n=384, k=4096, num_groups=128),
        ]

    benchmark_shapes(shape_args, args.num_warmup, args.num_run)


if __name__ == "__main__":
    main()
