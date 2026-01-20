"""
Benchmark comparing Triton vs Helion decode attention stage 1.

This script tests:
1. Correctness: Output equivalence between Triton and Helion implementations
2. Performance: Timing comparison across various batch sizes and sequence lengths
"""

import argparse
import torch
import triton

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _decode_att_m_fwd,
)
from sglang.srt.layers.attention.helion_ops.decode_attention_stage1 import (
    decode_attention_stage1_helion,
)

# Default config (MHA: H_Q == H_KV)
DEFAULT_HEAD_NUM = 64
DEFAULT_HEAD_DIM = 64


def create_test_inputs(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    max_kv_splits: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Create test inputs for decode attention stage 1."""
    total_tokens = batch * seq_len

    # Query: one token per batch item
    q = torch.randn(batch, num_heads, head_dim, dtype=dtype, device=device)

    # KV buffers: all previous tokens
    k_buffer = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device)
    v_buffer = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device)

    # CSR-style indexing
    b_seq_len = torch.full((batch,), seq_len, device=device)
    kv_indptr = torch.zeros((batch + 1,), dtype=torch.int32, device=device)
    kv_indptr[1 : batch + 1] = torch.cumsum(b_seq_len, dim=0)
    kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)

    # Split configuration
    num_kv_splits = torch.full((batch,), 4, dtype=torch.int32, device=device)

    # Output tensors for Triton
    att_out_triton = torch.empty(
        (batch, num_heads, max_kv_splits, head_dim),
        dtype=torch.float32,
        device=device,
    )
    att_lse_triton = torch.empty(
        (batch, num_heads, max_kv_splits, head_dim),
        dtype=torch.float32,
        device=device,
    )

    sm_scale = 1.0 / (head_dim**0.5)

    return {
        "q": q,
        "k_buffer": k_buffer,
        "v_buffer": v_buffer,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "num_kv_splits": num_kv_splits,
        "max_kv_splits": max_kv_splits,
        "sm_scale": sm_scale,
        "att_out_triton": att_out_triton,
        "att_lse_triton": att_lse_triton,
    }


def test_correctness(
    batch: int = 4,
    seq_len: int = 512,
    num_heads: int = DEFAULT_HEAD_NUM,
    head_dim: int = DEFAULT_HEAD_DIM,
    max_kv_splits: int = 8,
    rtol: float = 1e-2,
    atol: float = 1e-2,
):
    """Test correctness of Helion implementation against Triton."""
    print(f"\n=== Correctness Test: B={batch}, S={seq_len}, H={num_heads}, D={head_dim} ===")

    inputs = create_test_inputs(batch, seq_len, num_heads, head_dim, max_kv_splits)

    # Run Triton stage 1
    _decode_att_m_fwd(
        inputs["q"],
        inputs["k_buffer"],
        inputs["v_buffer"],
        inputs["att_out_triton"],
        inputs["att_lse_triton"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["num_kv_splits"],
        inputs["max_kv_splits"],
        inputs["sm_scale"],
        logit_cap=0.0,
    )

    # Run Helion stage 1
    att_out_helion, att_lse_helion = decode_attention_stage1_helion(
        inputs["q"],
        inputs["k_buffer"],
        inputs["v_buffer"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["num_kv_splits"],
        inputs["max_kv_splits"],
        inputs["sm_scale"],
    )

    # Compare outputs
    # Note: Triton stores LSE differently, we need to extract it
    # The Triton impl stores att_lse with shape [batch, heads, splits, head_dim] but uses only first element
    att_lse_triton_reduced = inputs["att_lse_triton"][:, :, :, 0]

    out_match = torch.allclose(
        inputs["att_out_triton"], att_out_helion, rtol=rtol, atol=atol
    )
    lse_match = torch.allclose(att_lse_triton_reduced, att_lse_helion, rtol=rtol, atol=atol)

    if out_match and lse_match:
        print("✓ PASSED: Outputs match within tolerance")
    else:
        print("✗ FAILED: Outputs do not match")
        if not out_match:
            diff_out = (inputs["att_out_triton"] - att_out_helion).abs()
            print(f"  att_out max diff: {diff_out.max().item():.6f}")
            print(f"  att_out mean diff: {diff_out.mean().item():.6f}")
        if not lse_match:
            diff_lse = (att_lse_triton_reduced - att_lse_helion).abs()
            print(f"  att_lse max diff: {diff_lse.max().item():.6f}")
            print(f"  att_lse mean diff: {diff_lse.mean().item():.6f}")

    return out_match and lse_match


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["S"],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "helion"],
        line_names=["Triton", "Helion"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Time (ms)",
        plot_name="decode-attention-stage1-comparison",
        args={},
    )
)
def benchmark_stage1(S, provider, B, H, D, max_kv_splits):
    """Benchmark decode attention stage 1."""
    inputs = create_test_inputs(B, S, H, D, max_kv_splits)

    if provider == "triton":
        fn = lambda: _decode_att_m_fwd(
            inputs["q"],
            inputs["k_buffer"],
            inputs["v_buffer"],
            inputs["att_out_triton"],
            inputs["att_lse_triton"],
            inputs["kv_indptr"],
            inputs["kv_indices"],
            inputs["num_kv_splits"],
            inputs["max_kv_splits"],
            inputs["sm_scale"],
            logit_cap=0.0,
        )
    else:  # helion
        fn = lambda: decode_attention_stage1_helion(
            inputs["q"],
            inputs["k_buffer"],
            inputs["v_buffer"],
            inputs["kv_indptr"],
            inputs["kv_indices"],
            inputs["num_kv_splits"],
            inputs["max_kv_splits"],
            inputs["sm_scale"],
        )

    # Warmup
    for _ in range(5):
        fn()

    # Benchmark
    ms = triton.testing.do_bench(fn, warmup=100, rep=500)
    return ms


def benchmark_manual(
    batch_sizes: list[int] = [1, 8, 32, 128],
    seq_lengths: list[int] = [128, 256, 512, 1024, 2048, 4096],
    num_heads: int = DEFAULT_HEAD_NUM,
    head_dim: int = DEFAULT_HEAD_DIM,
    max_kv_splits: int = 8,
):
    """Manual benchmark with detailed output."""
    print("\n=== Performance Benchmark ===")
    print(f"Config: H={num_heads}, D={head_dim}, max_splits={max_kv_splits}")
    print("-" * 80)
    print(f"{'Batch':>6} {'SeqLen':>8} {'Triton (ms)':>12} {'Helion (ms)':>12} {'Speedup':>10}")
    print("-" * 80)

    for B in batch_sizes:
        for S in seq_lengths:
            inputs = create_test_inputs(B, S, num_heads, head_dim, max_kv_splits)

            # Benchmark Triton
            def triton_fn():
                _decode_att_m_fwd(
                    inputs["q"],
                    inputs["k_buffer"],
                    inputs["v_buffer"],
                    inputs["att_out_triton"],
                    inputs["att_lse_triton"],
                    inputs["kv_indptr"],
                    inputs["kv_indices"],
                    inputs["num_kv_splits"],
                    inputs["max_kv_splits"],
                    inputs["sm_scale"],
                    logit_cap=0.0,
                )

            # Benchmark Helion
            def helion_fn():
                decode_attention_stage1_helion(
                    inputs["q"],
                    inputs["k_buffer"],
                    inputs["v_buffer"],
                    inputs["kv_indptr"],
                    inputs["kv_indices"],
                    inputs["num_kv_splits"],
                    inputs["max_kv_splits"],
                    inputs["sm_scale"],
                )

            # Warmup
            for _ in range(5):
                triton_fn()
                helion_fn()

            triton_ms = triton.testing.do_bench(triton_fn, warmup=100, rep=500)
            helion_ms = triton.testing.do_bench(helion_fn, warmup=100, rep=500)
            speedup = triton_ms / helion_ms if helion_ms > 0 else float("inf")

            print(f"{B:>6} {S:>8} {triton_ms:>12.4f} {helion_ms:>12.4f} {speedup:>10.2f}x")

    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark decode attention stage 1")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "correctness", "benchmark", "quick"],
        help="Test mode: all, correctness, benchmark, or quick",
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size for quick test")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length for quick test")
    args = parser.parse_args()

    if args.mode in ["all", "correctness"]:
        # Test correctness across various configs
        all_passed = True
        for B in [1, 4, 8]:
            for S in [128, 512, 1024]:
                passed = test_correctness(batch=B, seq_len=S)
                all_passed = all_passed and passed

        if all_passed:
            print("\n✓ All correctness tests passed!")
        else:
            print("\n✗ Some correctness tests failed!")

    if args.mode in ["all", "benchmark"]:
        benchmark_manual()

    if args.mode == "quick":
        print(f"\n=== Quick Test: B={args.batch}, S={args.seq_len} ===")
        test_correctness(batch=args.batch, seq_len=args.seq_len)

        inputs = create_test_inputs(args.batch, args.seq_len, DEFAULT_HEAD_NUM, DEFAULT_HEAD_DIM)

        def triton_fn():
            _decode_att_m_fwd(
                inputs["q"],
                inputs["k_buffer"],
                inputs["v_buffer"],
                inputs["att_out_triton"],
                inputs["att_lse_triton"],
                inputs["kv_indptr"],
                inputs["kv_indices"],
                inputs["num_kv_splits"],
                inputs["max_kv_splits"],
                inputs["sm_scale"],
                logit_cap=0.0,
            )

        def helion_fn():
            decode_attention_stage1_helion(
                inputs["q"],
                inputs["k_buffer"],
                inputs["v_buffer"],
                inputs["kv_indptr"],
                inputs["kv_indices"],
                inputs["num_kv_splits"],
                inputs["max_kv_splits"],
                inputs["sm_scale"],
            )

        for _ in range(5):
            triton_fn()
            helion_fn()

        triton_ms = triton.testing.do_bench(triton_fn, warmup=50, rep=200)
        helion_ms = triton.testing.do_bench(helion_fn, warmup=50, rep=200)

        print(f"\nTriton: {triton_ms:.4f} ms")
        print(f"Helion: {helion_ms:.4f} ms")
        print(f"Speedup: {triton_ms / helion_ms:.2f}x")


if __name__ == "__main__":
    main()
