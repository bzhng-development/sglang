import itertools
import os

import torch
import triton
from sgl_kernel import FusedSetKVBufferArg
from sgl_kernel.testing.rotary_embedding import (
    FlashInferRotaryEmbedding,
    MHATokenToKVPool,
    RotaryEmbedding,
    create_inputs,
)

from sglang.srt.bench_utils import bench_kineto

configs = [
    (batch_size, seq_len, save_kv_cache)
    for batch_size, seq_len in (
        (1, 1),
        (32, 1),
        (128, 1),
        (512, 1),
        (2, 512),
        (4, 4096),
    )
    for save_kv_cache in (False, True)
]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "save_kv_cache"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sglang"],
        line_names=["SGL Kernel"],
        styles=[("green", "-")],
        ylabel="us",
        plot_name="bench_rotary_embedding",
        args={},
    )
)
def benchmark(batch_size, seq_len, save_kv_cache, provider):
    device = torch.device("cuda")

    num_q_heads = 32
    num_kv_heads = 8
    head_size = 64
    dtype = torch.bfloat16

    config = dict(
        head_size=head_size,
        rotary_dim=64,
        max_position_embeddings=4096,
        base=8000,
        is_neox_style=True,
        dtype=dtype,
    )
    rope_flashinfer = FlashInferRotaryEmbedding(**config).to(device)
    pool_flashinfer = MHATokenToKVPool(head_num=num_kv_heads, head_dim=head_size)

    inputs = create_inputs(
        head_size=head_size,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        dtype=dtype,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )

    query_flashinfer, key_flashinfer = inputs["query"].clone(), inputs["key"].clone()

    bench_fn = lambda: rope_flashinfer.forward_cuda(
        inputs["pos_ids"],
        query_flashinfer,
        key_flashinfer,
        fused_set_kv_buffer_arg=(
            FusedSetKVBufferArg(
                value=inputs["value"],
                k_buffer=pool_flashinfer.k_buffer[0].view(-1, num_kv_heads * head_size),
                v_buffer=pool_flashinfer.v_buffer[0].view(-1, num_kv_heads * head_size),
                k_scale=None,
                v_scale=None,
                cache_loc=inputs["out_cache_loc"],
            )
            if save_kv_cache
            else None
        ),
    )

    time_s = bench_kineto(bench_fn, kernel_names="BatchQKApplyRotaryPosIds")
    return time_s * 1e6


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "save_kv_cache"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["mrope_triton"],
        line_names=["SGL mRoPE Triton"],
        styles=[("blue", "-")],
        ylabel="us",
        plot_name="bench_mrope",
        args={},
    )
)
def benchmark_mrope(batch_size, seq_len, save_kv_cache, provider):
    # Simple mRoPE benchmark using SGLang Triton path
    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_q_heads = 32
    num_kv_heads = 8
    head_size = 64
    rotary_dim = 64
    max_position = 4096
    base = 10000

    T = batch_size * seq_len

    # Prepare inputs
    positions = torch.randint(
        0, max_position // 4, (3, T), device=device, dtype=torch.int64
    )
    q = torch.randn(T, num_q_heads * head_size, dtype=dtype, device=device)
    k = torch.randn(T, num_kv_heads * head_size, dtype=dtype, device=device)

    # Create mRoPE helper
    from sglang.srt.layers.rotary_embedding import get_rope as sgl_get_rope

    rope = sgl_get_rope(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position=max_position,
        base=base,
        is_neox_style=True,
        rope_scaling={"type": "default", "mrope_section": [rotary_dim // 2, 0, 0]},
        dtype=dtype,
    ).to(device=device)

    def bench_fn():
        prev = os.environ.get("SGLANG_ROPE_TRITON")
        try:
            os.environ["SGLANG_ROPE_TRITON"] = "1"
            rope.forward(positions, q.clone(), k.clone())
        finally:
            if prev is None:
                os.environ.pop("SGLANG_ROPE_TRITON", None)
            else:
                os.environ["SGLANG_ROPE_TRITON"] = prev

    time_s = bench_kineto(bench_fn)
    return time_s * 1e6


if __name__ == "__main__":
    benchmark.run(print_data=True)
