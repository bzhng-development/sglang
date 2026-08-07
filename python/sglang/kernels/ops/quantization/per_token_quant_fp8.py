"""Per-token dynamic FP8 quantization (JIT).

JIT port of the sgl-kernel AOT ``sgl_per_token_quant_fp8`` small-batch
schedule (one token per CTA, block max-reduction), bit-identical semantics:
``scale[t] = rowmax(|input[t]|) / 448`` with no epsilon. Built for the
batch-1..O(1k) decode regime of latency-bound serving; the warp-batched
large-M schedule can join later behind the same op. Arch-generic
(sm90 / sm100 / sm120).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _per_token_quant_fp8_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "per_token_quant_fp8",
        *args,
        cuda_files=["gemm/per_token_quant_fp8.cuh"],
        cuda_wrappers=[("per_token_quant_fp8", f"per_token_quant_fp8<{args}>")],
    )


@register_custom_op(
    op_name="per_token_quant_fp8",
    mutates_args=["output_q", "output_s"],
)
def per_token_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    """Per-token dynamic FP8 (e4m3) quantization into caller-owned buffers.

    Args:
        input: (num_tokens, hidden_dim) float / half / bfloat16, contiguous.
        output_q: (num_tokens, hidden_dim) float8_e4m3fn.
        output_s: (num_tokens,) float32 per-token scales.
    """
    module = _per_token_quant_fp8_module(input.dtype)
    module.per_token_quant_fp8(input, output_q, output_s.view(-1))
