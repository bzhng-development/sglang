"""Fused decode pre-attention chain: per-head QK RMSNorm + rotate-half RoPE +
paged dual K/V store (JIT).

Collapses the norm / rope / rotate-half-cat / 4x index_copy chain of a
GQA decode step (Qwen3-family layers) into one kernel: one CTA per
(token, head). The caller gathers the position's cos/sin row (mrope or
standard — the kernel is position-encoding agnostic) and passes page slots;
K is stored twice — un-roped (shiftable cache) and roped (hot buffer for
length-limited attention) — pass the same tensor twice if one store
suffices. Inspired by the dsv4 fused indexer kernels. Arch-generic
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
def _fused_qk_norm_rope_store_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "fused_qk_norm_rope_store",
        *args,
        cuda_files=["gemm/fused_qk_norm_rope_store.cuh"],
        cuda_wrappers=[("fused_qk_norm_rope_store", f"fused_qk_norm_rope_store<{args}>")],
    )


@register_custom_op(
    op_name="fused_qk_norm_rope_store",
    mutates_args=["q_out", "k_cache", "k_hot", "v_cache", "v_hot"],
)
def fused_qk_norm_rope_store(
    qkv: torch.Tensor,
    q_out: torch.Tensor,
    k_cache: torch.Tensor,
    k_hot: torch.Tensor,
    v_cache: torch.Tensor,
    v_hot: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    slots: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    """Fused per-head QK RMSNorm + RoPE + paged K/V dual-store.

    Args:
        qkv: (num_tokens, (Hq + 2*Hkv) * D) packed q|k|v projection output.
        q_out: (num_tokens, Hq * D) roped, normed queries.
        k_cache / k_hot: (num_pages, Hkv, D) un-roped / roped K stores.
        v_cache / v_hot: (num_pages, Hkv, D) V stores.
        q_norm_weight / k_norm_weight: (D,) per-head RMSNorm weights.
        cos_sin: (num_tokens, 2, D) float32 gathered cos/sin rows.
        slots: (num_tokens,) int64 destination page per token.
        eps: RMSNorm epsilon.
    """
    module = _fused_qk_norm_rope_store_module(qkv.dtype)
    module.fused_qk_norm_rope_store(
        qkv,
        q_out,
        k_cache,
        k_hot,
        v_cache,
        v_hot,
        q_norm_weight,
        k_norm_weight,
        cos_sin,
        slots,
        eps,
    )
