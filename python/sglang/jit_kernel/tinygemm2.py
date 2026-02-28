"""
JIT wrapper for tinygemm2: SM90+ BF16 small GEMM with bias.
Computes: out = input @ weight.T + bias  (equivalent to F.linear)

Adapted from TensorRT-LLM/cpp/tensorrt_llm/kernels/tinygemm2/
via flashinfer PR #2587 (https://github.com/flashinfer-ai/flashinfer/pull/2587).
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit


@cache_once
def _get_module():
    return load_jit(
        "tinygemm2",
        cuda_files=["gemm/tinygemm2.cu"],
        cuda_wrappers=[("tinygemm2_op", "tinygemm2_op")],
    )


def tinygemm_bf16(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_pdl: bool = False,
) -> None:
    """SM90+ optimized small GEMM: out = input @ weight.T + bias.

    Args:
        input:  (batch_size, input_features)  BF16, contiguous. input_features % 64 == 0.
        weight: (output_features, input_features) BF16, contiguous. output_features % 16 == 0.
        out:    (batch_size, output_features) BF16, contiguous. Mutated in-place.
        bias:   (output_features,) BF16, contiguous. If None, zero bias is used.
        use_pdl: Enable Programmatic Dependent Launch (SM90+ stream serialization).
    """
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("tinygemm_bf16 requires SM90 (Hopper) or later")
    if bias is None:
        bias = torch.zeros(weight.shape[0], dtype=torch.bfloat16, device=input.device)
    _get_module().tinygemm2_op(input, weight, bias, out, use_pdl)
