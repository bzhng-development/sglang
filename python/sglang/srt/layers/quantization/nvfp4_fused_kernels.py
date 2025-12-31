"""
Fused RMSNorm + FP4 Quantization kernels for NVFP4 on Blackwell (SM100+).

Uses flashinfer CuTe-DSL kernels:
- rmsnorm_fp4quant: RMSNorm + FP4 quant
- add_rmsnorm_fp4quant: residual add + RMSNorm + FP4 quant

These kernels fuse RMSNorm with FP4 quantization to reduce kernel launch overhead
and memory bandwidth. They support NVFP4 format (block_size=16, E4M3 scales) with
optional global scale factor for proper dynamic range scaling.

Usage pattern (matches AMD MXFP4):
    # Fused path returns tuple that can be passed directly to FP4 linear
    q_prequant = fused_rmsnorm_fp4quant(q, weight, eps, global_scale)
    output = linear(q_prequant)  # Linear accepts tuple (fp4, scales)
"""

from typing import Optional, Tuple

import torch

from sglang.srt.utils import is_cuda, is_sm100_supported

# Detect if we can use the fused kernels
_fused_kernels_available = False
_rmsnorm_fp4quant = None
_add_rmsnorm_fp4quant = None

if is_cuda() and is_sm100_supported():
    try:
        from flashinfer import add_rmsnorm_fp4quant as _add_rmsnorm_fp4quant
        from flashinfer import rmsnorm_fp4quant as _rmsnorm_fp4quant

        _fused_kernels_available = True
    except ImportError:
        pass


def is_nvfp4_fused_kernels_available() -> bool:
    """Check if the fused RMSNorm + FP4 quantization kernels are available."""
    return _fused_kernels_available


def fused_rmsnorm_fp4quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    global_scale: Optional[torch.Tensor] = None,
    is_sf_swizzled_layout: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused RMSNorm + FP4 quantization for NVFP4.

    Returns a tuple (y_fp4, block_scale) that can be passed directly to FP4 linear
    layers (they detect tuple input and skip internal quantization).

    Args:
        input: Input tensor of shape (batch_size, hidden_size), dtype bf16/fp16
        weight: RMSNorm weight tensor of shape (hidden_size,)
        eps: Epsilon for RMSNorm numerical stability
        global_scale: Optional global scale factor of shape (1,), dtype float32.
                      For NVFP4, this is typically `layer.input_scale_inv`.
        is_sf_swizzled_layout: If True, output swizzled scales for GEMM (default).
                               If False, output row-major scales for TRT-LLM MoE.

    Returns:
        Tuple of (y_fp4, block_scale) - can be passed directly to FP4 linear
    """
    assert _rmsnorm_fp4quant is not None, "NVFP4 fused kernels not available"

    # Call the fused kernel - it auto-allocates outputs when not provided
    y_fp4, block_scale = _rmsnorm_fp4quant(
        input,
        weight,
        y_fp4=None,
        block_scale=None,
        global_scale=global_scale,
        eps=eps,
        block_size=16,  # NVFP4 format
        scale_format="e4m3",
        is_sf_swizzled_layout=is_sf_swizzled_layout,
    )

    return (y_fp4, block_scale)


def fused_add_rmsnorm_fp4quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    global_scale: Optional[torch.Tensor] = None,
    is_sf_swizzled_layout: bool = True,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Fused residual add + RMSNorm + FP4 quantization for NVFP4.

    Returns ((y_fp4, block_scale), new_residual) where the first element is a tuple
    that can be passed directly to FP4 linear layers.

    Args:
        input: Input tensor of shape (batch_size, hidden_size), dtype bf16/fp16
        residual: Residual tensor to add, same shape and dtype as input
        weight: RMSNorm weight tensor of shape (hidden_size,)
        eps: Epsilon for RMSNorm numerical stability
        global_scale: Optional global scale factor of shape (1,), dtype float32.
                      For NVFP4, this is typically `layer.input_scale_inv`.
        is_sf_swizzled_layout: If True, output swizzled scales for GEMM (default).
                               If False, output row-major scales for TRT-LLM MoE.

    Returns:
        Tuple of ((y_fp4, block_scale), new_residual):
        - (y_fp4, block_scale): Can be passed directly to FP4 linear
        - new_residual: Updated residual (input + residual) for next layer
    """
    assert _add_rmsnorm_fp4quant is not None, "NVFP4 fused kernels not available"

    # Compute new residual - needed for next layer
    new_residual = input + residual

    # Call the fused kernel
    y_fp4, block_scale = _add_rmsnorm_fp4quant(
        input,
        residual,
        weight,
        y_fp4=None,
        block_scale=None,
        global_scale=global_scale,
        eps=eps,
        block_size=16,  # NVFP4 format
        scale_format="e4m3",
        is_sf_swizzled_layout=is_sf_swizzled_layout,
    )

    return ((y_fp4, block_scale), new_residual)


__all__ = [
    "is_nvfp4_fused_kernels_available",
    "fused_rmsnorm_fp4quant",
    "fused_add_rmsnorm_fp4quant",
]
