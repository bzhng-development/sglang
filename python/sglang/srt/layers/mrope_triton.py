# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0

from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _triton_qwen2vl_mrope_forward(
    q_ptr,
    k_ptr,
    cos,  # [3, num_tokens, rd//2]
    sin,  # [3, num_tokens, rd//2]
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,  # head_size
    rd: tl.constexpr,  # rotary_dim
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,  # t section length (in half width)
    mrope_section_h: tl.constexpr,  # h section length (in half width)
):
    """
    mRoPE Triton kernel adapted from the Liger/vLLM kernel. For each token program instance:
      - Build cos/sin rows by selecting T/H/W segments according to mrope_section.
      - Apply Neox-style rotation to the first 'rd' dims of q/k, leave tail as pass-through.

    Data layout assumptions
      q_ptr: [num_tokens, n_qh * hd]
      k_ptr: [num_tokens, n_kh * hd]
      cos/sin: [3, num_tokens, rd//2] in row-major
    """
    pid = tl.program_id(0)  # token id

    # Locate start address for this token
    q_ptr = q_ptr + pid * (n_qh * hd)
    k_ptr = k_ptr + pid * (n_kh * hd)

    # Section boundaries on the half width (rd//2)
    t_end = mrope_section_t
    h_end = t_end + mrope_section_h

    half_rd = rd // 2

    # Build pointers for each modality plane for this token
    # cos/sin are laid out as: [T-plane][H-plane][W-plane], each sized [num_tokens, half_rd]
    t_cos = cos + pid * half_rd
    h_cos = t_cos + num_tokens * half_rd
    w_cos = h_cos + num_tokens * half_rd
    t_sin = sin + pid * half_rd
    h_sin = t_sin + num_tokens * half_rd
    w_sin = h_sin + num_tokens * half_rd

    # Assemble cos/sin rows according to section masks
    cos_offsets = tl.arange(0, pad_hd // 2)

    t_mask = cos_offsets < t_end
    h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
    w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)

    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)

    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    # Offsets/masks for left half (x1) of the rotary dimension
    first_half_q_offsets = (
        tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_half_k_offsets = (
        tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )

    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] < rd // 2
    )
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] < rd // 2
    )

    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(
        sin_row.dtype
    )
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(
        sin_row.dtype
    )

    # Offsets/masks for right half (x2) of the rotary dimension
    second_half_q_offsets = first_half_q_offsets + (rd // 2)
    second_half_k_offsets = first_half_k_offsets + (rd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask

    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(
        sin_row.dtype
    )
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(
        sin_row.dtype
    )

    # Neox-style rotation:
    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)


def apply_mrope_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
    head_size: int,
    is_neox_style: bool,
    *,
    positions: Optional[torch.Tensor] = None,
    mrope_positions: Optional[torch.Tensor] = None,
    mrope_section: Optional[List[int]] = None,
    stream=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply multimodal RoPE (mRoPE) to q/k in-place using a Triton kernel.

    Args:
        q: [T, num_q_heads * head_size], fp16/bf16. Contiguous along last dim.
        k: [T, num_kv_heads * head_size], fp16/bf16. Contiguous along last dim.
        cos_sin_cache: [P, rotary_dim] in fp32, where the last dim is [cos, sin] concatenation.
        rotary_dim: number of rotary dims (Dr). Must be even and Dr <= head_size.
        head_size: head dimension (Dh).
        is_neox_style: True for Neox-style rotation; GPT-J style is not supported by this kernel.
        positions: [T] (classic RoPE). Not supported here.
        mrope_positions: [3, T] (T/H/W multimodal positions). Required for mRoPE.
        mrope_section: list[int] of length 3, section sizes on the HALF-width (sum == rotary_dim // 2).
        stream: unused, reserved for future.

    Returns:
        (q, k): Tensors modified in-place, same objects returned for convenience.

    Invariants enforced:
        - q, k: 2D tensors, consistent T.
        - dtypes in {torch.bfloat16, torch.float16}.
        - rotary_dim even, rotary_dim <= head_size.
        - mrope_section sum equals rotary_dim // 2.
        - Exactly one of positions or mrope_positions is provided (this function handles only mRoPE).
    """
    # Validation
    if q.ndim != 2 or k.ndim != 2:
        raise ValueError("q and k must be 2D tensors of shape [T, H*Dh].")
    if q.shape[0] != k.shape[0]:
        raise ValueError("q and k must have the same num_tokens (first dimension).")
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype:
        raise ValueError("q and k must be bfloat16 or float16 and have the same dtype.")
    if rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be even.")
    if rotary_dim > head_size:
        raise ValueError("rotary_dim must be <= head_size.")
    if positions is not None:
        raise ValueError("apply_mrope_triton only supports mRoPE (mrope_positions).")
    if mrope_positions is None:
        raise ValueError("mrope_positions must be provided for mRoPE.")
    if mrope_positions.ndim != 2 or mrope_positions.shape[0] != 3:
        raise ValueError("mrope_positions must have shape [3, T].")
    if not is_neox_style:
        # Kernel currently implements Neox-style rotation only. Fallback should be handled by caller.
        raise NotImplementedError("Triton mRoPE supports Neox-style only.")

    T = q.shape[0]
    if mrope_positions.shape[1] != T:
        raise ValueError(
            f"mrope_positions shape mismatch: expected T={T}, got {mrope_positions.shape[1]}"
        )

    if q.shape[1] % head_size != 0 or k.shape[1] % head_size != 0:
        raise ValueError("The last dimension of q/k must be divisible by head_size.")
    n_qh = q.shape[1] // head_size
    n_kh = k.shape[1] // head_size

    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("q and k must be contiguous along the last dimension.")

    # Default section: text-only on T (units are half width)
    if mrope_section is None:
        mrope_section = [rotary_dim // 2, 0, 0]
    if len(mrope_section) != 3:
        raise ValueError("mrope_section must be a list of 3 integers: [t, h, w].")
    if sum(mrope_section) != (rotary_dim // 2):
        raise ValueError("Sum(mrope_section) must equal rotary_dim // 2.")

    # Gather cos/sin rows on device in fp32
    cos_sin_cache = cos_sin_cache.to(q.device, dtype=torch.float32)
    # cos_sin: [3, T, rotary_dim]
    cos_sin = cos_sin_cache[mrope_positions]
    # Split to cos/sin: [3, T, rotary_dim//2]
    cos, sin = cos_sin.chunk(2, dim=-1)

    # Triton padding helpers
    pad_hd = triton.next_power_of_2(head_size)
    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)

    # Ensure contiguous inputs for kernel (no-ops if already contiguous)
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    # Launch: one program per token
    _triton_qwen2vl_mrope_forward[(T,)](
        q,
        k,
        cos,
        sin,
        T,
        n_qh,
        n_kh,
        head_size,
        rotary_dim,
        pad_n_qh,
        pad_n_kh,
        pad_hd,
        mrope_section[0],
        mrope_section[1],
    )

    return q, k


__all__ = ["apply_mrope_triton"]
