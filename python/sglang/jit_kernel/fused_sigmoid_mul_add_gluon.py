"""Gluon implementation of the fused sigmoid-mul-add kernel.

Computes in-place:
    final_hidden_states[row, col] += sigmoid(gate[row, 0]) * shared_output[row, col]

This is equivalent to the standard Triton kernel in activation.py but written
using the Gluon (lower-level Triton) DSL, which gives explicit control over
tensor layouts and thread distribution.

Key Gluon concepts used:
    - @gluon.jit decorator instead of @triton.jit
    - Explicit BlockedLayout / SliceLayout for tensor distribution
    - gl.arange with layout parameter
    - gl.load / gl.store with mask
    - gl.exp for computing sigmoid manually (Gluon has no built-in sigmoid)
    - gl.program_id for grid indexing
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _fused_sigmoid_mul_add_gluon_kernel(
    gate_ptr,  # [num_tokens] (flattened from [num_tokens, 1])
    shared_ptr,  # [num_tokens, hidden_size]
    out_ptr,  # [num_tokens, hidden_size] (in-place)
    hidden_size,  # number of columns
    shared_stride_row,  # stride of shared_output along row dim
    out_stride_row,  # stride of final_hidden_states along row dim
    BLOCK_SIZE: gl.constexpr,
):
    # Each program handles one (row, col_block) tile.
    row = gl.program_id(0)
    col_block = gl.program_id(1)

    # Define a 1D blocked layout for distributing BLOCK_SIZE elements across
    # 4 warps (128 threads). Each thread handles 1 element at a time with
    # threads tiled contiguously for coalesced memory access.
    layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[32],
        warps_per_cta=[4],
        order=[0],
    )

    # Build column offsets with the explicit layout.
    col_offsets = col_block * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = col_offsets < hidden_size

    # Load gate value for this row (scalar, broadcast).
    # Cast to fp32 for the sigmoid computation.
    gate_val = gl.load(gate_ptr + row).to(gl.float32)

    # Gluon does not provide a built-in sigmoid, so we compute it manually:
    #   sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + gl.exp(-gate_val))

    # Compute row offsets for 2D tensors using strides.
    shared_offsets = row * shared_stride_row + col_offsets
    out_offsets = row * out_stride_row + col_offsets

    # Load tiles, cast to fp32 for arithmetic precision.
    shared_val = gl.load(shared_ptr + shared_offsets, mask=mask).to(gl.float32)
    out_val = gl.load(out_ptr + out_offsets, mask=mask).to(gl.float32)

    # Fused sigmoid * mul + add.
    result = out_val + sig * shared_val

    # Store result back (implicit cast to output dtype).
    gl.store(out_ptr + out_offsets, result, mask=mask)


def fused_sigmoid_mul_add_gluon(
    gate: torch.Tensor,
    shared_output: torch.Tensor,
    final_hidden_states: torch.Tensor,
) -> None:
    """Fused sigmoid-mul-add for shared expert gating (Gluon implementation).

    Computes in-place:
        final_hidden_states += sigmoid(gate) * shared_output

    Args:
        gate: [num_tokens, 1] -- output of the shared expert gate linear.
        shared_output: [num_tokens, hidden_size] -- shared expert output.
        final_hidden_states: [num_tokens, hidden_size] -- router expert output
            (modified in-place).
    """
    num_tokens, hidden_size = shared_output.shape

    # Flatten gate to 1-D for simpler pointer arithmetic.
    gate_flat = gate.view(-1)

    BLOCK_SIZE = 1024
    num_col_blocks = triton.cdiv(hidden_size, BLOCK_SIZE)
    grid = (num_tokens, num_col_blocks)

    _fused_sigmoid_mul_add_gluon_kernel[grid](
        gate_flat,
        shared_output,
        final_hidden_states,
        hidden_size,
        shared_output.stride(0),
        final_hidden_states.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
