# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Helion implementation of decode attention stage 1.

This implements the first stage of the two-stage decode attention kernel,
which computes partial attention outputs for each KV split.

Stage 1: For each (batch, head, split), compute attention over a portion of the KV cache
Stage 2: Reduce the partial outputs across splits (not implemented here)
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Minimum KV block size for split calculation (matches Triton impl)
_MIN_BLOCK_KV = 32


@helion.kernel(
    static_shapes=True,
)
def decode_attention_stage1_helion(
    q: torch.Tensor,  # [batch, num_heads, head_dim]
    k_buffer: torch.Tensor,  # [total_tokens, num_heads, head_dim]
    v_buffer: torch.Tensor,  # [total_tokens, num_heads, head_dim]
    kv_indptr: torch.Tensor,  # [batch + 1]
    kv_indices: torch.Tensor,  # [total_kv_len]
    num_kv_splits: torch.Tensor,  # [batch]
    max_kv_splits: int,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stage 1 of decode attention: compute partial attention for each KV split.

    This is the MHA variant (kv_group_num=1, no GQA support yet).
    """
    batch, num_heads, head_dim = q.shape
    total_tokens = k_buffer.shape[0]
    batch = hl.specialize(batch)
    head_dim = hl.specialize(head_dim)
    num_heads = hl.specialize(num_heads)
    max_kv_splits = hl.specialize(max_kv_splits)

    # Allocate outputs
    att_out = torch.zeros(
        [batch, num_heads, max_kv_splits, head_dim],
        dtype=torch.float32,
        device=q.device,
    )
    att_lse = torch.zeros(
        [batch, num_heads, max_kv_splits],
        dtype=torch.float32,
        device=q.device,
    )

    # Flatten K and V buffers to 1D for indirect indexing
    k_flat = k_buffer.view(-1)
    v_flat = v_buffer.view(-1)
    kv_stride = num_heads * head_dim

    # Use grid for batch (scalar index) since each batch has different seq_len
    for batch_idx in hl.grid(batch):
        # Get KV range for this batch (scalar values)
        kv_start = kv_indptr[batch_idx]
        kv_end = kv_indptr[batch_idx + 1]
        seq_len = kv_end - kv_start
        kv_splits = num_kv_splits[batch_idx]

        # Compute KV length per split
        raw_len = (seq_len + kv_splits - 1) // kv_splits
        kv_len_per_split = ((raw_len + _MIN_BLOCK_KV - 1) // _MIN_BLOCK_KV) * _MIN_BLOCK_KV

        # Tile over heads
        for tile_h in hl.tile(num_heads):
            # Load query: [tile_h, head_dim]
            q_vec = q[batch_idx, tile_h, :]

            # Process each split
            for split_id in hl.grid(max_kv_splits):
                split_start = split_id * kv_len_per_split
                split_end = torch.minimum(split_start + kv_len_per_split, seq_len)

                # Initialize online softmax state - shaped for tile_h
                e_max = hl.full([tile_h], float("-inf"), dtype=torch.float32)
                e_sum = hl.zeros([tile_h], dtype=torch.float32)
                acc = hl.zeros([tile_h, head_dim], dtype=torch.float32)

                # Only process if this split has work
                if split_end > split_start:
                    split_len = split_end - split_start

                    # Main KV loop
                    for tile_n in hl.tile(split_len):
                        # Mask for valid KV positions (both are scalar-derived now)
                        kv_mask = tile_n.index < split_len

                        # Get KV indices for this block
                        kv_pos = kv_start + split_start + tile_n.index
                        kv_loc = hl.load(kv_indices, [kv_pos], extra_mask=kv_mask)

                        # Compute flat indices: [tile_h, tile_n, head_dim]
                        # base = kv_loc * kv_stride + head_idx * head_dim
                        base_idx = kv_loc[None, :] * kv_stride + tile_h.index[:, None] * head_dim
                        head_offsets = hl.arange(head_dim)
                        flat_idx = base_idx[:, :, None] + head_offsets[None, None, :]

                        # Create 3D mask: [tile_h, tile_n, head_dim]
                        mask_3d = kv_mask[None, :, None]

                        # Load K and V: [tile_h, tile_n, head_dim]
                        k_blk = hl.load(k_flat, [flat_idx], extra_mask=mask_3d)
                        v_blk = hl.load(v_flat, [flat_idx], extra_mask=mask_3d)

                        # Compute attention: qk[h, n] = sum_d(q[h, d] * k[h, n, d])
                        qk = torch.sum(q_vec[:, None, :] * k_blk, dim=-1)  # [tile_h, tile_n]
                        qk = qk * sm_scale
                        qk = torch.where(kv_mask[None, :], qk, float("-inf"))

                        # Online softmax update
                        qk_max = torch.amax(qk, dim=-1)  # [tile_h]
                        n_e_max = torch.maximum(e_max, qk_max)
                        re_scale = torch.exp(e_max - n_e_max)
                        acc = acc * re_scale[:, None]
                        p = torch.exp(qk - n_e_max[:, None])  # [tile_h, tile_n]
                        # acc += sum over n: [tile_h, head_dim]
                        acc = acc + torch.sum(p[:, :, None] * v_blk, dim=1)
                        e_sum = e_sum * re_scale + torch.sum(p, dim=-1)
                        e_max = n_e_max

                    # Store results
                    att_out[batch_idx, tile_h, split_id, :] = acc / e_sum[:, None]
                    att_lse[batch_idx, tile_h, split_id] = e_max + torch.log(e_sum)

    return att_out, att_lse


def decode_attention_stage1_fwd(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    att_out: torch.Tensor,
    att_lse: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    num_kv_splits: torch.Tensor,
    max_kv_splits: int,
    sm_scale: float,
) -> None:
    """
    Wrapper that matches the Triton interface for easier comparison.
    Writes results to pre-allocated att_out and att_lse tensors.
    """
    out, lse = decode_attention_stage1_helion(
        q, k_buffer, v_buffer, kv_indptr, kv_indices, num_kv_splits, max_kv_splits, sm_scale
    )
    att_out.copy_(out)
    att_lse.copy_(lse)
