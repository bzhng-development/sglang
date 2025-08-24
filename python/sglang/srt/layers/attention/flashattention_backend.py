from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

from sgl_kernel import merge_state_v2
from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


@dataclass
class FlashAttentionMetadata:
    """Metadata to be init once in the model forward pass,
    each layer's forward pass can reuse the metadata.

    For each init metadata function, we will try set up them in below order
    """

    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor = None
    # Maximum sequence length for query
    max_seq_len_q: int = 1
    # Maximum sequence length for key
    max_seq_len_k: int = 0
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor = None
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor = None
    # Window size (typically used by Gemma)
    window_size: tuple = (-1, -1)
    # Page table, the index of KV Cache Tables/Blocks
    page_table: torch.Tensor = None

    # Encoder metadata
    # Cumulative sequence lengths for encoder key
    encoder_cu_seqlens_k: torch.Tensor = None
    # Maximum sequence length for encoder key
    encoder_max_seq_len_k: int = 0
    # Sequence lengths for the forward batch
    encoder_lens_int32: torch.Tensor = None
    # Page table for the encoder
    encoder_page_table: torch.Tensor = None

    @dataclass
    class LocalAttentionMetadata:
        local_query_start_loc: torch.Tensor = None  # cu_seqlens_q for local attention
        local_seqused_k: torch.Tensor = None  # sequence lengths for local attention
        local_block_table: torch.Tensor = None  # block table for local attention
        local_max_query_len: int = 0  # max query length for local attention
        local_max_seq_len: int = 0  # max sequence length for local attention

    local_attn_metadata: Optional[LocalAttentionMetadata] = None


# Copied from:
# https://github.com/houseroad/vllm/blob/4e45bfcaf928bdb9bd952b4ac922a3c205589ae8/vllm/v1/attention/backends/flash_attn.py
#
# Take in `query_start_loc_np` and `seq_lens_np` and break the sequences into
# local attention blocks, where each block is passed to the attention kernel
# as an independent local ("virtual") batch item.
#
# For example, if are performing a chunked prefill a batch of 3 sequences:
#   q_seqlens  = [4, 10, 5]
#   kv_seqlens = [6, 17, 9]
# Then normally for regular attention we would compute with an attention mask
#  for batch idx 0 (q_seqlens = 4, kv_seqlens = 6) like:
#   batch idx: 0 (q_seqlens = 4, kv_seqlens = 6)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 | 1 1 1 1 1
#               3 | 1 1 1 1 1 1
#
# for local attention (with attn_chunk_size = 4) we would compute with an
#  attention mask like:
#   batch idx: 0  (q_seqlens = 4, kv_seqlens = 6, attn_chunk_size = 4)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 |         1
#               3 |         1 1
#
# We can simulate this mask using standard flash-attention by breaking the
#  sequences into local ("virtual") batches, where each local batch item is a
#  local attention block, so in this case batch idx 0 would be broken up into:
#
#   local-batch idx: 0 (q_seqlens = 2, kv_seqlens = 4)  (batch 0)
#        k_toks >   0 1 2 3
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#   local-batch idx: 1 (q_seqlens = 2, kv_seqlens = 2) (batch 0)
#        k_toks >   4 5
#        q_toks v  _____________
#               2 | 1
#               3 | 1 1
#
# e.g. if we have:
#   attn_chunk_size = 4
#   query_start_loc_np = [0, 4, 14, 19] (q_seqlens = [4, 10, 5])
# Then this function would return:
#                           __b0__  ______b1______  __b2__ < orig batch indices
#   q_seqlens_local    = [   2,  2,  1,  4,  4,  1,  4,  1]
#   cu_seqlens_q_local = [0, 4,  6, 10, 14, 18, 19, 23, 24]
#   seqlens_k_local    = [   4,  2,  4,  4,  4,  1,  4,  1]
#   block_table_local  : shape[local_virtual_batches, pages_per_local_batch]
@torch.no_grad()  # metadata builder; keep out of autograd
def make_local_attention_virtual_batches(
    attn_chunk_size: int,
    query_start_loc: torch.Tensor,     # int32, device==block_table.device
    seq_lens: torch.Tensor,            # int32, device==block_table.device
    block_table: torch.Tensor,         # int32, [B, P]
    page_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        seqlens_q_local      : int32 [V]
        cu_seqlens_q_local   : int32 [V+1]
        seqlens_k_local      : int32 [V]
        block_table_local    : int32 [V, pages_per_local_batch]
    """
    dev = block_table.device
    i32, i64 = torch.int32, torch.int64

    # Normalize dtypes/devices once
    query_start_loc = query_start_loc.to(dev, i32, copy=False)
    seq_lens        = seq_lens.to(dev, i32, copy=False)
    block_table     = block_table.to(dev, i32, copy=False)

    # --- effective, page-aligned chunk size (CRITICAL) ---
    if seq_lens.numel() == 0:
        zero = torch.zeros(0, dtype=i32, device=dev)
        cu1  = torch.zeros(1, dtype=i32, device=dev)
        pages_per_local_batch = max(1, (attn_chunk_size // max(page_size, 1)))  # harmless
        return zero, cu1, zero, zero.view(0, pages_per_local_batch)

    max_seq_len = int(seq_lens.max().item())
    if page_size > 0:
        # floor to multiple of page_size, cap by actual max, and never below one page
        eff = min(attn_chunk_size, max_seq_len)
        eff = max(page_size, (eff // page_size) * page_size)
        pages_per_local_batch = eff // page_size
        PAGE = torch.tensor(page_size, dtype=i32, device=dev)
    else:
        # no paging: just cap by max length
        eff = min(attn_chunk_size, max_seq_len)
        pages_per_local_batch = eff
        PAGE = None

    ATTN = torch.tensor(eff, dtype=i32, device=dev)

    # q_seqlens per request
    q_seqlens = query_start_loc[1:] - query_start_loc[:-1]   # [B]
    B = q_seqlens.shape[0]

    # new_tokens_in_first_block: min(attn - ((k - q) % attn), q)
    q_tokens_in_first_block = torch.minimum(
        ATTN - torch.remainder(seq_lens - q_seqlens, ATTN),
        q_seqlens,
    )
    # tokens_in_last_block \in [1, ATTN]
    tokens_in_last_block = torch.remainder(seq_lens - 1, ATTN) + 1

    # local_blocks = 1 + ceil(max(q - first, 0)/ATTN)
    rem = torch.clamp_min(q_seqlens - q_tokens_in_first_block, 0)
    local_blocks = 1 + torch.div(rem + (ATTN - 1), ATTN, rounding_mode="floor")   # [B], i32

    # Prefix-sum => total V (virtual batches)
    cu_num_blocks = torch.cumsum(local_blocks, dim=0)                              # [B]
    V = int(cu_num_blocks[-1].item())

    # arange per virtual item and request id via bucketize (faster than repeat_interleave)
    arange = torch.arange(V, dtype=i32, device=dev)                                 # [V]
    # req_ids in [0, B-1]; bucketize returns int64
    req_ids = torch.bucketize(arange, cu_num_blocks, right=True).to(i64)          # [V]

    # Offset within each request’s block list: start index of that request’s region
    starts = torch.nn.functional.pad(cu_num_blocks[:-1], (1, 0))                   # [B]
    starts = starts.index_select(0, req_ids.to(i64)).to(i32)                        # [V]
    pos_in_req = arange - starts                                                    # [V]  (0..local_blocks[req]-1)
    rarange    = local_blocks.index_select(0, req_ids.to(i32)).to(i32) - pos_in_req - 1

    # Build seqlens_q_local branchlessly
    base = (q_seqlens - q_tokens_in_first_block).index_select(0, req_ids.to(i32))   # [V]
    first_mask = (pos_in_req == 0)
    rem_q = torch.minimum(base - ATTN * (pos_in_req - 1), ATTN)
    seqlens_q_local = torch.where(first_mask,
                                  q_tokens_in_first_block.index_select(0, req_ids.to(i32)),
                                  rem_q).clamp_min_(0)                              # [V], i32

    # cu_seqlens_q_local (length V+1, starting with 0)
    cu_seqlens_q_local = torch.nn.functional.pad(
        torch.cumsum(seqlens_q_local, dim=0, dtype=i32), (1, 0)
    )

    # seqlens_k_local: full chunks except last block of each request
    seqlens_k_local = ATTN.expand(V).clone()
    # last block index per request == cu_num_blocks-1; scatter tokens_in_last_block there
    last_block_positions = (cu_num_blocks - 1).to(i64)                              # [B]
    seqlens_k_local.index_copy_(
        0, last_block_positions, tokens_in_last_block.to(i32)
    )

    # Absolute key starts per virtual item
    # k_seqstarts_absolute = k_len(req)- (rarange*ATTN + tokens_in_last_block(req))
    k_len_req  = seq_lens.index_select(0, req_ids.to(i32))
    last_tok   = tokens_in_last_block.index_select(0, req_ids.to(i32))
    k_seqstarts_absolute = k_len_req - (rarange * ATTN + last_tok)                  # [V]
    if PAGE is not None:
        block_starts = torch.div(k_seqstarts_absolute, PAGE, rounding_mode="floor").clamp_min_(0)  # [V]
    else:
        block_starts = k_seqstarts_absolute.clamp_min_(0)

    # Build per-local-batch page indices: [V, pages_per_local_batch]
    per_block = torch.arange(pages_per_local_batch, dtype=i32, device=dev)          # [PBLK]
    block_indices = block_starts.unsqueeze(1) + per_block.unsqueeze(0)              # [V,PBLK]
    # Clamp to block_table width to avoid OOB
    max_col = block_table.shape[1] - 1
    if max_col >= 0:
        block_indices.clamp_(0, int(max_col))

    # Row selection once, then column gather with take_along_dim (better fusion)
    bt_rows = block_table.index_select(0, req_ids)                                  # [V, P]
    block_table_local = torch.take_along_dim(bt_rows, block_indices.to(i64), dim=1) # [V,PBLK]

    return seqlens_q_local, cu_seqlens_q_local, seqlens_k_local, block_table_local



def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


# TODO(hebiao064): remove this once we have a better way to handle the merge_state_v2 torch.compile issue
@torch._dynamo.disable()
def merge_state_v2_wrapper(o, s_a, o_exp, s_b):
    return merge_state_v2(o, s_a, o_exp, s_b)


class FlashAttentionBackend(AttentionBackend):
    """FlashAttention backend implementation.

    Note about the init:
    - If no spec decoding
        - FlashAttentionBackend will be init once when the server starts.
    - If spec decoding
        - FlashAttentionBackend will be init once for the target worker
        - FlashAttentionMultiStepBackend will be once for the draft worker
            - It will spawn num_steps FlashAttentionBackend for the draft worker

    Note about CUDA Graph:
    - We only support CUDA Graph for Decode (Normal Decode and Draft Decode) and Target Verify.
    - We don't support CUDA Graph for Extend and Draft Extend.
    - When server init, init_cuda_graph_state will be called first and then init_cuda_graph_capture will be called.
    - For each forward batch, init_replay_cuda_graph will be called first and then replay the graph.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        self.forward_metadata: FlashAttentionMetadata = None
        # extra metadata for handling speculative decoding topk > 1, extended draft decode and verify
        self.forward_metadata_spec_decode_expand: FlashAttentionMetadata = None
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.decode_cuda_graph_metadata = {}
        self.target_verify_metadata = {}
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
        self.page_size = model_runner.page_size
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        self.skip_prefill = skip_prefill
        self.is_hybrid = model_runner.is_hybrid
        if self.is_hybrid:
            self.full_to_swa_index_mapping = (
                model_runner.token_to_kv_pool.full_to_swa_index_mapping
            )
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id

        # Local attention settings
        self.attention_chunk_size = (
            model_runner.attention_chunk_size
            if hasattr(model_runner, "attention_chunk_size")
            else None
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize forward metadata hence all layers in the forward pass can reuse it."""
        metadata = FlashAttentionMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device

        if forward_batch.forward_mode.is_decode_or_idle():
            # Draft Decode
            if forward_batch.spec_info is not None:
                if self.topk <= 1:
                    metadata.cache_seqlens_int32 = (
                        seqlens_in_batch + (self.speculative_step_id + 1)
                    ).to(torch.int32)
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item() + (
                        self.speculative_step_id + 1
                    )
                    metadata.cu_seqlens_q = torch.arange(
                        0, batch_size + 1, dtype=torch.int32, device=device
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]
                else:
                    metadata.cache_seqlens_int32 = (seqlens_in_batch).to(torch.int32)
                    metadata.max_seq_len_q = self.topk
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                    metadata.cu_seqlens_q = torch.arange(
                        0,
                        batch_size * self.topk + 1,
                        step=self.topk,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]

                    metadata_expand = FlashAttentionMetadata()
                    decode_length = self.speculative_step_id + 1
                    metadata_expand.cache_seqlens_int32 = torch.full(
                        (seqlens_in_batch.numel() * self.topk,),
                        decode_length,
                        device=device,
                        dtype=torch.int32,
                    )
                    metadata_expand.max_seq_len_q = 1
                    metadata_expand.cu_seqlens_q = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata_expand.cu_seqlens_k = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() * decode_length + 1,
                        step=decode_length,
                        dtype=torch.int32,
                        device=device,
                    )
                    # shape: [bs, num_steps, topk] -> [bs x topk, num_steps]
                    cache_loc = forward_batch.out_cache_loc.view(
                        -1, self.speculative_num_steps
                    )
                    metadata_expand.page_table = (
                        cache_loc[:, :decode_length].contiguous().to(torch.int32)
                    )
                    self.forward_metadata_spec_decode_expand = metadata_expand
            else:
                # Normal Decode
                metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]
            # TODO: we need to test this part for llama 4 eagle case
            self._init_local_attn_metadata(forward_batch, metadata, device)
        elif forward_batch.forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata.cache_seqlens_int32 = (
                    forward_batch.seq_lens + self.speculative_num_draft_tokens
                ).to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = (
                    forward_batch.seq_lens_cpu.max().item()
                    + self.speculative_num_draft_tokens
                )
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                self._init_local_attn_metadata(forward_batch, metadata, device)
            else:
                metadata.cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                metadata_expand = FlashAttentionMetadata()

                metadata_expand.max_seq_len_q = 1
                metadata_expand.cu_seqlens_q = torch.arange(
                    0,
                    forward_batch.seq_lens.numel() * self.speculative_num_draft_tokens
                    + 1,
                    dtype=torch.int32,
                    device=device,
                )

                # create expand page table
                offsets = torch.arange(
                    self.speculative_num_draft_tokens, device=device
                ).unsqueeze(
                    0
                )  # shape: (1, self.speculative_num_draft_tokens)
                cols = offsets.expand(
                    forward_batch.seq_lens.numel(), -1
                ) + forward_batch.seq_lens.unsqueeze(1)
                cum_len = torch.nn.functional.pad(
                    torch.cumsum(
                        (
                            forward_batch.seq_lens + self.speculative_num_draft_tokens
                        ).repeat_interleave(self.speculative_num_draft_tokens),
                        dim=0,
                    ),
                    (1, 0),
                )[:-1]
                mask_extraction_indices = (
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                    + cum_len[:, None]
                ).view(1, -1)
                mask = forward_batch.spec_info.custom_mask[
                    mask_extraction_indices
                ].view(
                    -1, self.speculative_num_draft_tokens
                )  # (bsz * draft_num, draft_num)

                # shift table indices to avoid padding
                # non_masked_page_table [[8, 9, 10],   mask (display with int format) [[1, 0, 0],
                #                        [8, 9, 10],                                   [1, 1, 0],
                #                        [8, 9, 10]]                                   [1, 0, 1]]
                # if masked with padding [[8, 0, 0],   our mask without padding       [[8, 9, 10],
                #                        [8, 9, 0],                                    [8, 9, 10],
                #                        [8, 0, 10]]                                   [8, 10, 9]]
                # note here cache_seqlens_int32 is [1, 2, 2] so extra page indices will be ignored in each row
                col_indices = offsets.expand(
                    mask.shape[0], self.speculative_num_draft_tokens
                )
                # Build keys: if an entry is valid (mask==True), keep its original index;
                # if not, add self.speculative_num_draft_tokens so that it sorts after all valid entries.
                keys = torch.where(
                    mask, col_indices, col_indices + self.speculative_num_draft_tokens
                )
                _, sort_order = torch.sort(keys, dim=1)
                non_masked_page_table = (
                    forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, :
                    ]
                    .gather(1, cols)
                    .repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                )  # (bsz, draft_num)
                metadata_expand.page_table = non_masked_page_table.gather(1, sort_order)
                metadata_expand.cache_seqlens_int32 = mask.sum(dim=1).to(torch.int32)
                metadata_expand.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata_expand.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                self.forward_metadata_spec_decode_expand = metadata_expand
        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]

            if (
                any(forward_batch.extend_prefix_lens_cpu)
                or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
            ):
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = metadata.cu_seqlens_k

            # Setup local attention if enabled
            if forward_batch.forward_mode == ForwardMode.EXTEND:
                self._init_local_attn_metadata(forward_batch, metadata, device)

        # Encoder metadata for cross attention
        if forward_batch.encoder_lens is not None:
            assert (
                forward_batch.encoder_lens.numel() == 1
            ), "Only encoder size 1 is supported for now"

            metadata.encoder_lens_int32 = forward_batch.encoder_lens.to(torch.int32)
            metadata.encoder_cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(metadata.encoder_lens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
            metadata.encoder_max_seq_len_k = metadata.encoder_lens_int32.max().item()
            metadata.encoder_page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.encoder_max_seq_len_k
            ]

            # Currently only support forward_batch.encoder_lens.numel() == 1
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices,
                metadata.encoder_max_seq_len_k : (
                    metadata.encoder_max_seq_len_k + metadata.max_seq_len_k
                ),
            ]

        # Convert the page table to a strided format which is needed by FA3 API
        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )
            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )

        self.forward_metadata = metadata

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ):
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        window_size = (
            (layer.sliding_window_size, 0)
            if layer.sliding_window_size is not None and layer.sliding_window_size > -1
            else (-1, -1)
        )
        k_descale, v_descale = None, None
        # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # has corresponding quantization method so that layer.k_scale is not None,
        # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case.
        if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        causal = not layer.is_cross_attention

        # Check if we should use local attention
        use_local_attn = (
            self.attention_chunk_size is not None
            and metadata.local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        # We do cascade attention for Target Verify with topk > 1
        use_cascade_attn = (
            forward_batch.forward_mode.is_target_verify() and self.topk > 1
        )

        # For fa3 interface version compatibility, we put new fields into conditional keyword args
        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks

        # Get the appropriate page table based on whether we're using local attention
        if use_local_attn:
            local_metadata = metadata.local_attn_metadata
            page_table = local_metadata.local_block_table
            cu_seqlens_q = local_metadata.local_query_start_loc
            cache_seqlens = local_metadata.local_seqused_k
            max_seqlen_q = local_metadata.local_max_query_len
            max_seqlen_k = local_metadata.local_max_seq_len
        else:
            page_table = metadata.page_table
            cu_seqlens_q = metadata.cu_seqlens_q
            cache_seqlens = metadata.cache_seqlens_int32
            max_seqlen_q = metadata.max_seq_len_q
            max_seqlen_k = metadata.max_seq_len_k
            cu_seqlens_k = metadata.cu_seqlens_k

        # Use Flash Attention for prefill
        if not self.use_mla:
            # Do multi-head attention
            key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )
            if layer.is_cross_attention:
                page_table = metadata.encoder_page_table
                cache_seqlens = metadata.encoder_lens_int32
                cu_seqlens_k = metadata.encoder_cu_seqlens_k
                window_size = (-1, -1)

            result = flash_attn_with_kvcache(
                q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k_cache=key_cache,
                v_cache=value_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=False if use_cascade_attn else causal,
                window_size=window_size,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=use_cascade_attn,
                **kwargs,
            )

            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                    **kwargs,
                )
                o, _ = merge_state_v2_wrapper(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result
        else:
            if (
                forward_batch.attn_attend_prefix_cache is not None
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
            ):
                # Do multi-head attention with chunked prefix cache
                if forward_batch.attn_attend_prefix_cache:
                    assert not global_server_args_dict["disable_chunked_prefix_cache"]
                    # MHA for chunked prefix kv cache when running model with MLA
                    assert forward_batch.prefix_chunk_idx is not None
                    assert forward_batch.prefix_chunk_cu_seq_lens is not None
                    assert forward_batch.prefix_chunk_max_seq_lens is not None

                    chunk_idx = forward_batch.prefix_chunk_idx
                    assert chunk_idx >= 0

                    assert forward_batch.mha_return_lse
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                        softmax_scale=layer.scaling,
                        causal=False,
                        return_softmax_lse=True,
                    )
                else:
                    # MHA for extend part of sequence without attending prefix kv cache
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=metadata.cu_seqlens_q,
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=metadata.max_seq_len_q,
                        softmax_scale=layer.scaling,
                        causal=True,
                        return_softmax_lse=forward_batch.mha_return_lse,
                    )
                if forward_batch.mha_return_lse:
                    output, lse, *rest = output
                    lse = torch.transpose(lse, 0, 1).contiguous()
                    return output, lse
                return output
            else:
                # Do absorbed multi-latent attention
                kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                    layer.layer_id
                ).to(q.dtype)
                k_rope = kv_cache[:, :, layer.v_head_dim :]
                c_kv = kv_cache[:, :, : layer.v_head_dim]
                k_rope_cache = k_rope.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    layer.head_dim - layer.v_head_dim,
                )
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )
                if q_rope is not None:
                    q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                    q_rope = q_rope.view(
                        -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                    )
                else:
                    q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                    q_nope = q_all[:, :, : layer.v_head_dim]
                    q_rope = q_all[:, :, layer.v_head_dim :]

                result = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_rope,
                            k_cache=k_rope_cache,
                            v_cache=c_kv_cache,
                            qv=q_nope,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                        )
                    )
                    o, _ = merge_state_v2_wrapper(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata
        local_attn_metadata = getattr(metadata, "local_attn_metadata", None)
        use_local_attn = (
            self.attention_chunk_size is not None
            and local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        # When Spec Decode enabled, forward_decode would be called with two mode:
        # 1. DRAFT_DECODE: we enable cascade attention when top_k > 1
        # 2. IDLE: we don’t need cascade attention, spec_info will be none in this case
        use_cascade_attn = forward_batch.spec_info is not None and self.topk > 1

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        window_size = (
            (layer.sliding_window_size, 0)
            if layer.sliding_window_size is not None and layer.sliding_window_size > -1
            else (-1, -1)
        )
        causal = not layer.is_cross_attention

        # For fa3 interface version compatibility, we put new fields into conditional keyword args
        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks

        k_descale, v_descale = None, None
        # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # has corresponding quantization method so that layer.k_scale is not None,
        # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case.
        if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        if not self.use_mla:
            # Do multi-head attention

            key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )

            if layer.is_cross_attention:
                # Always use non-chunked logic for cross-attention
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=metadata.encoder_page_table,
                    cache_seqlens=metadata.encoder_lens_int32,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=metadata.encoder_cu_seqlens_k,
                    max_seqlen_q=1,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    **kwargs,
                )
            elif use_local_attn:
                # Use chunked (local) attention batching for self-attention
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=local_attn_metadata.local_block_table,
                    cache_seqlens=local_attn_metadata.local_seqused_k,
                    cu_seqlens_q=local_attn_metadata.local_query_start_loc,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=local_attn_metadata.local_max_query_len,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    **kwargs,
                )
            else:
                page_table = metadata.page_table
                cache_seqlens = metadata.cache_seqlens_int32
                cu_seqlens_k = metadata.cu_seqlens_k
                max_seqlen_q = metadata.max_seq_len_q
                q_reshaped = q.contiguous().view(
                    -1, layer.tp_q_head_num, layer.head_dim
                )

                # Default: single-token self-attention
                result = flash_attn_with_kvcache(
                    q=q_reshaped,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    **kwargs,
                )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_reshaped,
                            k_cache=key_cache,
                            v_cache=value_cache,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                            **kwargs,
                        )
                    )
                    o, _ = merge_state_v2(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result
        else:
            # Do absorbed multi-latent attention
            kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
                q.dtype
            )
            k_rope = kv_cache[:, :, layer.v_head_dim :]
            c_kv = kv_cache[:, :, : layer.v_head_dim]
            k_rope_cache = k_rope.view(
                -1,
                self.page_size,
                layer.tp_k_head_num,
                layer.head_dim - layer.v_head_dim,
            )
            c_kv_cache = c_kv.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )

            if q_rope is not None:
                q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope = q_rope.view(
                    -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
            else:
                q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                q_nope = q_all[:, :, : layer.v_head_dim]
                q_rope = q_all[:, :, layer.v_head_dim :]
            max_seqlen_q = metadata.max_seq_len_q

            result = flash_attn_with_kvcache(
                q=q_rope,
                k_cache=k_rope_cache,
                v_cache=c_kv_cache,
                qv=q_nope,
                page_table=metadata.page_table,
                cache_seqlens=metadata.cache_seqlens_int32,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k_new=metadata.cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=False if use_cascade_attn else causal,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=use_cascade_attn,  # softmax_lse is needed for merge states
            )
            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                )
                o, _ = merge_state_v2(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Initialize CUDA graph state for the attention backend.

        Args:
            max_bs (int): Maximum batch size to support in CUDA graphs

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        """
        max_num_pages = (self.max_context_len + self.page_size - 1) // self.page_size

        # This is being used by normal decode and draft decode when topk == 1
        self.decode_cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "page_table": torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            ),
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
        }
        # Only allocate local attention buffers if local attention is enabled
        # This prevents OOM errors when local attention is not being used
        if self.attention_chunk_size is not None:
            # Estimate maximum sizes for local attention metadata
            max_seq_len = self.max_context_len
            page_size = self.page_size or 1
            attn_chunk_size = self.attention_chunk_size
            max_virtual_batches = max_bs * (
                (max_seq_len + attn_chunk_size - 1) // attn_chunk_size
            )
            max_pages_per_block = (attn_chunk_size + page_size - 1) // page_size

            self.decode_cuda_graph_local_attn_metadata = {
                "local_query_start_loc": torch.zeros(
                    max_virtual_batches + 1, dtype=torch.int32, device=self.device
                ),
                "local_seqused_k": torch.zeros(
                    max_virtual_batches, dtype=torch.int32, device=self.device
                ),
                "local_block_table": torch.zeros(
                    max_virtual_batches,
                    max_pages_per_block,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

        # This is used by draft decode's first half of metadata when topk > 1
        if self.topk > 1:
            self.draft_decode_metadata_topk_normal = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.topk + 1,
                    step=self.topk,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    self.max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

            # This is used by draft decode's second half of metadata when topk > 1
            decode_length = self.speculative_step_id + 1
            self.draft_decode_metadata_topk_expand = {
                "cache_seqlens": torch.full(
                    (max_bs * self.topk,),
                    decode_length,
                    device=self.device,
                    dtype=torch.int32,
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.topk + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.arange(
                    0,
                    max_bs * self.topk * decode_length + 1,
                    step=decode_length,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "page_table": torch.zeros(
                    max_bs * self.topk,
                    decode_length,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

        if (
            self.speculative_num_draft_tokens is not None
            and self.speculative_num_draft_tokens > 0
        ):
            # "page_table_draft_decode" will be set only when spec decoding enabled to save memory
            self.decode_cuda_graph_metadata["page_table_draft_decode"] = torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            )

            self.target_verify_metadata = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
            }

            self.draft_extend_metadata = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.zeros(
                    max_bs + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
            }

        if self.topk > 1:
            self.target_verify_metadata_topk_normal = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    self.max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

            self.target_verify_metadata_topk_expand = {
                "cache_seqlens": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "page_table": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

        self.encoder_metadata = {
            "encoder_page_table": torch.zeros(
                max_bs,
                self.max_context_len,
                dtype=torch.int32,
                device=self.device,
            ),
            "encoder_lens_int32": torch.zeros(
                max_bs, dtype=torch.int32, device=self.device
            ),
            "encoder_cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
        }

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        """Initialize forward metadata for capturing CUDA graph."""
        metadata = FlashAttentionMetadata()

        # metadata_expand is needed for Spec Decoding when top k > 1
        metadata_expand = FlashAttentionMetadata()

        device = seq_lens.device
        if forward_mode.is_decode_or_idle():
            if spec_info is not None:
                # Draft Decode
                if self.topk <= 1:
                    # When topk = 1, we use the normal decode metadata
                    metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                        "cache_seqlens"
                    ][:bs]
                    metadata.max_seq_len_k = seq_lens.max().item() + (
                        self.speculative_step_id + 1
                    )
                    metadata.cu_seqlens_q = self.decode_cuda_graph_metadata[
                        "cu_seqlens_q"
                    ][: bs + 1]
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = self.decode_cuda_graph_metadata[
                        "page_table_draft_decode"
                    ][:bs, :]
                    self.decode_cuda_graph_metadata[bs] = metadata
                else:
                    # When top k > 1, we need two specific draft decode metadata, and then merge states
                    # 1. The first half of metadata for prefix tokens
                    metadata.cache_seqlens_int32 = (
                        self.draft_decode_metadata_topk_normal["cache_seqlens"][:bs]
                    )
                    metadata.max_seq_len_q = self.topk
                    metadata.max_seq_len_k = seq_lens.max().item()
                    metadata.cu_seqlens_q = self.draft_decode_metadata_topk_normal[
                        "cu_seqlens_q"
                    ][: bs + 1]
                    metadata.cu_seqlens_k = self.draft_decode_metadata_topk_normal[
                        "cu_seqlens_k"
                    ][: bs + 1]
                    metadata.page_table = self.draft_decode_metadata_topk_normal[
                        "page_table"
                    ][:bs, :]

                    # 2. The second half of metadata for draft tokens (per_batch_num_tokens = topk)
                    metadata_expand.cache_seqlens_int32 = (
                        self.draft_decode_metadata_topk_expand["cache_seqlens"][
                            : bs * self.topk
                        ]
                    )
                    metadata_expand.max_seq_len_q = 1
                    metadata_expand.cu_seqlens_q = (
                        self.draft_decode_metadata_topk_expand["cu_seqlens_q"][
                            : bs * self.topk + 1
                        ]
                    )
                    metadata_expand.cu_seqlens_k = (
                        self.draft_decode_metadata_topk_expand["cu_seqlens_k"][
                            : bs * self.topk + 1
                        ]
                    )
                    metadata_expand.page_table = self.draft_decode_metadata_topk_expand[
                        "page_table"
                    ][: bs * self.topk]
                    self.draft_decode_metadata_topk_normal[bs] = metadata
                    self.draft_decode_metadata_topk_expand[bs] = metadata_expand
            else:
                # Normal Decode
                # Get sequence information
                metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)
                batch_size = len(seq_lens)
                device = seq_lens.device
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
                # Precompute maximum sequence length
                metadata.max_seq_len_k = seq_lens.max().item()
                # Precompute page table
                metadata.page_table = self.decode_cuda_graph_metadata["page_table"][
                    :bs, :
                ]
                # Precompute cumulative sequence lengths
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                self.decode_cuda_graph_metadata[bs] = metadata

                if self.attention_chunk_size is not None:
                    self._update_local_attn_metadata_for_capture(metadata, batch_size)

        elif forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata.cache_seqlens_int32 = self.target_verify_metadata[
                    "cache_seqlens"
                ][:bs]
                metadata.cache_seqlens_int32.copy_(
                    (seq_lens + self.speculative_num_draft_tokens)
                )

                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = (
                    seq_lens.max().item() + self.speculative_num_draft_tokens
                )

                metadata.cu_seqlens_q = torch.arange(
                    0,
                    bs * self.speculative_num_draft_tokens + 1,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )

                metadata.cu_seqlens_k = self.target_verify_metadata["cu_seqlens_k"][
                    : (bs + 1)
                ]

                metadata.page_table = self.target_verify_metadata["page_table"][:bs, :]

                self.target_verify_metadata[bs] = metadata
            else:
                # When topk > 1, we need two specific target verify metadata, and then merge states
                # 1. The first half of metadata for prefix tokens
                metadata.cache_seqlens_int32 = self.target_verify_metadata_topk_normal[
                    "cache_seqlens"
                ][:bs]
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                # metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item(), do this in replay
                metadata.cu_seqlens_q = self.target_verify_metadata_topk_normal[
                    "cu_seqlens_q"
                ][: bs + 1]
                metadata.cu_seqlens_k = self.target_verify_metadata_topk_normal[
                    "cu_seqlens_k"
                ][: bs + 1]
                metadata.page_table = self.target_verify_metadata_topk_normal[
                    "page_table"
                ][:bs, :]

                # 2. The second half of metadata for draft tokens (per_batch_num_tokens = topk)
                metadata_expand.cache_seqlens_int32 = (
                    self.target_verify_metadata_topk_expand["cache_seqlens"][
                        : bs * self.speculative_num_draft_tokens
                    ]
                )
                metadata_expand.max_seq_len_q = 1
                metadata_expand.cu_seqlens_q = self.target_verify_metadata_topk_expand[
                    "cu_seqlens_q"
                ][: bs * self.speculative_num_draft_tokens + 1]
                metadata_expand.cu_seqlens_k = self.target_verify_metadata_topk_expand[
                    "cu_seqlens_k"
                ][: bs * self.speculative_num_draft_tokens + 1]

                metadata_expand.page_table = self.target_verify_metadata_topk_expand[
                    "page_table"
                ][: bs * self.speculative_num_draft_tokens]

                self.target_verify_metadata_topk_normal[bs] = metadata
                self.target_verify_metadata_topk_expand[bs] = metadata_expand
        elif forward_mode.is_draft_extend():
            metadata.cache_seqlens_int32 = self.draft_extend_metadata["cache_seqlens"][
                :bs
            ]
            metadata.cache_seqlens_int32.copy_(seq_lens)

            num_tokens_per_bs = num_tokens // bs
            metadata.max_seq_len_q = num_tokens_per_bs
            metadata.max_seq_len_k = seq_lens.max().item()

            metadata.cu_seqlens_q = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                num_tokens_per_bs,
                dtype=torch.int32,
                device=device,
            )

            metadata.cu_seqlens_k = self.draft_extend_metadata["cu_seqlens_k"][
                : (bs + 1)
            ]
            metadata.page_table = self.draft_extend_metadata["page_table"][:bs, :]

            self.draft_extend_metadata[bs] = metadata

        if encoder_lens is not None:
            encoder_bs = encoder_lens.numel()
            metadata.encoder_lens_int32 = self.encoder_metadata["encoder_lens_int32"][
                :encoder_bs
            ]
            metadata.encoder_cu_seqlens_k = self.encoder_metadata[
                "encoder_cu_seqlens_k"
            ][: (encoder_bs + 1)]

            metadata.encoder_page_table = self.encoder_metadata["encoder_page_table"][
                :bs, :
            ]

        self.forward_metadata = metadata
        self.forward_metadata_spec_decode_expand = metadata_expand

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        """Initialize forward metadata for replaying CUDA graph."""
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]
        device = seq_lens.device
        metadata = None
        metadata_expand = None

        if forward_mode.is_decode_or_idle():

            if spec_info is not None:
                # Draft Decode
                if self.topk <= 1:
                    # When topk = 1, we use the normal decode metadata
                    metadata = self.decode_cuda_graph_metadata[bs]
                    max_len = seq_lens_cpu.max().item()
                    metadata.max_seq_len_k = max_len + self.speculative_step_id + 1
                    max_seq_pages = (
                        metadata.max_seq_len_k + self.page_size - 1
                    ) // self.page_size

                    normal_decode_set_metadata(
                        metadata.cache_seqlens_int32,
                        metadata.cu_seqlens_k,
                        metadata.page_table,
                        self.req_to_token,
                        req_pool_indices,
                        self.decode_cuda_graph_metadata["strided_indices"],
                        max_seq_pages,
                        seq_lens,
                        self.speculative_step_id + 1,
                        self.page_size,
                    )

                else:
                    # When top k > 1, we need two specific draft decode metadata, and then merge states
                    # 1. The first half of metadata for prefix tokens
                    metadata = self.draft_decode_metadata_topk_normal[bs]
                    metadata.cache_seqlens_int32.copy_(seq_lens)
                    # metadata.max_seq_len_q = self.topk, already set in capture
                    metadata.max_seq_len_k = seq_lens_cpu.max().item()
                    # metadata.cu_seqlens_q already set in capture
                    metadata.cu_seqlens_k[1:].copy_(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        )
                    )

                    page_table = self.req_to_token[
                        req_pool_indices, : metadata.max_seq_len_k
                    ]

                    metadata.page_table[:, : metadata.max_seq_len_k].copy_(page_table)

                    # 2. The second half of metadata for draft tokens (per_batch_num_tokens = topk)
                    metadata_expand = self.draft_decode_metadata_topk_expand[bs]
                    decode_length = self.speculative_step_id + 1
                    # shape: [bs, num_steps, topk] -> [bs x topk, num_steps]
                    cache_loc = out_cache_loc.view(-1, self.speculative_num_steps)
                    metadata_expand.page_table[: cache_loc.shape[0]].copy_(
                        cache_loc[:, :decode_length]
                    )
                # TODO: Handle local attention metadata for draft decode when llama4 eagle is supported
            else:
                # Normal Decode
                metadata = self.decode_cuda_graph_metadata[bs]
                max_len = seq_lens_cpu.max().item()
                max_seq_pages = (max_len + self.page_size - 1) // self.page_size
                metadata.max_seq_len_k = max_len

                normal_decode_set_metadata(
                    metadata.cache_seqlens_int32,
                    metadata.cu_seqlens_k,
                    metadata.page_table,
                    self.req_to_token,
                    req_pool_indices,
                    self.decode_cuda_graph_metadata["strided_indices"],
                    max_seq_pages,
                    seq_lens,
                    0,
                    self.page_size,
                )

                self._update_local_attn_metadata_for_replay(
                    metadata,
                    bs,
                )
        elif forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata = self.target_verify_metadata[bs]
                metadata.cache_seqlens_int32.copy_(
                    (seq_lens + self.speculative_num_draft_tokens)
                )

                metadata.max_seq_len_k = (
                    seq_lens_cpu.max().item() + self.speculative_num_draft_tokens
                )
                metadata.cu_seqlens_k[1:].copy_(
                    torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
                )
                max_seq_pages = (
                    metadata.max_seq_len_k + self.page_size - 1
                ) // self.page_size
                page_indices = self.req_to_token[
                    req_pool_indices[:, None],
                    self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages],
                ]
                page_indices //= self.page_size
                metadata.page_table[:, :max_seq_pages].copy_(page_indices)
            else:
                # When topk > 1, we need two specific target verify metadata, and then merge states
                # 1. The first half of metadata for prefix tokens
                metadata = self.target_verify_metadata_topk_normal[bs]
                metadata.cache_seqlens_int32.copy_(seq_lens)
                # metadata.max_seq_len_q = self.speculative_num_draft_tokens, already set in capture
                metadata.max_seq_len_k = seq_lens_cpu.max().item()
                # metadata.cu_seqlens_q already set in capture
                metadata.cu_seqlens_k[1:].copy_(
                    torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
                )
                page_table = self.req_to_token[
                    req_pool_indices, : metadata.max_seq_len_k
                ]
                metadata.page_table[:, : metadata.max_seq_len_k].copy_(page_table)

                # 2. The second half of metadata for draft tokens (per_batch_num_tokens = topk)
                metadata_expand = self.target_verify_metadata_topk_expand[bs]

                # metadata_expand.max_seq_len_q = 1, already set in capture
                # metadata_expand.cu_seqlens_q already set in capture
                offsets = torch.arange(
                    self.speculative_num_draft_tokens, device=device
                ).unsqueeze(
                    0
                )  # shape: (1, self.speculative_num_draft_tokens)

                cols = offsets.expand(seq_lens.numel(), -1) + seq_lens.unsqueeze(1)
                cum_len = torch.nn.functional.pad(
                    torch.cumsum(
                        (
                            seq_lens + self.speculative_num_draft_tokens
                        ).repeat_interleave(self.speculative_num_draft_tokens),
                        dim=0,
                    ),
                    (1, 0),
                )[:-1]
                mask_extraction_indices = (
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                    + cum_len[:, None]
                ).view(1, -1)
                # avoid extracting padded seq indices which will be out of boundary
                mask_extraction_indices[
                    :,
                    spec_info.positions.numel() * self.speculative_num_draft_tokens :,
                ].fill_(0)
                mask = spec_info.custom_mask[mask_extraction_indices].view(
                    -1, self.speculative_num_draft_tokens
                )  # (bsz * draft_num, draft_num)

                col_indices = offsets.expand(
                    mask.shape[0], self.speculative_num_draft_tokens
                )
                keys = torch.where(
                    mask,
                    col_indices,
                    col_indices + self.speculative_num_draft_tokens,
                )
                _, sort_order = torch.sort(keys, dim=1)

                non_masked_page_table = (
                    self.req_to_token[req_pool_indices, :]
                    .gather(1, cols)
                    .repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                )  # (bsz, draft_num)

                metadata_expand.page_table.copy_(
                    non_masked_page_table.gather(1, sort_order)
                )
                metadata_expand.cache_seqlens_int32.copy_(mask.sum(dim=1))
                metadata_expand.cu_seqlens_k[1:].copy_(
                    torch.cumsum(
                        metadata_expand.cache_seqlens_int32,
                        dim=0,
                        dtype=torch.int32,
                    )
                )

        elif forward_mode.is_draft_extend():
            metadata = self.draft_extend_metadata[bs]
            metadata.cache_seqlens_int32.copy_(seq_lens)

            metadata.max_seq_len_k = seq_lens_cpu.max().item()
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
            )
            accept_length = spec_info.accept_length[:bs]
            if spec_info.accept_length_cpu:
                metadata.max_seq_len_q = max(spec_info.accept_length_cpu) + 1
            else:
                metadata.max_seq_len_q = 1

            metadata.cu_seqlens_q[1:].copy_(
                torch.cumsum(accept_length, dim=0, dtype=torch.int32)
            )

            max_seq_pages = (
                metadata.max_seq_len_k + self.page_size - 1
            ) // self.page_size
            page_indices = self.req_to_token[
                req_pool_indices[:, None],
                self.draft_extend_metadata["strided_indices"][:max_seq_pages],
            ]
            metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)

        if encoder_lens is not None:
            # Only support encoder size 1 for now
            metadata.encoder_max_seq_len_k = encoder_lens[0]
            metadata.encoder_lens_int32.copy_(encoder_lens[:1])
            metadata.encoder_cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.encoder_lens_int32, dim=0, dtype=torch.int32)
            )

            metadata.encoder_page_table[:, : metadata.encoder_max_seq_len_k].copy_(
                self.req_to_token[req_pool_indices, : metadata.encoder_max_seq_len_k]
            )

            # Update the regular page table
            page_table = self.req_to_token[
                req_pool_indices,
                metadata.encoder_max_seq_len_k : (
                    metadata.encoder_max_seq_len_k + metadata.max_seq_len_k
                ),
            ]
            metadata.page_table[:, : metadata.max_seq_len_k].copy_(page_table)

        self.forward_metadata = metadata
        self.forward_metadata_spec_decode_expand = metadata_expand

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 1

    def _init_local_attn_metadata(
        self, forwardbatch: ForwardBatch, metadata: FlashAttentionMetadata, device
    ):
        """Centralized utility to initialize local_attn_metadata if chunked attention is enabled."""
        if self.attention_chunk_size is None:
            metadata.local_attn_metadata = None
            return

        cu_seqlens_q = metadata.cu_seqlens_q
        cache_seqlens_int32 = metadata.cache_seqlens_int32
        if self.is_hybrid:
            page_table = self.full_to_swa_index_mapping[metadata.page_table].to(
                torch.int32
            )
        else:
            page_table = metadata.page_table
        if cu_seqlens_q is None or cache_seqlens_int32 is None or page_table is None:
            metadata.local_attn_metadata = None
            return

        (
            seqlens_q_local,
            cu_seqlens_q_local,
            seqlens_k_local,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q,
            cache_seqlens_int32,
            page_table,
            self.page_size,
        )

        local_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
            local_query_start_loc=cu_seqlens_q_local.to(device),
            local_seqused_k=seqlens_k_local.to(device),
            local_block_table=block_table_local.to(device),
            local_max_query_len=int(seqlens_q_local.max().item()) if seqlens_q_local.numel() > 0 else 0,
            local_max_seq_len=int(seqlens_k_local.max().item()) if seqlens_k_local.numel() > 0 else 0,
        )
        metadata.local_attn_metadata = local_metadata

    def _update_local_attn_metadata_for_capture(
        self, metadata: FlashAttentionMetadata, bs: int
    ):
        """Update local attention metadata during CUDA graph capture phase.

        This method calculates the exact buffer sizes needed for local attention metadata
        during the CUDA graph capture phase, optimizing memory usage by creating views of
        pre-allocated buffers with exactly the sizes needed.
        """
        seq_lens_capture = metadata.cache_seqlens_int32
        max_seq_len = int(seq_lens_capture.max().item())
        page_table_capture = metadata.page_table

        (
            seqlens_q_local,
            cu_seqlens_q_local,
            seqlens_k_local,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            metadata.cu_seqlens_q,
            seq_lens_capture,
            page_table_capture,
            self.page_size,
        )

        # Get exact dimensions from the calculation
        q_len = int(cu_seqlens_q_local.numel())
        k_len = int(seqlens_k_local.numel())
        b0 = int(block_table_local.shape[0]) if block_table_local.numel() > 0 else bs
        b1 = int(block_table_local.shape[1]) if block_table_local.numel() > 0 else 1

        # Create views of the pre-allocated buffers with exactly these sizes
        # This is the key optimization - we only use the memory we actually need
        local_query_start_loc = self.decode_cuda_graph_local_attn_metadata[
            "local_query_start_loc"
        ][:q_len]

        local_seqused_k = self.decode_cuda_graph_local_attn_metadata["local_seqused_k"][
            :k_len
        ]

        local_block_table = self.decode_cuda_graph_local_attn_metadata[
            "local_block_table"
        ][:b0, :b1]

        metadata.local_attn_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
            local_query_start_loc=local_query_start_loc,
            local_seqused_k=local_seqused_k,
            local_block_table=local_block_table,
            local_max_query_len=1,
            local_max_seq_len=max_seq_len,
        )

    def _update_local_attn_metadata_for_replay(
        self,
        metadata: FlashAttentionMetadata,
        bs: int,
    ):
        """Update preallocated local attention metadata in-place before CUDA graph replay."""
        if self.attention_chunk_size is None:
            return

        # Access preallocated buffers
        local_q_buf = self.decode_cuda_graph_local_attn_metadata[
            "local_query_start_loc"
        ]
        local_k_buf = self.decode_cuda_graph_local_attn_metadata["local_seqused_k"]
        local_block_buf = self.decode_cuda_graph_local_attn_metadata[
            "local_block_table"
        ]
        cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"]

        # Create a modified version for local attention that only processes the last token
        # This mimics the normal decode pattern
        cu_seqlens_q = torch.arange(
            bs + 1, device=cu_seqlens_q.device, dtype=cu_seqlens_q.dtype
        )
        seqlens = metadata.cache_seqlens_int32[:bs]
        # Slice the page_table to match the batch size and actual sequence length
        # This serves three important purposes:
        # 1. Ensures we only process the actual batch size (bs) and not the maximum batch size
        # 2. Limits the sequence length to prevent processing padding tokens or garbage values
        # 3. Prevents zeros in the block table which can cause garbage output during replay
        #
        # Without this slicing, the pre-allocated page_table may contain zeros or invalid indices
        # beyond the actual sequence length, leading to incorrect attention calculations
        max_seq_len = int(seqlens.max().item())
        if self.is_hybrid:
            sliced_page_table = self.full_to_swa_index_mapping[
                metadata.page_table[:bs, :max_seq_len]
            ].to(torch.int32)
        else:
            sliced_page_table = metadata.page_table[:bs, :max_seq_len]

        (
            seqlens_q_local,
            cu_seqlens_q_local,
            seqlens_k_local,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q,
            seqlens,
            sliced_page_table,
            self.page_size,
        )

        # Convert back to tensors
        device = local_q_buf.device
        cu_seqlens_q_local = cu_seqlens_q_local.to(device)
        seqlens_k_local = seqlens_k_local.to(device)
        block_table_local = block_table_local.to(device)
        # Get sizes
        q_len = cu_seqlens_q_local.shape[0]
        k_len = seqlens_k_local.shape[0]
        b0, b1 = block_table_local.shape

        # In-place updates into preallocated tensors and zero out the unused space
        local_q_buf[:q_len].copy_(cu_seqlens_q_local)
        local_q_buf[q_len:].fill_(0)
        local_k_buf[:k_len].copy_(seqlens_k_local)
        local_k_buf[k_len:].fill_(0)
        local_block_buf[:b0, :b1].copy_(block_table_local)
        local_block_buf[b0:, :].fill_(0)
        local_block_buf[:b0, b1:].fill_(0)

        if metadata.local_attn_metadata is not None:
            lam = metadata.local_attn_metadata
            lam.local_max_query_len = int(seqlens_q_local.max().item()) if seqlens_q_local.numel() > 0 else 0
            lam.local_max_seq_len = int(seqlens_k_local.max().item()) if seqlens_k_local.numel() > 0 else 0


class FlashAttentionMultiStepBackend:

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                FlashAttentionBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        forward_batch: ForwardBatch,
    ):
        assert forward_batch.spec_info is not None
        assert isinstance(forward_batch.spec_info, EagleDraftInput)

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        assert forward_batch.spec_info is not None
        assert isinstance(forward_batch.spec_info, EagleDraftInput)

        for i in range(self.speculative_num_steps - 1):
            # TODO: incrementally update the metadata for the later steps,
            # so that they do not need to recompute everything from scratch.
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                out_cache_loc=forward_batch.out_cache_loc,
            )


# @torch.compile(dynamic=True, backend=get_compiler_backend())
# TODO: fuse these kernels
# NOTE: torch.compile makes it slower in speculative decoding
def normal_decode_set_metadata(
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    strided_indices: torch.Tensor,
    max_seq_pages: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_len_delta: int,
    page_size: int,
):
    cache_seqlens_int32.copy_(seq_lens + seq_len_delta)
    cu_seqlens_k[1:].copy_(torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32))
    page_indices = req_to_token[
        req_pool_indices[:, None],
        strided_indices[:max_seq_pages][None, :],
    ]
    page_table[:, :max_seq_pages].copy_(page_indices // page_size)
