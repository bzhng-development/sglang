from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class ForwardMetadata:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    extend_prefix_lens: Optional[torch.Tensor] = None
    extend_seq_lens: Optional[torch.Tensor] = None
    req_pool_indices_cpu: Optional[torch.Tensor] = None
    seq_lens_cpu: Optional[torch.Tensor] = None
    extend_prefix_lens_cpu: Optional[torch.Tensor] = None
    extend_seq_lens_cpu: Optional[torch.Tensor] = None


class TorchNativeAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        self._graph_metadata: Dict[int, ForwardMetadata] = {}
        self._max_cuda_graph_bs = 0
        self._decode_chunk_size = 256
        self._chunk_token_buffer = None
        self._chunk_position = None

    @staticmethod
    def _to_cpu_tensor(data: Optional[torch.Tensor], *, dtype=torch.int32):
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data.detach().to(device="cpu", dtype=dtype)
        return torch.tensor(data, dtype=dtype)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        seq_lens_cpu_src = (
            forward_batch.seq_lens_cpu
            if forward_batch.seq_lens_cpu is not None
            else forward_batch.seq_lens
        )
        extend_prefix_cpu_src = (
            forward_batch.extend_prefix_lens_cpu
            if forward_batch.extend_prefix_lens_cpu is not None
            else forward_batch.extend_prefix_lens
        )
        extend_seq_cpu_src = (
            forward_batch.extend_seq_lens_cpu
            if forward_batch.extend_seq_lens_cpu is not None
            else forward_batch.extend_seq_lens
        )
        self.forward_metadata = ForwardMetadata(
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            extend_prefix_lens=forward_batch.extend_prefix_lens,
            extend_seq_lens=forward_batch.extend_seq_lens,
            req_pool_indices_cpu=self._to_cpu_tensor(forward_batch.req_pool_indices),
            seq_lens_cpu=self._to_cpu_tensor(seq_lens_cpu_src),
            extend_prefix_lens_cpu=self._to_cpu_tensor(extend_prefix_cpu_src),
            extend_seq_lens_cpu=self._to_cpu_tensor(extend_seq_cpu_src),
        )

    # NOTE: The optional ``kv_indices_buf`` argument is accepted for API parity
    # with other attention backends even though it is currently unused here.
    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        del (
            kv_indices_buf,
            max_num_tokens,
        )  # Unused but kept for signature compatibility
        self._graph_metadata.clear()
        self._max_cuda_graph_bs = max_bs
        if max_bs > 0:
            with torch.device(self.device):
                self._chunk_token_buffer = torch.zeros(
                    (max_bs, self._decode_chunk_size),
                    dtype=torch.int32,
                    device=self.device,
                )
                self._chunk_position = torch.arange(
                    self._decode_chunk_size,
                    device=self.device,
                    dtype=torch.int32,
                )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info,
    ):
        del encoder_lens, num_tokens  # Unused for torch native backend
        if not forward_mode.is_decode_or_idle():
            raise NotImplementedError(
                "TorchNativeAttnBackend only supports CUDA graph capture for decode/idle modes."
            )
        if spec_info is not None:
            raise NotImplementedError(
                "Speculative decoding is not yet supported by the torch native CUDA graph path."
            )

        metadata = ForwardMetadata(
            req_pool_indices=req_pool_indices[:bs],
            seq_lens=seq_lens[:bs],
            req_pool_indices_cpu=self._to_cpu_tensor(req_pool_indices[:bs]),
            seq_lens_cpu=self._to_cpu_tensor(seq_lens[:bs]),
        )
        self._graph_metadata[bs] = metadata
        self.forward_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info,
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        del seq_lens_sum
        del encoder_lens
        if not forward_mode.is_decode_or_idle():
            raise NotImplementedError(
                "TorchNativeAttnBackend only supports CUDA graph replay for decode/idle modes."
            )
        if spec_info is not None:
            raise NotImplementedError(
                "Speculative decoding is not yet supported by the torch native CUDA graph path."
            )

        metadata = self._graph_metadata.get(bs)
        if metadata is None:
            metadata = ForwardMetadata(
                req_pool_indices=req_pool_indices[:bs],
                seq_lens=seq_lens[:bs],
                req_pool_indices_cpu=self._to_cpu_tensor(req_pool_indices[:bs]),
                seq_lens_cpu=self._to_cpu_tensor(seq_lens[:bs]),
            )
            self._graph_metadata[bs] = metadata
        else:
            metadata.req_pool_indices = req_pool_indices[:bs]
            metadata.seq_lens = seq_lens[:bs]
            metadata.req_pool_indices_cpu = self._to_cpu_tensor(req_pool_indices[:bs])
            if seq_lens_cpu is not None:
                metadata.seq_lens_cpu = self._to_cpu_tensor(seq_lens_cpu[:bs])
            else:
                metadata.seq_lens_cpu = self._to_cpu_tensor(seq_lens[:bs])

        self.forward_metadata = metadata

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the extend forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        metadata = self.forward_metadata
        seq_lens_cpu = metadata.seq_lens_cpu if metadata is not None else None
        extend_prefix_lens_cpu = (
            metadata.extend_prefix_lens_cpu
            if metadata is not None and metadata.extend_prefix_lens_cpu is not None
            else None
        )
        extend_seq_lens_cpu = (
            metadata.extend_seq_lens_cpu
            if metadata is not None and metadata.extend_seq_lens_cpu is not None
            else None
        )

        if seq_lens_cpu is None:
            seq_lens_cpu = self._to_cpu_tensor(seq_lens)
        if extend_prefix_lens_cpu is None:
            extend_prefix_lens_cpu = self._to_cpu_tensor(extend_prefix_lens)
        if extend_seq_lens_cpu is None:
            extend_seq_lens_cpu = self._to_cpu_tensor(extend_seq_lens)
        if extend_prefix_lens_cpu is None or extend_seq_lens_cpu is None:
            raise RuntimeError(
                "Extend metadata is unavailable for torch native backend."
            )

        seq_lens_list = (
            seq_lens_cpu.tolist()
            if isinstance(seq_lens_cpu, torch.Tensor)
            else list(seq_lens_cpu)
        )
        extend_prefix_list = (
            extend_prefix_lens_cpu.tolist()
            if isinstance(extend_prefix_lens_cpu, torch.Tensor)
            else list(extend_prefix_lens_cpu)
        )
        extend_seq_list = (
            extend_seq_lens_cpu.tolist()
            if isinstance(extend_seq_lens_cpu, torch.Tensor)
            else list(extend_seq_lens_cpu)
        )

        if not (len(seq_lens_list) == len(extend_prefix_list) == len(extend_seq_list)):
            raise RuntimeError(
                "Mismatched extend metadata lengths for torch native backend."
            )

        req_pool_indices_device = (
            req_pool_indices
            if req_pool_indices.device == req_to_token.device
            else req_pool_indices.to(device=req_to_token.device)
        )
        if req_pool_indices_device.dtype != torch.long:
            req_pool_indices_device = req_pool_indices_device.to(dtype=torch.long)
        req_to_token_rows = torch.index_select(req_to_token, 0, req_pool_indices_device)

        if req_to_token_rows.size(0) != len(seq_lens_list):
            raise RuntimeError(
                "Sequence metadata and req_to_token rows have different lengths."
            )

        if len(seq_lens_list) != req_to_token_rows.size(0):
            raise RuntimeError(
                "Sequence length metadata mismatch with req_to_token rows for torch native backend."
            )

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q = 0
        for seq_idx, (seq_len_kv, extend_seq_len_q, prefill_seq_len_q) in enumerate(
            zip(seq_lens_list, extend_seq_list, extend_prefix_list)
        ):
            seq_len_kv = int(seq_len_kv)
            extend_seq_len_q = int(extend_seq_len_q)
            prefill_seq_len_q = int(prefill_seq_len_q)

            if extend_seq_len_q == 0:
                continue

            end_q = start_q + extend_seq_len_q

            per_req_query = query[:, start_q:end_q, :]
            per_req_query_redudant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )
            per_req_query_redudant.zero_()
            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            per_req_tokens = req_to_token_rows[seq_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out_redudant = (
                scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q = end_q
        return output

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run decode forward using the torch native SDPA op without Python-side data dependent control flow.

        This implementation keeps tensor shapes static while relying on attention masks to drop
        padded KV tokens. It enables CUDA graph replay to provide correct results for varying
        sequence lengths across iterations.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        if self._chunk_token_buffer is None or self._chunk_position is None:
            raise RuntimeError("CUDA graph buffers are not initialized for torch native backend decode path.")

        seq_lens_device = (
            seq_lens
            if seq_lens.device == req_to_token.device
            else seq_lens.to(device=req_to_token.device)
        )
        seq_lens_device = seq_lens_device.to(dtype=torch.int32)

        req_pool_indices_device = (
            req_pool_indices
            if req_pool_indices.device == req_to_token.device
            else req_pool_indices.to(device=req_to_token.device)
        )
        if req_pool_indices_device.dtype != torch.long:
            req_pool_indices_device = req_pool_indices_device.to(dtype=torch.long)

        req_to_token_rows = torch.index_select(req_to_token, 0, req_pool_indices_device)

        batch_size = query.shape[0]
        num_q_heads = query.shape[1]
        head_dim = query.shape[2]
        num_k_heads = k_cache.shape[1]
        group_size = num_q_heads // num_k_heads if enable_gqa else 1

        query_fp32 = query.to(dtype=torch.float32)
        output_fp32 = torch.zeros_like(query_fp32, dtype=torch.float32)
        max_logit = torch.full(
            (batch_size, num_q_heads, 1),
            float("-inf"),
            dtype=torch.float32,
            device=query.device,
        )
        exp_sum = torch.zeros(
            (batch_size, num_q_heads, 1),
            dtype=torch.float32,
            device=query.device,
        )
        value_accum = torch.zeros_like(query_fp32, dtype=torch.float32)

        chunk_size = self._decode_chunk_size
        max_context_len = req_to_token_rows.shape[1]
        chunk_count = math.ceil(max_context_len / chunk_size) if max_context_len > 0 else 0
        neg_inf = torch.finfo(torch.float32).min
        if scaling is None:
            scale_tensor = None
        elif isinstance(scaling, torch.Tensor):
            scale_tensor = scaling.to(dtype=torch.float32, device=query.device)
        else:
            scale_tensor = torch.tensor(
                float(scaling), dtype=torch.float32, device=query.device
            )

        chunk_position = self._chunk_position

        for chunk_idx in range(chunk_count):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, max_context_len)
            chunk_width = chunk_end - chunk_start
            if chunk_width <= 0:
                break

            if chunk_width == chunk_size:
                chunk_tokens = req_to_token_rows[:, chunk_start:chunk_end]
            else:
                chunk_tokens = self._chunk_token_buffer[:batch_size]
                chunk_tokens[:, :chunk_width] = req_to_token_rows[:, chunk_start:chunk_end]
                if chunk_width < chunk_size:
                    chunk_tokens[:, chunk_width:] = 0

            token_positions = chunk_position + chunk_start
            valid_mask = token_positions.view(1, -1) < seq_lens_device.view(batch_size, 1)
            if chunk_width < chunk_size:
                valid_mask[:, chunk_width:] = False

            flat_tokens = chunk_tokens.reshape(-1).to(dtype=torch.long)

            key_chunk = k_cache.index_select(0, flat_tokens)
            value_chunk = v_cache.index_select(0, flat_tokens)

            key_chunk = key_chunk.view(batch_size, chunk_size, num_k_heads, head_dim).permute(0, 2, 1, 3)
            value_chunk = value_chunk.view(batch_size, chunk_size, num_k_heads, head_dim).permute(0, 2, 1, 3)

            invalid_mask = (~valid_mask).unsqueeze(1).unsqueeze(-1)
            key_chunk = key_chunk.masked_fill(invalid_mask, 0)
            value_chunk = value_chunk.masked_fill(invalid_mask, 0)

            if enable_gqa and group_size > 1:
                key_chunk = key_chunk.repeat_interleave(group_size, dim=1)
                value_chunk = value_chunk.repeat_interleave(group_size, dim=1)

            key_chunk_fp32 = key_chunk.to(dtype=torch.float32)
            value_chunk_fp32 = value_chunk.to(dtype=torch.float32)

            q_chunk = query_fp32.unsqueeze(2)
            scores = torch.matmul(q_chunk, key_chunk_fp32.transpose(-1, -2)).squeeze(2)
            if scale_tensor is not None:
                scores = scores * scale_tensor

            scores = scores.masked_fill(~valid_mask.unsqueeze(1), neg_inf)

            chunk_max = scores.max(dim=-1, keepdim=True).values
            has_chunk = valid_mask.any(dim=-1, keepdim=True).unsqueeze(1)
            chunk_max = torch.where(has_chunk, chunk_max, torch.full_like(chunk_max, neg_inf))

            new_max = torch.maximum(max_logit, chunk_max)
            safe_new_max = torch.where(
                torch.isfinite(new_max),
                new_max,
                torch.zeros_like(new_max),
            )
            safe_max_logit = torch.where(
                torch.isfinite(max_logit),
                max_logit,
                torch.zeros_like(max_logit),
            )

            exp_prev_factor = torch.where(
                torch.isfinite(max_logit),
                torch.exp(safe_max_logit - safe_new_max),
                torch.zeros_like(max_logit),
            )

            exp_chunk = torch.exp(scores - safe_new_max)
            exp_chunk = exp_chunk * valid_mask.unsqueeze(1)

            exp_sum = exp_prev_factor * exp_sum + exp_chunk.sum(dim=-1, keepdim=True)

            value_prev_factor = exp_prev_factor
            value_chunk_contrib = torch.matmul(
                exp_chunk.unsqueeze(-2), value_chunk_fp32
            ).squeeze(-2)
            value_accum = value_prev_factor * value_accum + value_chunk_contrib

            max_logit = torch.where(torch.isfinite(new_max), new_max, max_logit)

        denom = exp_sum
        output_fp32 = torch.where(
            denom > 0,
            value_accum / denom,
            torch.zeros_like(value_accum),
        )

        output.copy_(output_fp32.to(dtype=output.dtype))

        zero_seq_mask = seq_lens_device <= 0
        if zero_seq_mask.any():
            output[zero_seq_mask] = 0

        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        metadata = self.forward_metadata
        req_pool_indices = (
            metadata.req_pool_indices
            if metadata is not None and metadata.req_pool_indices is not None
            else forward_batch.req_pool_indices
        )
        seq_lens = (
            metadata.seq_lens
            if metadata is not None and metadata.seq_lens is not None
            else forward_batch.seq_lens
        )
        extend_prefix_lens = (
            metadata.extend_prefix_lens
            if metadata is not None and metadata.extend_prefix_lens is not None
            else forward_batch.extend_prefix_lens
        )
        extend_seq_lens = (
            metadata.extend_seq_lens
            if metadata is not None and metadata.extend_seq_lens is not None
            else forward_batch.extend_seq_lens
        )

        self._run_sdpa_forward_extend(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            req_pool_indices,
            seq_lens,
            extend_prefix_lens,
            extend_seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=causal,
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        metadata = self.forward_metadata
        req_pool_indices = (
            metadata.req_pool_indices
            if metadata is not None and metadata.req_pool_indices is not None
            else forward_batch.req_pool_indices
        )
        seq_lens = (
            metadata.seq_lens
            if metadata is not None and metadata.seq_lens is not None
            else forward_batch.seq_lens
        )

        self._run_sdpa_forward_decode(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            req_pool_indices,
            seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=False,
        )

        return o

    def support_triton(self):
        return False
