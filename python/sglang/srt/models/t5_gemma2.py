# Copyright 2025 SGLang Team
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
"""T5Gemma2 encoder-decoder model with merged self+cross attention.

Architecture overview (google/t5gemma-2-4b-4b):
  - Encoder: bidirectional Gemma3-style text encoder + SigLIP vision tower +
    multimodal projector.  34 layers, SWA (sliding_window=1024, every 6th full).
  - Decoder: Gemma3-style causal decoder.  34 layers, same SWA pattern.
    The decoder has NO separate cross-attention weights.  Instead, the merged
    self+cross attention shares K/V projections between decoder self-attention
    and encoder cross-attention.
  - Tied word embeddings across encoder, decoder, and lm_head.

For serving, the encoder runs once per request during prefill and the output is
cached in ``_encoder_cache``.  During each decode step, each decoder layer
re-projects the cached encoder output through its own K/V projections and
performs manual cross-attention (following the Whisper/NemotronParse pattern).
Decoder self-attention uses RadixAttention for paged KV-cache management.
The cross-attention contribution is added to the self-attention output *before*
the output projection so that o_proj is applied once.
"""

import copy
import logging
from typing import Any, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.layernorm import Gemma3RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gemma3_causal import (
    Gemma3MLP,
    Gemma3RotaryEmbedding,
    get_attention_sliding_window_size,
)
from sglang.srt.models.siglip import SiglipVisionModel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


def _make_rope_config(base_config, attention_type: str):
    """Build a config object that Gemma3RotaryEmbedding can consume.

    T5Gemma2 stores RoPE parameters under ``rope_parameters.{full_attention,
    sliding_attention}`` (v5 nested format).  Gemma3RotaryEmbedding (post-v5
    patch) reads ``config.rope_parameters`` as a flat dict with ``rope_type``
    and ``rope_theta``.  This helper flattens the nested format into a
    per-layer-type config.
    """
    cfg = copy.deepcopy(base_config)
    rope_params = getattr(base_config, "rope_parameters", None)
    if rope_params and isinstance(rope_params, dict) and attention_type in rope_params:
        rp = rope_params[attention_type]
        cfg.rope_parameters = {
            "rope_type": rp.get("rope_type", "default"),
            "rope_theta": rp.get("rope_theta", 10000),
        }
        if "factor" in rp:
            cfg.rope_parameters["factor"] = rp["factor"]
    else:
        cfg.rope_parameters = {"rope_type": "default", "rope_theta": 10000}
    return cfg


# ---------------------------------------------------------------------------
# Multimodal projector (same arch as Gemma3)
# ---------------------------------------------------------------------------
class T5Gemma2MultiModalProjector(nn.Module):
    def __init__(self, encoder_config):
        super().__init__()
        vision_cfg = encoder_config.vision_config
        hidden_size = encoder_config.text_config.hidden_size
        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(vision_cfg.hidden_size, hidden_size)
        )
        self.mm_soft_emb_norm = Gemma3RMSNorm(
            vision_cfg.hidden_size, eps=vision_cfg.layer_norm_eps
        )
        self.patches_per_image = vision_cfg.image_size // vision_cfg.patch_size
        tokens_per_side = int(encoder_config.mm_tokens_per_image**0.5)
        kernel_size = self.patches_per_image // tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)

    def forward(self, vision_outputs: torch.Tensor) -> torch.Tensor:
        b, seq_len, hidden = vision_outputs.shape
        x = vision_outputs.transpose(1, 2).reshape(
            b, hidden, self.patches_per_image, self.patches_per_image
        )
        x = self.avg_pool(x).flatten(2).transpose(1, 2)
        x = self.mm_soft_emb_norm(x)
        return torch.matmul(x, self.mm_input_projection_weight).type_as(vision_outputs)


# ---------------------------------------------------------------------------
# Encoder attention (bidirectional, no KV cache, uses SDPA)
# ---------------------------------------------------------------------------
class T5Gemma2EncoderAttention(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.query_pre_attn_scalar**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.is_sliding = config.layer_types[layer_id] == "sliding_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).flatten(1, 2)
            v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).flatten(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, scale=self.scaling, is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output, _ = self.o_proj(attn_output)
        return output


# ---------------------------------------------------------------------------
# Encoder layer
# ---------------------------------------------------------------------------
class T5Gemma2EncoderLayer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = T5Gemma2EncoderAttention(
            layer_id, config, quant_config, prefix=add_prefix("self_attn", prefix)
        )
        self.mlp = Gemma3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.pre_self_attn_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_self_attn_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.is_sliding = self.self_attn.is_sliding

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: Tuple[torch.Tensor, torch.Tensor],
        position_embeddings_local: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        pos_emb = (
            position_embeddings_local
            if self.is_sliding
            else position_embeddings_global
        )

        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, pos_emb)
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Decoder attention -- causal self-attention via RadixAttention + manual
# cross-attention with encoder hidden states using the same K/V projections.
# The cross-attention output is added to self-attention output BEFORE o_proj.
# ---------------------------------------------------------------------------
class T5Gemma2DecoderAttention(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.query_pre_attn_scalar**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.is_sliding = config.layer_types[layer_id] == "sliding_attention"

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=0.0,
            sliding_window_size=(
                get_attention_sliding_window_size(config) if self.is_sliding else None
            ),
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    # -- Manual cross-attention using the same Q/K/V projections as self-attn --
    def _cross_attention(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Returns cross-attention output in [total_tokens, num_heads * head_dim]
        (pre-o_proj) so it can be added to the self-attention output."""
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim

        # Decoder Q (no RoPE for cross-attention -- positions are meaningless
        # between encoder and decoder coordinate spaces)
        qkv_dec, _ = self.qkv_proj(hidden_states)
        q, _, _ = qkv_dec.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, num_heads, head_dim)
        q = self.q_norm(q)
        q = q * self.scaling

        # Encoder K, V (same projections, no RoPE)
        qkv_enc, _ = self.qkv_proj(encoder_hidden_states)
        _, k_enc, v_enc = qkv_enc.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        k_enc = k_enc.view(-1, num_kv_heads, head_dim)
        v_enc = v_enc.view(-1, num_kv_heads, head_dim)
        k_enc = self.k_norm(k_enc)

        # Expand KV heads for GQA
        if num_kv_heads != num_heads:
            n_rep = num_heads // num_kv_heads
            k_enc = (
                k_enc.unsqueeze(2).expand(-1, -1, n_rep, -1).reshape(-1, num_heads, head_dim)
            )
            v_enc = (
                v_enc.unsqueeze(2).expand(-1, -1, n_rep, -1).reshape(-1, num_heads, head_dim)
            )

        # [total_tokens, heads, dim] -> [heads, total_tokens, dim]
        q = q.transpose(0, 1)
        k_enc = k_enc.transpose(0, 1)
        v_enc = v_enc.transpose(0, 1)

        q_len = q.shape[1]
        kv_len = k_enc.shape[1]

        attn_weights = torch.bmm(q, k_enc.transpose(1, 2))

        # Block-diagonal mask so each request only attends to its own encoder output
        batch_size = forward_batch.batch_size if forward_batch else 1
        if batch_size > 1 and kv_len > 0:
            encoder_len_per_req = kv_len // batch_size
            if encoder_len_per_req * batch_size == kv_len:
                is_decode = forward_batch.forward_mode.is_decode()
                if is_decode:
                    # Decode: one token per request
                    mask = torch.zeros(
                        (q_len, kv_len), device=q.device, dtype=torch.bool
                    )
                    for i in range(batch_size):
                        enc_s = i * encoder_len_per_req
                        enc_e = enc_s + encoder_len_per_req
                        mask[i, enc_s:enc_e] = True
                    attn_weights = attn_weights.masked_fill(
                        ~mask.unsqueeze(0), float("-inf")
                    )
                else:
                    # Extend: variable-length decoder tokens per request
                    seq_lens = forward_batch.extend_seq_lens
                    if seq_lens is not None:
                        seq_lens_list = (
                            seq_lens.tolist()
                            if hasattr(seq_lens, "tolist")
                            else list(seq_lens)
                        )
                        mask = torch.zeros(
                            (q_len, kv_len), device=q.device, dtype=torch.bool
                        )
                        q_start = 0
                        for i, dec_len in enumerate(seq_lens_list):
                            enc_s = i * encoder_len_per_req
                            enc_e = enc_s + encoder_len_per_req
                            q_end = q_start + dec_len
                            mask[q_start:q_end, enc_s:enc_e] = True
                            q_start = q_end
                        attn_weights = attn_weights.masked_fill(
                            ~mask.unsqueeze(0), float("-inf")
                        )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            v_enc.dtype
        )
        attn_output = torch.bmm(attn_weights, v_enc)
        # [heads, total_tokens, dim] -> [total_tokens, heads * dim]
        attn_output = attn_output.transpose(0, 1).reshape(q_len, num_heads * head_dim)
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        forward_batch: ForwardBatch,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # --- Decoder self-attention through RadixAttention (paged KV cache) ---
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = q.transpose(0, 1).unsqueeze(0)
        q = self.q_norm(q)

        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k = k.transpose(0, 1).unsqueeze(0)
        k = self.k_norm(k)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        attn_output = self.attn(q, k, v, forward_batch=forward_batch)

        if attn_output.dim() == 4 and attn_output.shape[0] == 1:
            attn_output = attn_output.squeeze(0)
            attn_output = attn_output.flatten(-2, -1)

        # --- Add cross-attention contribution (pre-o_proj) ---
        if encoder_hidden_states is not None:
            cross_output = self._cross_attention(
                hidden_states, encoder_hidden_states, forward_batch
            )
            attn_output = attn_output + cross_output

        output, _ = self.o_proj(attn_output)
        return output


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------
class T5Gemma2DecoderLayer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = T5Gemma2DecoderAttention(
            layer_id, config, quant_config, prefix=add_prefix("self_attn", prefix)
        )
        self.mlp = Gemma3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.pre_self_attn_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_self_attn_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.is_sliding = self.self_attn.is_sliding

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: Tuple[torch.Tensor, torch.Tensor],
        position_embeddings_local: Tuple[torch.Tensor, torch.Tensor],
        forward_batch: ForwardBatch,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pos_emb = (
            position_embeddings_local
            if self.is_sliding
            else position_embeddings_global
        )

        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, pos_emb, forward_batch, encoder_hidden_states
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Text encoder (bidirectional Gemma3-style layers)
# ---------------------------------------------------------------------------
class T5Gemma2TextEncoder(nn.Module):
    def __init__(
        self,
        config,
        embed_tokens: nn.Module,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        text_cfg = config.text_config
        self.embed_tokens = embed_tokens
        self.hidden_scale = text_cfg.hidden_size**0.5

        self.layers = nn.ModuleList(
            [
                T5Gemma2EncoderLayer(
                    i,
                    text_cfg,
                    quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(text_cfg.num_hidden_layers)
            ]
        )
        self.norm = Gemma3RMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps)

        global_rope_cfg = _make_rope_config(text_cfg, "full_attention")
        local_rope_cfg = _make_rope_config(text_cfg, "sliding_attention")
        self.rotary_emb_global = Gemma3RotaryEmbedding(config=global_rope_cfg)
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=local_rope_cfg)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is not None:
            hidden_states = input_embeds * self.hidden_scale
        else:
            hidden_states = self.embed_tokens(input_ids) * self.hidden_scale

        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)

        bsz, seq_len, _ = hidden_states.shape
        positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        pos_global = self.rotary_emb_global(hidden_states, positions)
        pos_local = self.rotary_emb_local(hidden_states, positions)

        for layer in self.layers:
            hidden_states = layer(hidden_states, pos_global, pos_local)

        hidden_states = self.norm(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Full encoder: text encoder + SigLIP vision tower + projector
# ---------------------------------------------------------------------------
class T5Gemma2Encoder(nn.Module):
    def __init__(
        self,
        config,
        embed_tokens: nn.Module,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.text_encoder = T5Gemma2TextEncoder(
            config, embed_tokens, quant_config, prefix=prefix
        )

        if hasattr(config, "vision_config") and config.vision_config is not None:
            self.vision_tower = SiglipVisionModel(
                config=config.vision_config,
                quant_config=quant_config,
                prefix=add_prefix("vision_tower", prefix),
            )
            self.multi_modal_projector = T5Gemma2MultiModalProjector(config)
        else:
            self.vision_tower = None
            self.multi_modal_projector = None

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.text_encoder(input_ids=input_ids, input_embeds=input_embeds)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class T5Gemma2Decoder(nn.Module):
    def __init__(
        self,
        config,
        embed_tokens: nn.Module,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = embed_tokens
        self.hidden_scale = config.hidden_size**0.5

        self.layers = nn.ModuleList(
            [
                T5Gemma2DecoderLayer(
                    i,
                    config,
                    quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        global_rope_cfg = _make_rope_config(config, "full_attention")
        local_rope_cfg = _make_rope_config(config, "sliding_attention")
        self.rotary_emb_global = Gemma3RotaryEmbedding(config=global_rope_cfg)
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=local_rope_cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) * self.hidden_scale

        if positions.dim() == 1:
            positions = positions.unsqueeze(0)

        pos_global = self.rotary_emb_global(hidden_states, positions)
        pos_local = self.rotary_emb_local(hidden_states, positions)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                pos_global,
                pos_local,
                forward_batch,
                encoder_hidden_states,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------
class T5Gemma2ForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        encoder_config = config.encoder
        decoder_config = config.decoder

        text_cfg = encoder_config.text_config
        vocab_size = getattr(config, "vocab_size", text_cfg.vocab_size)
        hidden_size = text_cfg.hidden_size

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        self.encoder = T5Gemma2Encoder(
            encoder_config,
            self.embed_tokens,
            quant_config,
            prefix=add_prefix("model.encoder", prefix),
        )
        self.decoder = T5Gemma2Decoder(
            decoder_config,
            self.embed_tokens,
            quant_config,
            prefix=add_prefix("model.decoder", prefix),
        )

        self.lm_head = ParallelLMHead(vocab_size, hidden_size, quant_config=quant_config)
        self.logits_processor = LogitsProcessor(decoder_config)

        self._encoder_cache: dict = {}

    def pad_input_ids(self, input_ids: List[int], _mm_inputs: MultimodalInputs):
        return input_ids

    def get_attention_sliding_window_size(self):
        return get_attention_sliding_window_size(self.config.decoder)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        is_decode = forward_batch.forward_mode.is_decode()

        # --- Encoder output retrieval / computation ---
        if is_decode:
            encoder_outputs = self._gather_encoder_outputs(forward_batch)
        else:
            encoder_outputs = self._encode_and_cache(forward_batch)

        # --- Decoder ---
        decoder_outputs = self.decoder(
            input_ids, positions, forward_batch, encoder_outputs
        )

        return self.logits_processor(
            input_ids=input_ids,
            hidden_states=decoder_outputs,
            lm_head=self.lm_head,
            logits_metadata=forward_batch,
        )

    # -- helpers --

    def _gather_encoder_outputs(self, forward_batch: ForwardBatch):
        """Look up cached encoder outputs for each request in the batch."""
        if forward_batch.req_pool_indices is None:
            return None
        req_indices = forward_batch.req_pool_indices.tolist()
        parts = [
            self._encoder_cache[idx]
            for idx in req_indices
            if idx in self._encoder_cache
        ]
        return torch.cat(parts, dim=0) if parts else None

    def _encode_and_cache(self, forward_batch: ForwardBatch):
        """Run encoder on new requests during prefill and cache the output."""
        mm_inputs_list = forward_batch.mm_inputs or []
        req_indices = (
            forward_batch.req_pool_indices.tolist()
            if forward_batch.req_pool_indices is not None
            else []
        )

        parts: List[torch.Tensor] = []
        for req_idx, mm_input in zip(req_indices, mm_inputs_list):
            if mm_input is None or not mm_input.mm_items:
                continue

            pixel_values = mm_input.mm_items[0].feature
            if pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)

            if self.encoder.vision_tower is not None:
                dtype = next(self.encoder.vision_tower.parameters()).dtype
                vision_out = self.encoder.vision_tower(
                    pixel_values=pixel_values.to(dtype)
                )
                image_features = self.encoder.multi_modal_projector(vision_out)
                enc_out = image_features.squeeze(0)
            else:
                enc_out = torch.zeros(
                    0, self.config.encoder.text_config.hidden_size,
                    device=pixel_values.device,
                    dtype=pixel_values.dtype,
                )

            self._encoder_cache[req_idx] = enc_out
            parts.append(enc_out)

        return torch.cat(parts, dim=0) if parts else None

    # -----------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            # ---- Tied embeddings ----
            if name == "model.encoder.embed_tokens.weight":
                name = "embed_tokens.weight"
            elif name == "model.decoder.embed_tokens.weight":
                continue  # tied to encoder
            elif name == "lm_head.weight":
                if name not in params_dict:
                    continue

            # ---- Remap HF prefix -> module prefix ----
            if name.startswith("model.encoder."):
                name = name.replace("model.encoder.", "encoder.", 1)
                if name.startswith("encoder.layers."):
                    name = name.replace(
                        "encoder.layers.", "encoder.text_encoder.layers.", 1
                    )
                elif name.startswith("encoder.norm."):
                    name = name.replace(
                        "encoder.norm.", "encoder.text_encoder.norm.", 1
                    )
            elif name.startswith("model.decoder."):
                name = name.replace("model.decoder.", "decoder.", 1)

            # Vision model out_proj -> proj
            if "vision_model" in name:
                name = name.replace(".self_attn.out_proj", ".self_attn.proj")

            # ---- Stacked param fusion (qkv, gate_up) ----
            loaded = False
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    loaded = True
                    break
                if name not in params_dict:
                    loaded = True
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded = True
                break

            if not loaded:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.debug("Skipping weight: %s", name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)

        return loaded_params


EntryClass = [T5Gemma2ForConditionalGeneration]
