from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.base import DeRouterPlugin


class FusedMoEDeRouter(DeRouterPlugin):
    """De-router for SGLang FusedMoE (StandardDispatcher / Triton path).

    ``sorted_token_ids`` from ``moe_align_block_size()`` maps dispatch positions to
    original flatten indices (``token_idx * top_k + k``).  Padding positions have
    values ``>= num_tokens * top_k``.

    The routed tensor is **not physically permuted** — the Triton kernel uses
    ``sorted_token_ids`` for indirect addressing.  So we actually just need to
    validate that the tensor is in the original order.  However, for tensors
    that have been physically reordered by the dispatch kernel (e.g. after
    up/gate GEMM, activation, or down GEMM), the inverse scatter restores the
    original token order.
    """

    def compute_forward_permutation(
        self,
        aux_tensors: dict[str, torch.Tensor],
        *,
        num_tokens: int,
        top_k: int,
        num_routed: int,
    ) -> torch.Tensor:
        sorted_token_ids: torch.Tensor = aux_tensors["sorted_token_ids"]

        total_slots: int = num_tokens * top_k
        forward_perm: torch.Tensor = sorted_token_ids.long().clone()
        forward_perm[forward_perm >= total_slots] = -1

        return forward_perm
