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
    validate that the tensor is in the original order.  However, for dump points
    *after* the kernel (③④⑤), the tensor **is** in sorted order and needs
    inverse scatter.
    """

    def de_route(
        self,
        routed_tensor: torch.Tensor,
        aux_tensors: dict[str, torch.Tensor],
        *,
        num_tokens: int,
        top_k: int,
    ) -> torch.Tensor:
        sorted_token_ids: torch.Tensor = aux_tensors["sorted_token_ids"]

        total_slots: int = num_tokens * top_k
        valid_mask: torch.Tensor = sorted_token_ids < total_slots
        valid_ids: torch.Tensor = sorted_token_ids[valid_mask]
        valid_routed: torch.Tensor = routed_tensor[valid_mask]

        trailing_shape: list[int] = list(routed_tensor.shape[1:])
        output: torch.Tensor = torch.zeros(
            [total_slots] + trailing_shape,
            dtype=routed_tensor.dtype,
            device=routed_tensor.device,
        )
        output[valid_ids] = valid_routed

        return output
