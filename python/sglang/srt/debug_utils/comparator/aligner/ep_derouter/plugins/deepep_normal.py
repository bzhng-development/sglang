from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.base import DeRouterPlugin


class DeepEPNormalDeRouter(DeRouterPlugin):
    """De-router for SGLang DeepEP Normal dispatch path.

    Uses ``deepep_normal_src2dst`` (output of ``deepep_compute_src2dst_triton_kernel``).

    ``src2dst[canonical_flat_idx] = dispatch_pos`` where
    ``canonical_flat_idx = token_idx * top_k + k_idx``.
    Invalid positions (token not handled by this EP rank) have negative values.

    We invert this to get ``forward_perm[dispatch_pos] = canonical_flat_idx``.
    """

    @property
    def required_aux_dump_names(self) -> frozenset[str]:
        return frozenset({"deepep_normal_src2dst"})

    def compute_forward_permutation(
        self,
        aux_tensors: dict[str, torch.Tensor],
        *,
        num_tokens: int,
        top_k: int,
        num_routed: int,
    ) -> torch.Tensor:
        src2dst: torch.Tensor = aux_tensors["deepep_normal_src2dst"].long()
        total_slots: int = num_tokens * top_k
        forward_perm: torch.Tensor = torch.full(
            (num_routed,), -1, dtype=torch.long, device=src2dst.device
        )

        valid: torch.Tensor = (src2dst >= 0) & (src2dst < num_routed)
        canonical_indices: torch.Tensor = torch.arange(
            total_slots, device=src2dst.device
        )
        forward_perm[src2dst[valid]] = canonical_indices[valid]

        return forward_perm
