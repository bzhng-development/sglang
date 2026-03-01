from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.base import DeRouterPlugin
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.utils import (
    compute_within_group_indices,
)


class MegatronA2ADeRouter(DeRouterPlugin):
    """De-router for Megatron AlltoAll dispatch path.

    ``megatron_a2a_reversed_local_input_permutation_mapping`` from
    ``moe_utils.permute()`` is a **token-level** index (values in
    ``[0, num_tokens)``): position ``i`` in the permuted tensor holds original
    token ``sorted_indices[i]``.

    Since ``sorted_indices`` is token-level (not flatten), when a token appears
    in multiple experts each occurrence corresponds to a different k slot.
    """

    @property
    def required_aux_dump_names(self) -> frozenset[str]:
        return frozenset(
            {"megatron_a2a_reversed_local_input_permutation_mapping"}
        )

    def compute_forward_permutation(
        self,
        aux_tensors: dict[str, torch.Tensor],
        *,
        num_tokens: int,
        top_k: int,
        num_routed: int,
    ) -> torch.Tensor:
        sorted_indices: torch.Tensor = aux_tensors[
            "megatron_a2a_reversed_local_input_permutation_mapping"
        ]

        token_ids: torch.Tensor = sorted_indices.long()
        k_indices: torch.Tensor = compute_within_group_indices(token_ids)
        forward_perm: torch.Tensor = token_ids * top_k + k_indices

        total_slots: int = num_tokens * top_k
        forward_perm[forward_perm >= total_slots] = -1
        return forward_perm
