from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.base import DeRouterPlugin


class DeepEPNormalDeRouter(DeRouterPlugin):
    """De-router for SGLang DeepEP Normal dispatch path.

    After dispatch, tokens from all source ranks are laid out contiguously per
    expert.  ``deepep_normal_rank_prefix_matrix[src_rank]`` gives the starting
    offset for tokens coming from ``src_rank``.

    We reconstruct the global flatten index for each received token using:
    1. The source rank prefix offsets to determine which source rank each
       received token belongs to.
    2. The position within that rank's contribution gives the original token
       index on that rank.

    NOTE: The exact encoding of ``rank_prefix_matrix`` is TBD until we have
    real dumps.  The current implementation treats it as a 1-D tensor of
    cumulative counts per source rank: ``rank_prefix_matrix[r]`` =
    total number of tokens from ranks ``< r``.
    """

    @property
    def required_aux_dump_names(self) -> frozenset[str]:
        return frozenset({"deepep_normal_rank_prefix_matrix"})

    def compute_forward_permutation(
        self,
        aux_tensors: dict[str, torch.Tensor],
        *,
        num_tokens: int,
        top_k: int,
        num_routed: int,
    ) -> torch.Tensor:
        rank_prefix_matrix: torch.Tensor = aux_tensors[
            "deepep_normal_rank_prefix_matrix"
        ]
        total_slots: int = num_tokens * top_k

        rank_prefix: torch.Tensor = rank_prefix_matrix.to(dtype=torch.long).flatten()
        num_ranks: int = rank_prefix.shape[0]

        device: torch.device = rank_prefix.device
        if num_ranks > 1:
            rank_bounds: torch.Tensor = rank_prefix.clone()
            if rank_bounds[-1] < num_routed:
                rank_bounds = torch.cat(
                    [rank_bounds, torch.tensor([num_routed], device=device)]
                )
        else:
            rank_bounds = torch.tensor([0, num_routed], device=device)

        positions: torch.Tensor = torch.arange(
            num_routed, dtype=torch.long, device=device
        )
        source_ranks: torch.Tensor = (
            torch.searchsorted(rank_bounds, positions, right=True) - 1
        )
        local_positions: torch.Tensor = positions - rank_bounds[source_ranks]

        tokens_per_rank: int = num_tokens // num_ranks
        global_token_idx: torch.Tensor = (
            source_ranks * tokens_per_rank + local_positions // top_k
        )
        k_idx: torch.Tensor = local_positions % top_k
        flatten_idx: torch.Tensor = global_token_idx * top_k + k_idx

        flatten_idx[flatten_idx >= total_slots] = -1
        return flatten_idx
