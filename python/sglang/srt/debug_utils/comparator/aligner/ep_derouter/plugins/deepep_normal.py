from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.base import DeRouterPlugin


class DeepEPNormalDeRouter(DeRouterPlugin):
    """De-router for SGLang DeepEP Normal dispatch path.

    After dispatch, tokens from all source ranks are laid out contiguously per
    expert.  ``rank_prefix_matrix[src_rank]`` gives the starting offset for
    tokens coming from ``src_rank``.  ``recv_topk_ids[recv_idx, k]`` is the
    **local** expert id assigned to the token.

    We reconstruct the global flatten index for each received token using:
    1. The source rank prefix offsets to determine which source rank each
       received token belongs to.
    2. The position within that rank's contribution gives the original token
       index on that rank.
    3. ``recv_topk_ids`` maps each received row to a (token, k) slot.

    NOTE: The exact encoding of ``rank_prefix_matrix`` is TBD until we have
    real dumps.  The current implementation treats it as a 1-D tensor of
    cumulative counts per source rank: ``rank_prefix_matrix[r]`` =
    total number of tokens from ranks ``< r``.
    """

    def de_route(
        self,
        routed_tensor: torch.Tensor,
        aux_tensors: dict[str, torch.Tensor],
        *,
        num_tokens: int,
        top_k: int,
    ) -> torch.Tensor:
        rank_prefix_matrix: torch.Tensor = aux_tensors["rank_prefix_matrix"]
        recv_topk_ids: torch.Tensor = aux_tensors["recv_topk_ids"]
        num_recv_tokens_per_expert: torch.Tensor = aux_tensors[
            "num_recv_tokens_per_expert"
        ]

        total_slots: int = num_tokens * top_k
        num_recv: int = routed_tensor.shape[0]

        # Build expert offset: prefix sum of num_recv_tokens_per_expert
        expert_offsets: torch.Tensor = torch.zeros(
            num_recv_tokens_per_expert.shape[0] + 1,
            dtype=torch.long,
            device=routed_tensor.device,
        )
        expert_offsets[1:] = num_recv_tokens_per_expert.cumsum(dim=0)

        # For each received token, compute its position within its expert.
        # Tokens are grouped by expert contiguously.
        # recv_topk_ids tells us the expert assignment for each received token's
        # top_k slot. We need to figure out the global (token, k) identity.

        # The rank_prefix_matrix provides the cumulative token count per source rank.
        # Source rank r contributed tokens from index rank_prefix[r] to rank_prefix[r+1].
        # Within each source rank's contribution, the tokens are in order of the
        # original token index on that rank, with each token appearing top_k times.

        # Build source rank assignment for each received token
        rank_prefix: torch.Tensor = rank_prefix_matrix.to(
            dtype=torch.long, device=routed_tensor.device
        ).flatten()
        num_ranks: int = rank_prefix.shape[0]

        # rank_prefix is cumulative: token at position pos came from rank r
        # where rank_prefix[r] <= pos < rank_prefix[r+1]
        # Position within rank: pos - rank_prefix[r]
        # That maps to original token: (rank * tokens_per_rank + local_pos)

        # Compute cumulative bounds for rank assignment
        if num_ranks > 1:
            rank_bounds: torch.Tensor = rank_prefix.clone()
            if rank_bounds[-1] < num_recv:
                rank_bounds = torch.cat(
                    [rank_bounds, torch.tensor([num_recv], device=rank_bounds.device)]
                )
        else:
            rank_bounds = torch.tensor([0, num_recv], device=routed_tensor.device)

        # Assign each position to its source rank via searchsorted
        positions: torch.Tensor = torch.arange(
            num_recv, dtype=torch.long, device=routed_tensor.device
        )
        source_ranks: torch.Tensor = (
            torch.searchsorted(rank_bounds, positions, right=True) - 1
        )
        local_positions: torch.Tensor = positions - rank_bounds[source_ranks]

        # Global token index: source_rank * tokens_per_rank + local_token_idx
        # where local_token_idx = local_positions // top_k  (within rank)
        # and k_idx = local_positions % top_k
        tokens_per_rank: int = num_tokens // num_ranks
        global_token_idx: torch.Tensor = (
            source_ranks * tokens_per_rank + local_positions // top_k
        )
        k_idx: torch.Tensor = local_positions % top_k
        flatten_idx: torch.Tensor = global_token_idx * top_k + k_idx

        # Scatter into output
        trailing_shape: list[int] = list(routed_tensor.shape[1:])
        output: torch.Tensor = torch.zeros(
            [total_slots] + trailing_shape,
            dtype=routed_tensor.dtype,
            device=routed_tensor.device,
        )

        valid_mask: torch.Tensor = flatten_idx < total_slots
        output[flatten_idx[valid_mask]] = routed_tensor[valid_mask]

        return output
