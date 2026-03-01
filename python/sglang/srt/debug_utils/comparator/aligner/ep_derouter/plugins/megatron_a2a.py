from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.base import DeRouterPlugin


class MegatronA2ADeRouter(DeRouterPlugin):
    """De-router for Megatron AlltoAll dispatch path.

    ``reversed_local_input_permutation_mapping`` from ``moe_utils.permute()`` is
    a **token-level** index (values in ``[0, num_tokens)``): position ``i`` in
    the permuted tensor holds original token ``sorted_indices[i]``.

    ``tokens_per_expert[e]`` gives the number of tokens routed to local expert
    ``e``.  Tokens in the permuted tensor are grouped by expert, so the first
    ``tokens_per_expert[0]`` rows belong to expert 0, etc.

    Since ``sorted_indices`` is token-level (not flatten), we need
    ``tokens_per_expert`` to reconstruct the k-index: if a token appears in
    multiple experts, each occurrence corresponds to a different k slot.
    """

    def de_route(
        self,
        routed_tensor: torch.Tensor,
        aux_tensors: dict[str, torch.Tensor],
        *,
        num_tokens: int,
        top_k: int,
    ) -> torch.Tensor:
        sorted_indices: torch.Tensor = aux_tensors[
            "reversed_local_input_permutation_mapping"
        ]
        tokens_per_expert: torch.Tensor = aux_tensors["tokens_per_expert"]

        total_slots: int = num_tokens * top_k
        trailing_shape: list[int] = list(routed_tensor.shape[1:])

        output: torch.Tensor = torch.zeros(
            [total_slots] + trailing_shape,
            dtype=routed_tensor.dtype,
            device=routed_tensor.device,
        )

        # Track how many times each token has been seen so far, to assign k-index
        token_k_counter: torch.Tensor = torch.zeros(
            num_tokens, dtype=torch.long, device=routed_tensor.device
        )

        # sorted_indices[i] = original token index at permuted position i
        token_ids: torch.Tensor = sorted_indices.long()
        num_permuted: int = token_ids.shape[0]

        for i in range(num_permuted):
            token_idx: int = int(token_ids[i].item())
            k_idx: int = int(token_k_counter[token_idx].item())
            flatten_idx: int = token_idx * top_k + k_idx
            if flatten_idx < total_slots:
                output[flatten_idx] = routed_tensor[i]
            token_k_counter[token_idx] += 1

        return output
