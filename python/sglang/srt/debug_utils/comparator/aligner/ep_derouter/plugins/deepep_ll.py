from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.base import DeRouterPlugin


class DeepEPLLDeRouter(DeRouterPlugin):
    """De-router for SGLang DeepEP Low-Latency dispatch path.

    Routed tensor is 3D: ``(num_experts, expected_m, hidden_size)``.
    ``masked_m[expert_i]`` specifies how many of the ``expected_m`` rows
    are valid for expert ``expert_i``.

    ``packed_recv_src_info`` has shape ``(num_experts, expected_m)`` and encodes
    the source identity of each received token.  Based on DeepEP test code, the
    encoding is::

        packed_recv_src_info[e][j] % num_tokens == original_token_index

    More precisely the packing is ``(source_rank * tokens_per_rank + token_index_in_rank)``
    which we can decode with modular arithmetic.
    """

    def de_route(
        self,
        routed_tensor: torch.Tensor,
        aux_tensors: dict[str, torch.Tensor],
        *,
        num_tokens: int,
        top_k: int,
    ) -> torch.Tensor:
        packed_recv_src_info: torch.Tensor = aux_tensors["packed_recv_src_info"]
        masked_m: torch.Tensor = aux_tensors["masked_m"]

        num_experts: int = routed_tensor.shape[0]
        expected_m: int = routed_tensor.shape[1]
        trailing_shape: list[int] = list(routed_tensor.shape[2:])
        total_slots: int = num_tokens * top_k

        output: torch.Tensor = torch.zeros(
            [total_slots] + trailing_shape,
            dtype=routed_tensor.dtype,
            device=routed_tensor.device,
        )

        # Track how many times each token has been seen to assign k-index
        token_k_counter: torch.Tensor = torch.zeros(
            num_tokens, dtype=torch.long, device=routed_tensor.device
        )

        for expert_i in range(num_experts):
            valid_count: int = int(masked_m[expert_i].item())
            if valid_count == 0:
                continue

            # Extract valid rows for this expert
            expert_src_info: torch.Tensor = packed_recv_src_info[expert_i, :valid_count]
            expert_hidden: torch.Tensor = routed_tensor[expert_i, :valid_count]

            # Decode token index from packed format
            token_indices: torch.Tensor = expert_src_info.long() % num_tokens

            # Assign k-index for each token occurrence
            for j in range(valid_count):
                token_idx: int = int(token_indices[j].item())
                k_idx: int = int(token_k_counter[token_idx].item())
                flatten_idx: int = token_idx * top_k + k_idx
                if flatten_idx < total_slots:
                    output[flatten_idx] = expert_hidden[j]
                token_k_counter[token_idx] += 1

        return output
