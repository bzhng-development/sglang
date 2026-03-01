from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.ep_derouter.base import DeRouterPlugin
from sglang.srt.debug_utils.comparator.aligner.ep_derouter.plugins.utils import (
    compute_within_group_indices,
)


class DeepEPLLDeRouter(DeRouterPlugin):
    """De-router for SGLang DeepEP Low-Latency dispatch path.

    Routed tensor is 3D: ``(num_experts, expected_m, hidden_size)``.
    ``masked_m[expert_i]`` specifies how many of the ``expected_m`` rows
    are valid for expert ``expert_i``.

    ``packed_recv_src_info`` has shape ``(num_experts, expected_m)`` and encodes
    the source identity of each received token.  The encoding is::

        packed_recv_src_info[e][j] % num_tokens == original_token_index
    """

    def flatten_routed_tensor(
        self,
        routed_tensor: torch.Tensor,
        aux_tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        masked_m: torch.Tensor = aux_tensors["masked_m"]
        return _extract_valid_rows(routed_tensor, masked_m)

    def compute_forward_permutation(
        self,
        aux_tensors: dict[str, torch.Tensor],
        *,
        num_tokens: int,
        top_k: int,
        num_routed: int,
    ) -> torch.Tensor:
        packed_recv_src_info: torch.Tensor = aux_tensors["packed_recv_src_info"]
        masked_m: torch.Tensor = aux_tensors["masked_m"]

        flat_src_info: torch.Tensor = _extract_valid_rows(
            packed_recv_src_info.unsqueeze(-1), masked_m
        ).squeeze(-1)

        token_ids: torch.Tensor = flat_src_info.long() % num_tokens
        k_indices: torch.Tensor = compute_within_group_indices(token_ids)
        forward_perm: torch.Tensor = token_ids * top_k + k_indices

        total_slots: int = num_tokens * top_k
        forward_perm[forward_perm >= total_slots] = -1
        return forward_perm


def _extract_valid_rows(
    tensor_3d: torch.Tensor, masked_m: torch.Tensor
) -> torch.Tensor:
    """Extract valid rows from a 3D ``[num_experts, expected_m, ...]`` tensor.

    For each expert ``e``, takes the first ``masked_m[e]`` rows and concatenates
    them into a 2D tensor.
    """
    expected_m: int = tensor_3d.shape[1]
    arange: torch.Tensor = torch.arange(expected_m, device=masked_m.device)
    valid_mask: torch.Tensor = arange.unsqueeze(0) < masked_m.unsqueeze(1)
    return tensor_3d[valid_mask]
