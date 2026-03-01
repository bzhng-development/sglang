from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class DeRouterPlugin(ABC):
    """Base class for de-routing routed MoE tensors back to global (token, top_k) order."""

    @abstractmethod
    def de_route(
        self,
        routed_tensor: torch.Tensor,
        aux_tensors: dict[str, torch.Tensor],
        *,
        num_tokens: int,
        top_k: int,
    ) -> torch.Tensor:
        """Restore routed tensor to shape ``[num_tokens * top_k, ...]`` in global order.

        Args:
            routed_tensor: The dispatched tensor (shape varies by dispatch path).
            aux_tensors: Auxiliary tensors dumped alongside (e.g. sorted_token_ids).
            num_tokens: Total number of tokens before dispatch.
            top_k: Number of experts per token.

        Returns:
            Tensor of shape ``[num_tokens * top_k, ...]`` in canonical
            ``(token_i * top_k + k)`` order.
        """
