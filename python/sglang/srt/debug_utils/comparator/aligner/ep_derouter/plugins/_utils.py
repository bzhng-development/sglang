from __future__ import annotations

import torch


def compute_within_group_indices(token_ids: torch.Tensor) -> torch.Tensor:
    """Vectorized k-counter: for each occurrence of a token, compute its within-group index.

    Given ``token_ids = [A, B, A, A, B]``, returns ``[0, 0, 1, 2, 1]``.

    Algorithm: stable argsort → detect group boundaries → cumulative reset → scatter back.
    Complexity: O(n log n).

    Args:
        token_ids: 1-D tensor of token identifiers (values in ``[0, num_tokens)``).

    Returns:
        1-D ``torch.long`` tensor of the same length, where ``result[i]`` is how many
        times ``token_ids[i]`` has appeared before position ``i``.
    """
    n: int = token_ids.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=token_ids.device)

    sorted_order: torch.Tensor = torch.argsort(token_ids, stable=True)
    sorted_tids: torch.Tensor = token_ids[sorted_order]

    same_as_prev: torch.Tensor = torch.cat([
        torch.tensor([False], device=token_ids.device),
        sorted_tids[1:] == sorted_tids[:-1],
    ])

    positions: torch.Tensor = torch.arange(n, dtype=torch.long, device=token_ids.device)
    reset_positions: torch.Tensor = torch.where(
        ~same_as_prev, positions, torch.zeros_like(positions)
    )
    group_starts: torch.Tensor = torch.cummax(reset_positions, dim=0).values
    within_group: torch.Tensor = positions - group_starts

    result: torch.Tensor = torch.empty_like(token_ids, dtype=torch.long)
    result[sorted_order] = within_group
    return result
