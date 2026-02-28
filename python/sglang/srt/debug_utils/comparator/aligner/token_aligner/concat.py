from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.dims import (
    TOKEN_DIM_NAME,
    resolve_dim_by_name,
)
from sglang.srt.debug_utils.comparator.utils import Pair

_UNNAMED_TOKEN_DIM_FALLBACK: int = 0


def execute_concat(
    tensor_of_step_pair: Pair[dict[int, torch.Tensor]],
) -> Pair[torch.Tensor]:
    """Concat all steps in order, then truncate to min(total_x, total_y) tokens."""
    some_tensor: torch.Tensor = next(iter(tensor_of_step_pair.x.values()))
    token_dim: int = _resolve_token_dim(some_tensor)

    x: torch.Tensor = _concat_steps(tensor_of_step_pair.x, dim=token_dim)
    y: torch.Tensor = _concat_steps(tensor_of_step_pair.y, dim=token_dim)
    common: int = min(x.shape[token_dim], y.shape[token_dim])
    return Pair(
        x=x.narrow(dim=token_dim, start=0, length=common),
        y=y.narrow(dim=token_dim, start=0, length=common),
    )


def _resolve_token_dim(tensor: torch.Tensor) -> int:
    if tensor.names[0] is None:
        return _UNNAMED_TOKEN_DIM_FALLBACK
    return resolve_dim_by_name(tensor, TOKEN_DIM_NAME)


def _concat_steps(tensor_of_step: dict[int, torch.Tensor], *, dim: int) -> torch.Tensor:
    return torch.cat([tensor_of_step[s] for s in sorted(tensor_of_step)], dim=dim)
