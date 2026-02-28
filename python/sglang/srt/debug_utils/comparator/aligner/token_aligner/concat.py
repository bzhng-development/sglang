from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.utils import Pair


def execute_concat(
    tensor_of_step_pair: Pair[dict[int, torch.Tensor]],
) -> Pair[torch.Tensor]:
    """Concat all steps in order, then truncate to min(total_x, total_y) tokens."""
    x: torch.Tensor = _concat_steps(tensor_of_step_pair.x)
    y: torch.Tensor = _concat_steps(tensor_of_step_pair.y)
    common: int = min(x.shape[0], y.shape[0])
    return Pair(x=x[:common], y=y[:common])


def _concat_steps(tensor_of_step: dict[int, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [tensor_of_step[s] for s in sorted(tensor_of_step)], dim=0
    )
