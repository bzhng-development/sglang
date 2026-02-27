from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange

from sglang.srt.debug_utils.comparator.dims import (
    DimSpec,
    is_squeeze_dim,
    make_singleton_name,
    parse_dims,
)
from sglang.srt.debug_utils.comparator.utils import Pair, _FrozenBase
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink

# --- types ---


class AxisAlignerPlan(_FrozenBase):
    squeeze_x: list[str]  # singleton dim names to squeeze on x side
    squeeze_y: list[str]  # singleton dim names to squeeze on y side
    swap_pattern: Optional[str]  # einops pattern, None if no swap needed


# --- planner ---


def compute_axis_aligner_plan(
    dims_str_pair: Pair[Optional[str]],
) -> Optional[AxisAlignerPlan]:
    if dims_str_pair.x is None or dims_str_pair.y is None:
        return None

    squeeze_x, x_names = _extract_squeeze_and_names(parse_dims(dims_str_pair.x))
    squeeze_y, y_names = _extract_squeeze_and_names(parse_dims(dims_str_pair.y))

    swap_pattern: Optional[str] = _compute_swap_pattern(x_names, y_names)

    if not squeeze_x and not squeeze_y and swap_pattern is None:
        return None

    return AxisAlignerPlan(
        squeeze_x=squeeze_x,
        squeeze_y=squeeze_y,
        swap_pattern=swap_pattern,
    )


def _extract_squeeze_and_names(
    dim_specs: list[DimSpec],
) -> tuple[list[str], list[str]]:
    """Split dim_specs into (singleton_names_for_squeeze, non_squeeze_dim_names)."""
    squeeze_names: list[str] = []
    dim_names: list[str] = []
    sq_idx: int = 0

    for spec in dim_specs:
        if is_squeeze_dim(spec):
            squeeze_names.append(make_singleton_name(sq_idx))
            sq_idx += 1
        else:
            dim_names.append(spec.name)

    return squeeze_names, dim_names


def _compute_swap_pattern(
    x_names: list[str], y_names: list[str]
) -> Optional[str]:
    if x_names == y_names:
        return None

    if set(x_names) != set(y_names):
        # Local import to avoid circular dependency:
        # output_types -> aligner/entrypoint/types -> axis_aligner -> output_types
        from sglang.srt.debug_utils.comparator.output_types import GeneralWarning

        warning_sink.add(
            GeneralWarning(
                category="axis_aligner_dim_mismatch",
                message=(
                    f"AxisAligner: dim name sets differ (x={x_names}, y={y_names}), "
                    f"skipping axis swap"
                ),
            )
        )
        return None

    return f"{' '.join(x_names)} -> {' '.join(y_names)}"


# --- executor ---


def execute_axis_aligner_plan(
    tensor: torch.Tensor, plan: AxisAlignerPlan, *, side: str
) -> torch.Tensor:
    squeeze_names: list[str] = plan.squeeze_x if side == "x" else plan.squeeze_y

    for name in squeeze_names:
        dim_idx: int = list(tensor.names).index(name)
        tensor = tensor.squeeze(dim_idx)

    if plan.swap_pattern is not None and side == "x":
        tensor = rearrange(tensor.rename(None), plan.swap_pattern)

    return tensor
