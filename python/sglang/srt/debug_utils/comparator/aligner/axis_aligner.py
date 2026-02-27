from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange

from sglang.srt.debug_utils.comparator.dims import (
    filter_squeeze_dims,
    is_singleton_name,
    parse_dims,
    resolve_dim_names_with_singletons,
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

    x_specs = parse_dims(dims_str_pair.x)
    y_specs = parse_dims(dims_str_pair.y)

    x_resolved: list[str] = resolve_dim_names_with_singletons(x_specs)
    y_resolved: list[str] = resolve_dim_names_with_singletons(y_specs)

    squeeze_x: list[str] = [n for n in x_resolved if is_singleton_name(n)]
    squeeze_y: list[str] = [n for n in y_resolved if is_singleton_name(n)]

    x_names: list[str] = [s.name for s in filter_squeeze_dims(x_specs)]
    y_names: list[str] = [s.name for s in filter_squeeze_dims(y_specs)]

    swap_pattern: Optional[str] = None

    if x_names != y_names:
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
        else:
            swap_pattern = f"{' '.join(x_names)} -> {' '.join(y_names)}"

    if not squeeze_x and not squeeze_y and swap_pattern is None:
        return None

    return AxisAlignerPlan(
        squeeze_x=squeeze_x,
        squeeze_y=squeeze_y,
        swap_pattern=swap_pattern,
    )


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
