from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange

from sglang.srt.debug_utils.comparator.dims import (
    DimSpec,
    _SingletonDimUtil,
    fused_sub_names,
    is_fused,
    parse_dims,
)
from sglang.srt.debug_utils.comparator.log_sink import log_sink
from sglang.srt.debug_utils.comparator.utils import Pair, _FrozenBase

# --- types ---


class FlattenGroup(_FrozenBase):
    """Consecutive physical axes to be flattened into one."""

    dim_indices: list[int]
    target_name: str


class FlattenPlan(_FrozenBase):
    groups: list[FlattenGroup]


class AxisAlignerPlan(_FrozenBase):
    pre_flatten: Pair[Optional[FlattenPlan]]
    pattern: Pair[Optional[str]]  # einops pattern per side, None = no-op


# --- planner ---


def compute_axis_aligner_plan(
    dims_str_pair: Pair[Optional[str]],
) -> Optional[AxisAlignerPlan]:
    if dims_str_pair.x is None or dims_str_pair.y is None:
        return None

    dims_pair: Pair[str] = Pair(x=dims_str_pair.x, y=dims_str_pair.y)

    specs_pair: Pair[list[DimSpec]] = dims_pair.map(
        lambda s: parse_dims(s).dims
    )

    # Expand semantic names: fused dims → sub-dim names, regular → [name]
    semantic_pair: Pair[list[str]] = specs_pair.map(_expand_semantic_names)

    # Check if semantic name sets match (after removing squeeze dims)
    filtered_semantic_pair: Pair[list[str]] = Pair(
        x=_filter_squeeze_from_names(
            semantic_names=semantic_pair.x, specs=specs_pair.x
        ),
        y=_filter_squeeze_from_names(
            semantic_names=semantic_pair.y, specs=specs_pair.y
        ),
    )

    target_order: Optional[list[str]] = _resolve_target_order(
        x_names=filtered_semantic_pair.x, y_names=filtered_semantic_pair.y
    )
    if target_order is None:
        return None

    # Compute flatten plans: flatten the *separate* side to match the fused side
    pre_flatten: Pair[Optional[FlattenPlan]] = Pair(
        x=_compute_flatten_plan(
            this_specs=specs_pair.x, other_specs=specs_pair.y
        ),
        y=_compute_flatten_plan(
            this_specs=specs_pair.y, other_specs=specs_pair.x
        ),
    )

    # After flatten, compute the physical dim names for einops pattern
    post_flatten_names: Pair[list[str]] = Pair(
        x=_names_after_flatten(specs=specs_pair.x, flatten_plan=pre_flatten.x),
        y=_names_after_flatten(specs=specs_pair.y, flatten_plan=pre_flatten.y),
    )

    pattern: Pair[Optional[str]] = post_flatten_names.map(
        lambda names: _build_pattern(source=names, target=target_order)
    )

    if (
        pre_flatten.x is None
        and pre_flatten.y is None
        and pattern.x is None
        and pattern.y is None
    ):
        return None

    return AxisAlignerPlan(pre_flatten=pre_flatten, pattern=pattern)


def _expand_semantic_names(specs: list[DimSpec]) -> list[str]:
    """Expand DimSpecs into flat semantic name list.

    Fused dims expand to sub-dim names; regular dims keep their name.
    Each DimSpec contributes one or more semantic names.
    """
    result: list[str] = []
    for spec in specs:
        if is_fused(spec):
            result.extend(fused_sub_names(spec))
        else:
            result.append(spec.name)
    return result


def _filter_squeeze_from_names(
    *, semantic_names: list[str], specs: list[DimSpec]
) -> list[str]:
    """Remove names that came from squeeze dims."""
    result: list[str] = []
    idx: int = 0
    for spec in specs:
        if _SingletonDimUtil.is_squeeze(spec):
            idx += 1
            continue
        if is_fused(spec):
            n_sub: int = len(spec.sub_dims)
            result.extend(semantic_names[idx : idx + n_sub])
            idx += n_sub
        else:
            result.append(semantic_names[idx])
            idx += 1
    return result


def _resolve_target_order(
    x_names: list[str], y_names: list[str]
) -> Optional[list[str]]:
    """Determine canonical dim order from expanded semantic names.

    Returns y_names if semantic name sets match, else None.
    """
    if x_names == y_names:
        return y_names

    if set(x_names) != set(y_names):
        # Local import to avoid circular dependency:
        # output_types -> aligner/entrypoint/types -> axis_aligner -> output_types
        from sglang.srt.debug_utils.comparator.output_types import ErrorLog

        log_sink.add(
            ErrorLog(
                category="axis_aligner_dim_mismatch",
                message=(
                    f"AxisAligner: dim name sets differ (x={x_names}, y={y_names}), "
                    f"skipping axis swap"
                ),
            )
        )
        return None

    return y_names


def _compute_flatten_plan(
    *, this_specs: list[DimSpec], other_specs: list[DimSpec]
) -> Optional[FlattenPlan]:
    """Determine if *this* side needs flatten to align with the *other* side.

    For each fused dim on the *other* side, check if *this* side has those
    sub-dims as consecutive separate axes — if so, flatten them.
    """
    other_fused_groups: dict[frozenset[str], str] = {}
    for spec in other_specs:
        if is_fused(spec):
            sub_names: list[str] = fused_sub_names(spec)
            other_fused_groups[frozenset(sub_names)] = spec.tensor_name

    if not other_fused_groups:
        return None

    # Build name→index mapping for this side (only non-squeeze, non-fused)
    this_name_to_idx: dict[str, int] = {}
    for phys_idx, spec in enumerate(this_specs):
        if _SingletonDimUtil.is_squeeze(spec):
            continue
        if is_fused(spec):
            continue
        this_name_to_idx[spec.name] = phys_idx

    groups: list[FlattenGroup] = []
    for sub_name_set, target_name in other_fused_groups.items():
        indices: list[int] = []
        for sub_name in sub_name_set:
            if sub_name not in this_name_to_idx:
                break
            indices.append(this_name_to_idx[sub_name])
        else:
            # All sub-names found; verify they are consecutive
            indices.sort()
            if _are_consecutive(indices):
                groups.append(
                    FlattenGroup(dim_indices=indices, target_name=target_name)
                )

    return FlattenPlan(groups=groups) if groups else None


def _are_consecutive(indices: list[int]) -> bool:
    return all(
        indices[i] + 1 == indices[i + 1] for i in range(len(indices) - 1)
    )


def _names_after_flatten(
    *, specs: list[DimSpec], flatten_plan: Optional[FlattenPlan]
) -> list[str]:
    """Compute the physical dim names after applying flatten.

    Squeeze dims use their original name ("1"). Fused dims use tensor_name.
    Flatten groups merge multiple names into one.
    """
    names: list[str] = [spec.tensor_name for spec in specs]

    if flatten_plan is None:
        return names

    # Build set of indices that are part of a flatten group
    consumed: set[int] = set()
    insert_map: dict[int, str] = {}  # first index of group → target_name
    for group in flatten_plan.groups:
        consumed.update(group.dim_indices)
        insert_map[group.dim_indices[0]] = group.target_name

    result: list[str] = []
    for i, name in enumerate(names):
        if i in insert_map:
            result.append(insert_map[i])
        elif i not in consumed:
            result.append(name)

    return result


def _build_pattern(*, source: list[str], target: list[str]) -> Optional[str]:
    """Build an einops rearrange pattern from source dim names to target dim names.

    Returns None if source already matches target (no rearrange needed).
    """
    if source == target:
        return None

    return f"{' '.join(source)} -> {' '.join(target)}"


# --- executor ---


def execute_axis_aligner_plan(
    tensor: torch.Tensor, plan: AxisAlignerPlan, *, side: str
) -> torch.Tensor:
    flatten_plan: Optional[FlattenPlan] = (
        plan.pre_flatten.x if side == "x" else plan.pre_flatten.y
    )
    pattern: Optional[str] = plan.pattern.x if side == "x" else plan.pattern.y

    if flatten_plan is not None:
        tensor = _execute_flatten(tensor, flatten_plan)

    if pattern is not None:
        tensor = rearrange(tensor.rename(None), pattern)

    return tensor


def _execute_flatten(
    tensor: torch.Tensor, plan: FlattenPlan
) -> torch.Tensor:
    """Flatten groups of consecutive dims via reshape.

    Processes groups in reverse index order so earlier indices remain valid.
    """
    result: torch.Tensor = tensor.rename(None)
    sorted_groups: list[FlattenGroup] = sorted(
        plan.groups, key=lambda g: g.dim_indices[0], reverse=True
    )

    for group in sorted_groups:
        shape: list[int] = list(result.shape)
        start: int = group.dim_indices[0]
        end: int = group.dim_indices[-1] + 1
        merged_size: int = 1
        for idx in group.dim_indices:
            merged_size *= shape[idx]
        new_shape: list[int] = shape[:start] + [merged_size] + shape[end:]
        result = result.reshape(new_shape)

    return result
