from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange

from sglang.srt.debug_utils.comparator.dims import (
    SQUEEZE_DIM_NAME,
    DimSpec,
    _SingletonDimUtil,
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
    flatten: Pair[Optional[FlattenPlan]]
    pattern: Pair[Optional[str]]  # einops pattern per side, None = no-op


# --- planner ---


def compute_axis_aligner_plan(
    dims_str_pair: Pair[Optional[str]],
) -> Optional[AxisAlignerPlan]:
    if dims_str_pair.x is None or dims_str_pair.y is None:
        return None

    dims_pair: Pair[str] = Pair(x=dims_str_pair.x, y=dims_str_pair.y)
    specs_pair: Pair[list[DimSpec]] = dims_pair.map(lambda s: parse_dims(s).dims)

    # Verify both sides share the same semantic name set (expanded, no squeeze)
    if not _semantic_names_match(specs_pair):
        return None

    # Compute flatten plans: flatten the *separate* side to match the fused side
    flatten_plan: Pair[Optional[FlattenPlan]] = Pair(
        x=_compute_flatten_plan(this_specs=specs_pair.x, other_specs=specs_pair.y),
        y=_compute_flatten_plan(this_specs=specs_pair.y, other_specs=specs_pair.x),
    )

    # After flatten, compute the physical dim names for einops pattern.
    # These are what we use for einops rearrange (squeeze dims as "1", fused as "a__b").
    post_flatten_names: Pair[list[str]] = Pair(
        x=_names_after_flatten(specs=specs_pair.x, flatten_plan=flatten_plan.x),
        y=_names_after_flatten(specs=specs_pair.y, flatten_plan=flatten_plan.y),
    )

    # Target order: y's post-flatten names, with squeeze dims filtered out
    target_order: list[str] = [n for n in post_flatten_names.y if n != SQUEEZE_DIM_NAME]

    pattern: Pair[Optional[str]] = post_flatten_names.map(
        lambda names: _build_pattern(source=names, target=target_order)
    )

    if (
        flatten_plan.x is None
        and flatten_plan.y is None
        and pattern.x is None
        and pattern.y is None
    ):
        return None

    return AxisAlignerPlan(flatten=flatten_plan, pattern=pattern)


def _semantic_names_match(specs_pair: Pair[list[DimSpec]]) -> bool:
    """Check that both sides share the same semantic name set (ignoring squeeze dims).

    Fused dims expand to sub-dim names for comparison.
    """
    names_pair: Pair[list[str]] = specs_pair.map(_expand_non_squeeze)

    if set(names_pair.x) == set(names_pair.y):
        return True

    # Local import to avoid circular dependency:
    # output_types -> aligner/entrypoint/types -> axis_aligner -> output_types
    from sglang.srt.debug_utils.comparator.output_types import ErrorLog

    log_sink.add(
        ErrorLog(
            category="axis_aligner_dim_mismatch",
            message=(
                f"AxisAligner: dim name sets differ (x={names_pair.x}, y={names_pair.y}), "
                f"skipping axis swap"
            ),
        )
    )
    return False


def _expand_non_squeeze(specs: list[DimSpec]) -> list[str]:
    """Expand DimSpecs to flat semantic names, skipping squeeze dims."""
    result: list[str] = []
    for spec in specs:
        if _SingletonDimUtil.is_squeeze(spec):
            continue
        result.extend(spec.sub_dims)
    return result


def _compute_flatten_plan(
    *, this_specs: list[DimSpec], other_specs: list[DimSpec]
) -> Optional[FlattenPlan]:
    """Determine if *this* side needs flatten to align with the *other* side.

    For each fused dim on the *other* side, check if *this* side has those
    sub-dims as consecutive separate axes — if so, flatten them.
    """
    other_fused_groups: dict[frozenset[str], str] = {}
    for spec in other_specs:
        if spec.is_fused:
            other_fused_groups[frozenset(spec.sub_dims)] = spec.sanitized_name

    if not other_fused_groups:
        return None

    # Build name→index mapping for this side (only non-squeeze, non-fused)
    this_name_to_idx: dict[str, int] = {}
    for phys_idx, spec in enumerate(this_specs):
        if _SingletonDimUtil.is_squeeze(spec):
            continue
        if spec.is_fused:
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
    return all(indices[i] + 1 == indices[i + 1] for i in range(len(indices) - 1))


def _names_after_flatten(
    *, specs: list[DimSpec], flatten_plan: Optional[FlattenPlan]
) -> list[str]:
    """Compute the physical dim names after applying flatten.

    Squeeze dims use their original name ("1"). Fused dims use sanitized_name.
    Flatten groups merge multiple names into one.
    """
    names: list[str] = [spec.sanitized_name for spec in specs]

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
        plan.flatten.x if side == "x" else plan.flatten.y
    )
    pattern: Optional[str] = plan.pattern.x if side == "x" else plan.pattern.y

    if flatten_plan is not None:
        tensor = _execute_flatten(tensor, flatten_plan)

    if pattern is not None:
        tensor = rearrange(tensor.rename(None), pattern)

    return tensor


def _execute_flatten(tensor: torch.Tensor, plan: FlattenPlan) -> torch.Tensor:
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
