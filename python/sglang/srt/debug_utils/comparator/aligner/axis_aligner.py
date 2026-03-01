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


class AxisAlignerPlan(_FrozenBase):
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

    # Build einops source tokens per side (fused dims → "(a b)", squeeze → "1")
    source_pair: Pair[list[str]] = specs_pair.map(_build_einops_tokens)

    # Target order: y's semantic names (no squeeze), each fused group in "(a b)" form
    target_tokens: list[str] = [t for t in source_pair.y if t != SQUEEZE_DIM_NAME]

    pattern: Pair[Optional[str]] = source_pair.map(
        lambda tokens: _build_pattern(source=tokens, target=target_tokens)
    )

    if pattern.x is None and pattern.y is None:
        return None

    return AxisAlignerPlan(pattern=pattern)


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


def _build_einops_tokens(specs: list[DimSpec]) -> list[str]:
    """Convert DimSpecs to einops-compatible tokens.

    Fused dims become ``"(a b)"``; squeeze dims stay ``"1"``; plain dims use their name.
    """
    tokens: list[str] = []
    for spec in specs:
        if spec.is_fused:
            tokens.append(f"({' '.join(spec.sub_dims)})")
        else:
            tokens.append(spec.name)
    return tokens


def _build_pattern(*, source: list[str], target: list[str]) -> Optional[str]:
    """Build an einops rearrange pattern from source tokens to target tokens.

    Returns None if source already matches target (no rearrange needed).
    """
    if source == target:
        return None

    return f"{' '.join(source)} -> {' '.join(target)}"


# --- executor ---


def execute_axis_aligner_plan(
    tensor: torch.Tensor, plan: AxisAlignerPlan, *, side: str
) -> torch.Tensor:
    pattern: Optional[str] = plan.pattern.x if side == "x" else plan.pattern.y

    if pattern is not None:
        tensor = rearrange(tensor.rename(None), pattern)

    return tensor
