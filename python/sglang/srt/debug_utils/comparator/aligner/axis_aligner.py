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

    if not _semantic_names_match(specs_pair):
        return None

    source_tokens_pair: Pair[list[str]] = specs_pair.map(_build_einops_tokens)

    # Target: y's tokens with squeeze removed. Fused dims stay fused — both sides
    # flatten toward the fused representation (unflatten is ambiguous without sizes).
    target_tokens: list[str] = _build_fused_target(specs_pair)

    pattern: Pair[Optional[str]] = source_tokens_pair.map(
        lambda tokens: _build_pattern(source=tokens, target=target_tokens)
    )

    if pattern.x is None and pattern.y is None:
        return None

    return AxisAlignerPlan(pattern=pattern)


def _semantic_names_match(specs_pair: Pair[list[DimSpec]]) -> bool:
    """Check that both sides share the same semantic name set (ignoring squeeze dims)."""
    names_pair: Pair[list[str]] = specs_pair.map(_expand_and_skip_squeeze)

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


def _expand_and_skip_squeeze(specs: list[DimSpec]) -> list[str]:
    """Expand DimSpecs to flat semantic names, skipping squeeze dims."""
    return [name for spec in specs if not _SingletonDimUtil.is_squeeze(spec) for name in spec.sub_dims]


def _build_einops_tokens(specs: list[DimSpec]) -> list[str]:
    """Convert DimSpecs to einops-compatible tokens.

    Fused dims become ``"(a b)"``; squeeze dims stay ``"1"``; plain dims use their name.
    """
    return [
        f"({' '.join(spec.sub_dims)})" if spec.is_fused else spec.name
        for spec in specs
    ]


def _build_fused_target(specs_pair: Pair[list[DimSpec]]) -> list[str]:
    """Build target token list that prefers the fused representation.

    For each semantic name group, if *either* side has it as a fused dim, the target
    uses the fused ``(a b)`` token. This ensures the separate side always flattens
    (einops can flatten without extra size info, but unflatten is ambiguous).

    Dim order follows y; squeeze dims are excluded.
    """
    # Map each sub-dim name → (fused_token, set_of_siblings) from both sides
    fused_lookup: dict[str, tuple[str, frozenset[str]]] = {}
    for spec in (*specs_pair.x, *specs_pair.y):
        if spec.is_fused:
            token: str = f"({' '.join(spec.sub_dims)})"
            siblings: frozenset[str] = frozenset(spec.sub_dims)
            for sub_name in spec.sub_dims:
                fused_lookup.setdefault(sub_name, (token, siblings))

    # Walk y's semantic names in order, emitting fused tokens or plain names
    result: list[str] = []
    consumed: set[str] = set()

    for spec in specs_pair.y:
        if _SingletonDimUtil.is_squeeze(spec):
            continue

        names: list[str] = spec.sub_dims
        if any(n in consumed for n in names):
            continue

        entry: Optional[tuple[str, frozenset[str]]] = fused_lookup.get(names[0])
        if entry is not None:
            fused_token, siblings = entry
            result.append(fused_token)
            consumed.update(siblings)
        else:
            result.append(spec.name)
            consumed.update(names)

    return result


def _build_pattern(*, source: list[str], target: list[str]) -> Optional[str]:
    """Build an einops rearrange pattern. Returns None if already matching."""
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
