"""DP-aware tensor bundle comparison.

When dp_size > 1, groups ValueWithMeta items by dp_rank, unshardes each
group independently, concatenates results along the token dimension, then
applies cross-side token alignment and comparison.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

import torch

from sglang.srt.debug_utils.comparator.aligner.axis_aligner import (
    AxisAlignerPlan,
    compute_axis_aligner_plan,
    execute_axis_aligner_plan,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.executor import (
    _execute_step_plans,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.planner import (
    _compute_per_step_plans,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import (
    AlignerPerStepPlan,
    AlignerPlan,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.executor import (
    execute_token_aligner,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerPlan,
)
from sglang.srt.debug_utils.comparator.dims import (
    SEQ_DIM_NAME,
    TOKEN_DIM_NAME,
    apply_dim_names,
    resolve_dim_names,
)
from sglang.srt.debug_utils.comparator.dp_utils import (
    extract_dp_rank_from_meta,
    extract_dp_size_from_meta,
)
from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    GeneralWarning,
    SkipRecord,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.comparator import (
    compare_tensor_pair,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink
from sglang.srt.debug_utils.dump_loader import ValueWithMeta

_FAILED_SIDE_MAP: dict[str, str] = {"x": "baseline", "y": "target"}


# ── DP detection ──────────────────────────────────────────────────


def any_side_has_dp(all_pair: Pair[list[ValueWithMeta]]) -> bool:
    """Check if either side has dp_size > 1."""
    for items in (all_pair.x, all_pair.y):
        if items and extract_dp_size_from_meta(items[0].meta) > 1:
            return True
    return False


def group_by_dp_rank(
    items: list[ValueWithMeta],
) -> dict[int, list[ValueWithMeta]]:
    """Group ValueWithMeta items by dp_rank."""
    groups: dict[int, list[ValueWithMeta]] = defaultdict(list)
    for item in items:
        dp_rank: int = extract_dp_rank_from_meta(item.meta)
        groups[dp_rank].append(item)
    return dict(groups)


# ── DP-aware unshard ──────────────────────────────────────────────


def unshard_per_dp_rank(
    *,
    items: list[ValueWithMeta],
    thd_seq_lens_by_step: Optional[dict[int, list[int]]],
) -> dict[int, torch.Tensor]:
    """Group items by dp_rank, unshard each group independently, return step→tensor mapping.

    For each dp_rank group, compute and execute per-step plans (unshard + reorder).
    Then concat across dp_ranks for each step.
    """
    dp_groups: dict[int, list[ValueWithMeta]] = group_by_dp_rank(items)

    if len(dp_groups) <= 1:
        metas: list[dict[str, Any]] = [it.meta for it in items]
        tensors: list[torch.Tensor] = _apply_dim_names_from_meta(
            tensors=[it.value for it in items], metas=metas
        )
        step_plans: list[AlignerPerStepPlan] = _compute_per_step_plans(
            metas=metas, thd_seq_lens_by_step=thd_seq_lens_by_step
        )
        return _execute_step_plans(tensors=tensors, step_plans=step_plans)

    per_dp_step_tensors: list[dict[int, torch.Tensor]] = []
    for dp_rank in sorted(dp_groups.keys()):
        group_items: list[ValueWithMeta] = dp_groups[dp_rank]
        group_metas: list[dict[str, Any]] = [it.meta for it in group_items]
        group_tensors: list[torch.Tensor] = _apply_dim_names_from_meta(
            tensors=[it.value for it in group_items], metas=group_metas
        )
        step_plans = _compute_per_step_plans(
            metas=group_metas, thd_seq_lens_by_step=thd_seq_lens_by_step
        )
        step_result: dict[int, torch.Tensor] = _execute_step_plans(
            tensors=group_tensors, step_plans=step_plans
        )
        per_dp_step_tensors.append(step_result)

    return _concat_step_tensors_across_dp(per_dp_step_tensors)


def _concat_step_tensors_across_dp(
    per_dp: list[dict[int, torch.Tensor]],
) -> dict[int, torch.Tensor]:
    """Concat step tensors from multiple dp_ranks along the token dim (dim 0)."""
    all_steps: set[int] = set()
    for dp_result in per_dp:
        all_steps.update(dp_result.keys())

    merged: dict[int, torch.Tensor] = {}
    for step in sorted(all_steps):
        parts: list[torch.Tensor] = [
            dp_result[step] for dp_result in per_dp if step in dp_result
        ]
        if len(parts) == 1:
            merged[step] = parts[0]
        else:
            token_dim: int = _resolve_token_dim(parts[0])
            stripped: list[torch.Tensor] = [p.rename(None) for p in parts]
            concated: torch.Tensor = torch.cat(stripped, dim=token_dim)
            if parts[0].names[0] is not None:
                concated = concated.refine_names(*parts[0].names)
            merged[step] = concated

    return merged


def _resolve_token_dim(tensor: torch.Tensor) -> int:
    """Resolve the token/seq dimension for DP concat."""
    if tensor.names[0] is not None:
        names: tuple[Optional[str], ...] = tensor.names
        for target_name in (TOKEN_DIM_NAME, SEQ_DIM_NAME):
            if target_name in names:
                return list(names).index(target_name)
    return 0


# ── DP tensor comparison ──────────────────────────────────────────


def compare_bundle_pair_tensor_type_dp(
    *,
    name: str,
    valid_pair: Pair[list[ValueWithMeta]],
    token_aligner_plan: Optional[TokenAlignerPlan],
    diff_threshold: float,
    thd_seq_lens_by_step_pair: Pair[Optional[dict[int, list[int]]]] = Pair(
        x=None, y=None
    ),
    viz_output_dir: Optional[Path] = None,
    compute_per_token: bool = False,
) -> Union[ComparisonRecord, SkipRecord]:
    """DP-aware tensor comparison: group by dp_rank, unshard per group, concat, then align."""
    if not valid_pair.x or not valid_pair.y:
        reason = "baseline_load_failed" if not valid_pair.x else "target_load_failed"
        return SkipRecord(name=name, reason=reason)

    # Per-side: unshard within dp_rank groups, concat across dp_ranks → dict[step, tensor]
    step_tensors_x: dict[int, torch.Tensor] = unshard_per_dp_rank(
        items=valid_pair.x,
        thd_seq_lens_by_step=thd_seq_lens_by_step_pair.x,
    )
    step_tensors_y: dict[int, torch.Tensor] = unshard_per_dp_rank(
        items=valid_pair.y,
        thd_seq_lens_by_step=thd_seq_lens_by_step_pair.y,
    )

    if not step_tensors_x or not step_tensors_y:
        failed_side_xy: str = "x" if not step_tensors_x else "y"
        side_name: str = _FAILED_SIDE_MAP[failed_side_xy]
        return SkipRecord(name=name, reason=f"{side_name}_load_failed")

    # Cross-side: token alignment
    if token_aligner_plan is not None:
        combined: Pair[torch.Tensor] = execute_token_aligner(
            plan=token_aligner_plan,
            tensor_of_step_pair=Pair(x=step_tensors_x, y=step_tensors_y),
        )
    else:
        assert len(step_tensors_x) == 1 and len(step_tensors_y) == 1
        combined = Pair(
            x=list(step_tensors_x.values())[0],
            y=list(step_tensors_y.values())[0],
        )

    # Cross-side: axis alignment
    metas_pair: Pair[list[dict[str, Any]]] = valid_pair.map(
        lambda items: [it.meta for it in items]
    )
    dims_str_pair: Pair[Optional[str]] = metas_pair.map(
        lambda metas: metas[0].get("dims") if metas else None
    )
    axis_aligner_plan: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
        dims_str_pair=dims_str_pair
    )
    if axis_aligner_plan is not None:
        combined = Pair(
            x=execute_axis_aligner_plan(
                tensor=combined.x, plan=axis_aligner_plan, side="x"
            ),
            y=execute_axis_aligner_plan(
                tensor=combined.y, plan=axis_aligner_plan, side="y"
            ),
        )

    plan: AlignerPlan = AlignerPlan(
        per_step_plans=Pair(x=[], y=[]),
        token_aligner_plan=token_aligner_plan,
        axis_aligner_plan=axis_aligner_plan,
    )

    # Resolve seq_dim for per-token computation
    seq_dim: Optional[int] = _resolve_seq_dim(combined.y) if compute_per_token else None

    # Compare
    aligned_baseline: torch.Tensor = combined.x.rename(None)
    aligned_target: torch.Tensor = combined.y.rename(None)

    info = compare_tensor_pair(
        x_baseline=aligned_baseline,
        x_target=aligned_target,
        name=name,
        diff_threshold=diff_threshold,
        seq_dim=seq_dim,
    )
    record = ComparisonRecord(**info.model_dump(), aligner_plan=plan)

    if viz_output_dir is not None:
        _try_generate_viz(
            baseline=aligned_baseline,
            target=aligned_target,
            name=name,
            viz_output_dir=viz_output_dir,
        )

    return record


# ── shared helpers (duplicated from bundle_comparator to avoid circular imports) ──


def _resolve_seq_dim(tensor: torch.Tensor) -> Optional[int]:
    """Find the token/seq dimension index from the tensor's named dims."""
    if tensor.names[0] is None:
        return None

    names: tuple[Optional[str], ...] = tensor.names
    for target_name in (TOKEN_DIM_NAME, SEQ_DIM_NAME):
        if target_name in names:
            return list(names).index(target_name)

    return None


def _apply_dim_names_from_meta(
    *,
    tensors: list[torch.Tensor],
    metas: list[dict[str, Any]],
) -> list[torch.Tensor]:
    if not metas:
        return tensors

    dims_str: Optional[str] = metas[0].get("dims")
    if dims_str is None:
        return tensors

    dim_names: list[str] = resolve_dim_names(dims_str)
    return [apply_dim_names(t, dim_names) for t in tensors]


def _try_generate_viz(
    *,
    baseline: torch.Tensor,
    target: torch.Tensor,
    name: str,
    viz_output_dir: Path,
) -> None:
    from sglang.srt.debug_utils.comparator.visualizer import (
        generate_comparison_figure,
    )
    from sglang.srt.debug_utils.comparator.visualizer.preprocessing import (
        _sanitize_filename,
    )

    filename: str = _sanitize_filename(name) + ".png"
    output_path: Path = viz_output_dir / filename

    try:
        generate_comparison_figure(
            baseline=baseline,
            target=target,
            name=name,
            output_path=output_path,
        )
    except Exception as exc:
        warning_sink.add(
            GeneralWarning(
                category="visualizer",
                message=f"Visualization failed for {name}: {exc}",
            )
        )
