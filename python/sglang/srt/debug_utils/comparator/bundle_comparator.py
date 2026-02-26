"""Compare two tensor bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import torch

from sglang.srt.debug_utils.comparator.aligner.entrypoint.executor import (
    AlignerResult,
    execute_aligner_plan,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.planner import (
    compute_aligner_plan,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import AlignerPlan
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerPlan,
)
from sglang.srt.debug_utils.comparator.dims import apply_dim_names, parse_dim_names
from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    ScalarRecord,
    SkipRecord,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.comparator import (
    compare_tensor_pair,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink
from sglang.srt.debug_utils.dump_loader import ValueWithMeta

_FAILED_SIDE_MAP: dict[str, str] = {"x": "baseline", "y": "target"}


def compare_bundle_pair(
    *,
    name: str,
    filenames_pair: Pair[list[str]],
    baseline_path: Path,
    target_path: Path,
    token_aligner_plan: Optional[TokenAlignerPlan],
    diff_threshold: float,
    thd_seq_lens_by_step_pair: Pair[Optional[dict[int, list[int]]]] = Pair(
        x=None, y=None
    ),
) -> Union[ComparisonRecord, SkipRecord, ScalarRecord]:
    with warning_sink.context() as collected_warnings:
        result = _compare_bundle_pair_raw(
            name=name,
            filenames_pair=filenames_pair,
            baseline_path=baseline_path,
            target_path=target_path,
            token_aligner_plan=token_aligner_plan,
            diff_threshold=diff_threshold,
            thd_seq_lens_by_step_pair=thd_seq_lens_by_step_pair,
        )

    return result.model_copy(update={"warnings": collected_warnings})


def _compare_bundle_pair_raw(
    *,
    name: str,
    filenames_pair: Pair[list[str]],
    baseline_path: Path,
    target_path: Path,
    token_aligner_plan: Optional[TokenAlignerPlan],
    diff_threshold: float,
    thd_seq_lens_by_step_pair: Pair[Optional[dict[int, list[int]]]] = Pair(
        x=None, y=None
    ),
) -> Union[ComparisonRecord, SkipRecord, ScalarRecord]:
    # 1. Load all non-None values
    all_pair: Pair[list[ValueWithMeta]] = Pair(
        x=_load_all_values(filenames=filenames_pair.x, base_path=baseline_path),
        y=_load_all_values(filenames=filenames_pair.y, base_path=target_path),
    )

    if not all_pair.x or not all_pair.y:
        reason = "baseline_load_failed" if not all_pair.x else "target_load_failed"
        return SkipRecord(name=name, reason=reason)

    # 2. Classify into tensor vs non-tensor
    tensor_pair: Pair[list[ValueWithMeta]] = all_pair.map(
        lambda items: [it for it in items if isinstance(it.value, torch.Tensor)]
    )
    scalar_pair: Pair[list[ValueWithMeta]] = all_pair.map(
        lambda items: [it for it in items if not isinstance(it.value, torch.Tensor)]
    )

    has_tensors: bool = bool(tensor_pair.x) or bool(tensor_pair.y)
    has_scalars: bool = bool(scalar_pair.x) or bool(scalar_pair.y)

    # 3. Both sides have tensors → existing tensor comparison path
    if has_tensors and not has_scalars:
        return _compare_tensor_pair(
            name=name,
            valid_pair=tensor_pair,
            token_aligner_plan=token_aligner_plan,
            diff_threshold=diff_threshold,
            thd_seq_lens_by_step_pair=thd_seq_lens_by_step_pair,
        )

    # 4. Both sides have scalars (no tensors) → scalar comparison
    if has_scalars and not has_tensors:
        return _compare_scalar_pair(name=name, scalar_pair=scalar_pair)

    # 5. Mixed: one side tensor, other side non-tensor → type mismatch
    return SkipRecord(name=name, reason="type_mismatch")


def _compare_tensor_pair(
    *,
    name: str,
    valid_pair: Pair[list[ValueWithMeta]],
    token_aligner_plan: Optional[TokenAlignerPlan],
    diff_threshold: float,
    thd_seq_lens_by_step_pair: Pair[Optional[dict[int, list[int]]]] = Pair(
        x=None, y=None
    ),
) -> Union[ComparisonRecord, SkipRecord]:
    if not valid_pair.x or not valid_pair.y:
        reason = "baseline_load_failed" if not valid_pair.x else "target_load_failed"
        return SkipRecord(name=name, reason=reason)

    # Plan (meta only, no tensor)
    metas_pair: Pair[list[dict[str, Any]]] = valid_pair.map(
        lambda items: [it.meta for it in items]
    )
    plan: AlignerPlan = compute_aligner_plan(
        metas_pair=metas_pair,
        token_aligner_plan=token_aligner_plan,
        thd_seq_lens_by_step_pair=thd_seq_lens_by_step_pair,
    )

    # Apply dim names to tensors, then execute
    tensors_pair: Pair[list[torch.Tensor]] = Pair(
        x=_apply_dim_names_from_meta(
            tensors=[it.value for it in valid_pair.x],
            metas=metas_pair.x,
        ),
        y=_apply_dim_names_from_meta(
            tensors=[it.value for it in valid_pair.y],
            metas=metas_pair.y,
        ),
    )
    aligner_result: AlignerResult = execute_aligner_plan(
        tensors_pair=tensors_pair, plan=plan
    )

    if aligner_result.tensors is None:
        assert aligner_result.failed_side_xy is not None
        side_name: str = _FAILED_SIDE_MAP[aligner_result.failed_side_xy]
        reason: str = f"{side_name}_load_failed"
        return SkipRecord(name=name, reason=reason)

    # Compare
    info = compare_tensor_pair(
        x_baseline=aligner_result.tensors.x.rename(None),
        x_target=aligner_result.tensors.y.rename(None),
        name=name,
        diff_threshold=diff_threshold,
    )
    return ComparisonRecord(**info.model_dump(), aligner_plan=plan)


def _compare_scalar_pair(
    *,
    name: str,
    scalar_pair: Pair[list[ValueWithMeta]],
) -> ScalarRecord:
    baseline_value: Any = scalar_pair.x[0].value
    target_value: Any = scalar_pair.y[0].value

    try:
        values_equal: bool = bool(baseline_value == target_value)
    except Exception:
        values_equal = False

    return ScalarRecord(
        name=name,
        baseline_value=repr(baseline_value),
        target_value=repr(target_value),
        baseline_type=type(baseline_value).__name__,
        target_type=type(target_value).__name__,
        values_equal=values_equal,
    )


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

    dim_names: list[str] = parse_dim_names(dims_str)
    return [apply_dim_names(t, dim_names) for t in tensors]


def _load_all_values(filenames: list[str], base_path: Path) -> list[ValueWithMeta]:
    return [
        item
        for f in filenames
        if (item := ValueWithMeta.load(base_path / f)).value is not None
    ]
