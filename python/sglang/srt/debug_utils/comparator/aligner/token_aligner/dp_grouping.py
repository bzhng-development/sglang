"""DP-aware grouping and merging for the aux_loader pipeline.

When dp_size > 1, rows must be split by dp_rank so that each dp_rank group
is independently unshareded (TP/CP/EP), then the per-group results are
concatenated to form the global TokenAlignerGlobalAux.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import polars as pl

from sglang.srt.debug_utils.comparator.aligner.token_aligner.aux_plugins import (
    _AuxFrameworkPlugin,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerGlobalAux,
    TokenAlignerStepAux,
)
from sglang.srt.debug_utils.comparator.dims import TokenLayout
from sglang.srt.debug_utils.comparator.dp_utils import (
    extract_dp_rank_from_meta_with_plugin,
)
from sglang.srt.debug_utils.dump_loader import ValueWithMeta, filter_rows


def build_rank_to_dp_rank(
    *,
    df: pl.DataFrame,
    dump_path: Path,
    plugin: _AuxFrameworkPlugin,
) -> dict[int, int]:
    """Build a mapping from global rank → dp_rank by loading one file per rank."""
    unique_ranks: list[int] = sorted(df["rank"].unique().to_list())
    rank_to_dp_rank: dict[int, int] = {}

    for rank in unique_ranks:
        rank_rows: list[dict[str, Any]] = filter_rows(df, conditions={"rank": rank})
        if not rank_rows:
            continue

        value: ValueWithMeta = ValueWithMeta.load(dump_path / rank_rows[0]["filename"])
        dp_rank: int = extract_dp_rank_from_meta_with_plugin(
            meta=value.meta, plugin_name=plugin.name
        )
        rank_to_dp_rank[rank] = dp_rank

    return rank_to_dp_rank


def load_multi_dp_groups(
    *,
    dump_path: Path,
    df: pl.DataFrame,
    plugin: _AuxFrameworkPlugin,
    dp_ranks: list[int],
    rank_to_dp_rank: dict[int, int],
    load_single_dp_group_fn: Any,
) -> Optional[TokenAlignerGlobalAux]:
    """Load aux for multiple DP groups, merge by concatenating per step.

    Args:
        load_single_dp_group_fn: callback with signature
            (dump_path, df, plugin, dp_rank) -> Optional[TokenAlignerGlobalAux]
    """
    dp_rank_to_global_ranks: dict[int, list[int]] = defaultdict(list)
    for global_rank, dp_rank in rank_to_dp_rank.items():
        dp_rank_to_global_ranks[dp_rank].append(global_rank)

    per_dp: list[tuple[int, TokenAlignerGlobalAux]] = []
    for dp_rank in dp_ranks:
        global_ranks: list[int] = dp_rank_to_global_ranks[dp_rank]
        dp_df: pl.DataFrame = df.filter(pl.col("rank").is_in(global_ranks))

        result: Optional[TokenAlignerGlobalAux] = load_single_dp_group_fn(
            dump_path=dump_path, df=dp_df, plugin=plugin, dp_rank=dp_rank
        )
        if result is None:
            continue
        per_dp.append((dp_rank, result))

    if not per_dp:
        return None

    if len(per_dp) == 1:
        return per_dp[0][1]

    return _merge_dp_global_auxs(per_dp)


def _merge_dp_global_auxs(
    per_dp: list[tuple[int, TokenAlignerGlobalAux]],
) -> TokenAlignerGlobalAux:
    """Merge multiple DP groups' GlobalAux into one by concatenating step auxs."""
    first: TokenAlignerGlobalAux = per_dp[0][1]
    framework: str = first.framework
    layout: TokenLayout = first.layout

    all_steps: set[int] = set()
    for _, aux in per_dp:
        all_steps.update(aux.step_auxs.keys())

    merged_step_auxs: dict[int, TokenAlignerStepAux] = {}
    for step in sorted(all_steps):
        step_parts: list[TokenAlignerStepAux] = []
        for _, aux in per_dp:
            if step in aux.step_auxs:
                step_parts.append(aux.step_auxs[step])
        merged_step_auxs[step] = _concat_step_auxs(step_parts)

    merged_thd: Optional[dict[int, list[int]]] = _merge_thd_seq_lens(per_dp)

    return TokenAlignerGlobalAux(
        step_auxs=merged_step_auxs,
        framework=framework,
        layout=layout,
        thd_seq_lens_by_step=merged_thd,
    )


def _concat_step_auxs(parts: list[TokenAlignerStepAux]) -> TokenAlignerStepAux:
    """Concatenate multiple TokenAlignerStepAux (from different dp_ranks)."""
    if len(parts) == 1:
        return parts[0]

    input_ids: list[int] = []
    positions: list[int] = []
    seq_lens: list[int] = []
    seq_ids = []

    for part in parts:
        input_ids.extend(part.input_ids)
        positions.extend(part.positions)
        seq_lens.extend(part.seq_lens)
        seq_ids.extend(part.seq_ids)

    return TokenAlignerStepAux(
        input_ids=input_ids,
        positions=positions,
        seq_lens=seq_lens,
        seq_ids=seq_ids,
    )


def _merge_thd_seq_lens(
    per_dp: list[tuple[int, TokenAlignerGlobalAux]],
) -> Optional[dict[int, list[int]]]:
    """Merge thd_seq_lens_by_step across DP groups."""
    has_any: bool = any(aux.thd_seq_lens_by_step is not None for _, aux in per_dp)
    if not has_any:
        return None

    all_steps: set[int] = set()
    for _, aux in per_dp:
        if aux.thd_seq_lens_by_step is not None:
            all_steps.update(aux.thd_seq_lens_by_step.keys())

    merged: dict[int, list[int]] = {}
    for step in sorted(all_steps):
        combined: list[int] = []
        for _, aux in per_dp:
            if (
                aux.thd_seq_lens_by_step is not None
                and step in aux.thd_seq_lens_by_step
            ):
                combined.extend(aux.thd_seq_lens_by_step[step])
        merged[step] = combined

    return merged or None
