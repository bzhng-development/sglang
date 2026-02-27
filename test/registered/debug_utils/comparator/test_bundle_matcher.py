import sys
from typing import Any

import polars as pl
import pytest

from sglang.srt.debug_utils.comparator.bundle_matcher import (
    TensorBundleInfo,
    TensorFileInfo,
    _rows_to_tensor_infos,
    match_bundles,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)


def _make_row(
    *,
    name: str,
    step: int = 0,
    rank: int = 0,
    layer_id: int | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    if filename is None:
        layer_part: str = f"___layer_id={layer_id}" if layer_id is not None else ""
        filename = f"name={name}___step={step}___rank={rank}{layer_part}.pt"
    row: dict[str, Any] = {
        "name": name,
        "step": step,
        "rank": rank,
        "filename": filename,
    }
    if layer_id is not None:
        row["layer_id"] = layer_id
    return row


def _make_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(rows)

    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    normalized: list[dict[str, Any]] = [
        {k: row.get(k, None) for k in all_keys} for row in rows
    ]
    return pl.DataFrame(normalized)


class TestMatchBundles:
    def test_single_tensor_single_step(self) -> None:
        target_df: pl.DataFrame = _make_df([_make_row(name="t_a")])
        baseline_df: pl.DataFrame = _make_df([_make_row(name="t_a")])

        results: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=Pair(x=baseline_df, y=target_df),
            skip_keys={"filename"},
        )

        assert len(results) == 1
        assert len(results[0].x) == 1
        assert len(results[0].y) == 1
        assert results[0].y[0].name == "t_a"

    def test_multiple_names_separate_bundles(self) -> None:
        target_df: pl.DataFrame = _make_df(
            [
                _make_row(name="t_a"),
                _make_row(name="t_b"),
            ]
        )
        baseline_df: pl.DataFrame = _make_df(
            [
                _make_row(name="t_a"),
                _make_row(name="t_b"),
            ]
        )

        results: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=Pair(x=baseline_df, y=target_df),
            skip_keys={"filename"},
        )

        assert len(results) == 2
        result_names: list[str] = [r.y[0].name for r in results]
        assert "t_a" in result_names
        assert "t_b" in result_names

    def test_skip_rank_groups_across_ranks(self) -> None:
        target_df: pl.DataFrame = _make_df(
            [
                _make_row(name="t_a", rank=0),
                _make_row(name="t_a", rank=1),
            ]
        )
        baseline_df: pl.DataFrame = _make_df(
            [
                _make_row(name="t_a", rank=0),
                _make_row(name="t_a", rank=1),
            ]
        )

        results: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=Pair(x=baseline_df, y=target_df),
            skip_keys={"filename", "rank"},
        )

        assert len(results) == 1
        assert len(results[0].y) == 2

    def test_baseline_missing_tensor(self) -> None:
        target_df: pl.DataFrame = _make_df(
            [
                _make_row(name="t_a"),
                _make_row(name="t_extra"),
            ]
        )
        baseline_df: pl.DataFrame = _make_df(
            [
                _make_row(name="t_a"),
            ]
        )

        results: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=Pair(x=baseline_df, y=target_df),
            skip_keys={"filename"},
        )

        assert len(results) == 2
        extra_pair: Pair[TensorBundleInfo] = [
            r for r in results if r.y[0].name == "t_extra"
        ][0]
        assert extra_pair.x == []

    def test_empty_target_returns_empty(self) -> None:
        target_df: pl.DataFrame = _make_df([])
        baseline_df: pl.DataFrame = _make_df([_make_row(name="t_a")])

        results: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=Pair(x=baseline_df, y=target_df),
            skip_keys={"filename"},
        )

        assert results == []

    def test_skip_step_groups_across_steps(self) -> None:
        target_df: pl.DataFrame = _make_df(
            [
                _make_row(name="t_a", step=0),
                _make_row(name="t_a", step=1),
            ]
        )
        baseline_df: pl.DataFrame = _make_df(
            [
                _make_row(name="t_a", step=0),
                _make_row(name="t_a", step=1),
            ]
        )

        results: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=Pair(x=baseline_df, y=target_df),
            skip_keys={"filename", "step"},
        )

        assert len(results) == 1
        assert len(results[0].y) == 2


class TestMatchBundlesWithLayerId:
    """Verify that nullable layer_id column participates correctly in grouping.

    This is the only net-new behavior vs existing tests: rows with and without
    layer_id coexist in the same DataFrame, and layer_id values (including null)
    must be part of the grouping key.
    """

    LOGICAL_SKIP_KEYS: set[str] = {"filename", "rank", "dump_index", "recompute_status"}

    def test_layer_id_groups_separately_from_non_layer(self) -> None:
        """Mix of layer tensors (with layer_id) and non-layer tensors (without).
        Should form separate bundles by (name, layer_id) with rank skipped."""
        target_df: pl.DataFrame = _make_df(
            [
                _make_row(name="hidden", rank=0, layer_id=0),
                _make_row(name="hidden", rank=0, layer_id=1),
                _make_row(name="embed_tokens", rank=0),
            ]
        )
        baseline_df: pl.DataFrame = _make_df(
            [
                _make_row(name="hidden", rank=1, layer_id=0),
                _make_row(name="hidden", rank=1, layer_id=1),
                _make_row(name="embed_tokens", rank=1),
            ]
        )

        results: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=Pair(x=baseline_df, y=target_df),
            skip_keys=self.LOGICAL_SKIP_KEYS,
        )

        assert len(results) == 3
        for pair in results:
            assert len(pair.x) == 1
            assert len(pair.y) == 1


class TestRowsToTensorInfos:
    def test_filters_extra_columns(self) -> None:
        rows: list[dict[str, Any]] = [
            {"filename": "a.pt", "name": "t_a", "step": 0, "rank": 7}
        ]
        infos: list[TensorFileInfo] = _rows_to_tensor_infos(rows)

        assert len(infos) == 1
        assert infos[0] == TensorFileInfo(filename="a.pt", name="t_a", step=0)

    def test_empty_rows(self) -> None:
        infos: list[TensorFileInfo] = _rows_to_tensor_infos([])
        assert infos == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
