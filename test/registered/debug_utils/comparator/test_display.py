import sys
from pathlib import Path

import polars as pl
import pytest
import torch

from sglang.srt.debug_utils.comparator.display import (
    _extract_parallel_info,
    print_input_ids_and_positions,
    print_rank_info,
    render_polars_by_rich,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


def _save_dump_file(
    directory: Path,
    *,
    name: str,
    step: int,
    rank: int,
    dump_index: int,
    value: torch.Tensor,
    meta: dict,
) -> str:
    filename = f"name={name}___step={step}___rank={rank}___dump_index={dump_index}.pt"
    torch.save({"value": value, "meta": meta}, directory / filename)
    return filename


def _make_df(rows: list[dict]) -> pl.DataFrame:
    df = pl.DataFrame(rows)
    df = df.with_columns(
        pl.col("step").cast(int),
        pl.col("rank").cast(int),
        pl.col("dump_index").cast(int),
    )
    return df


class TestRenderPolarsByRich:
    def test_renders_without_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        df = pl.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
        render_polars_by_rich(df, title="test table")

        captured: str = capsys.readouterr().out
        assert "test table" in captured
        assert "col_a" in captured
        assert "col_b" in captured

    def test_renders_empty_dataframe(self, capsys: pytest.CaptureFixture[str]) -> None:
        df = pl.DataFrame({"a": [], "b": []})
        render_polars_by_rich(df, title="empty")

        captured: str = capsys.readouterr().out
        assert "empty" in captured


class TestPrintRankInfo:
    def test_prints_rank_info(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        sglang_info = {
            "tp_rank": 0,
            "tp_size": 2,
            "pp_rank": 0,
            "pp_size": 1,
        }
        filename: str = _save_dump_file(
            tmp_path,
            name="model_input_ids",
            step=0,
            rank=0,
            dump_index=0,
            value=torch.tensor([1, 2, 3]),
            meta={"sglang_parallel_info": sglang_info},
        )
        df = _make_df(
            [{"filename": filename, "name": "model_input_ids", "step": 0, "rank": 0, "dump_index": 0}]
        )

        print_rank_info(df, dump_dir=tmp_path, label="baseline")

        captured: str = capsys.readouterr().out
        assert "baseline ranks" in captured
        assert "0/2" in captured  # tp=0/2

    def test_no_output_when_no_input_ids(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        df = _make_df(
            [{"filename": "f.pt", "name": "some_other", "step": 0, "rank": 0, "dump_index": 0}]
        )
        print_rank_info(df, dump_dir=tmp_path, label="test")
        assert capsys.readouterr().out == ""

    def test_deduplicates_ranks(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        meta = {"sglang_parallel_info": {"tp_rank": 0, "tp_size": 1}}
        f1: str = _save_dump_file(
            tmp_path, name="model_input_ids", step=0, rank=0, dump_index=0,
            value=torch.tensor([1]), meta=meta,
        )
        f2: str = _save_dump_file(
            tmp_path, name="model_input_ids", step=1, rank=0, dump_index=1,
            value=torch.tensor([2]), meta=meta,
        )
        df = _make_df([
            {"filename": f1, "name": "model_input_ids", "step": 0, "rank": 0, "dump_index": 0},
            {"filename": f2, "name": "model_input_ids", "step": 1, "rank": 0, "dump_index": 1},
        ])

        print_rank_info(df, dump_dir=tmp_path, label="test")

        captured: str = capsys.readouterr().out
        # rank 0 should appear only once
        assert captured.count("0/1") == 1


class TestPrintInputIdsAndPositions:
    def test_prints_ids_and_positions(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f_ids: str = _save_dump_file(
            tmp_path, name="model_input_ids", step=0, rank=0, dump_index=0,
            value=torch.tensor([10, 20, 30]), meta={},
        )
        f_pos: str = _save_dump_file(
            tmp_path, name="model_positions", step=0, rank=0, dump_index=1,
            value=torch.tensor([0, 1, 2]), meta={},
        )
        df = _make_df([
            {"filename": f_ids, "name": "model_input_ids", "step": 0, "rank": 0, "dump_index": 0},
            {"filename": f_pos, "name": "model_positions", "step": 0, "rank": 0, "dump_index": 1},
        ])

        print_input_ids_and_positions(df, dump_dir=tmp_path, label="target")

        captured: str = capsys.readouterr().out
        assert "target input_ids & positions" in captured
        assert "10" in captured
        assert "num_tokens" in captured

    def test_no_output_when_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        df = _make_df(
            [{"filename": "f.pt", "name": "weight", "step": 0, "rank": 0, "dump_index": 0}]
        )
        print_input_ids_and_positions(df, dump_dir=tmp_path, label="test")
        assert capsys.readouterr().out == ""

    def test_with_mock_tokenizer(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f_ids: str = _save_dump_file(
            tmp_path, name="model_input_ids", step=0, rank=0, dump_index=0,
            value=torch.tensor([1, 2]), meta={},
        )
        df = _make_df(
            [{"filename": f_ids, "name": "model_input_ids", "step": 0, "rank": 0, "dump_index": 0}]
        )

        class _MockTokenizer:
            def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
                return f"decoded:{ids}"

        print_input_ids_and_positions(
            df, dump_dir=tmp_path, label="test", tokenizer=_MockTokenizer()
        )

        captured: str = capsys.readouterr().out
        assert "decoded_text" in captured
        assert "decoded:" in captured


class TestExtractParallelInfo:
    def test_extracts_rank_size_pairs(self) -> None:
        info: dict = {
            "tp_rank": 1,
            "tp_size": 4,
            "pp_rank": 0,
            "pp_size": 2,
        }
        row_data: dict = {}
        _extract_parallel_info(row_data=row_data, info=info)

        assert row_data["tp"] == "1/4"
        assert row_data["pp"] == "0/2"

    def test_skips_error_info(self) -> None:
        row_data: dict = {}
        _extract_parallel_info(row_data=row_data, info={"error": True, "tp_rank": 0, "tp_size": 1})
        assert row_data == {}

    def test_skips_empty_info(self) -> None:
        row_data: dict = {}
        _extract_parallel_info(row_data=row_data, info={})
        assert row_data == {}

    def test_ignores_rank_without_size(self) -> None:
        row_data: dict = {}
        _extract_parallel_info(row_data=row_data, info={"tp_rank": 0})
        assert "tp" not in row_data


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
