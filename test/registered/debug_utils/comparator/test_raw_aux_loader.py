import sys
from pathlib import Path

import polars as pl
import pytest
import torch

from sglang.srt.debug_utils.comparator.raw_aux_loader import RawAuxLoader
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


def _save_tensor(
    dump_dir: Path,
    *,
    name: str,
    step: int,
    rank: int,
    layer_id: int,
    value: torch.Tensor,
    dump_index: int = 0,
) -> str:
    filename: str = (
        f"name={name}___step={step}___rank={rank}"
        f"___layer_id={layer_id}___dump_index={dump_index}.pt"
    )
    torch.save({"value": value, "meta": {}}, dump_dir / filename)
    return filename


def _build_df(filenames: list[str]) -> pl.DataFrame:
    """Build a DataFrame from filenames, parsing metadata from each filename."""
    rows: list[dict] = []
    for filename in filenames:
        stem: str = Path(filename).stem
        meta: dict = {}
        for kv in stem.split("___"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                meta[k] = v
        rows.append(
            {
                "name": meta["name"],
                "step": int(meta["step"]),
                "rank": int(meta["rank"]),
                "layer_id": int(meta["layer_id"]),
                "filename": filename,
            }
        )
    return pl.DataFrame(rows)


class TestRawAuxLoaderLoad:
    def test_single_tensor(self, tmp_path: Path) -> None:
        tensor: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
        filename: str = _save_tensor(
            tmp_path, name="aux_a", step=0, rank=0, layer_id=0, value=tensor
        )
        df: pl.DataFrame = _build_df([filename])
        loader = RawAuxLoader(df=df, dump_dir=tmp_path)

        result: dict[str, torch.Tensor] = loader.load(
            step=0, rank=0, layer_id=0, aux_names=frozenset({"aux_a"})
        )

        assert "aux_a" in result
        assert torch.equal(result["aux_a"], tensor)

    def test_multiple_aux_names(self, tmp_path: Path) -> None:
        tensor_a: torch.Tensor = torch.tensor([1.0])
        tensor_b: torch.Tensor = torch.tensor([2.0])
        filenames: list[str] = [
            _save_tensor(
                tmp_path, name="aux_a", step=0, rank=0, layer_id=0, value=tensor_a
            ),
            _save_tensor(
                tmp_path, name="aux_b", step=0, rank=0, layer_id=0, value=tensor_b
            ),
        ]
        df: pl.DataFrame = _build_df(filenames)
        loader = RawAuxLoader(df=df, dump_dir=tmp_path)

        result: dict[str, torch.Tensor] = loader.load(
            step=0, rank=0, layer_id=0, aux_names=frozenset({"aux_a", "aux_b"})
        )

        assert len(result) == 2
        assert torch.equal(result["aux_a"], tensor_a)
        assert torch.equal(result["aux_b"], tensor_b)

    def test_missing_name_returns_empty(self, tmp_path: Path) -> None:
        tensor: torch.Tensor = torch.tensor([1.0])
        filename: str = _save_tensor(
            tmp_path, name="aux_a", step=0, rank=0, layer_id=0, value=tensor
        )
        df: pl.DataFrame = _build_df([filename])
        loader = RawAuxLoader(df=df, dump_dir=tmp_path)

        result: dict[str, torch.Tensor] = loader.load(
            step=0, rank=0, layer_id=0, aux_names=frozenset({"nonexistent"})
        )

        assert result == {}

    def test_partial_match_returns_found_only(self, tmp_path: Path) -> None:
        tensor: torch.Tensor = torch.tensor([1.0])
        filename: str = _save_tensor(
            tmp_path, name="aux_a", step=0, rank=0, layer_id=0, value=tensor
        )
        df: pl.DataFrame = _build_df([filename])
        loader = RawAuxLoader(df=df, dump_dir=tmp_path)

        result: dict[str, torch.Tensor] = loader.load(
            step=0, rank=0, layer_id=0, aux_names=frozenset({"aux_a", "missing"})
        )

        assert len(result) == 1
        assert "aux_a" in result

    def test_different_step_rank_layer(self, tmp_path: Path) -> None:
        tensor_s0: torch.Tensor = torch.tensor([10.0])
        tensor_s1: torch.Tensor = torch.tensor([20.0])
        filenames: list[str] = [
            _save_tensor(
                tmp_path, name="aux", step=0, rank=0, layer_id=0, value=tensor_s0
            ),
            _save_tensor(
                tmp_path, name="aux", step=1, rank=0, layer_id=0, value=tensor_s1
            ),
        ]
        df: pl.DataFrame = _build_df(filenames)
        loader = RawAuxLoader(df=df, dump_dir=tmp_path)

        result_s0: dict[str, torch.Tensor] = loader.load(
            step=0, rank=0, layer_id=0, aux_names=frozenset({"aux"})
        )
        result_s1: dict[str, torch.Tensor] = loader.load(
            step=1, rank=0, layer_id=0, aux_names=frozenset({"aux"})
        )

        assert torch.equal(result_s0["aux"], tensor_s0)
        assert torch.equal(result_s1["aux"], tensor_s1)

    def test_caching_returns_same_object(self, tmp_path: Path) -> None:
        tensor: torch.Tensor = torch.tensor([1.0])
        filename: str = _save_tensor(
            tmp_path, name="aux", step=0, rank=0, layer_id=0, value=tensor
        )
        df: pl.DataFrame = _build_df([filename])
        loader = RawAuxLoader(df=df, dump_dir=tmp_path)

        result1: dict[str, torch.Tensor] = loader.load(
            step=0, rank=0, layer_id=0, aux_names=frozenset({"aux"})
        )
        result2: dict[str, torch.Tensor] = loader.load(
            step=0, rank=0, layer_id=0, aux_names=frozenset({"aux"})
        )

        assert result1 is result2


class TestRawAuxLoaderDuplicateRows:
    def test_duplicate_rows_raises(self, tmp_path: Path) -> None:
        """When multiple rows match (step, rank, layer_id, name), load raises."""
        tensor: torch.Tensor = torch.tensor([1.0])
        filename_a: str = _save_tensor(
            tmp_path,
            name="aux",
            step=0,
            rank=0,
            layer_id=0,
            value=tensor,
            dump_index=0,
        )
        filename_b: str = _save_tensor(
            tmp_path,
            name="aux",
            step=0,
            rank=0,
            layer_id=0,
            value=tensor,
            dump_index=1,
        )
        df: pl.DataFrame = _build_df([filename_a, filename_b])
        loader = RawAuxLoader(df=df, dump_dir=tmp_path)

        with pytest.raises(ValueError, match="Expected exactly 1 row"):
            loader.load(step=0, rank=0, layer_id=0, aux_names=frozenset({"aux"}))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
