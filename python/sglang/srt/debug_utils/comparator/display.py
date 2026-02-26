from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any, Optional

import polars as pl

from sglang.srt.debug_utils.dump_loader import ValueWithMeta


def render_polars_as_text(df: pl.DataFrame, *, title: Optional[str] = None) -> str:
    from rich.console import Console
    from rich.table import Table

    table = Table(title=title)
    for col in df.columns:
        table.add_column(col)
    for row in df.iter_rows():
        table.add_row(*[str(v) for v in row])

    buf = StringIO()
    Console(file=buf, force_terminal=False, width=200).print(table)
    return buf.getvalue().rstrip("\n")


def collect_rank_info(
    df: pl.DataFrame, dump_dir: Path, label: str
) -> Optional[list[dict[str, Any]]]:
    unique_rows: pl.DataFrame = (
        df.filter(pl.col("name") == "model_input_ids")
        .sort("rank")
        .unique(subset=["rank"], keep="first")
    )
    if unique_rows.is_empty():
        return None

    table_rows: list[dict[str, Any]] = []
    for row in unique_rows.to_dicts():
        item: ValueWithMeta = ValueWithMeta.load(dump_dir / row["filename"])
        meta: dict[str, Any] = item.meta

        row_data: dict[str, Any] = {"rank": row["rank"]}
        _extract_parallel_info(
            row_data=row_data, info=meta.get("sglang_parallel_info", {})
        )
        _extract_parallel_info(
            row_data=row_data, info=meta.get("megatron_parallel_info", {})
        )
        table_rows.append(row_data)

    return table_rows if table_rows else None


def collect_input_ids_and_positions(
    df: pl.DataFrame,
    dump_dir: Path,
    label: str,
    *,
    tokenizer: Any = None,
) -> Optional[list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = df.filter(
        pl.col("name").is_in(["model_input_ids", "model_positions"])
    ).to_dicts()
    if not rows:
        return None

    data_by_step_rank: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key: tuple[int, int] = (row["step"], row["rank"])
        if key not in data_by_step_rank:
            data_by_step_rank[key] = {}
        item: ValueWithMeta = ValueWithMeta.load(dump_dir / row["filename"])
        if item.value is not None:
            data_by_step_rank[key][row["name"]] = item.value

    table_rows: list[dict[str, Any]] = []
    for (step, rank), data in sorted(data_by_step_rank.items()):
        ids = data.get("model_input_ids")
        pos = data.get("model_positions")

        ids_str: str = str(ids.flatten().tolist()) if ids is not None else "N/A"
        pos_str: str = str(pos.flatten().tolist()) if pos is not None else "N/A"

        row_data: dict[str, Any] = {
            "step": step,
            "rank": rank,
            "num_tokens": ids.numel() if ids is not None else None,
            "input_ids": ids_str,
            "positions": pos_str,
        }

        if tokenizer is not None and ids is not None:
            decoded: str = tokenizer.decode(
                ids.flatten().tolist(), skip_special_tokens=False
            )
            row_data["decoded_text"] = repr(decoded)

        table_rows.append(row_data)

    return table_rows if table_rows else None


def _extract_parallel_info(row_data: dict[str, Any], info: dict[str, Any]) -> None:
    if not info or info.get("error"):
        return

    for key in sorted(info.keys()):
        if key.endswith("_rank"):
            base: str = key[:-5]
            size_key: str = f"{base}_size"
            if size_key in info:
                row_data[base] = f"{info[key]}/{info[size_key]}"
