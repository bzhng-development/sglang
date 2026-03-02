"""Loads auxiliary tensors by (step, rank, layer_id) from a dump directory.

Caches results by (step, rank, layer_id, frozenset(aux_names)) since
multiple tensor bundles in the same layer may share the same aux tensors.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import polars as pl
import torch

from sglang.srt.debug_utils.dump_loader import ValueWithMeta, filter_rows


def _cast_for_filter(value: int, dtype: Optional[pl.DataType]) -> Union[int, str]:
    """Cast a Python int to match the polars column dtype (may be String)."""
    if dtype == pl.String:
        return str(value)
    return value


class RawAuxLoader:
    """Load auxiliary tensors on demand with LRU caching."""

    def __init__(self, df: pl.DataFrame, dump_dir: Path) -> None:
        self._df = df
        self._dump_dir = dump_dir

    @lru_cache(maxsize=32)
    def available_names(self, *, step: int, layer_id: int) -> frozenset[str]:
        """Return all unique aux tensor names for the given (step, layer_id)."""
        step_val = _cast_for_filter(step, self._df.schema.get("step"))
        layer_val = _cast_for_filter(layer_id, self._df.schema.get("layer_id"))
        filtered = self._df.filter(
            (pl.col("step") == step_val) & (pl.col("layer_id") == layer_val)
        )
        return frozenset(filtered["name"].unique().to_list())

    @lru_cache(maxsize=32)
    def load(
        self,
        *,
        step: int,
        rank: int,
        layer_id: int,
        aux_names: frozenset[str],
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}

        for aux_name in aux_names:
            rows = filter_rows(
                self._df,
                conditions={
                    "name": aux_name,
                    "step": step,
                    "rank": rank,
                    "layer_id": layer_id,
                },
            )
            if not rows:
                continue
            if len(rows) != 1:
                raise ValueError(
                    f"Expected exactly 1 row for aux tensor {aux_name!r} "
                    f"(step={step}, rank={rank}, layer_id={layer_id}), "
                    f"got {len(rows)}"
                )

            item: ValueWithMeta = ValueWithMeta.load(
                self._dump_dir / rows[0]["filename"]
            )
            if isinstance(item.value, torch.Tensor):
                result[aux_name] = item.value

        return result
