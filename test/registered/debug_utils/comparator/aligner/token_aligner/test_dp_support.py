"""Tests for DP (Data Parallel) support in the comparator."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import polars as pl
import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.token_aligner.aux_loader import (
    _build_rank_to_dp_rank,
    _extract_dp_rank_from_meta,
    load_and_normalize_aux,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.aux_plugins import (
    _MegatronPlugin,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.seq_info_builder import (
    build_seqs_info,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    PositionalSeqId,
    SGLangSeqId,
    TokenAlignerGlobalAux,
    TokenAlignerSeqsInfo,
    TokenAlignerStepAux,
)
from sglang.srt.debug_utils.comparator.bundle_comparator import (
    _any_side_has_dp,
)
from sglang.srt.debug_utils.comparator.bundle_comparator import (
    _extract_dp_size_from_meta,
    _group_by_dp_rank,
)
from sglang.srt.debug_utils.comparator.dims import TokenLayout
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.dump_loader import ValueWithMeta
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="default", nightly=True)


# ── helpers ───────────────────────────────────────────────────────


def _save_pt(
    dump_path: Path,
    *,
    name: str,
    step: int,
    rank: int,
    value: object,
    meta: Optional[dict[str, Any]] = None,
) -> str:
    filename: str = f"name={name}___step={step}___rank={rank}.pt"
    payload: dict[str, Any] = {"value": value, "meta": meta or {}}
    torch.save(payload, dump_path / filename)
    return filename


def _make_df_from_filenames(filenames: list[str]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for fn in filenames:
        parts: dict[str, str] = {}
        stem: str = fn.removesuffix(".pt")
        for kv in stem.split("___"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                parts[k] = v
        rows.append(
            {
                "filename": fn,
                "name": parts["name"],
                "step": int(parts["step"]),
                "rank": int(parts["rank"]),
            }
        )
    return pl.DataFrame(rows)


def _megatron_meta(
    *, tp_rank: int = 0, tp_size: int = 1, dp_rank: int = 0, dp_size: int = 1
) -> dict[str, Any]:
    return {
        "megatron_parallel_info": {
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "dp_rank": dp_rank,
            "dp_size": dp_size,
        }
    }


def _sglang_meta(
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    attn_dp_rank: int = 0,
    attn_dp_size: int = 1,
) -> dict[str, Any]:
    return {
        "sglang_parallel_info": {
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "attn_dp_rank": attn_dp_rank,
            "attn_dp_size": attn_dp_size,
        }
    }


# ── PositionalSeqId ──────────────────────────────────────────────


class TestPositionalSeqIdDpRank:
    def test_default_dp_rank_is_zero(self) -> None:
        sid = PositionalSeqId(step=0, seq_index=0)
        assert sid.dp_rank == 0

    def test_different_dp_ranks_are_distinct(self) -> None:
        sid0 = PositionalSeqId(step=0, seq_index=0, dp_rank=0)
        sid1 = PositionalSeqId(step=0, seq_index=0, dp_rank=1)
        assert sid0 != sid1

    def test_same_dp_rank_are_equal(self) -> None:
        sid0 = PositionalSeqId(step=0, seq_index=0, dp_rank=0)
        sid1 = PositionalSeqId(step=0, seq_index=0, dp_rank=0)
        assert sid0 == sid1

    def test_backward_compat_hash(self) -> None:
        """PositionalSeqId with dp_rank=0 can be used in dicts."""
        sid = PositionalSeqId(step=0, seq_index=0)
        d: dict[PositionalSeqId, int] = {sid: 42}
        assert d[PositionalSeqId(step=0, seq_index=0)] == 42


# ── dp_rank extraction ────────────────────────────────────────────


class TestExtractDpRank:
    def test_megatron_dp_rank(self) -> None:
        meta = _megatron_meta(dp_rank=3, dp_size=4)
        assert _extract_dp_rank_from_meta(meta, plugin_name="megatron") == 3

    def test_sglang_attn_dp_rank(self) -> None:
        meta = _sglang_meta(attn_dp_rank=2, attn_dp_size=4)
        assert _extract_dp_rank_from_meta(meta, plugin_name="sglang") == 2

    def test_missing_dp_rank_returns_zero(self) -> None:
        meta: dict[str, Any] = {"megatron_parallel_info": {"tp_rank": 0, "tp_size": 2}}
        assert _extract_dp_rank_from_meta(meta, plugin_name="megatron") == 0

    def test_no_parallel_info_returns_zero(self) -> None:
        meta: dict[str, Any] = {}
        assert _extract_dp_rank_from_meta(meta, plugin_name="megatron") == 0


class TestExtractDpSize:
    def test_megatron_dp_size(self) -> None:
        meta = _megatron_meta(dp_size=4)
        assert _extract_dp_size_from_meta(meta) == 4

    def test_sglang_attn_dp_size(self) -> None:
        meta = _sglang_meta(attn_dp_size=2)
        assert _extract_dp_size_from_meta(meta) == 2

    def test_no_dp_returns_one(self) -> None:
        meta: dict[str, Any] = {}
        assert _extract_dp_size_from_meta(meta) == 1


# ── bundle_comparator DP detection ───────────────────────────────


class TestAnyHasDp:
    def test_no_dp(self) -> None:
        items_x = [ValueWithMeta(value=torch.zeros(3), meta=_megatron_meta(dp_size=1))]
        items_y = [ValueWithMeta(value=torch.zeros(3), meta=_megatron_meta(dp_size=1))]
        pair: Pair[list[ValueWithMeta]] = Pair(x=items_x, y=items_y)
        assert not _any_side_has_dp(pair)

    def test_baseline_has_dp(self) -> None:
        items_x = [ValueWithMeta(value=torch.zeros(3), meta=_megatron_meta(dp_size=2))]
        items_y = [ValueWithMeta(value=torch.zeros(3), meta=_megatron_meta(dp_size=1))]
        pair: Pair[list[ValueWithMeta]] = Pair(x=items_x, y=items_y)
        assert _any_side_has_dp(pair)

    def test_target_has_dp(self) -> None:
        items_x = [ValueWithMeta(value=torch.zeros(3), meta=_megatron_meta(dp_size=1))]
        items_y = [ValueWithMeta(value=torch.zeros(3), meta=_megatron_meta(dp_size=2))]
        pair: Pair[list[ValueWithMeta]] = Pair(x=items_x, y=items_y)
        assert _any_side_has_dp(pair)


class TestGroupByDpRank:
    def test_single_dp_rank(self) -> None:
        items = [
            ValueWithMeta(value=torch.zeros(3), meta=_megatron_meta(dp_rank=0)),
            ValueWithMeta(value=torch.ones(3), meta=_megatron_meta(dp_rank=0)),
        ]
        groups = _group_by_dp_rank(items)
        assert list(groups.keys()) == [0]
        assert len(groups[0]) == 2

    def test_multiple_dp_ranks(self) -> None:
        items = [
            ValueWithMeta(value=torch.zeros(3), meta=_megatron_meta(dp_rank=0)),
            ValueWithMeta(value=torch.ones(3), meta=_megatron_meta(dp_rank=1)),
            ValueWithMeta(value=torch.full((3,), 2.0), meta=_megatron_meta(dp_rank=0)),
            ValueWithMeta(value=torch.full((3,), 3.0), meta=_megatron_meta(dp_rank=1)),
        ]
        groups = _group_by_dp_rank(items)
        assert sorted(groups.keys()) == [0, 1]
        assert len(groups[0]) == 2
        assert len(groups[1]) == 2


# ── aux_loader DP grouping ───────────────────────────────────────


class TestBuildRankToDpRank:
    def test_megatron_dp2(self, tmp_path: Path) -> None:
        """4 ranks, TP=2, DP=2 → rank 0,1 are dp=0, rank 2,3 are dp=1."""
        filenames: list[str] = []
        for rank, dp_rank in [(0, 0), (1, 0), (2, 1), (3, 1)]:
            fn = _save_pt(
                tmp_path,
                name="input_ids",
                step=0,
                rank=rank,
                value=torch.tensor([rank * 10]),
                meta=_megatron_meta(
                    tp_rank=rank % 2, tp_size=2, dp_rank=dp_rank, dp_size=2
                ),
            )
            filenames.append(fn)

        df = _make_df_from_filenames(filenames)
        mapping = _build_rank_to_dp_rank(
            df=df, dump_path=tmp_path, plugin=_MegatronPlugin()
        )
        assert mapping == {0: 0, 1: 0, 2: 1, 3: 1}


class TestLoadAndNormalizeAuxDp:
    """Test that load_and_normalize_aux correctly handles DP > 1."""

    def test_megatron_dp2_merges_step_auxs(self, tmp_path: Path) -> None:
        """DP=2, no TP: 2 ranks with different input_ids, merged into one step_aux."""
        filenames: list[str] = []
        for rank, dp_rank in [(0, 0), (1, 1)]:
            for aux_name, value in [
                ("input_ids", torch.tensor([10 + rank, 20 + rank])),
                ("cu_seqlens_q", torch.tensor([0, 2])),
            ]:
                fn = _save_pt(
                    tmp_path,
                    name=aux_name,
                    step=0,
                    rank=rank,
                    value=value,
                    meta=_megatron_meta(dp_rank=dp_rank, dp_size=2),
                )
                filenames.append(fn)

        df = _make_df_from_filenames(filenames)
        result = load_and_normalize_aux(dump_path=tmp_path, df=df)

        assert result is not None
        assert 0 in result.step_auxs
        step_aux = result.step_auxs[0]

        assert step_aux.input_ids == [10, 20, 11, 21]
        assert len(step_aux.seq_lens) == 2
        assert len(step_aux.seq_ids) == 2

        # seq_ids should be different due to different dp_rank
        assert step_aux.seq_ids[0] != step_aux.seq_ids[1]

    def test_sglang_dp2_with_rids(self, tmp_path: Path) -> None:
        """DP=2 SGLang with rids: seq_ids use SGLangSeqId (not PositionalSeqId), so dp_rank doesn't matter."""
        filenames: list[str] = []
        for rank, dp_rank in [(0, 0), (1, 1)]:
            for aux_name, value in [
                ("input_ids", torch.tensor([10 + rank, 20 + rank, 30 + rank])),
                ("positions", torch.tensor([0, 1, 0])),
                ("seq_lens", torch.tensor([2, 1])),
            ]:
                fn = _save_pt(
                    tmp_path,
                    name=aux_name,
                    step=0,
                    rank=rank,
                    value=value,
                    meta=_sglang_meta(attn_dp_rank=dp_rank, attn_dp_size=2),
                )
                filenames.append(fn)

            fn = _save_pt(
                tmp_path,
                name="rids",
                step=0,
                rank=rank,
                value=[f"req_A_{dp_rank}", f"req_B_{dp_rank}"],
                meta=_sglang_meta(attn_dp_rank=dp_rank, attn_dp_size=2),
            )
            filenames.append(fn)

        df = _make_df_from_filenames(filenames)
        result = load_and_normalize_aux(dump_path=tmp_path, df=df)

        assert result is not None
        step_aux = result.step_auxs[0]

        assert step_aux.input_ids == [10, 20, 30, 11, 21, 31]
        assert len(step_aux.seq_ids) == 4
        # First dp_rank's seqs, then second dp_rank's seqs
        assert isinstance(step_aux.seq_ids[0], SGLangSeqId)

    def test_dp1_unchanged_behavior(self, tmp_path: Path) -> None:
        """DP=1 should produce the same result as the non-DP path."""
        fn = _save_pt(
            tmp_path,
            name="input_ids",
            step=0,
            rank=0,
            value=torch.tensor([10, 20, 30]),
            meta=_megatron_meta(dp_rank=0, dp_size=1),
        )
        fn2 = _save_pt(
            tmp_path,
            name="cu_seqlens_q",
            step=0,
            rank=0,
            value=torch.tensor([0, 3]),
            meta=_megatron_meta(dp_rank=0, dp_size=1),
        )
        df = _make_df_from_filenames([fn, fn2])

        result = load_and_normalize_aux(dump_path=tmp_path, df=df)

        assert result is not None
        step_aux = result.step_auxs[0]
        assert step_aux.input_ids == [10, 20, 30]
        assert step_aux.seq_lens == [3]


# ── seq_info_builder with DP ─────────────────────────────────────


class TestSeqInfoBuilderDp:
    def test_dp2_positional_seq_ids_are_distinct(self) -> None:
        """Two dp_ranks with same step and seq_index produce different SeqIds."""
        step_aux = TokenAlignerStepAux(
            input_ids=[10, 20, 30, 40],
            positions=[0, 1, 0, 1],
            seq_lens=[2, 2],
            seq_ids=[
                PositionalSeqId(step=0, seq_index=0, dp_rank=0),
                PositionalSeqId(step=0, seq_index=0, dp_rank=1),
            ],
        )

        global_aux = TokenAlignerGlobalAux(
            step_auxs={0: step_aux},
            framework="megatron",
            layout=TokenLayout.T,
        )

        seqs_info: TokenAlignerSeqsInfo = build_seqs_info(global_aux)

        assert len(seqs_info.sequences) == 2
        sid0 = PositionalSeqId(step=0, seq_index=0, dp_rank=0)
        sid1 = PositionalSeqId(step=0, seq_index=0, dp_rank=1)
        assert sid0 in seqs_info.sequences
        assert sid1 in seqs_info.sequences
        assert seqs_info.sequences[sid0].input_ids == [10, 20]
        assert seqs_info.sequences[sid1].input_ids == [30, 40]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
