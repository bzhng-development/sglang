"""Tests for override_config — unit + integration."""

from __future__ import annotations

import sys
import textwrap
from argparse import Namespace
from pathlib import Path

import pytest
import torch

import sglang.srt.debug_utils.dumper as _dumper_module
from sglang.srt.debug_utils.comparator.entrypoint import run
from sglang.srt.debug_utils.comparator.output_types import (
    AnyRecord,
    ComparisonRecord,
    NonTensorRecord,
    SummaryRecord,
    parse_record_json,
)
from sglang.srt.debug_utils.comparator.override_config import (
    DimsOverrider,
    MetaOverrideRule,
    _load_yaml_rules,
    _merge_per_side_cli_rules,
    _parse_cli_override_arg,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.dumper import DumperConfig, _Dumper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)

_FIXED_EXP_NAME = "my_exp_name"


# ───────────────────── Unit: MetaOverrideRule ─────────────────────


class TestMetaOverrideRule:
    """Pydantic validation for MetaOverrideRule."""

    def test_shared_dims_only(self) -> None:
        """Shared 'dims' field is accepted alone."""
        rule = MetaOverrideRule(match="hidden", dims="b s h d")
        assert rule.effective_baseline_dims() == "b s h d"
        assert rule.effective_target_dims() == "b s h d"

    def test_per_side_dims(self) -> None:
        """baseline_dims / target_dims accepted without shared dims."""
        rule = MetaOverrideRule(
            match="logits", baseline_dims="b s v(tp)", target_dims="b s v(ep)"
        )
        assert rule.effective_baseline_dims() == "b s v(tp)"
        assert rule.effective_target_dims() == "b s v(ep)"

    def test_per_side_partial(self) -> None:
        """Only baseline_dims without target_dims is valid."""
        rule = MetaOverrideRule(match="logits", baseline_dims="b s v(tp)")
        assert rule.effective_baseline_dims() == "b s v(tp)"
        assert rule.effective_target_dims() is None

    def test_mutual_exclusion(self) -> None:
        """Cannot specify both 'dims' and 'baseline_dims'."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            MetaOverrideRule(match="x", dims="b s", baseline_dims="b s")

    def test_mutual_exclusion_target(self) -> None:
        """Cannot specify both 'dims' and 'target_dims'."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            MetaOverrideRule(match="x", dims="b s", target_dims="b s")

    def test_none_rejected(self) -> None:
        """Must specify at least one dims field."""
        with pytest.raises(ValueError, match="Must specify"):
            MetaOverrideRule(match="x")

    def test_extra_field_rejected(self) -> None:
        """Extra fields are rejected by _StrictBase."""
        with pytest.raises(Exception):
            MetaOverrideRule(match="x", dims="b s", bogus="y")


# ──────────────────── Unit: _parse_cli_override_arg ────────────────────


class TestParseCLIOverrideArg:
    """CLI arg parsing for 'name:dims_string' format."""

    def test_basic(self) -> None:
        """Standard 'name:dims' parsing."""
        name, dims_str = _parse_cli_override_arg("hidden_states:b s h d")
        assert name == "hidden_states"
        assert dims_str == "b s h d"

    def test_colon_in_dims(self) -> None:
        """Extra colons in dims are kept (maxsplit=1)."""
        name, dims_str = _parse_cli_override_arg("x:a:b")
        assert name == "x"
        assert dims_str == "a:b"

    def test_whitespace_trimmed(self) -> None:
        """Leading/trailing whitespace around name and dims is stripped."""
        name, dims_str = _parse_cli_override_arg("  foo  :  b s  ")
        assert name == "foo"
        assert dims_str == "b s"

    def test_missing_colon(self) -> None:
        """No colon raises ValueError."""
        with pytest.raises(ValueError, match="Invalid override format"):
            _parse_cli_override_arg("no_colon_here")

    def test_empty_name(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid override format"):
            _parse_cli_override_arg(":b s h")

    def test_empty_dims(self) -> None:
        """Empty dims raises ValueError."""
        with pytest.raises(ValueError, match="Invalid override format"):
            _parse_cli_override_arg("foo:")


# ──────────────────── Unit: _merge_per_side_cli_rules ────────────────────


class TestMergePerSideCliRules:
    """Merging --override-baseline-dims and --override-target-dims."""

    def test_same_pattern_merged(self) -> None:
        """Same match pattern from both flags produces one merged rule."""
        rules = _merge_per_side_cli_rules(
            override_baseline_dims=["hidden:b s h(tp)"],
            override_target_dims=["hidden:b s h(ep)"],
        )
        assert len(rules) == 1
        assert rules[0].baseline_dims == "b s h(tp)"
        assert rules[0].target_dims == "b s h(ep)"

    def test_different_patterns_stay_separate(self) -> None:
        """Different match patterns produce separate rules."""
        rules = _merge_per_side_cli_rules(
            override_baseline_dims=["hidden:b s h"],
            override_target_dims=["logits:b s v"],
        )
        assert len(rules) == 2

    def test_order_preserved(self) -> None:
        """First-seen pattern determines order."""
        rules = _merge_per_side_cli_rules(
            override_baseline_dims=["b_first:x", "second:y"],
            override_target_dims=["second:z", "t_only:w"],
        )
        names: list[str] = [r.match for r in rules]
        assert names == ["b_first", "second", "t_only"]

    def test_baseline_only(self) -> None:
        """Only baseline dims specified → target_dims is None."""
        rules = _merge_per_side_cli_rules(
            override_baseline_dims=["hidden:b s h"],
            override_target_dims=[],
        )
        assert len(rules) == 1
        assert rules[0].baseline_dims == "b s h"
        assert rules[0].target_dims is None


# ──────────────────── Unit: DimsOverrider ────────────────────


class TestDimsOverrider:
    """DimsOverrider logic: matching, priority, apply_to_metas."""

    def test_first_match_wins(self) -> None:
        """First matching rule takes effect; later rules ignored."""
        overrider = DimsOverrider(
            rules=[
                MetaOverrideRule(match="hidden", dims="FIRST"),
                MetaOverrideRule(match="hidden", dims="SECOND"),
            ]
        )
        result: Pair[list[dict]] = overrider.apply_to_metas(
            name="hidden_states",
            baseline_metas=[{"dims": "old"}],
            target_metas=[{"dims": "old"}],
        )
        assert result.x[0]["dims"] == "FIRST"
        assert result.y[0]["dims"] == "FIRST"

    def test_regex_contains_match(self) -> None:
        """match is a regex contains search, not exact match."""
        overrider = DimsOverrider(
            rules=[MetaOverrideRule(match=r"\.q_proj\.", dims="h d")]
        )
        result = overrider.apply_to_metas(
            name="layers.0.q_proj.weight",
            baseline_metas=[{"dims": "old"}],
            target_metas=[{"dims": "old"}],
        )
        assert result.x[0]["dims"] == "h d"

    def test_no_match_preserves_original(self) -> None:
        """No matching rule leaves metas untouched."""
        overrider = DimsOverrider(
            rules=[MetaOverrideRule(match="logits", dims="b s v")]
        )
        original_meta: dict = {"dims": "original"}
        result = overrider.apply_to_metas(
            name="hidden_states",
            baseline_metas=[original_meta],
            target_metas=[original_meta],
        )
        assert result.x[0]["dims"] == "original"
        assert result.y[0]["dims"] == "original"

    def test_per_side_override(self) -> None:
        """baseline_dims / target_dims override independently."""
        overrider = DimsOverrider(
            rules=[
                MetaOverrideRule(
                    match="logits",
                    baseline_dims="b s v(tp)",
                    target_dims="b s v(ep)",
                )
            ]
        )
        result = overrider.apply_to_metas(
            name="logits",
            baseline_metas=[{"dims": "old"}],
            target_metas=[{"dims": "old"}],
        )
        assert result.x[0]["dims"] == "b s v(tp)"
        assert result.y[0]["dims"] == "b s v(ep)"

    def test_partial_per_side_preserves_other(self) -> None:
        """Only baseline_dims specified → target meta unchanged."""
        overrider = DimsOverrider(
            rules=[MetaOverrideRule(match="logits", baseline_dims="b s v(tp)")]
        )
        result = overrider.apply_to_metas(
            name="logits",
            baseline_metas=[{"dims": "old_b"}],
            target_metas=[{"dims": "old_t"}],
        )
        assert result.x[0]["dims"] == "b s v(tp)"
        assert result.y[0]["dims"] == "old_t"

    def test_is_empty(self) -> None:
        """Empty overrider reports is_empty=True."""
        assert DimsOverrider(rules=[]).is_empty
        assert not DimsOverrider(rules=[MetaOverrideRule(match="x", dims="d")]).is_empty

    def test_multiple_metas(self) -> None:
        """All metas in the list are updated when a rule matches."""
        overrider = DimsOverrider(rules=[MetaOverrideRule(match="hidden", dims="NEW")])
        result = overrider.apply_to_metas(
            name="hidden",
            baseline_metas=[{"dims": "a"}, {"dims": "b"}],
            target_metas=[{"dims": "c"}],
        )
        assert result.x[0]["dims"] == "NEW"
        assert result.x[1]["dims"] == "NEW"
        assert result.y[0]["dims"] == "NEW"

    def test_meta_without_dims_key(self) -> None:
        """Override adds 'dims' even if original meta lacks it."""
        overrider = DimsOverrider(rules=[MetaOverrideRule(match="hidden", dims="NEW")])
        result = overrider.apply_to_metas(
            name="hidden",
            baseline_metas=[{"other": "val"}],
            target_metas=[{}],
        )
        assert result.x[0]["dims"] == "NEW"
        assert result.y[0]["dims"] == "NEW"


# ──────────────────── Unit: from_args_and_config ────────────────────


class TestFromArgsAndConfig:
    """DimsOverrider.from_args_and_config merges CLI + YAML rules."""

    def test_cli_before_yaml(self, tmp_path: Path) -> None:
        """CLI rules are ordered before YAML rules (CLI wins on conflict)."""
        yaml_path = tmp_path / "override.yaml"
        yaml_path.write_text(textwrap.dedent("""\
            dims:
              - match: "hidden"
                dims: "FROM_YAML"
        """))

        overrider = DimsOverrider.from_args_and_config(
            override_dims=["hidden:FROM_CLI"],
            override_baseline_dims=[],
            override_target_dims=[],
            override_config=yaml_path,
        )

        result = overrider.apply_to_metas(
            name="hidden",
            baseline_metas=[{"dims": "old"}],
            target_metas=[{"dims": "old"}],
        )
        assert result.x[0]["dims"] == "FROM_CLI"

    def test_no_config_no_cli(self) -> None:
        """Empty CLI + no YAML yields empty overrider."""
        overrider = DimsOverrider.from_args_and_config(
            override_dims=[],
            override_baseline_dims=[],
            override_target_dims=[],
            override_config=None,
        )
        assert overrider.is_empty

    def test_per_side_same_pattern_merged(self) -> None:
        """--override-baseline-dims and --override-target-dims with same pattern merge into one rule."""
        overrider = DimsOverrider.from_args_and_config(
            override_dims=[],
            override_baseline_dims=["hidden:b s h(tp)"],
            override_target_dims=["hidden:b s h(ep)"],
            override_config=None,
        )

        result = overrider.apply_to_metas(
            name="hidden",
            baseline_metas=[{"dims": "old"}],
            target_metas=[{"dims": "old"}],
        )
        assert result.x[0]["dims"] == "b s h(tp)"
        assert result.y[0]["dims"] == "b s h(ep)"


# ──────────────────── Unit: _load_yaml_rules ────────────────────


class TestLoadYamlRules:
    """YAML loading and validation."""

    def test_valid_yaml(self, tmp_path: Path) -> None:
        """Valid YAML with dims rules loads correctly."""
        yaml_path = tmp_path / "override.yaml"
        yaml_path.write_text(textwrap.dedent("""\
            dims:
              - match: "hidden"
                dims: "b s h d"
              - match: "logits"
                baseline_dims: "b s v(tp)"
                target_dims: "b s v(ep)"
        """))
        rules = _load_yaml_rules(yaml_path)
        assert len(rules) == 2
        assert rules[0].dims == "b s h d"
        assert rules[1].baseline_dims == "b s v(tp)"

    def test_empty_yaml(self, tmp_path: Path) -> None:
        """Empty YAML file returns no rules."""
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        rules = _load_yaml_rules(yaml_path)
        assert rules == []

    def test_unknown_top_key_rejected(self, tmp_path: Path) -> None:
        """Unknown top-level key is rejected by OverrideConfig."""
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("unknown_key: 42\n")
        with pytest.raises(Exception):
            _load_yaml_rules(yaml_path)

    def test_dims_only_top_level(self, tmp_path: Path) -> None:
        """Only 'dims' key with no entries returns empty list."""
        yaml_path = tmp_path / "minimal.yaml"
        yaml_path.write_text("dims: []\n")
        rules = _load_yaml_rules(yaml_path)
        assert rules == []


# ──────────────────── Integration: entrypoint + override ────────────────────


class TestEntrypointDimsOverride:
    """E2E: dump with wrong dims → --override-dims corrects at comparison time."""

    def test_override_dims_fixes_wrong_dims(self, tmp_path: Path, capsys) -> None:
        """Tensor dumped with wrong dims='h d' is fixed by --override-dims to 't h(tp)'."""
        torch.manual_seed(42)

        full_tensor: torch.Tensor = torch.randn(10, 8)
        tp_chunks: list[torch.Tensor] = list(full_tensor.chunk(2, dim=1))

        target_full: torch.Tensor = full_tensor + torch.randn(10, 8) * 0.001
        target_tp_chunks: list[torch.Tensor] = list(target_full.chunk(2, dim=1))

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        # Dump with WRONG dims "h d" instead of correct "t h(tp)"
        for tp_rank in range(2):
            _create_rank_dump(
                baseline_dir,
                rank=tp_rank,
                name="hidden",
                tensor=tp_chunks[tp_rank],
                dims="h d",
                parallel_info={"tp_rank": tp_rank, "tp_size": 2},
            )
            _create_rank_dump(
                target_dir,
                rank=tp_rank,
                name="hidden",
                tensor=target_tp_chunks[tp_rank],
                dims="h d",
                parallel_info={"tp_rank": tp_rank, "tp_size": 2},
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="logical",
            override_dims=["hidden:t h(tp)"],
        )
        records = _run_and_parse(args, capsys)

        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

    def test_override_baseline_dims_only(self, tmp_path: Path, capsys) -> None:
        """--override-baseline-dims overrides only the baseline side."""
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(10, 8)
        target: torch.Tensor = tensor + torch.randn(10, 8) * 0.001

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        _create_rank_dump(
            baseline_dir, rank=0, name="hidden", tensor=tensor, dims="x y"
        )
        _create_rank_dump(target_dir, rank=0, name="hidden", tensor=target, dims="t h")

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="raw",
            override_baseline_dims=["hidden:t h"],
        )
        records = _run_and_parse(args, capsys)

        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

    def test_override_target_dims_only(self, tmp_path: Path, capsys) -> None:
        """--override-target-dims overrides only the target side."""
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(10, 8)
        target: torch.Tensor = tensor + torch.randn(10, 8) * 0.001

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        _create_rank_dump(
            baseline_dir, rank=0, name="hidden", tensor=tensor, dims="t h"
        )
        _create_rank_dump(target_dir, rank=0, name="hidden", tensor=target, dims="x y")

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="raw",
            override_target_dims=["hidden:t h"],
        )
        records = _run_and_parse(args, capsys)

        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

    def test_override_config_yaml(self, tmp_path: Path, capsys) -> None:
        """--override-config YAML overrides dims."""
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(10, 8)
        target: torch.Tensor = tensor + torch.randn(10, 8) * 0.001

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        _create_rank_dump(
            baseline_dir, rank=0, name="hidden", tensor=tensor, dims="x y"
        )
        _create_rank_dump(target_dir, rank=0, name="hidden", tensor=target, dims="x y")

        yaml_path: Path = tmp_path / "override.yaml"
        yaml_path.write_text(textwrap.dedent("""\
            dims:
              - match: "hidden"
                dims: "t h"
        """))

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="raw",
            override_config=str(yaml_path),
        )
        records = _run_and_parse(args, capsys)

        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

    def test_no_match_uses_original_dims(self, tmp_path: Path, capsys) -> None:
        """When override regex doesn't match, original dims from dump are used."""
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(10, 8)
        target: torch.Tensor = tensor + torch.randn(10, 8) * 0.001

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        _create_rank_dump(
            baseline_dir, rank=0, name="hidden", tensor=tensor, dims="t h"
        )
        _create_rank_dump(target_dir, rank=0, name="hidden", tensor=target, dims="t h")

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="raw",
            override_dims=["no_match_pattern:b s d"],
        )
        records = _run_and_parse(args, capsys)

        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

    def test_selective_match_multi_tensor(self, tmp_path: Path, capsys) -> None:
        """Override matches only 'logits'; 'hidden' uses original dims."""
        torch.manual_seed(42)

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        hidden_b: torch.Tensor = torch.randn(10, 8)
        hidden_t: torch.Tensor = hidden_b + torch.randn(10, 8) * 0.001
        logits_b: torch.Tensor = torch.randn(10, 4)
        logits_t: torch.Tensor = logits_b + torch.randn(10, 4) * 0.001

        for name, b_tensor, t_tensor, dims in [
            ("hidden", hidden_b, hidden_t, "t h"),
            ("logits", logits_b, logits_t, "x y"),
        ]:
            _create_rank_dump(
                baseline_dir, rank=0, name=name, tensor=b_tensor, dims=dims
            )
            _create_rank_dump(target_dir, rank=0, name=name, tensor=t_tensor, dims=dims)

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="raw",
            override_dims=["logits:t v"],
        )
        records = _run_and_parse(args, capsys)

        comparisons = _get_comparisons(records)
        assert len(comparisons) == 2
        assert all(c.diff is not None and c.diff.passed for c in comparisons)

    def test_multiple_cli_override_dims(self, tmp_path: Path, capsys) -> None:
        """Multiple --override-dims for different tensors."""
        torch.manual_seed(42)

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        hidden_b: torch.Tensor = torch.randn(10, 8)
        hidden_t: torch.Tensor = hidden_b + torch.randn(10, 8) * 0.001
        logits_b: torch.Tensor = torch.randn(10, 4)
        logits_t: torch.Tensor = logits_b + torch.randn(10, 4) * 0.001

        for name, b_tensor, t_tensor in [
            ("hidden", hidden_b, hidden_t),
            ("logits", logits_b, logits_t),
        ]:
            _create_rank_dump(
                baseline_dir, rank=0, name=name, tensor=b_tensor, dims="x y"
            )
            _create_rank_dump(
                target_dir, rank=0, name=name, tensor=t_tensor, dims="x y"
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="raw",
            override_dims=["hidden:t h", "logits:t v"],
        )
        records = _run_and_parse(args, capsys)

        comparisons = _get_comparisons(records)
        assert len(comparisons) == 2
        assert all(c.diff is not None and c.diff.passed for c in comparisons)

    def test_per_side_dims_different_parallelism(self, tmp_path: Path, capsys) -> None:
        """baseline TP-sharded, target EP-sharded — per-side override fixes both."""
        torch.manual_seed(42)
        full_tensor: torch.Tensor = torch.randn(10, 8)
        target_full: torch.Tensor = full_tensor + torch.randn(10, 8) * 0.001

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        b_chunks: list[torch.Tensor] = list(full_tensor.chunk(2, dim=1))
        for tp_rank in range(2):
            _create_rank_dump(
                baseline_dir,
                rank=tp_rank,
                name="hidden",
                tensor=b_chunks[tp_rank],
                dims="x y",
                parallel_info={"tp_rank": tp_rank, "tp_size": 2},
            )

        t_chunks: list[torch.Tensor] = list(target_full.chunk(2, dim=1))
        for ep_rank in range(2):
            _create_rank_dump(
                target_dir,
                rank=ep_rank,
                name="hidden",
                tensor=t_chunks[ep_rank],
                dims="x y",
                parallel_info={"ep_rank": ep_rank, "ep_size": 2},
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="logical",
            override_baseline_dims=["hidden:t h(tp)"],
            override_target_dims=["hidden:t h(ep)"],
        )
        records = _run_and_parse(args, capsys)

        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

    def test_yaml_first_match_wins_e2e(self, tmp_path: Path, capsys) -> None:
        """YAML with two matching rules: first rule wins in real pipeline."""
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(10, 8)
        target: torch.Tensor = tensor + torch.randn(10, 8) * 0.001

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        _create_rank_dump(
            baseline_dir, rank=0, name="hidden", tensor=tensor, dims="x y"
        )
        _create_rank_dump(target_dir, rank=0, name="hidden", tensor=target, dims="x y")

        yaml_path: Path = tmp_path / "override.yaml"
        yaml_path.write_text(textwrap.dedent("""\
            dims:
              - match: "hidden"
                dims: "t h"
              - match: "hidden"
                dims: "a b"
        """))

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="raw",
            override_config=str(yaml_path),
        )
        records = _run_and_parse(args, capsys)

        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

    def test_cli_overrides_yaml_e2e(self, tmp_path: Path, capsys) -> None:
        """CLI --override-dims wins over YAML rule for the same tensor."""
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(10, 8)
        target: torch.Tensor = tensor + torch.randn(10, 8) * 0.001

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        _create_rank_dump(
            baseline_dir, rank=0, name="hidden", tensor=tensor, dims="x y"
        )
        _create_rank_dump(target_dir, rank=0, name="hidden", tensor=target, dims="x y")

        yaml_path: Path = tmp_path / "override.yaml"
        yaml_path.write_text(textwrap.dedent("""\
            dims:
              - match: "hidden"
                dims: "a b"
        """))

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="raw",
            override_dims=["hidden:t h"],
            override_config=str(yaml_path),
        )
        records = _run_and_parse(args, capsys)

        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

    def test_override_injects_dims_when_absent(self, tmp_path: Path, capsys) -> None:
        """Override injects dims into meta even when dump had no dims annotation."""
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(10, 8)
        target: torch.Tensor = tensor + torch.randn(10, 8) * 0.001

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        _create_rank_dump(baseline_dir, rank=0, name="hidden", tensor=tensor, dims=None)
        _create_rank_dump(target_dir, rank=0, name="hidden", tensor=target, dims=None)

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="raw",
            override_dims=["hidden:t h"],
        )
        records = _run_and_parse(args, capsys)

        comparisons = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].diff is not None
        assert comparisons[0].diff.passed

    def test_non_tensor_unaffected_by_override(self, tmp_path: Path, capsys) -> None:
        """Non-tensor values pass through without error even with active override."""
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(4, 4)

        baseline_dir: Path = tmp_path / "baseline"
        target_dir: Path = tmp_path / "target"
        baseline_dir.mkdir()
        target_dir.mkdir()

        for side_dir in [baseline_dir, target_dir]:
            _create_non_tensor_rank_dump(
                side_dir,
                rank=0,
                name="sm_scale",
                value=0.125,
                extra_tensor_dumps=[("hidden", tensor)],
            )

        args = _make_args(
            baseline_dir / _FIXED_EXP_NAME,
            target_dir / _FIXED_EXP_NAME,
            grouping="raw",
            override_dims=["hidden:x y"],
        )
        records = _run_and_parse(args, capsys)

        non_tensors: list[NonTensorRecord] = [
            r for r in records if isinstance(r, NonTensorRecord)
        ]
        assert len(non_tensors) == 1
        assert non_tensors[0].name == "sm_scale"
        assert non_tensors[0].values_equal

        comparisons: list[ComparisonRecord] = _get_comparisons(records)
        assert len(comparisons) == 1
        assert comparisons[0].name == "hidden"

        summary: SummaryRecord = [r for r in records if isinstance(r, SummaryRecord)][0]
        assert summary.failed == 0


# ──────────────────── Test helpers ────────────────────


def _get_comparisons(records: list[AnyRecord]) -> list[ComparisonRecord]:
    return [r for r in records if isinstance(r, ComparisonRecord)]


def _make_args(baseline_path: Path, target_path: Path, **overrides) -> Namespace:
    defaults: dict = dict(
        baseline_path=str(baseline_path),
        target_path=str(target_path),
        start_step=0,
        end_step=1000000,
        diff_threshold=1e-3,
        filter=None,
        output_format="json",
        grouping="logical",
        viz_bundle_details=False,
        viz_output_dir="/tmp/comparator_viz/",
        visualize_per_token=None,
        override_dims=[],
        override_baseline_dims=[],
        override_target_dims=[],
        override_config=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _run_and_parse(args: Namespace, capsys: pytest.CaptureFixture) -> list[AnyRecord]:
    capsys.readouterr()
    run(args)
    return [
        parse_record_json(line) for line in capsys.readouterr().out.strip().splitlines()
    ]


def _create_rank_dump(
    directory: Path,
    *,
    rank: int,
    name: str,
    tensor: torch.Tensor,
    dims: str | None = None,
    parallel_info: dict | None = None,
) -> Path:
    """Create a dump file via the real dumper, as if running on the given rank."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_dumper_module, "_get_rank", lambda: rank)

        dumper = _Dumper(
            config=DumperConfig(
                enable=True,
                dir=str(directory),
                exp_name=_FIXED_EXP_NAME,
            )
        )

        static_meta: dict = {"world_rank": rank, "world_size": 1}
        if parallel_info is not None:
            static_meta["sglang_parallel_info"] = parallel_info
        dumper.__dict__["_static_meta"] = static_meta

        dumper.dump(name, tensor, dims=dims)
        dumper.step()

    return directory / _FIXED_EXP_NAME


def _create_non_tensor_rank_dump(
    directory: Path,
    *,
    rank: int,
    name: str,
    value: object,
    extra_tensor_dumps: list[tuple[str, torch.Tensor]] | None = None,
) -> Path:
    """Create a dump with a non-tensor value via the real dumper."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_dumper_module, "_get_rank", lambda: rank)

        dumper = _Dumper(
            config=DumperConfig(
                enable=True,
                dir=str(directory),
                exp_name=_FIXED_EXP_NAME,
            )
        )
        dumper.__dict__["_static_meta"] = {"world_rank": rank, "world_size": 1}

        dumper.dump(name, value)
        for extra_name, extra_tensor in extra_tensor_dumps or []:
            dumper.dump(extra_name, extra_tensor)
        dumper.step()

    return directory / _FIXED_EXP_NAME


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
