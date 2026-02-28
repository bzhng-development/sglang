"""Tests for meta_overrider — unit tests."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from sglang.srt.debug_utils.comparator.meta_overrider import (
    MetaOverrider,
    MetaOverrideRule,
    _load_yaml_rules,
    _parse_cli_override_arg,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


# ───────────────────── Unit: MetaOverrideRule ─────────────────────


class TestMetaOverrideRule:
    """Pydantic validation for MetaOverrideRule."""

    def test_shared_dims_both(self) -> None:
        """Default side='both' applies dims to both sides."""
        rule = MetaOverrideRule(match="hidden", dims="b s h d")
        assert rule.dims == "b s h d"
        assert rule.side == "both"

    def test_side_baseline(self) -> None:
        """side='baseline' is accepted."""
        rule = MetaOverrideRule(match="logits", dims="b s v(tp)", side="baseline")
        assert rule.dims == "b s v(tp)"
        assert rule.side == "baseline"

    def test_side_target(self) -> None:
        """side='target' is accepted."""
        rule = MetaOverrideRule(match="logits", dims="b s v(ep)", side="target")
        assert rule.dims == "b s v(ep)"
        assert rule.side == "target"

    def test_invalid_side_rejected(self) -> None:
        """Invalid side value is rejected."""
        with pytest.raises(Exception):
            MetaOverrideRule(match="x", dims="b s", side="invalid")

    def test_dims_required(self) -> None:
        """Must specify dims."""
        with pytest.raises(Exception):
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


# ──────────────────── Unit: MetaOverrider ────────────────────


class TestMetaOverrider:
    """MetaOverrider logic: matching, priority, apply_to_metas."""

    def test_first_match_wins(self) -> None:
        """First matching rule takes effect; later rules ignored."""
        overrider = MetaOverrider(
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
        overrider = MetaOverrider(
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
        overrider = MetaOverrider(
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

    def test_side_baseline_only(self) -> None:
        """side='baseline' overrides only baseline; target unchanged."""
        overrider = MetaOverrider(
            rules=[
                MetaOverrideRule(match="logits", dims="b s v(tp)", side="baseline")
            ]
        )
        result = overrider.apply_to_metas(
            name="logits",
            baseline_metas=[{"dims": "old_b"}],
            target_metas=[{"dims": "old_t"}],
        )
        assert result.x[0]["dims"] == "b s v(tp)"
        assert result.y[0]["dims"] == "old_t"

    def test_side_target_only(self) -> None:
        """side='target' overrides only target; baseline unchanged."""
        overrider = MetaOverrider(
            rules=[
                MetaOverrideRule(match="logits", dims="b s v(ep)", side="target")
            ]
        )
        result = overrider.apply_to_metas(
            name="logits",
            baseline_metas=[{"dims": "old_b"}],
            target_metas=[{"dims": "old_t"}],
        )
        assert result.x[0]["dims"] == "old_b"
        assert result.y[0]["dims"] == "b s v(ep)"

    def test_is_empty(self) -> None:
        """Empty overrider reports is_empty=True."""
        assert MetaOverrider(rules=[]).is_empty
        assert not MetaOverrider(rules=[MetaOverrideRule(match="x", dims="d")]).is_empty

    def test_multiple_metas(self) -> None:
        """All metas in the list are updated when a rule matches."""
        overrider = MetaOverrider(rules=[MetaOverrideRule(match="hidden", dims="NEW")])
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
        overrider = MetaOverrider(rules=[MetaOverrideRule(match="hidden", dims="NEW")])
        result = overrider.apply_to_metas(
            name="hidden",
            baseline_metas=[{"other": "val"}],
            target_metas=[{}],
        )
        assert result.x[0]["dims"] == "NEW"
        assert result.y[0]["dims"] == "NEW"


# ──────────────────── Unit: from_args_and_config ────────────────────


class TestFromArgsAndConfig:
    """MetaOverrider.from_args_and_config merges CLI + YAML rules."""

    def test_cli_before_yaml(self, tmp_path: Path) -> None:
        """CLI rules are ordered before YAML rules (CLI wins on conflict)."""
        yaml_path = tmp_path / "override.yaml"
        yaml_path.write_text(textwrap.dedent("""\
            overrides:
              - match: "hidden"
                dims: "FROM_YAML"
        """))

        overrider = MetaOverrider.from_args_and_config(
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
        overrider = MetaOverrider.from_args_and_config(
            override_dims=[],
            override_baseline_dims=[],
            override_target_dims=[],
            override_config=None,
        )
        assert overrider.is_empty

    def test_per_side_cli_produces_separate_rules(self) -> None:
        """--override-baseline-dims and --override-target-dims produce separate rules with side field."""
        overrider = MetaOverrider.from_args_and_config(
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
        # Each side matched by its own first-matching rule
        assert result.x[0]["dims"] == "b s h(tp)"
        assert result.y[0]["dims"] == "b s h(ep)"


# ──────────────────── Unit: _load_yaml_rules ────────────────────


class TestLoadYamlRules:
    """YAML loading and validation."""

    def test_valid_yaml(self, tmp_path: Path) -> None:
        """Valid YAML with override rules loads correctly."""
        yaml_path = tmp_path / "override.yaml"
        yaml_path.write_text(textwrap.dedent("""\
            overrides:
              - match: "hidden"
                dims: "b s h d"
              - match: "logits"
                dims: "b s v(tp)"
                side: baseline
        """))
        rules = _load_yaml_rules(yaml_path)
        assert len(rules) == 2
        assert rules[0].dims == "b s h d"
        assert rules[0].side == "both"
        assert rules[1].dims == "b s v(tp)"
        assert rules[1].side == "baseline"

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

    def test_overrides_empty_list(self, tmp_path: Path) -> None:
        """Only 'overrides' key with no entries returns empty list."""
        yaml_path = tmp_path / "minimal.yaml"
        yaml_path.write_text("overrides: []\n")
        rules = _load_yaml_rules(yaml_path)
        assert rules == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
