"""Meta overrider: replace metadata fields (e.g. dims) without re-running dumps."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Optional

import yaml

from sglang.srt.debug_utils.comparator.utils import Pair, _StrictBase


class MetaOverrideRule(_StrictBase):
    """Single override rule: regex match → replacement dims string(s)."""

    match: str
    dims: str
    side: Literal["both", "baseline", "target"] = "both"


class MetaOverrideConfig(_StrictBase):
    """YAML top-level config for overriding comparator behavior."""

    dims: list[MetaOverrideRule] = []


class MetaOverrider:
    """Holds compiled override rules and applies first-match-wins replacement."""

    def __init__(self, rules: list[MetaOverrideRule]) -> None:
        self._rules: list[MetaOverrideRule] = rules
        self._compiled: list[tuple[re.Pattern[str], MetaOverrideRule]] = [
            (re.compile(rule.match), rule) for rule in rules
        ]

    @property
    def is_empty(self) -> bool:
        return len(self._rules) == 0

    @classmethod
    def from_args_and_config(
        cls,
        *,
        override_dims: list[str],
        override_baseline_dims: list[str],
        override_target_dims: list[str],
        override_config: Optional[Path],
    ) -> "MetaOverrider":
        cli_rules: list[MetaOverrideRule] = [
            MetaOverrideRule(match=name, dims=dims_str, side="both")
            for name, dims_str in _parse_cli_override_args(override_dims)
        ]
        cli_rules.extend(
            MetaOverrideRule(match=name, dims=dims_str, side="baseline")
            for name, dims_str in _parse_cli_override_args(override_baseline_dims)
        )
        cli_rules.extend(
            MetaOverrideRule(match=name, dims=dims_str, side="target")
            for name, dims_str in _parse_cli_override_args(override_target_dims)
        )

        yaml_rules: list[MetaOverrideRule] = (
            _load_yaml_rules(override_config) if override_config is not None else []
        )

        return cls(rules=cli_rules + yaml_rules)

    def apply_to_metas(
        self,
        *,
        name: str,
        baseline_metas: list[dict[str, Any]],
        target_metas: list[dict[str, Any]],
    ) -> Pair[list[dict[str, Any]]]:
        """First-match-wins per side: each side is overridden by the first matching rule that covers it."""
        result_baseline: list[dict[str, Any]] = baseline_metas
        result_target: list[dict[str, Any]] = target_metas
        baseline_matched: bool = False
        target_matched: bool = False

        for pattern, rule in self._compiled:
            if baseline_matched and target_matched:
                break
            if not pattern.search(name):
                continue

            if not baseline_matched and rule.side in ("both", "baseline"):
                result_baseline = _apply_dims_to_metas(metas=baseline_metas, new_dims=rule.dims)
                baseline_matched = True

            if not target_matched and rule.side in ("both", "target"):
                result_target = _apply_dims_to_metas(metas=target_metas, new_dims=rule.dims)
                target_matched = True

        return Pair(x=result_baseline, y=result_target)


def _parse_cli_override_arg(raw: str) -> tuple[str, str]:
    """Parse 'name:dims_string' from a CLI --override-* argument."""
    parts: list[str] = raw.split(":", maxsplit=1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError(
            f"Invalid override format: {raw!r}; expected 'name:dims_string'"
        )
    return parts[0].strip(), parts[1].strip()


def _parse_cli_override_args(raw_args: list[str]) -> list[tuple[str, str]]:
    """Parse multiple CLI override arguments."""
    return [_parse_cli_override_arg(raw) for raw in raw_args]


def _load_yaml_rules(path: Path) -> list[MetaOverrideRule]:
    """Load override rules from a YAML config file."""
    with open(path) as f:
        raw_data: Any = yaml.safe_load(f)

    if raw_data is None:
        return []

    config: MetaOverrideConfig = MetaOverrideConfig.model_validate(raw_data)
    return config.dims


def _apply_dims_to_metas(
    *,
    metas: list[dict[str, Any]],
    new_dims: Optional[str],
) -> list[dict[str, Any]]:
    """Replace 'dims' in each meta dict if new_dims is provided."""
    if new_dims is None:
        return metas

    return [{**meta, "dims": new_dims} for meta in metas]
