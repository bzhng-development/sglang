"""Meta overrider: replace metadata fields (e.g. dims) without re-running dumps."""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import model_validator

from sglang.srt.debug_utils.comparator.utils import Pair, _StrictBase


class MetaOverrideRule(_StrictBase):
    """Single override rule: regex match → replacement dims string(s)."""

    match: str
    dims: Optional[str] = None
    baseline_dims: Optional[str] = None
    target_dims: Optional[str] = None

    @model_validator(mode="after")
    def _validate_dims_fields(self) -> "MetaOverrideRule":
        has_shared: bool = self.dims is not None
        has_per_side: bool = (
            self.baseline_dims is not None or self.target_dims is not None
        )

        if has_shared and has_per_side:
            raise ValueError(
                "Cannot specify both 'dims' and 'baseline_dims'/'target_dims'; "
                "use either shared 'dims' or per-side overrides"
            )
        if not has_shared and not has_per_side:
            raise ValueError(
                "Must specify either 'dims' or at least one of "
                "'baseline_dims'/'target_dims'"
            )

        return self

    def effective_baseline_dims(self) -> Optional[str]:
        return self.dims if self.dims is not None else self.baseline_dims

    def effective_target_dims(self) -> Optional[str]:
        return self.dims if self.dims is not None else self.target_dims


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
            MetaOverrideRule(match=name, dims=dims_str)
            for name, dims_str in _parse_cli_override_args(override_dims)
        ]

        cli_rules.extend(
            _merge_per_side_cli_rules(
                override_baseline_dims=override_baseline_dims,
                override_target_dims=override_target_dims,
            )
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
        """First-match-wins: find the first matching rule, apply its dims."""
        for pattern, rule in self._compiled:
            if pattern.search(name):
                new_baseline: list[dict[str, Any]] = _apply_dims_to_metas(
                    metas=baseline_metas,
                    new_dims=rule.effective_baseline_dims(),
                )
                new_target: list[dict[str, Any]] = _apply_dims_to_metas(
                    metas=target_metas,
                    new_dims=rule.effective_target_dims(),
                )
                return Pair(x=new_baseline, y=new_target)

        return Pair(x=baseline_metas, y=target_metas)


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


def _merge_per_side_cli_rules(
    *,
    override_baseline_dims: list[str],
    override_target_dims: list[str],
) -> list[MetaOverrideRule]:
    """Merge --override-baseline-dims and --override-target-dims into unified rules.

    Same match pattern from both flags → one rule with both baseline_dims and target_dims.
    Uses OrderedDict to preserve CLI order (first appearance wins).
    """
    merged: OrderedDict[str, dict[str, Optional[str]]] = OrderedDict()

    for name, dims_str in _parse_cli_override_args(override_baseline_dims):
        if name not in merged:
            merged[name] = {"baseline_dims": None, "target_dims": None}
        merged[name]["baseline_dims"] = dims_str

    for name, dims_str in _parse_cli_override_args(override_target_dims):
        if name not in merged:
            merged[name] = {"baseline_dims": None, "target_dims": None}
        merged[name]["target_dims"] = dims_str

    return [
        MetaOverrideRule(
            match=name,
            baseline_dims=sides["baseline_dims"],
            target_dims=sides["target_dims"],
        )
        for name, sides in merged.items()
    ]


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
