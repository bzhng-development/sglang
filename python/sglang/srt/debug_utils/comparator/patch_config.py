"""Dims override: patch dims strings in .pt metadata without re-running dumps."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import model_validator

from sglang.srt.debug_utils.comparator.utils import Pair, _StrictBase


class DimsOverrideRule(_StrictBase):
    """Single dims override rule: regex match → replacement dims string(s)."""

    match: str
    dims: Optional[str] = None
    baseline_dims: Optional[str] = None
    target_dims: Optional[str] = None

    @model_validator(mode="after")
    def _validate_dims_fields(self) -> "DimsOverrideRule":
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


class PatchConfig(_StrictBase):
    """YAML top-level config for patching comparator behavior."""

    dims: list[DimsOverrideRule] = []


class DimsOverrider:
    """Holds compiled override rules and applies first-match-wins replacement."""

    def __init__(self, rules: list[DimsOverrideRule]) -> None:
        self._rules: list[DimsOverrideRule] = rules
        self._compiled: list[tuple[re.Pattern[str], DimsOverrideRule]] = [
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
        patch_config: Optional[Path],
    ) -> "DimsOverrider":
        cli_rules: list[DimsOverrideRule] = []

        for raw in override_dims:
            name, dims_str = _parse_cli_override_arg(raw)
            cli_rules.append(DimsOverrideRule(match=name, dims=dims_str))

        for raw in override_baseline_dims:
            name, dims_str = _parse_cli_override_arg(raw)
            cli_rules.append(DimsOverrideRule(match=name, baseline_dims=dims_str))

        for raw in override_target_dims:
            name, dims_str = _parse_cli_override_arg(raw)
            cli_rules.append(DimsOverrideRule(match=name, target_dims=dims_str))

        yaml_rules: list[DimsOverrideRule] = (
            _load_yaml_rules(patch_config) if patch_config is not None else []
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


def _load_yaml_rules(path: Path) -> list[DimsOverrideRule]:
    """Load dims override rules from a YAML patch config file."""
    with open(path) as f:
        raw_data: Any = yaml.safe_load(f)

    if raw_data is None:
        return []

    config: PatchConfig = PatchConfig.model_validate(raw_data)
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
