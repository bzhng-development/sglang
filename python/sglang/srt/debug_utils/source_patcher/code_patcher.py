import importlib
import inspect
import textwrap
import types
from collections.abc import Callable
from typing import Any, Optional

import yaml

from sglang.srt.debug_utils.source_patcher.source_editor import apply_edits
from sglang.srt.debug_utils.source_patcher.types import (
    EditSpec,
    PatchConfig,
    PatchSpec,
    PatchState,
)


def apply_patches_from_config(
    yaml_content: str,
    *,
    extra_imports: Optional[list[str]] = None,
) -> list[PatchState]:
    """Parse a YAML config string and apply all patches.

    Args:
        yaml_content: YAML string with patch specifications.
        extra_imports: Import lines to prepend to every replacement block
            (e.g. ["from pkg import foo"]).  The caller (dumper) uses this
            so users don't have to write boilerplate in YAML.
    """
    raw: dict[str, Any] = yaml.safe_load(yaml_content)
    config: PatchConfig = PatchConfig(**raw)

    if extra_imports:
        config = _inject_extra_imports(config=config, extra_imports=extra_imports)

    return _apply_specs(config.patches)


class CodePatcher:
    """Context manager that patches functions on enter and restores on exit."""

    def __init__(self, *, patches: list[PatchSpec]) -> None:
        self._patches = patches
        self._states: list[PatchState] = []

    def __enter__(self) -> "CodePatcher":
        self._states = _apply_specs(self._patches)
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        for state in reversed(self._states):
            state.restore()
        self._states.clear()


def patch_function(*, target: Callable[..., Any], edits: list[EditSpec]) -> PatchState:
    """Patch a function by modifying its source and replacing __code__.

    1. inspect.getsource -> get original source
    2. apply_edits -> modify source text
    3. compile + exec -> get new code object
    4. replace target.__code__

    Returns PatchState that can restore the original code.
    """
    original_code: types.CodeType = target.__code__

    source: str = inspect.getsource(target)
    modified_source: str = apply_edits(source=source, edits=edits)

    modified_source = textwrap.dedent(modified_source)

    code: types.CodeType = compile(modified_source, inspect.getfile(target), "exec")
    temp_namespace: dict[str, Any] = {}
    exec(code, target.__globals__, temp_namespace)

    new_fn: Any = temp_namespace[target.__name__]
    target.__code__ = new_fn.__code__

    return PatchState(target_fn=target, original_code=original_code)


# --------------------------------- private ---------------------------------


def _apply_specs(specs: list[PatchSpec]) -> list[PatchState]:
    states: list[PatchState] = []
    for spec in specs:
        target_fn: Callable[..., Any] = _resolve_target(spec.target)
        print(f"[source_patcher] patching {spec.target}")
        state: PatchState = patch_function(target=target_fn, edits=spec.edits)
        states.append(state)
    return states


def _inject_extra_imports(
    *, config: PatchConfig, extra_imports: list[str]
) -> PatchConfig:
    """Prepend extra import lines to every replacement in the config."""
    import_block: str = "\n".join(extra_imports)
    new_patches: list[PatchSpec] = []

    for spec in config.patches:
        new_edits: list[EditSpec] = []
        for edit in spec.edits:
            new_replacement: str = import_block + "\n" + edit.replacement
            new_edits.append(EditSpec(match=edit.match, replacement=new_replacement))
        new_patches.append(PatchSpec(target=spec.target, edits=new_edits))

    return PatchConfig(patches=new_patches)


def _resolve_target(qualified_name: str) -> Callable[..., Any]:
    """Resolve 'pkg.mod.Class.method' to the actual function object.

    Tries progressively shorter module paths from right to left,
    then uses getattr for the remaining attribute chain.
    """
    parts: list[str] = qualified_name.split(".")

    target: Any = None
    for split_idx in range(len(parts), 0, -1):
        module_path: str = ".".join(parts[:split_idx])
        try:
            target = importlib.import_module(module_path)
            attr_parts: list[str] = parts[split_idx:]
            break
        except ImportError:
            continue
    else:
        raise ImportError(f"could not import any module prefix of '{qualified_name}'")

    for attr_name in attr_parts:
        target = getattr(target, attr_name)

    if isinstance(target, classmethod):
        target = target.__func__
    if not callable(target):
        raise TypeError(
            f"resolved target '{qualified_name}' is not callable: {type(target)}"
        )

    return target
