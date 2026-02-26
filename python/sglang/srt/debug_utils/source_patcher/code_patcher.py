import importlib
import inspect
import textwrap
import types
from collections.abc import Callable
from typing import Any, Optional

from sglang.srt.debug_utils.source_patcher.source_editor import apply_edits
from sglang.srt.debug_utils.source_patcher.types import EditSpec, PatchSpec, PatchState


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


def patch_function(*, target: Callable[..., Any], edits: list[EditSpec]) -> PatchState:
    """Patch a function by modifying its source and replacing __code__.

    1. inspect.getsource → get original source
    2. apply_edits → modify source text
    3. compile + exec → get new code object
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


class CodePatcher:
    """Context manager that patches functions on enter and restores on exit."""

    def __init__(self, *, patches: list[PatchSpec]) -> None:
        self._patches = patches
        self._states: list[PatchState] = []

    def __enter__(self) -> "CodePatcher":
        for spec in self._patches:
            target_fn: Callable[..., Any] = _resolve_target(spec.target)
            state: PatchState = patch_function(target=target_fn, edits=spec.edits)
            self._states.append(state)
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
