import importlib.util
import sys
from pathlib import Path

import pytest

from sglang.srt.debug_utils.source_patcher.code_patcher import (
    CodePatcher,
    _resolve_target,
    patch_function,
)
from sglang.srt.debug_utils.source_patcher.types import EditSpec, PatchSpec

_FIXTURES_DIR = Path(__file__).parent / "_fixtures"
_SAMPLE_MODULE_PATH = _FIXTURES_DIR / "sample_module.py"
_SAMPLE_MODULE_NAME = "_source_patcher_fixtures.sample_module"


def _load_fixture_module():
    """Load sample_module.py by file path and register it in sys.modules."""
    if _SAMPLE_MODULE_NAME in sys.modules:
        return sys.modules[_SAMPLE_MODULE_NAME]

    spec = importlib.util.spec_from_file_location(
        _SAMPLE_MODULE_NAME, _SAMPLE_MODULE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SAMPLE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _get_sample_class():
    module = _load_fixture_module()
    return module.SampleClass


def _get_standalone_fn():
    module = _load_fixture_module()
    return module.standalone_function


class TestPatchFunction:
    def test_basic_patch_changes_behavior(self) -> None:
        cls = _get_sample_class()
        obj = cls()
        assert obj.greet("world") == "hello world"

        state = patch_function(
            target=cls.greet,
            edits=[
                EditSpec(
                    match='greeting = f"hello {name}"',
                    replacement='greeting = f"patched {name}"',
                )
            ],
        )
        try:
            assert obj.greet("world") == "patched world"
        finally:
            state.restore()

        assert obj.greet("world") == "hello world"

    def test_globals_preserved_after_patch(self) -> None:
        cls = _get_sample_class()
        obj = cls()
        assert obj.uses_global() == "value=global_value"

        state = patch_function(
            target=cls.uses_global,
            edits=[
                EditSpec(
                    match='return f"value={GLOBAL_VAR}"',
                    replacement='return f"patched_value={GLOBAL_VAR}"',
                )
            ],
        )
        try:
            assert obj.uses_global() == "patched_value=global_value"
        finally:
            state.restore()

    def test_function_identity_preserved(self) -> None:
        cls = _get_sample_class()
        fn_id_before = id(cls.greet)

        state = patch_function(
            target=cls.greet,
            edits=[
                EditSpec(
                    match='greeting = f"hello {name}"',
                    replacement='greeting = f"patched {name}"',
                )
            ],
        )
        try:
            assert id(cls.greet) == fn_id_before
        finally:
            state.restore()

    def test_patch_standalone_function(self) -> None:
        fn = _get_standalone_fn()
        assert fn(2, 3) == 5

        state = patch_function(
            target=fn,
            edits=[
                EditSpec(
                    match="return a + b",
                    replacement="return a * b",
                )
            ],
        )
        try:
            assert fn(2, 3) == 6
        finally:
            state.restore()

        assert fn(2, 3) == 5


class TestResolveTarget:
    def test_resolve_class_method(self) -> None:
        _load_fixture_module()
        target = _resolve_target(f"{_SAMPLE_MODULE_NAME}.SampleClass.greet")
        cls = _get_sample_class()
        assert target is cls.greet

    def test_resolve_standalone_function(self) -> None:
        _load_fixture_module()
        target = _resolve_target(f"{_SAMPLE_MODULE_NAME}.standalone_function")
        fn = _get_standalone_fn()
        assert target is fn

    def test_resolve_nonexistent_raises(self) -> None:
        _load_fixture_module()
        with pytest.raises((ImportError, AttributeError)):
            _resolve_target(f"{_SAMPLE_MODULE_NAME}.NonexistentClass.method")


class TestCodePatcher:
    def test_context_manager_patches_and_restores(self) -> None:
        _load_fixture_module()
        cls = _get_sample_class()
        obj = cls()
        assert obj.greet("world") == "hello world"

        patches = [
            PatchSpec(
                target=f"{_SAMPLE_MODULE_NAME}.SampleClass.greet",
                edits=[
                    EditSpec(
                        match='greeting = f"hello {name}"',
                        replacement='greeting = f"ctx_patched {name}"',
                    )
                ],
            )
        ]

        with CodePatcher(patches=patches):
            assert obj.greet("world") == "ctx_patched world"

        assert obj.greet("world") == "hello world"

    def test_context_manager_multiple_patches(self) -> None:
        _load_fixture_module()
        cls = _get_sample_class()
        obj = cls()

        patches = [
            PatchSpec(
                target=f"{_SAMPLE_MODULE_NAME}.SampleClass.greet",
                edits=[
                    EditSpec(
                        match='greeting = f"hello {name}"',
                        replacement='greeting = f"p1 {name}"',
                    )
                ],
            ),
            PatchSpec(
                target=f"{_SAMPLE_MODULE_NAME}.SampleClass.compute",
                edits=[
                    EditSpec(
                        match="result = x * 2 + 1",
                        replacement="result = x * 100",
                    )
                ],
            ),
        ]

        with CodePatcher(patches=patches):
            assert obj.greet("world") == "p1 world"
            assert obj.compute(5) == 500

        assert obj.greet("world") == "hello world"
        assert obj.compute(5) == 11

    def test_restores_on_exception(self) -> None:
        _load_fixture_module()
        cls = _get_sample_class()
        obj = cls()

        patches = [
            PatchSpec(
                target=f"{_SAMPLE_MODULE_NAME}.SampleClass.greet",
                edits=[
                    EditSpec(
                        match='greeting = f"hello {name}"',
                        replacement='greeting = f"err_patched {name}"',
                    )
                ],
            )
        ]

        with pytest.raises(RuntimeError):
            with CodePatcher(patches=patches):
                assert obj.greet("world") == "err_patched world"
                raise RuntimeError("test error")

        assert obj.greet("world") == "hello world"
