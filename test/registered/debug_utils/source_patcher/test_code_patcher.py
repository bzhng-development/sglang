import os
import tempfile
from pathlib import Path

import pytest
import yaml

from sglang.srt.debug_utils.source_patcher.code_patcher import (
    CodePatcher,
    _resolve_target,
    apply_patches_from_env,
    patch_function,
)
from sglang.srt.debug_utils.source_patcher.types import EditSpec, PatchSpec

_FIXTURE_MODULE = (
    "test.registered.debug_utils.source_patcher._fixtures.sample_module"
)


def _get_sample_class():
    from test.registered.debug_utils.source_patcher._fixtures.sample_module import (
        SampleClass,
    )

    return SampleClass


def _get_standalone_fn():
    from test.registered.debug_utils.source_patcher._fixtures.sample_module import (
        standalone_function,
    )

    return standalone_function


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
        target = _resolve_target(f"{_FIXTURE_MODULE}.SampleClass.greet")
        cls = _get_sample_class()
        assert target is cls.greet

    def test_resolve_standalone_function(self) -> None:
        target = _resolve_target(f"{_FIXTURE_MODULE}.standalone_function")
        fn = _get_standalone_fn()
        assert target is fn

    def test_resolve_nonexistent_raises(self) -> None:
        with pytest.raises((ImportError, AttributeError)):
            _resolve_target(f"{_FIXTURE_MODULE}.NonexistentClass.method")


class TestCodePatcher:
    def test_context_manager_patches_and_restores(self) -> None:
        cls = _get_sample_class()
        obj = cls()
        assert obj.greet("world") == "hello world"

        patches = [
            PatchSpec(
                target=f"{_FIXTURE_MODULE}.SampleClass.greet",
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
        cls = _get_sample_class()
        obj = cls()

        patches = [
            PatchSpec(
                target=f"{_FIXTURE_MODULE}.SampleClass.greet",
                edits=[
                    EditSpec(
                        match='greeting = f"hello {name}"',
                        replacement='greeting = f"p1 {name}"',
                    )
                ],
            ),
            PatchSpec(
                target=f"{_FIXTURE_MODULE}.SampleClass.compute",
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
        cls = _get_sample_class()
        obj = cls()

        patches = [
            PatchSpec(
                target=f"{_FIXTURE_MODULE}.SampleClass.greet",
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


class TestApplyPatchesFromEnv:
    def test_no_env_var_is_noop(self) -> None:
        old = os.environ.pop("SOURCE_PATCHER_CONFIG", None)
        try:
            apply_patches_from_env()
        finally:
            if old is not None:
                os.environ["SOURCE_PATCHER_CONFIG"] = old

    def test_patches_applied_from_yaml(self, tmp_path: Path) -> None:
        cls = _get_sample_class()
        obj = cls()
        assert obj.greet("world") == "hello world"

        config = {
            "patches": [
                {
                    "target": f"{_FIXTURE_MODULE}.SampleClass.greet",
                    "edits": [
                        {
                            "match": 'greeting = f"hello {name}"',
                            "replacement": 'greeting = f"yaml_patched {name}"',
                        }
                    ],
                }
            ]
        }

        config_path = tmp_path / "patch_config.yaml"
        config_path.write_text(yaml.dump(config))

        old = os.environ.get("SOURCE_PATCHER_CONFIG")
        os.environ["SOURCE_PATCHER_CONFIG"] = str(config_path)
        try:
            apply_patches_from_env()
            assert obj.greet("world") == "yaml_patched world"
        finally:
            if old is not None:
                os.environ["SOURCE_PATCHER_CONFIG"] = old
            else:
                del os.environ["SOURCE_PATCHER_CONFIG"]

            state = patch_function(
                target=cls.greet,
                edits=[
                    EditSpec(
                        match='greeting = f"yaml_patched {name}"',
                        replacement='greeting = f"hello {name}"',
                    )
                ],
            )
