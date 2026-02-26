from pathlib import Path

import yaml

from sglang.srt.debug_utils.source_patcher.subprocess_patcher import (
    SubprocessPatcher,
)
from sglang.srt.debug_utils.source_patcher.types import EditSpec, PatchSpec


def _make_patches() -> list[PatchSpec]:
    return [
        PatchSpec(
            target="some.module.Class.method",
            edits=[
                EditSpec(match="x = 1", replacement="x = 2"),
                EditSpec(match="return x", replacement="return x + 1"),
            ],
        ),
        PatchSpec(
            target="some.other.function",
            edits=[
                EditSpec(match="pass", replacement="return 42"),
            ],
        ),
    ]


class TestSubprocessPatcher:
    def test_creates_yaml_config_on_enter(self) -> None:
        patches = _make_patches()
        with SubprocessPatcher(patches=patches) as sp:
            config_path = Path(sp.env_vars["SOURCE_PATCHER_CONFIG"])
            assert config_path.exists()

            raw = yaml.safe_load(config_path.read_text())
            assert len(raw["patches"]) == 2
            assert raw["patches"][0]["target"] == "some.module.Class.method"
            assert len(raw["patches"][0]["edits"]) == 2

    def test_env_vars_returns_correct_key(self) -> None:
        patches = _make_patches()
        with SubprocessPatcher(patches=patches) as sp:
            env = sp.env_vars
            assert "SOURCE_PATCHER_CONFIG" in env
            assert Path(env["SOURCE_PATCHER_CONFIG"]).exists()

    def test_cleanup_on_exit(self) -> None:
        patches = _make_patches()
        with SubprocessPatcher(patches=patches) as sp:
            config_path = Path(sp.env_vars["SOURCE_PATCHER_CONFIG"])
            assert config_path.exists()

        assert not config_path.exists()

    def test_roundtrip_yaml_to_patchspec(self) -> None:
        patches = _make_patches()
        with SubprocessPatcher(patches=patches) as sp:
            config_path = Path(sp.env_vars["SOURCE_PATCHER_CONFIG"])
            raw = yaml.safe_load(config_path.read_text())

            loaded_patches = [
                PatchSpec(**patch_raw) for patch_raw in raw["patches"]
            ]
            assert len(loaded_patches) == 2
            assert loaded_patches[0].target == patches[0].target
            assert loaded_patches[0].edits[0].match == "x = 1"
            assert loaded_patches[0].edits[0].replacement == "x = 2"

    def test_cleanup_on_exception(self) -> None:
        patches = _make_patches()
        config_path: Path = Path("/nonexistent")

        try:
            with SubprocessPatcher(patches=patches) as sp:
                config_path = Path(sp.env_vars["SOURCE_PATCHER_CONFIG"])
                raise RuntimeError("test error")
        except RuntimeError:
            pass

        assert not config_path.exists()
