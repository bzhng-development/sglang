"""Tests for SubprocessPatcher."""

import yaml

from sglang.srt.debug_utils.source_patcher import (
    EditSpec,
    PatchConfig,
    PatchSpec,
    SubprocessPatcher,
)


class TestSubprocessPatcher:
    def test_creates_yaml_config_and_env_vars(self, tmp_path) -> None:
        specs = [
            PatchSpec(
                target="some.module.Cls.method",
                edits=[EditSpec(match="old", replacement="new")],
            )
        ]

        with SubprocessPatcher(patches=specs) as sp:
            env = sp.env_vars
            assert "DUMPER_SOURCE_PATCHER_CONFIG" in env

            config_path = sp.config_path
            assert config_path.exists()

            loaded = yaml.safe_load(config_path.read_text())
            parsed = PatchConfig(**loaded)
            assert len(parsed.patches) == 1
            assert parsed.patches[0].target == "some.module.Cls.method"
            assert parsed.patches[0].edits[0].match == "old"
            assert parsed.patches[0].edits[0].replacement == "new"

    def test_cleanup_on_exit(self) -> None:
        specs = [
            PatchSpec(
                target="a.b.C.d",
                edits=[EditSpec(match="x", replacement="y")],
            )
        ]

        with SubprocessPatcher(patches=specs) as sp:
            config_path = sp.config_path
            assert config_path.exists()

        assert not config_path.exists()

    def test_multiple_patches(self) -> None:
        specs = [
            PatchSpec(
                target="mod1.Cls1.fn1",
                edits=[EditSpec(match="a", replacement="b")],
            ),
            PatchSpec(
                target="mod2.Cls2.fn2",
                edits=[
                    EditSpec(match="c", replacement="d"),
                    EditSpec(match="e", replacement="f"),
                ],
            ),
        ]

        with SubprocessPatcher(patches=specs) as sp:
            loaded = yaml.safe_load(sp.config_path.read_text())
            parsed = PatchConfig(**loaded)
            assert len(parsed.patches) == 2
            assert len(parsed.patches[1].edits) == 2

    def test_env_vars_raises_outside_context(self) -> None:
        specs = [
            PatchSpec(
                target="a.b.c",
                edits=[EditSpec(match="x", replacement="y")],
            )
        ]
        sp = SubprocessPatcher(patches=specs)

        try:
            sp.env_vars
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass
