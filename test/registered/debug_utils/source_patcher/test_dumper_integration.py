"""Test dumper.apply_source_patches() integration with source_patcher."""

import importlib.util
import sys
from pathlib import Path

import yaml

from sglang.srt.debug_utils.dumper import DumperConfig, _Dumper
from sglang.srt.debug_utils.source_patcher.code_patcher import patch_function
from sglang.srt.debug_utils.source_patcher.types import EditSpec

_FIXTURES_DIR = Path(__file__).parent / "_fixtures"
_SAMPLE_MODULE_PATH = _FIXTURES_DIR / "sample_module.py"
_SAMPLE_MODULE_NAME = "_source_patcher_fixtures.sample_module"


def _load_fixture_module():
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


class TestDumperApplySourcePatches:
    def test_no_config_is_noop(self) -> None:
        config = DumperConfig(source_patcher_config=None)
        d = _Dumper(config=config)
        d.apply_source_patches()

    def test_patches_applied_from_yaml(self, tmp_path: Path) -> None:
        _load_fixture_module()
        module = sys.modules[_SAMPLE_MODULE_NAME]
        cls = module.SampleClass
        obj = cls()
        assert obj.greet("world") == "hello world"

        original_code = cls.greet.__code__

        patch_config = {
            "patches": [
                {
                    "target": f"{_SAMPLE_MODULE_NAME}.SampleClass.greet",
                    "edits": [
                        {
                            "match": 'greeting = f"hello {name}"',
                            "replacement": 'greeting = f"dumper_patched {name}"',
                        }
                    ],
                }
            ]
        }

        config_path = tmp_path / "patch_config.yaml"
        config_path.write_text(yaml.dump(patch_config))

        config = DumperConfig(source_patcher_config=str(config_path))
        d = _Dumper(config=config)

        try:
            d.apply_source_patches()
            assert obj.greet("world") == "dumper_patched world"
        finally:
            cls.greet.__code__ = original_code

        assert obj.greet("world") == "hello world"
