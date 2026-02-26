import tempfile
from pathlib import Path
from typing import Optional

import yaml

from sglang.srt.debug_utils.source_patcher.types import PatchSpec


class SubprocessPatcher:
    """Write a temporary YAML patch config and provide env vars for child processes.

    Usage:
        specs = [PatchSpec(target="pkg.mod.Cls.method", edits=[...])]
        with SubprocessPatcher(patches=specs) as sp:
            subprocess.Popen(..., env={**os.environ, **sp.env_vars})
        # temp file cleaned up on exit
    """

    def __init__(self, *, patches: list[PatchSpec]) -> None:
        self._patches = patches
        self._config_path: Optional[Path] = None
        self._tmpdir: Optional[tempfile.TemporaryDirectory[str]] = None

    def __enter__(self) -> "SubprocessPatcher":
        self._tmpdir = tempfile.TemporaryDirectory(prefix="source_patcher_")
        config_dict: dict = {"patches": [spec.model_dump() for spec in self._patches]}
        self._config_path = Path(self._tmpdir.name) / "patch_config.yaml"
        self._config_path.write_text(yaml.dump(config_dict))
        return self

    def __exit__(
        self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: object
    ) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
            self._config_path = None

    @property
    def env_vars(self) -> dict[str, str]:
        if self._config_path is None:
            raise RuntimeError("SubprocessPatcher must be used as a context manager")
        return {"DUMPER_SOURCE_PATCHER_CONFIG": str(self._config_path)}

    @property
    def config_path(self) -> Path:
        if self._config_path is None:
            raise RuntimeError("SubprocessPatcher must be used as a context manager")
        return self._config_path
