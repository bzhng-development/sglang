import tempfile
from pathlib import Path
from typing import Any, Optional

import yaml

from sglang.srt.debug_utils.source_patcher.types import PatchSpec


class SubprocessPatcher:
    """Context manager that writes a YAML patch config for child processes.

    Usage:
        patches = [PatchSpec(target="...", edits=[...])]
        with SubprocessPatcher(patches=patches) as sp:
            subprocess.Popen(["python", ...], env={**os.environ, **sp.env_vars})
    """

    def __init__(self, *, patches: list[PatchSpec]) -> None:
        self._patches = patches
        self._config_path: Optional[Path] = None
        self._tmpdir: Optional[tempfile.TemporaryDirectory[str]] = None

    def __enter__(self) -> "SubprocessPatcher":
        self._tmpdir = tempfile.TemporaryDirectory(prefix="source_patcher_config_")
        config_dir = Path(self._tmpdir.name)
        self._config_path = config_dir / "patch_config.yaml"

        config: dict[str, Any] = {
            "patches": [spec.model_dump() for spec in self._patches]
        }
        self._config_path.write_text(yaml.dump(config, default_flow_style=False))
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
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
