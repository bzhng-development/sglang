"""E2E test: source patcher + dumper + comparator on SGLang server.

Patches Qwen3DecoderLayer.forward to insert dumper.dump() calls,
launches 1-GPU baseline and 2-GPU TP=2 target servers, runs inference,
verifies patched dump fields exist, then runs comparator to verify
numerical consistency.

The dumper.apply_source_patches() auto-injects ``from ... import dumper``
so the YAML only needs ``dumper.dump(...)`` calls.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import requests
import yaml

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="nightly-2-gpu", nightly=True)

MODEL = "Qwen/Qwen3-0.6B"

PATCH_CONFIG: dict = {
    "patches": [
        {
            "target": "sglang.srt.models.qwen3.Qwen3DecoderLayer.forward",
            "edits": [
                {
                    "match": (
                        "hidden_states = self.self_attn(\n"
                        "    positions=positions,\n"
                        "    hidden_states=hidden_states,\n"
                        "    forward_batch=forward_batch,\n"
                        ")"
                    ),
                    "replacement": (
                        "hidden_states = self.self_attn(\n"
                        "    positions=positions,\n"
                        "    hidden_states=hidden_states,\n"
                        "    forward_batch=forward_batch,\n"
                        ")\n"
                        "dumper.dump('patched_attn_output', hidden_states)"
                    ),
                },
                {
                    "match": "hidden_states = self.mlp(hidden_states)",
                    "replacement": (
                        "hidden_states = self.mlp(hidden_states)\n"
                        "dumper.dump('patched_mlp_output', hidden_states)"
                    ),
                },
            ],
        }
    ]
}


def _run_server_and_generate(
    *,
    dump_dir: Path,
    config_path: Path,
    tp: int,
    base_url: str,
) -> None:
    """Launch SGLang server with source patcher + dumper, send a generate request."""
    env = {
        **os.environ,
        "DUMPER_SOURCE_PATCHER_CONFIG": str(config_path),
        "DUMPER_SERVER_PORT": "reuse",
    }

    proc = popen_launch_server(
        MODEL,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=["--tp", str(tp), "--max-total-tokens", "128"],
        env=env,
    )
    try:
        requests.post(
            f"{base_url}/dumper/configure",
            json={"enable": True, "dir": str(dump_dir)},
        ).raise_for_status()

        resp = requests.post(
            f"{base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"max_new_tokens": 8},
            },
        )
        assert resp.status_code == 200, f"Generate failed: {resp.text}"
    finally:
        kill_process_tree(proc.pid)


def _find_exp_dir(dump_dir: Path) -> Path:
    """Find the experiment directory (dump_*) under the dump base dir."""
    candidates = list(dump_dir.glob("dump_*"))
    assert (
        len(candidates) >= 1
    ), f"No dump_* dir found in {dump_dir}, contents: {list(dump_dir.iterdir())}"
    return candidates[0]


def _verify_patched_fields(dump_dir: Path, field_names: list[str]) -> None:
    """Verify that patched dump fields exist as .pt files."""
    for field in field_names:
        matches = list(dump_dir.rglob(f"*name={field}*.pt"))
        assert len(matches) > 0, (
            f"Expected patched field '{field}' not found under {dump_dir}. "
            f"Available files: {sorted(f.name for f in dump_dir.rglob('*.pt'))[:20]}"
        )


class TestSourcePatcherE2ESGLang:
    """E2E: patch Qwen3 forward -> dump -> compare 1gpu vs 2gpu-tp2."""

    @pytest.mark.timeout(300)
    def test_patch_dump_and_compare(self, tmp_path: Path) -> None:
        patched_fields = ["patched_attn_output", "patched_mlp_output"]
        base_url = DEFAULT_URL_FOR_TEST

        config_path = tmp_path / "patch_config.yaml"
        config_path.write_text(yaml.dump(PATCH_CONFIG))

        # Run 1: baseline (1 GPU)
        baseline_dir = tmp_path / "baseline"
        _run_server_and_generate(
            dump_dir=baseline_dir,
            config_path=config_path,
            tp=1,
            base_url=base_url,
        )

        baseline_exp = _find_exp_dir(baseline_dir)
        _verify_patched_fields(baseline_dir, patched_fields)

        # Run 2: target (2 GPU TP=2)
        target_dir = tmp_path / "target"
        _run_server_and_generate(
            dump_dir=target_dir,
            config_path=config_path,
            tp=2,
            base_url=base_url,
        )

        target_exp = _find_exp_dir(target_dir)
        _verify_patched_fields(target_dir, patched_fields)

        # Compare baseline vs target (raw grouping to avoid token aligner issues,
        # filter to only compare our patched fields)
        result = subprocess.run(
            [
                "python",
                "-m",
                "sglang.srt.debug_utils.comparator",
                "--baseline-path",
                str(baseline_exp),
                "--target-path",
                str(target_exp),
                "--output-format",
                "json",
                "--grouping",
                "raw",
                "--filter",
                "patched_",
            ],
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode == 0
        ), f"Comparator failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

        records: list[dict] = [
            json.loads(line)
            for line in result.stdout.strip().splitlines()
            if line.strip()
        ]
        assert len(records) > 0, "Comparator produced no output records"

        summary = next(
            (r for r in records if r.get("type") == "summary"),
            None,
        )
        assert (
            summary is not None
        ), f"No summary record found. Records: {[r.get('type') for r in records]}"
        assert summary["total"] > 0, "No comparisons were made"

        comparison_names: set[str] = {
            r.get("name", "") for r in records if r.get("type") == "comparison"
        }
        for field in patched_fields:
            assert any(field in name for name in comparison_names), (
                f"Patched field '{field}' not in comparison records. "
                f"Got: {sorted(comparison_names)}"
            )
