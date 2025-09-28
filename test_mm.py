#!/usr/bin/env python3

"""
Automation script for multimodal benchmarking.

The workflow is:
1. Launch the SGLang server in the background and capture logs.
2. Run ``sglang.bench_serving`` against the running server and save logs.
3. Stop the server.
4. Optionally checkout a user-provided Git branch and repeat steps 1-3.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 30000
KILL_SCRIPT = REPO_ROOT / "scripts" / "killall_sglang.sh"
START_STOP_SLEEP_SECONDS = 60

SERVE_COMMAND: List[str] = [
    "python",
    "-m",
    "sglang.launch_server",
    "--model-path",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "--mem-fraction-static",
    "0.8",
    "--chat-template",
    "qwen2-vl",
    "--tp",
    "1",
    "--disable-radix-cache",
    "--cuda-graph-bs",
    "256",
    "--cuda-graph-max-bs",
    "256",
    "--chunked-prefill-size",
    "8192",
    "--max-prefill-tokens",
    "8192",
    "--max-running-requests",
    "256",
    "--enable-multimodal",
]

SERVE_ENV: Dict[str, str] = {"SGLANG_VLM_CACHE_SIZE_MB": "0"}

BENCH_COMMAND_TEMPLATE: List[str] = [
    "python",
    "-m",
    "sglang.bench_serving",
    "--backend",
    "sglang",
    "--host",
    "{host}",
    "--port",
    "{port}",
    "--model",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "--dataset-name",
    "random-image",
    "--random-image-num-images",
    "1",
    "--random-image-resolution",
    "720p",
    "--random-input-len",
    "128",
    "--random-output-len",
    "128",
    "--num-prompts",
    "500",
    "--request-rate",
    "20",
    "--max-concurrency",
    "64",
    "--apply-chat-template",
    "--warmup-requests",
    "50",
    "--output-details",
]

ACCURACY_COMMAND_TEMPLATE = (
    "python benchmark/mmmu/bench_sglang.py --port {port} --concurrency {concurrency}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the SGLang server, run multimodal benchmarks, and optionally compare against another branch.",
    )
    parser.add_argument(
        "--branch",
        help="Git branch to checkout for the second benchmark run. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "bench_runs",
        help="Directory used to store server and benchmark logs.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Host used by the benchmark client to reach the server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port used by both the server and benchmark client.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=120,
        help="Seconds to wait for the server port to become available.",
    )
    parser.add_argument(
        "--post-start-delay",
        type=float,
        default=5.0,
        help="Seconds to sleep after the port is reachable to allow warm-up.",
    )
    parser.add_argument(
        "--allow-dirty-tree",
        action="store_true",
        help="Allow branch checkout even when the git worktree has uncommitted changes.",
    )
    parser.add_argument(
        "--no-timestamp-subdir",
        action="store_true",
        help="Disable creation of a timestamped subdirectory under the output directory.",
    )
    parser.add_argument(
        "--skip-second-run",
        action="store_true",
        help="Skip the second benchmark run entirely (ignores --branch).",
    )
    parser.add_argument(
        "--accuracy-concurrency",
        type=int,
        default=16,
        help="Concurrency to pass to the accuracy evaluation script.",
    )
    return parser.parse_args()


def ensure_port_available(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        if sock.connect_ex((host, port)) == 0:
            raise SystemExit(
                f"Port {host}:{port} is already in use. Stop the existing process before running this script."
            )


def wait_for_port(host: str, port: int, timeout: int) -> bool:
    end_time = time.time() + timeout
    while time.time() < end_time:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(1)
    return False


def build_bench_command(host: str, port: int) -> List[str]:
    return [arg.format(host=host, port=port) for arg in BENCH_COMMAND_TEMPLATE]


def build_accuracy_command(port: int, concurrency: int) -> List[str]:
    return shlex.split(
        ACCURACY_COMMAND_TEMPLATE.format(port=port, concurrency=concurrency)
    )


def launch_server(log_path: Path) -> Tuple[subprocess.Popen, Iterable[Path]]:
    env = os.environ.copy()
    env.update(SERVE_ENV)
    log_file = open(log_path, "w")
    process = subprocess.Popen(
        SERVE_COMMAND,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
        env=env,
    )
    return process, (log_file,)


def stop_server(process: subprocess.Popen, resources: Iterable[object]) -> None:
    subprocess.run([str(KILL_SCRIPT)], cwd=REPO_ROOT, check=False)
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    for resource in resources:
        try:
            resource.close()  # type: ignore[attr-defined]
        except Exception:
            pass


def stream_process_output(command: List[str], log_path: Path) -> None:
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        exit_code = process.wait()
    if exit_code != 0:
        raise RuntimeError(
            f"Command {' '.join(command)} failed with exit code {exit_code}. See {log_path} for details."
        )


def run_benchmark_cycle(
    tag: str,
    host: str,
    port: int,
    output_dir: Path,
    startup_timeout: int,
    post_start_delay: float,
    accuracy_concurrency: int,
) -> Dict[str, Path]:
    server_log = output_dir / f"{tag}_server.log"
    bench_log = output_dir / f"{tag}_bench.log"
    accuracy_log = output_dir / f"{tag}_accuracy.log"

    ensure_port_available(host, port)
    print(f"[run:{tag}] Launching server...")
    server_process, resources = launch_server(server_log)

    try:
        if not wait_for_port(host, port, startup_timeout):
            raise RuntimeError(
                f"Server failed to open {host}:{port} within {startup_timeout} seconds. Check {server_log}."
            )
        if post_start_delay:
            time.sleep(post_start_delay)
        time.sleep(START_STOP_SLEEP_SECONDS)
        print(f"[run:{tag}] Running benchmark...")
        bench_command = build_bench_command(host, port)
        stream_process_output(bench_command, bench_log)
        print(f"[run:{tag}] Running accuracy evaluation...")
        accuracy_command = build_accuracy_command(port, accuracy_concurrency)
        stream_process_output(accuracy_command, accuracy_log)
    finally:
        time.sleep(START_STOP_SLEEP_SECONDS)
        print(f"[run:{tag}] Stopping server...")
        stop_server(server_process, resources)

    return {
        "server_log": server_log,
        "bench_log": bench_log,
        "accuracy_log": accuracy_log,
    }


def ensure_git_available() -> None:
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=REPO_ROOT,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit("This script must be run inside a git repository.") from exc


def get_current_branch_label() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    branch = result.stdout.strip()
    if branch == "HEAD":
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        branch = result.stdout.strip()
    return branch


def is_git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return bool(result.stdout.strip())


def checkout_branch(branch: str) -> None:
    print(f"[git] Checking out '{branch}'...")
    subprocess.run(["git", "checkout", branch], cwd=REPO_ROOT, check=True)


def sanitize_tag(tag: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in tag)
    return cleaned or "run"


def prompt_for_branch() -> str:
    try:
        return input(
            "Enter git branch to checkout for second run (leave blank to skip): "
        ).strip()
    except EOFError:
        return ""


def create_output_directory(base_dir: Path, use_timestamp: bool) -> Path:
    base_dir = base_dir.expanduser().resolve()
    if use_timestamp:
        base_dir = base_dir / time.strftime("%Y%m%d-%H%M%S")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def summarize_runs(results: Dict[str, Dict[str, Path]]) -> None:
    print("\nSummary of generated logs:")
    for tag, paths in results.items():
        print(f"  {tag}:")
        for kind, path in paths.items():
            print(f"    {kind}: {path}")


def main() -> None:
    ensure_git_available()
    args = parse_args()

    output_dir = create_output_directory(args.output_dir, not args.no_timestamp_subdir)
    print(f"Logs will be stored under {output_dir}")

    branch_name = args.branch or ""
    if not branch_name and not args.skip_second_run:
        branch_name = prompt_for_branch()

    if args.skip_second_run:
        branch_name = ""

    results = {}

    baseline_tag = sanitize_tag(get_current_branch_label())
    print(f"Starting baseline run on '{baseline_tag}'...")
    results[f"baseline_{baseline_tag}"] = run_benchmark_cycle(
        tag=f"baseline_{baseline_tag}",
        host=args.host,
        port=args.port,
        output_dir=output_dir,
        startup_timeout=args.startup_timeout,
        post_start_delay=args.post_start_delay,
        accuracy_concurrency=args.accuracy_concurrency,
    )

    if branch_name:
        original_branch = get_current_branch_label()
        if is_git_dirty() and not args.allow_dirty_tree:
            raise SystemExit(
                "Git worktree has uncommitted changes. Commit/stash them or re-run with --allow-dirty-tree."
            )

        try:
            checkout_branch(branch_name)
            branch_tag = sanitize_tag(get_current_branch_label())
            results[f"candidate_{branch_tag}"] = run_benchmark_cycle(
                tag=f"candidate_{branch_tag}",
                host=args.host,
                port=args.port,
                output_dir=output_dir,
                startup_timeout=args.startup_timeout,
                post_start_delay=args.post_start_delay,
                accuracy_concurrency=args.accuracy_concurrency,
            )
        finally:
            checkout_branch(original_branch)

    summarize_runs(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
