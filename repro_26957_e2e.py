#!/usr/bin/env python3
"""End-to-end reproduction of https://github.com/sgl-project/sglang/issues/26957

trtllm_mha + FROZEN_KV_MTP (NEXTN) on a hybrid-SWA Gemma-4 model crashes with a
CUDA illegal memory access once the SWA pool fills past the SWA-vs-full layer
ratio. Root cause: TRTLLMHAAttnBackend cached SWA pool state at __init__ from
the draft's own (non-SWA) token_to_kv_pool, so the SWA->full page-index
translation was dead code on the draft backend; SWA layers then received raw
full-pool page indices and read the smaller SWA k-cache out of bounds.

Auto-detects BUGGY (cached use_sliding_window_kv_pool) vs FIXED
(_resolve_swa_kv_pool). Exit 3 = reproduced crash, 0 = no crash, 1 = setup error.

Env knobs (all optional): MODEL, DRAFT, PORT, MEM_FRACTION, CONTEXT_LEN,
NUM_PROMPTS, INPUT_LEN, OUTPUT_LEN, LAUNCH_TIMEOUT, PREFILL_BACKEND,
DECODE_BACKEND. On B200 (SM100) leave the backends unset (exact issue command).
On H200 (SM90) set PREFILL_BACKEND=triton DECODE_BACKEND=trtllm_mha.
"""

import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_FILE = os.path.join(
    ROOT, "python/sglang/srt/layers/attention/trtllm_mha_backend.py"
)

MODEL = os.environ.get("MODEL", "google/Gemma-4-26B-A4B-IT")
DRAFT = os.environ.get("DRAFT", "gg-hf-am/gemma-4-26B-A4B-it-assistant")
PORT = int(os.environ.get("PORT", "18000"))
MEM_FRACTION = os.environ.get("MEM_FRACTION", "0.85")
CONTEXT_LEN = os.environ.get("CONTEXT_LEN", "16384")
NUM_PROMPTS = os.environ.get("NUM_PROMPTS", "80")
INPUT_LEN = os.environ.get("INPUT_LEN", "8000")
OUTPUT_LEN = os.environ.get("OUTPUT_LEN", "1000")
LAUNCH_TIMEOUT = int(os.environ.get("LAUNCH_TIMEOUT", "1200"))
PREFILL_BACKEND = os.environ.get("PREFILL_BACKEND")
DECODE_BACKEND = os.environ.get("DECODE_BACKEND")

SERVER_LOG = os.path.join(ROOT, "repro_26957_server.log")
BENCH_LOG = os.path.join(ROOT, "repro_26957_bench.log")

CRASH_PATTERNS = re.compile(
    r"illegal memory access"
    r"|cudaErrorIllegalAddress"
    r"|CUDA error: an illegal memory access"
    r"|AcceleratorError",
    re.IGNORECASE,
)


def detect_code_variant() -> str:
    try:
        with open(BACKEND_FILE) as f:
            src = f.read()
    except OSError:
        return "unknown"
    if "_resolve_swa_kv_pool" in src:
        return "FIXED (allocator-resolved _swa_kv_pool)"
    if "use_sliding_window_kv_pool" in src:
        return "BUGGY (cached use_sliding_window_kv_pool at __init__)"
    return "unknown"


def launch_server():
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--host", "0.0.0.0", "--port", str(PORT),
        "--model-path", MODEL, "--served-model-name", MODEL, "--tp", "1",
    ]
    if PREFILL_BACKEND or DECODE_BACKEND:
        cmd += ["--prefill-attention-backend", PREFILL_BACKEND or "triton"]
        cmd += ["--decode-attention-backend", DECODE_BACKEND or "trtllm_mha"]
    else:
        cmd += ["--attention-backend", "trtllm_mha"]
    cmd += [
        "--context-length", CONTEXT_LEN, "--mem-fraction-static", MEM_FRACTION,
        "--chunked-prefill-size", "8192", "--max-prefill-tokens", "8192",
        "--cuda-graph-max-bs", "128", "--max-running-requests", "64",
        "--speculative-algorithm", "NEXTN", "--speculative-draft-model-path", DRAFT,
        "--speculative-num-steps", "3", "--speculative-num-draft-tokens", "4",
        "--speculative-eagle-topk", "1",
    ]
    print("[repro] launching server:\n  " + " ".join(cmd), flush=True)
    log = open(SERVER_LOG, "w")
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, cwd=ROOT,
    )
    return proc, log


def server_healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def tail(path: str, n: int = 60) -> str:
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(no log)"


def log_has_crash(path: str) -> bool:
    try:
        with open(path) as f:
            return bool(CRASH_PATTERNS.search(f.read()))
    except OSError:
        return False


def wait_until_ready(proc) -> bool:
    deadline = time.time() + LAUNCH_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"[repro] server exited early (code {proc.returncode})", flush=True)
            return False
        if server_healthy():
            print("[repro] server is healthy.", flush=True)
            return True
        time.sleep(3)
    print("[repro] timed out waiting for server health.", flush=True)
    return False


def run_bench() -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend", "sglang-oai-chat", "--base-url", f"http://127.0.0.1:{PORT}",
        "--model", MODEL, "--tokenizer", MODEL, "--dataset-name", "random",
        "--random-input-len", INPUT_LEN, "--random-output-len", OUTPUT_LEN,
        "--random-range-ratio", "1.0", "--num-prompts", NUM_PROMPTS,
        "--warmup-requests", "2", "--seed", "1",
    ]
    print("[repro] starting load:\n  " + " ".join(cmd), flush=True)
    log = open(BENCH_LOG, "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)


def kill(proc):
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            time.sleep(5)
        except Exception:
            pass
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def main():
    print("=" * 72)
    print("Issue #26957 e2e reproduction")
    print(f"  code variant : {detect_code_variant()}")
    print(f"  model        : {MODEL}")
    print(f"  draft        : {DRAFT}")
    print(f"  backend      : trtllm_mha   spec: NEXTN (FROZEN_KV MTP)")
    print("=" * 72, flush=True)

    server, server_log = launch_server()
    reproduced = False
    try:
        if not wait_until_ready(server):
            print("\n[repro] --- server log tail ---\n" + tail(SERVER_LOG, 80))
            if log_has_crash(SERVER_LOG):
                print("[repro] crash signature found during startup.")
                reproduced = True
            sys.exit(3 if reproduced else 1)
        bench = run_bench()
        while bench.poll() is None:
            if server.poll() is not None:
                print(f"[repro] *** server died during load (code {server.returncode}) ***", flush=True)
                reproduced = True
                break
            if log_has_crash(SERVER_LOG):
                print("[repro] *** crash signature in server log ***", flush=True)
                reproduced = True
                break
            time.sleep(2)
        time.sleep(3)
        if not reproduced:
            reproduced = log_has_crash(SERVER_LOG) or server.poll() is not None
        kill(bench)
        print("\n[repro] --- server log tail ---\n" + tail(SERVER_LOG, 80))
        print("\n[repro] --- bench log tail ---\n" + tail(BENCH_LOG, 25))
    finally:
        kill(server)
        try:
            server_log.close()
        except Exception:
            pass
    print("=" * 72)
    if reproduced:
        print("RESULT: REPRODUCED — server crashed under load (issue #26957).")
        print("=" * 72)
        sys.exit(3)
    print("RESULT: NO CRASH — load completed without a crash signature.")
    print("=" * 72)
    sys.exit(0)


if __name__ == "__main__":
    main()
