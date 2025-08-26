# Repository Guidelines

## Project Structure & Module Organization
- `python/sglang/`: Core Python library and entry points (e.g., `sglang.launch_server`).
- `sgl-kernel/`: C++/CUDA kernels and Python bindings (CMake-based); wheel build targets.
- `sgl-router/`: Rust router service and benchmarks (`cargo` workspace).
- `test/`: Python unit/integration tests (`lang/`, `srt/`).
- `docs/`, `examples/`, `benchmark/`, `docker/`, `scripts/`: Supporting materials and tools.

## Build, Test, and Development Commands
- Python dev install: `pip install -e "python[dev]"` (or `pip install -e python`).
- Run server locally: `python -m sglang.launch_server --model <hf_model> --port 30000`.
- Root formatting helpers: `make format` (formats modified Python files); `pre-commit run --all-files`.
- Kernel (C++/CUDA): `make -C sgl-kernel build`, tests: `make -C sgl-kernel test`.
- Router (Rust): `make -C sgl-router build`, tests: `make -C sgl-router test`.
- Python tests (all): `python -m unittest discover -s test -p 'test_*.py'`.
- Suite runners: `python test/lang/run_suite.py --suite per-commit` and `python test/srt/run_suite.py --suite per-commit`.

## Coding Style & Naming Conventions
- Indentation: 4 spaces (2 for `*.{json,yaml,yml}`), LF EOL (`.editorconfig`).
- Python: Black + isort profile, run `isort` then `black`. Lint docs/examples with `ruff`.
- C++/CUDA: `clang-format` (config in repo). Use `.cuh/.cu/.cc` naming.
- Rust: `cargo fmt`, keep idiomatic module and test naming.
- Naming: `snake_case` for files/functions, `CamelCase` for classes, concise module paths (e.g., `sglang.srt.*`).

## Testing Guidelines
- Framework: Python `unittest` (primary). Rust uses `cargo test`; kernels have Makefile-driven tests.
- Location/pattern: place under `test/` with `test_*.py` names. Keep fast, deterministic, and GPU-light where possible.
- Suites: add to `test/*/run_suite.py` (e.g., `per-commit`, `nightly`) when relevant.

## Commit & Pull Request Guidelines
- Commit style: imperative, scoped tags in brackets (e.g., `[router]`, `[kernel]`, `[Docs]`, `[fix]`), concise subject, details in body; reference issues/PRs.
- PRs must include: clear description, rationale, minimal diff, tests or rationale for no tests, performance notes (before/after if applicable), and environment details (GPU/driver/torch).
- Ensure CI passes and run `pre-commit` locally before pushing.

## Security & Configuration Tips
- Prefer environment variables for secrets; never commit keys.
- Match extras to hardware: e.g., `pip install -e "python[all]"` for CUDA, `all_cpu`/`all_hip`/`all_xpu`/`all_hpu` as needed (see `python/pyproject.toml`).
- Pin large-model tests to lightweight configs locally; use Dockerfiles in `docker/` for reproducible setups.

Workspace-specific instructions for coding in the SGLang codebase. Apply every rule below. Generate code that integrates cleanly with existing modules, preserves invariants, and prioritizes correctness over stylistic invention or speculative features.

Scope and sources of truth
- Obey the repository's architecture and division of responsibilities: TokenizerManager ↔ Scheduler ↔ TpModelWorker/TpModelWorkerClient ↔ ModelRunner ↔ AttentionBackend, with DetokenizerManager handling decode and Runtime/Engine providing service entry points. Do not move responsibilities across layers.
- Use only facts present in the code, docstrings, and the walkthrough provided here. If an external behavior is unknown, write a brief in-line comment `# Unknown: explain why` or raise `NotImplementedError("explain missing fact")` rather than guessing.
- Preserve public interfaces consumed by the FastAPI endpoints, OpenAI-compatible adapters, CLI flags, ZMQ message schemas, and dataclasses used for request/response payloads.

Architecture map (non-negotiable responsibilities)
- TokenizerManager: tokenize inputs; coordinate weight updates; block new work during updates; manage ZMQ sockets to Scheduler and DetokenizerManager.
- Scheduler: maintain waiting_queue/new_batch/running_batch; build ScheduleBatch; run batches via TpModelWorker/Client; handle cache bookkeeping; implement extend-first execution; support chunked prefill; implement retract_decode when memory is tight.
- TpModelWorker: single-GPU executor façade; construct ForwardBatch; call ModelRunner.forward; call ModelRunner.sample when generating; expose update_weights.
- TpModelWorkerClient: same single-GPU scope; multithreaded overlap of compute and GPU→CPU copies; preserve TpModelWorker-compatible surface.
- ModelRunner: device/distributed init; attention backend selection; CUDA graphs wiring; memory pool; forward_decode/forward_extend/forward_idle; sample; update_weights/load_model.
- Attention backends: implement init_forward_metadata, forward_extend, forward_decode; manage KV index layouts; support ragged/paged attention as appropriate; integrate optional CUDA graphs (capture/replay).
- DataParallelController: orchestrate DP groups and their TP workers; load-balance per policy (e.g., round-robin today); route requests via ZMQ; unify TP/DP orchestration.
- DetokenizerManager: incremental decode; EOS/stop-string trimming; LimitedCapacityDict for decode state; return BatchStrOut to TokenizerManager.
- Engine (no HTTP) vs Runtime (HTTP/FastAPI): do not mix concerns; Engine is embedded mode; Runtime exposes `/v1/*` endpoints.

Concurrency, processes, and IPC
- Process boundaries: TokenizerManager and DetokenizerManager are separate processes; Scheduler per TP rank; optional DataParallelController above them; ZMQ sockets are the only cross-process channel. Do not introduce cross-process shared memory unless an existing pattern exists.
- TpModelWorkerClient overlap threads: forward thread executes model work; copy thread performs async GPU→CPU transfers and logprob processing; maintain thread-safe queues and futures; avoid Python GIL-heavy work on hot paths.

Data types, shapes, and placement
- Always specify dtype and device for tensors; align with model_config.dtype.
- Respect page size semantics and attention backend's KV index layout. Page size > 1 may be logically converted to 1 by some kernels; do not depend on implicit conversions—pass explicit arguments (`--page-size N`) through server_args where needed.
- For multimodal models (e.g., Qwen2.5-VL), carry `mm_items` and `mrope_positions` through ForwardBatch; replace `<|image_pad|>` placeholders with hashed identifiers at scheduling time to preserve prefix cache sharing.

Attention backend guidance
- Supported backends include flashinfer (default non-Hopper), fa3 (default Hopper), triton, torch_native, flashmla, trtllm_mla, ascend. Choose by server_args.attention_backend; never hard-code.
- Backend interface:
  - init_forward_metadata(decode|extend): compute indices (paged or ragged), wrapper selection, flags.
  - forward_extend: set KV for new tokens, run attention.
  - forward_decode: set KV for single-step token, run attention.
  - Optional CUDA graphs: implement init_cuda_graph_state once; provide capture/replay metadata initializers; replay path must be fast.
- When adding a backend, provide both non-CUDA-graph and CUDA-graph paths; prefer split_kv heuristics in plan functions; do not regress existing backends.

Scheduler rules (implementation details you must preserve)
- waiting_queue: ordered by longest prefix or policy; admits new and retracted requests; ensure O(log n) or better operations where feasible.
- get_next_batch_to_run:
  - Merge last_batch into running_batch first.
  - Prefer extend batches via get_new_batch_prefill; form cur_batch accordingly.
  - If no extend batch, filter running_batch to live decode requests and form cur_batch; consider retractions.
- process_batch_result:
  - Decode mode: update per-request state, token/logprob buffers, memory accounting, statistics.
  - Extend mode: insert new tokens into caches; update prefix indices and last_node; manage lock refs.
- Radix cache steps:
  - cache_unfinished_req: insert token ids and kv_indices; free KV in token_to_kv_pool as appropriate; update prefix_indices and last_node; inc/dec locks carefully.
  - cache_finished_req: persist final tokens; release req_to_token_pool; adjust lock refs.

Error handling and recovery
- Raise explicit exceptions with actionable messages; do not swallow errors.
- Partial failures:
  - For multi-request batches, isolate failed requests; return error to TokenizerManager; continue others when safe.
  - For weight updates, abort new scheduling until update completes; ensure allocator consistency; on failure, roll back to last known good weights or exit worker cleanly with clear logs.
- Network/IPC: detect ZMQ disconnects; implement backoff/retry where the project already does; never spin-wait.

Logging, metrics, observability
- Use existing logging channels and severities; include request ids, tp_rank/dp_rank, gpu_id, attention backend, forward_mode.
- Emit performance counters the Scheduler expects (tokens/s, batch sizes, cache hits, retractions, chunking events).
- For new backends or scheduling policies, add counters behind feature flags without breaking current metric consumers.


Prohibitions

* Do not invent APIs, config keys, or message schemas.
* Do not move responsibilities across layers or collapse processes.
* Do not introduce implicit CPU↔CUDA copies or dtype changes.
* Do not degrade latency/throughput without a behind-a-flag option and a factual note.
* Do not add placeholders or TODOs in production paths; use `NotImplementedError` with reason if unavoidable.

Output expectations for code generation

* Provide complete, runnable patches with imports, types, and device/dtype annotations.
* Keep functions small; reuse existing helpers and conventions; align filenames and module paths with current layout.
* Include concise inline comments explaining shapes, dtypes, devices, and ordering assumptions adjacent to non-trivial logic.
