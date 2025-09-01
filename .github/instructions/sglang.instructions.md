````
Workspace-specific instructions for coding in the SGLang codebase. Apply every rule below. Generate code that integrates cleanly with existing modules, preserves invariants, and prioritizes correctness over stylistic invention or speculative features.

Scope and sources of truth
- Obey the repository’s architecture and division of responsibilities: TokenizerManager ↔ Scheduler ↔ TpModelWorker/TpModelWorkerClient ↔ ModelRunner ↔ AttentionBackend, with DetokenizerManager handling decode and Runtime/Engine providing service entry points. Do not move responsibilities across layers.
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

Core invariants and contracts
- KV cache memory model:
  - req_to_token_pool maps (req, token_position) → out_cache_loc.
  - token_to_kv_pool maps (layer_id, out_cache_loc, head, head_dim) → KV tensors.
  - tree_cache holds token-level prefix structure across requests; keys are token ids; values are KV indices; it is request-agnostic.
- Execution modes:
  - extend mode is the default for prefill-like work; decode mode generates one token per active sequence step.
  - forward_extend writes KV for new tokens and attends over prefix+new tokens; forward_decode writes KV for the single next token and attends over full prefix.
- Scheduling:
  - If a new extend-able batch exists, it becomes the current batch; otherwise decode; retract decode requests if memory pressure exceeds thresholds.
  - Chunk prefill: split by remaining_tokens; produce being_chunked_request fragments and exclude them from decode until extend completes.
- Weight updates:
  - TokenizerManager coordinates updates; block new work; wait for outstanding requests to finish; broadcast UpdateWeightReq downstream; TpModelWorker/ModelRunner perform the actual load/update; maintain dtype/device consistency.
- Time/seed/device:
  - Keep deterministic seeding across processes where possible; honor server_args seeds; never create implicit CPU↔CUDA moves; annotate device for new tensors.

Concurrency, processes, and IPC
- Process boundaries: TokenizerManager and DetokenizerManager are separate processes; Scheduler per TP rank; optional DataParallelController above them; ZMQ sockets are the only cross-process channel. Do not introduce cross-process shared memory unless an existing pattern exists.
- TpModelWorkerClient overlap threads: forward thread executes model work; copy thread performs async GPU→CPU transfers and logprob processing; maintain thread-safe queues and futures; avoid Python GIL-heavy work on hot paths.

Data types, shapes, and placement
- Always specify dtype and device for tensors; align with model_config.dtype.
- Respect page size semantics and attention backend’s KV index layout. Page size > 1 may be logically converted to 1 by some kernels; do not depend on implicit conversions—pass explicit arguments (`--page-size N`) through server_args where needed.
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

Configuration and flags
- All new behavior must be controlled via server_args or environment variables already used in the codebase; expose new flags in server_arguments docs and FastAPI validators.
- Do not change defaults silently; document new options; maintain backward-compatible behavior; avoid surprising performance changes without a flag.

Testing and reproducibility
- Unit tests: functions with deterministic outputs (token index planning, KV slices, cache operations, prefix matching).
- Integration tests: launch Engine or Runtime with small models; validate extend→decode transitions, chunked prefill, retractions, and EOS trimming.
- Backend parity tests: run the same forward batch across backends (as feasible) and assert shape/dtype/device and tolerances for logits.
- Weight update tests: ensure TokenizerManager stalls new work; verify TpModelWorker and ModelRunner load new weights and dtype matches.
- Determinism: fix seeds; check stable decode under constrained settings; ensure tests are not flaky under multithreading in TpModelWorkerClient.

Performance constraints
- No quadratic-time algorithms in hot paths; prefer precomputed indices and contiguous memory where possible.
- Avoid device synchronizations (`torch.cuda.synchronize`) on the critical path unless required by semantics (e.g., graph capture boundaries).
- Respect memory pool watermarks; do not allocate per-token Python objects inside loops; batch tensor ops.

Coding style and patterns
- Python: type-annotate all new public functions; minimal, explicit imports; keep functions small and composable; retain existing naming and directory layout.
- Match existing dataclasses and Pydantic models for request/response; extend them, don’t replace.
- Comments: short, factual, adjacent to logic; include units, dtypes, and device placement notes.
- No TODO placeholders in shipping code; use `NotImplementedError` for deliberately unimplemented, unreachable, or model-specific branches with a one-line justification.

Integration with OpenAI-compatible endpoints
- Do not break `/v1/chat/completions` adapters; keep sampling params normalization consistent; preserve streaming and non-streaming code paths.
- Maintain DetokenizerManager trimming behavior; do not regress surrogate handling or stop-string trimming.

Multimodal specifics (Qwen2.5-VL example)
- TokenizerManager must compute `mrope_positions` after token expansion; carry them through Scheduler→ForwardBatch.
- Model forward must inject vision embeddings into `<|image_pad|>` spans, replacing token embeddings; ensure dimensions align via the model’s patch merger and that RoPE positions stay consistent.

Git plumbing and commands (use in examples and tests)
- Launch examples:
  ```bash
  python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --attention-backend flashinfer
  python3 -m sglang.launch_server --tp 8 --model deepseek-ai/DeepSeek-V3 --attention-backend fa3 --trust-remote-code
  python3 -m sglang.launch_server --tp 8 --model deepseek-ai/DeepSeek-R1 --attention-backend flashmla --kv-cache-dtype fp8_e4m3 --trust-remote-code
````

* Make targets and scripts must remain runnable as documented; do not change invocation semantics.

Change patterns and examples (generate code in these shapes)

1. Adding or modifying an Attention Backend skeleton

```python
# python/sglang/srt/layers/attention/my_backend.py
from .base import AttentionBackendBase

class MyBackend(AttentionBackendBase):
    def __init__(self, model_config, server_args):
        super().__init__(model_config, server_args)
        self._init_wrappers_and_buffers()
        self._init_cuda_graph_state_if_enabled()

    def init_forward_metadata(self, meta):
        # Compute indices and select wrappers; honor ragged/paged settings
        meta.decode = self._should_decode(meta)
        meta.indices = self._build_indices(meta)
        meta.flags = self._build_flags(meta)

    def forward_extend(self, tensors, meta):
        # Save KV for new tokens and run attention without host sync
        self._save_kv_extend(tensors, meta)
        return self._run_attention(tensors, meta)

    def forward_decode(self, tensors, meta):
        # Single-token step; write KV and run attention
        self._save_kv_decode(tensors, meta)
        return self._run_attention(tensors, meta)

    def init_cuda_graph_state(self):
        # Optional; allocate persistent buffers
        self.graph_state = self._allocate_persistent_buffers()
```

2. Weight update flow in TpModelWorker/ModelRunner

```python
# python/sglang/srt/managers/tp_worker.py
def update_weights(self, recv_req):
    # Validate dtype and shape; maintain device placement
    model_path, load_format = recv_req.model_path, recv_req.load_format
    self.model_runner.update_weights(model_path=model_path, load_format=load_format)
    return UpdateWeightReqOutput(ok=True)
```

```python
# python/sglang/srt/model_executor/model_runner.py
def update_weights(self, model_path: str, load_format: str):
    # Block concurrent forward; keep allocator consistent
    with self._update_lock:
        new_state = self._load_state(model_path, load_format, dtype=self.model_config.dtype, device=self.device)
        self.model.load_state_dict(new_state, strict=True)
        torch.cuda.empty_cache()
```

3. Scheduler extend→decode correctness hooks

```python
# python/sglang/srt/managers/scheduler.py
def process_batch_result(self, out):
    if out.mode == "extend":
        self.tree_cache.cache_unfinished_req(out.req)
    else:
        self._update_decode_state(out)
    self._maintain_running_sets()
```

4. Multimodal ForwardBatch propagation

```python
# ensure ForwardBatch carries mrope_positions
@dataclass
class ForwardBatch:
    input_ids: torch.Tensor
    positions: torch.Tensor
    mrope_positions: Optional[torch.Tensor] = None
    # device/dtype explicitly set by ModelRunner
```

Review checklist for generated changes (must hold before you stop)

* Public surfaces preserved: FastAPI endpoints, adapters, CLI flags, ZMQ payloads, dataclass fields.
* KV cache invariants hold: req\_to\_token\_pool, token\_to\_kv\_pool, tree\_cache coherence; lock refs balanced; no leaks.
* Batch flow correct: extend prioritized; decode retraction only under memory pressure; chunk prefill fragments excluded from decode.
* Attention backend indices consistent with forward mode; no device sync on hot path; page size option respected.
* Weight update safe: TokenizerManager gates new work; TpModelWorker/ModelRunner dtype match; allocator stable; errors surfaced clearly.
* Multimodal paths: mrope\_positions computed and passed; vision embeddings injected over `<|image_pad|>`; shapes aligned.
* Tests updated/added for new logic; determinism guarded; negative cases included; documented commands runnable.
* Logs/metrics present for new code paths; counters do not break consumers; severity appropriate.

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

```
```
