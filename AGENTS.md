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
