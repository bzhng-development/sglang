# SGLang Internal Developer Documentation

Internal engineering documentation for the SGLang SRT runtime.

## Purpose

Navigate and maintain a 500K-line codebase spanning 467+ files with 5,594+ functions across distributed systems, ML model execution, memory management, and kernel implementations.

## Layout

```
docs/dev/
├── refs/api/           # pdoc-generated HTML API reference
├── refs/code-index/    # AST-extracted function indexes
├── runbooks/           # Operational procedures
├── architecture/       # Module dependency graphs
└── search/             # Local search interface
```

## Function Index

The function index provides multiple formats for different workflows:

### Formats Available

1. **Full JSON** (`function-index.json`) - Complete metadata for programmatic analysis
2. **Minimal JSON** (`function-index-min.json`) - Lightweight for tooling
3. **Minimal Text** (`function-index-min.txt`) - Token-efficient for grep/LLM
4. **Readable Text** (`function-index-readable.txt`) - Browser-friendly formatted docs
5. **Per-Directory** (`by_directory/`) - Module-specific minimal and readable versions

### Generation

```bash
# Generate all formats
python tools/devdocs/function_index/generate_function_index.py \
  --root python/sglang/srt \
  --out docs/dev/refs/code-index \
  --format both \
  --per-directory
```

### Viewing

```bash
# Browser viewing (recommended)
open docs/dev/refs/code-index/function-index-readable.txt
open docs/dev/refs/code-index/by_directory/managers_readable.txt

# Terminal search
grep "schedule" docs/dev/refs/code-index/function-index-min.txt

# Programmatic queries
jq '.[] | select(.class == "Scheduler") | .qualname' \
  docs/dev/refs/code-index/function-index.json
```

## API Documentation

```bash
# Browse locally
cd docs/dev/refs/api && python -m http.server 8000
# Open http://localhost:8000
```

Key entry points:
- `sglang/srt/managers/scheduler.html` - Scheduling logic
- `sglang/srt/model_executor/model_runner.html` - Model execution
- `sglang/srt/mem_cache/memory_pool.html` - Memory management

## Coverage Metrics

Current statistics:
- **Total Functions**: 5,594
- **Documented**: 914 (16.3%)
- **Files**: 431
- **Average per file**: 13.0

Check coverage:
```bash
cat docs/dev/refs/code-index/coverage.md
```

## Regeneration

After code changes:
```bash
# Update function indexes
python tools/devdocs/function_index/generate_function_index.py \
  --root python/sglang/srt \
  --out docs/dev/refs/code-index \
  --format both \
  --per-directory

# Regenerate pdoc HTML
PDOC_ALLOW_EXEC=1 pdoc ./sglang -o docs/dev/refs/api
```
