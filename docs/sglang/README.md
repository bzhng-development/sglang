# SGLang Internal Developer Documentation

Internal engineering documentation for the SGLang complete codebase.

## Purpose

Navigate and maintain a 500K-line codebase spanning 489+ files with 6,263+ functions across distributed systems, ML model execution, memory management, and kernel implementations.

## Layout

```
docs/sglang/
├── llm-txt-ref/        # AST-extracted function indexes and coverage
└── README.md           # This documentation

docs/dev/refs/api/      # pdoc3-generated HTML API reference
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
# Generate all formats for complete codebase
python3 tools/devdocs/function_index/generate_function_index.py \
  --root python/sglang \
  --out docs/sglang/llm-txt-ref \
  --format both \
  --per-directory
```

### Viewing

```bash
# Browser viewing (recommended)
open docs/sglang/llm-txt-ref/function-index-readable.txt
open docs/sglang/llm-txt-ref/by_directory/managers_readable.txt

# Terminal search
grep "schedule" docs/sglang/llm-txt-ref/function-index-min.txt

# Programmatic queries
jq '.[] | select(.class == "Scheduler") | .qualname' \
  docs/sglang/llm-txt-ref/function-index.json
```

## API Documentation

### Installation

```bash
# Install pdoc3 for HTML generation
uv pip install pdoc3
```

### Generation

```bash
# Generate HTML API documentation
pdoc3 --html --output-dir docs --skip-errors sglang
```

### Viewing

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
- **Total Functions**: 6,263
- **Documented**: 1,008 (16.1%)
- **Files**: 489
- **Average per file**: 12.8

Check coverage:
```bash
cat docs/sglang/llm-txt-ref/coverage.md
```

## Regeneration

After code changes:
```bash
# Update function indexes (complete codebase)
python3 tools/devdocs/function_index/generate_function_index.py \
  --root python/sglang \
  --out docs/sglang/llm-txt-ref \
  --format both \
  --per-directory

# Regenerate pdoc3 HTML
pdoc3 --html --output-dir docs --skip-errors sglang
```
