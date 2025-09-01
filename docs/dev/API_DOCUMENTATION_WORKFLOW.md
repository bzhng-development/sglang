# SGLang API Documentation Workflow

This document explains the different ways to generate API documentation for SGLang.

## Overview

SGLang uses multiple documentation generation approaches:

1. **Sphinx-based Documentation** - Main documentation system
2. **pdoc3 Manual Workflow** - Alternative API documentation
3. **Function Index Generation** - Custom function indexing

## 1. Sphinx Documentation (Primary)

The primary documentation system uses Sphinx with the following workflow:

### Location
- Configuration: `docs/conf.py`
- Build system: `docs/Makefile`
- CI/CD: `.github/workflows/release-docs.yml`

### Building
```bash
cd docs
make clean
make compile  # Execute notebooks
make html     # Generate HTML docs
```

### Deployment
- Automatically deployed via GitHub Actions to `sgl-project.github.io`
- Triggered on pushes to main branch affecting docs or Python code

## 2. pdoc3 Manual Workflow (Alternative)

pdoc3 provides an alternative way to generate API documentation directly from Python docstrings.

### Prerequisites
```bash
# Install sglang from source
pip install -e python/

# Install pdoc3
pip install pdoc3
```

### Usage
```bash
# Generate HTML documentation
pdoc3 --html --output-dir docs/dev/refs/api --skip-errors sglang

# Or for a specific module
pdoc3 --html --output-dir docs/dev/refs/api --skip-errors sglang.srt
```

### Output Structure
```
docs/dev/refs/api/
├── sglang/
│   ├── index.html           # Main module index
│   ├── lang/                # Language API
│   ├── srt/                 # SRT runtime API
│   └── ...                  # Other modules
└── search.js                # Search functionality
```

### Advantages
- **Fast generation** - No complex build process
- **Complete coverage** - Includes all modules and functions
- **Docstring-based** - Uses actual Python docstrings
- **Interactive search** - Built-in search functionality

### Disadvantages
- **Manual process** - Not automated in CI/CD
- **Requires source install** - Must install sglang first
- **No notebook integration** - Doesn't include tutorial notebooks

## 3. Function Index Generation (Custom)

The custom function index generator provides structured function listings.

### Location
- Script: `tools/devdocs/function_index/generate_function_index.py`

### Usage
```bash
python3 tools/devdocs/function_index/generate_function_index.py
```

### Output
- `docs/dev/refs/code-index/function-index.json` - Detailed function metadata
- `docs/dev/refs/code-index/function-index-min.txt` - Minimal format with types
- `docs/dev/refs/code-index/function-index-readable.txt` - Human-readable format
- `docs/dev/refs/code-index/coverage.md` - Documentation coverage report

### Features
- **Type information** - Includes parameter and return types
- **Full docstrings** - Complete multi-line docstring support
- **Coverage metrics** - Documentation coverage analysis
- **Multiple formats** - JSON, minimal text, and readable text

## Recommendations

### For Contributors
- **Use Sphinx** for main documentation and tutorials
- **Use pdoc3** for quick API reference during development
- **Use function index** for code analysis and coverage metrics

### For Users
- **Sphinx docs** at [sgl-project.github.io](https://sgl-project.github.io) for comprehensive guides
- **pdoc3 output** for detailed API reference with search
- **Function index** for quick function lookups

## Troubleshooting

### Common Issues

1. **Duplication in API docs**
   - Issue: Multiple directories with same content (e.g., `sglang/srt/` and `srt/sglang/srt/`)
   - Cause: Multiple pdoc3 runs or incorrect module paths
   - Solution: Clean output directory before regeneration

2. **Missing modules in pdoc3**
   - Issue: Some modules not appearing in documentation
   - Cause: Import errors or missing dependencies
   - Solution: Use `--skip-errors` flag and ensure all dependencies are installed

3. **Function index errors**
   - Issue: Script fails on certain files
   - Cause: Syntax errors or encoding issues in source files
   - Solution: Check error messages and fix source file issues

### Best Practices

1. **Clean before rebuild**
   ```bash
   rm -rf docs/dev/refs/api
   pdoc3 --html --output-dir docs/dev/refs/api --skip-errors sglang
   ```

2. **Check for duplicates**
   ```bash
   find docs/dev/refs/api -name "*.html" | sort | uniq -c | sort -nr
   ```

3. **Validate links**
   - Ensure pdoc URLs in function index match actual generated structure
   - Check cross-references between different documentation systems

## Configuration

### pdoc3 URL Generation
The function index generator creates pdoc URLs using this pattern:
```python
func['pdoc_url'] = f"docs/dev/refs/api/{module}.html#{module}.{qualname}"
```

This assumes pdoc3 generates files at `docs/dev/refs/api/{module}.html`.

### Customization
To customize the pdoc3 output location, modify the `enrich_with_urls` function in `tools/devdocs/function_index/generate_function_index.py`.
