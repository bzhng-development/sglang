# SGLang Documentation Guide

**Comprehensive guide to SGLang's documentation systems, tools, and workflows.**

---

## Table of Contents

1. [Overview](#overview)
2. [Documentation Locations](#documentation-locations)
3. [Documentation Systems](#documentation-systems)
4. [Quick Start Guide](#quick-start-guide)
5. [Detailed Workflows](#detailed-workflows)
6. [Troubleshooting](#troubleshooting)
7. [Maintenance](#maintenance)

---

## Overview

SGLang uses **three complementary documentation systems**:

1. **Sphinx Documentation** - Primary user-facing docs at https://docs.sglang.ai
2. **API Documentation** - Developer API reference (pdoc-generated)
3. **Function Index** - Code analysis and coverage tools

Each serves different needs and audiences, with specific tools and workflows.

---

## Documentation Locations

### Directory Structure
```
sglang-code/
├── docs/                           # PRIMARY: Main documentation
│   ├── DOCUMENTATION_GUIDE.md      # THIS FILE (consolidated guide)
│   ├── README.md                   # Basic workflow (DEPRECATED - use this guide)
│   ├── conf.py                     # Sphinx configuration
│   ├── Makefile                    # Build system
│   ├── serve.sh                    # Local dev server
│   ├── generate_api_docs.sh        # API doc helper script
│   ├── sglang/                     # Developer documentation
│   │   ├── llm-txt-ref/            # Function analysis (AST-extracted)
│   │   └── README.md               # Developer reference
│   └── dev/                        # Developer documentation
│       ├── refs/                   # Reference materials
│       │   └── api/                # Generated API docs (pdoc3)
│       └── runbooks/               # Development runbooks
└── python/docs/                    # GENERATED: Raw API output (can be removed)
    └── sglang/                     # Complete module documentation
```

### What Goes Where

| Location | Purpose | Audience | Auto-Generated |
|----------|---------|----------|----------------|
| `docs/` | User guides, tutorials | End users | ❌ Manual |
| `docs/dev/refs/api/` | Clean API reference | Developers | ✅ pdoc3 |
| `docs/sglang/llm-txt-ref/` | Code analysis | Contributors | ✅ Custom tool |
| `python/docs/` | Raw API dump | None (cleanup target) | ✅ pdoc |

---

## Documentation Systems

### 1. Sphinx Documentation (Primary)

**Purpose**: Official user documentation with tutorials and guides

**Technology Stack**:
- Sphinx + MyST + Jupyter notebooks
- Auto-deployment via GitHub Actions
- Live at https://docs.sglang.ai

**Key Files**:
- `docs/conf.py` - Sphinx configuration
- `docs/Makefile` - Build commands
- `docs/index.rst` - Main entry point
- `docs/**/*.md` - Markdown content
- `docs/**/*.ipynb` - Jupyter notebooks

**Workflow**:
```bash
cd docs

# Install dependencies
pip install -r requirements.txt

# Build documentation
make clean
make compile  # Execute notebooks (10+ mins)
make html     # Generate HTML

# Serve locally
make serve    # Auto-reload on changes
# OR
bash serve.sh # Alternative serving method
```

**Deployment**: Automated via `.github/workflows/release-docs.yml`

### 2. API Documentation (Developer Reference)

**Purpose**: Clean, curated API reference for developers
**Technology Stack**:
- pdoc3 (legacy, stable version)
- Selective module inclusion
- Interactive search functionality

**Output Location**: `docs/dev/refs/api/`

**Generation Commands**:
```bash
cd docs

# Method 1: Using helper script (RECOMMENDED)
./generate_api_docs.sh --clean

# Method 2: Direct pdoc command
pip install -e ../python/
uv pip install pdoc3
rm -rf dev/refs/api
pdoc3 --html --output-dir docs --skip-errors sglang

# Method 3: With UV package manager
uv pip install pdoc3
pdoc3 --html --output-dir docs --skip-errors sglang
```

**Key Entry Points**:
- `docs/dev/refs/api/index.html` - Main index
- `docs/dev/refs/api/sglang.html` - Module overview
- `docs/dev/refs/api/sglang/srt.html` - SRT runtime API
- `docs/dev/refs/api/sglang/lang.html` - Language API

**Content**: 66 HTML files (16MB) covering core API modules only

### 3. Function Index (Code Analysis)

**Purpose**: Function signatures, coverage analysis, and code metrics

**Technology Stack**:
- Custom AST-based Python tool
- Multiple output formats
- Enhanced with types and full docstrings

**Tool Location**: `tools/devdocs/function_index/generate_function_index.py`

**Generation**:
```bash
python3 tools/devdocs/function_index/generate_function_index.py \
  --root python/sglang \
  --out docs/sglang/llm-txt-ref \
  --format both \
  --per-directory

# Output files created in docs/sglang/llm-txt-ref/:
# - function-index.json (complete metadata)
# - function-index-min.txt (minimal with types)
# - function-index-readable.txt (human-readable)
# - coverage.md (documentation coverage report)
# - by_directory/ (per-module breakdowns)
```

**Features**:
- ✅ Parameter and return types in minimal format
- ✅ Complete multi-line docstrings in readable format
- ✅ Coverage analysis and metrics
- ✅ Cross-references to API docs

---

## Quick Start Guide

### For New Contributors

1. **Set up documentation environment**:
   ```bash
   cd docs
   pip install -r requirements.txt
   pip install -e ../python/
   ```

2. **Make changes** to `.md` or `.ipynb` files

3. **Test locally**:
   ```bash
   make html  # Quick build
   make serve # Live preview
   ```

4. **Submit PR** - docs auto-deploy on merge

### For API Documentation Updates

1. **Generate clean API docs**:
   ```bash
   cd docs
   ./generate_api_docs.sh --clean
   ```

2. **Review output** at `dev/refs/api/sglang/index.html`

3. **Commit changes** if needed

### For Code Analysis

1. **Generate function index**:
   ```bash
   python3 tools/devdocs/function_index/generate_function_index.py \
     --root python/sglang \
     --out docs/sglang/llm-txt-ref \
     --format both \
     --per-directory
   ```

2. **Check coverage**:
   ```bash
   cat docs/sglang/llm-txt-ref/coverage.md
   ```

---

## Detailed Workflows

### Adding New Documentation

1. **User guides**: Add to `docs/` with `.md` or `.ipynb`
2. **API changes**: Regenerate API docs
3. **Development guides**: Add to `docs/dev/`

### Updating Existing Content

1. **Text changes**: Edit `.md` files directly
2. **Code examples**: Update `.ipynb` notebooks
3. **API changes**: Auto-reflected after regeneration

### Managing Jupyter Notebooks

```bash
# Execute notebooks before commit
make compile

# Strip output for clean commits
pip install nbstripout
find . -name '*.ipynb' -exec nbstripout {} \;

# Pre-commit hook handles this automatically
pre-commit run --all-files
```

---

## Troubleshooting

### Common Issues

#### 1. Duplicate API Documentation
**Problem**: Multiple versions of API docs in different locations

**Solution**:
```bash
# Remove redundant python/docs
rm -rf python/docs

# Use only docs/dev/refs/api
cd docs
./generate_api_docs.sh --clean
```

#### 2. Missing Modules in API Docs
**Problem**: Some modules not appearing in pdoc output

**Cause**: Import errors or missing dependencies

**Solution**:
```bash
# Ensure sglang is installed
pip install -e ../python/

# Generate documentation
pdoc sglang -o dev/refs/api
```

#### 3. Function Index Failures
**Problem**: Script fails on certain files

**Cause**: Syntax errors or encoding issues

**Solution**:
```bash
# Check specific file
python3 -m py_compile python/sglang/srt/problematic_file.py

# Fix encoding issues
find python/sglang -name "*.py" -exec file {} \; | grep -v ASCII
```

#### 4. Sphinx Build Errors
**Problem**: Notebooks fail to execute

**Cause**: Missing dependencies or long execution times

**Solution**:
```bash
# Skip notebook execution temporarily
make html  # Builds without executing

# Or fix specific notebook
jupyter nbconvert --to notebook --execute problematic_notebook.ipynb --inplace
```

### Tool Version Issues

#### pdoc vs pdoc3 Confusion
**Current Status**:
- `docs/dev/refs/api/` uses pdoc 15.0.4 (newer, cleaner)
- `python/docs/` used pdoc3 0.11.6 (older, now removed)

**Recommendation**: Use modern pdoc for consistency:
```bash
pip install pdoc  # Modern version
pdoc sglang -o dev/refs/api
```

---

## Maintenance

### Regular Tasks

1. **Weekly**: Regenerate API docs after major changes
2. **Monthly**: Update function index and check coverage
3. **Before releases**: Full documentation rebuild and review

### Cleanup Tasks

1. **Remove redundant directories**:
   ```bash
   rm -rf python/docs  # Raw pdoc output - not needed
   ```

2. **Clean old builds**:
   ```bash
   cd docs
   make clean
   rm -rf _build logs
   ```

3. **Update documentation dependencies**:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

### Automation Opportunities

Current automation:
- ✅ Main docs deploy on push to main
- ❌ API docs generation (manual)
- ❌ Function index updates (manual)

Potential improvements:
- [ ] Auto-generate API docs on API changes
- [ ] Daily function index updates
- [ ] Link validation checks

---

## Commands Reference

### Essential Commands

```bash
# Main documentation
cd docs
make clean && make html && make serve

# API documentation
./generate_api_docs.sh --clean --serve

# Function analysis
python3 ../tools/devdocs/function_index/generate_function_index.py

# Pre-commit checks
pre-commit run --all-files

# Clean everything
make clean
rm -rf dev/refs/api dev/refs/code-index python/docs
```

### Helper Scripts

- `docs/generate_api_docs.sh` - API documentation generation
- `docs/serve.sh` - Local documentation server
- `tools/devdocs/function_index/generate_function_index.py` - Function analysis

---

## File Locations Summary

### Files to Use
- `docs/DOCUMENTATION_GUIDE.md` - **THIS FILE** (authoritative guide)
- `docs/generate_api_docs.sh` - API generation helper
- `tools/devdocs/function_index/generate_function_index.py` - Function analysis

### Files Being Deprecated
- `docs/README.md` - Basic workflow (superseded by this guide)
- `docs/dev/API_DOCUMENTATION_WORKFLOW.md` - Detailed workflow (consolidated here)
- `docs/dev/refs/README.md` - Directory-specific guide (consolidated here)

### Directories to Clean Up
- `python/docs/` - Raw pdoc output (can be removed)

---

## Getting Help

1. **Documentation issues**: Check this guide first
2. **Build problems**: See troubleshooting section
3. **Tool questions**: Check commands reference
4. **Contributions**: Follow quick start guide

For complex issues, create an issue with:
- Command that failed
- Error message
- Environment details (OS, Python version)

---

*This guide consolidates all SGLang documentation workflows into a single authoritative source. Last updated: September 2025*
