# SGLang SRT Function Analysis - Generation Guide

This document explains how to regularly generate and update the comprehensive function analysis for the SGLang SRT codebase.

## Quick Start

```bash
# From SGLang repository root
./scripts/generate_function_analysis.sh
```

## Overview

The generation script creates two types of analysis:

### 1. **Detailed Analysis** (Full AST metadata)
- Complete function signatures with type annotations
- Return type information
- Class context for methods
- Documentation summaries
- Line numbers
- **Output**: `function_analysis/*.txt` files

### 2. **Minimal Analysis** (Token-efficient)
- Ultra-compact format for LLM context
- Just function names + basic parameters
- **56% smaller** than detailed version
- **Output**: `function_analysis/minimal/*.txt` files

## Script Location

```
scripts/generate_function_analysis.sh
```

## Usage Options

### Basic Generation
```bash
./scripts/generate_function_analysis.sh
```
Generates both detailed and minimal versions.

### Minimal Only (Faster)
```bash
./scripts/generate_function_analysis.sh --minimal-only
```
Generates only token-efficient minimal versions. Ideal for regular updates when you primarily need the compact format.

### Clean Regeneration
```bash
./scripts/generate_function_analysis.sh --clean
```
Removes existing analysis and regenerates everything from scratch.

### Help
```bash
./scripts/generate_function_analysis.sh --help
```

## What Gets Generated

### Directory Structure
```
function_analysis/
├── README.md                           # Main documentation
├── GENERATION_GUIDE.md                 # This file
├── srt_complete_directory_summary.txt  # Overview report
│
├── srt_functions_index.txt             # Main SRT codebase (detailed)
├── root_functions_index.txt            # Root-level files (detailed)
├── <directory>_functions_index.txt     # Per-directory analysis (detailed)
│
└── minimal/                            # Token-efficient versions
    ├── README.md                       # Minimal format documentation
    ├── srt_minimal.txt                 # Main SRT codebase (minimal)
    ├── root_minimal.txt                # Root-level files (minimal)
    └── <directory>_minimal.txt         # Per-directory analysis (minimal)
```

### Covered Directories
The script analyzes all SGLang SRT directories:
- `configs/`, `connector/`, `constrained/`, `debug_utils/`
- `disaggregation/`, `distributed/`, `entrypoints/`, `eplb/`
- `function_call/`, `layers/`, `lora/`, `managers/`
- `mem_cache/`, `metrics/`, `model_executor/`, `model_loader/`
- `models/`, `multimodal/`, `sampling/`, `speculative/`
- `tokenizer/`, `weight_sync/`
- Plus root-level Python files

## Requirements

### System Requirements
- **Python 3.9+** (for `ast.unparse` support)
- **ripgrep (rg)** - preferred for speed, falls back to `find`+`grep`
- Must run from **SGLang repository root**

### Installation
```bash
# macOS
brew install ripgrep

# Ubuntu/Debian
apt install ripgrep

# Or use fallback (find+grep) - works everywhere
```

## Regular Usage Workflow

### For Development Work
```bash
# Quick minimal update (recommended for frequent use)
./scripts/generate_function_analysis.sh --minimal-only

# Use the compact versions for LLM context
cat function_analysis/minimal/layers_minimal.txt
```

### For Comprehensive Analysis
```bash
# Full regeneration (after major changes)
./scripts/generate_function_analysis.sh --clean

# View detailed analysis
cat function_analysis/layers_functions_index.txt
```

### Automation (CI/CD)
```bash
# In CI pipeline or git hooks
./scripts/generate_function_analysis.sh --minimal-only
```

## Performance

### Generation Speed
- **Minimal only**: ~10-30 seconds
- **Full analysis**: ~30-90 seconds
- **Clean regeneration**: ~60-120 seconds

### Output Sizes
- **Detailed**: ~1.5MB total
- **Minimal**: ~679KB total (56% reduction)

## Validation

The script includes automatic validation:
- ✅ Checks Python 3.9+ availability
- ✅ Verifies SGLang SRT directory structure
- ✅ Validates ripgrep or find+grep availability
- ✅ Creates output directories automatically
- ✅ Handles errors gracefully with colored logging

## Integration Examples

### Git Hook (Pre-commit)
```bash
# .git/hooks/pre-commit
#!/bin/bash
./scripts/generate_function_analysis.sh --minimal-only
git add function_analysis/minimal/
```

### CI/CD Pipeline
```yaml
# .github/workflows/function-analysis.yml
name: Update Function Analysis
on:
  push:
    paths: ['python/sglang/srt/**/*.py']

jobs:
  update-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install ripgrep
        run: sudo apt-get install ripgrep
      - name: Generate function analysis
        run: ./scripts/generate_function_analysis.sh --minimal-only
      - name: Commit updates
        run: |
          git add function_analysis/
          git commit -m "Auto-update function analysis" || exit 0
```

### Makefile Integration
```makefile
# Makefile
.PHONY: func-analysis func-analysis-minimal

func-analysis:
	./scripts/generate_function_analysis.sh

func-analysis-minimal:
	./scripts/generate_function_analysis.sh --minimal-only
```

## Troubleshooting

### Common Issues

**"SGLang SRT directory not found"**
- Ensure you're running from the repository root
- Check that `python/sglang/srt/` exists

**"Python 3.9+ required"**
- Upgrade Python: `brew install python@3.9` or similar
- Some older systems may need explicit `python3.9` command

**"ripgrep not found"**
- Install ripgrep or script will use find+grep (slower but works)
- No action needed if find+grep are available

### Debug Mode
```bash
# Add debug output
bash -x ./scripts/generate_function_analysis.sh --minimal-only
```

## Customization

### Modify Excluded Directories
Edit the `excluded_dirs` array in the script:
```bash
# In generate_detailed_analysis() function
local excluded_dirs=("configs" "connector" ... "models")
```

### Change Output Location
Modify the `OUTPUT_DIR` variable:
```bash
OUTPUT_DIR="${REPO_ROOT}/my_custom_analysis"
```

### Add Custom Processing
The script creates temporary AST extraction scripts in `/tmp/`. You can modify these for custom analysis needs.

## Best Practices

### Regular Updates
- **Daily development**: Use `--minimal-only`
- **Weekly/major changes**: Full regeneration
- **Release preparation**: `--clean` regeneration

### LLM Context Usage
- Use **minimal versions** for token efficiency
- Combine multiple minimal files for broader context
- Search with `rg "pattern" function_analysis/minimal/*_minimal.txt`

### Version Control
- **Commit analysis files** - they're useful for code review
- **Gitignore temporary files** if any leak
- **Track changes** in analysis to spot API evolution

## Support

If you encounter issues:
1. Check the script's colored log output for specific error messages
2. Verify all requirements are met
3. Try the `--clean` option to reset everything
4. Check file permissions on the `scripts/` directory

The script is designed to be robust and provide clear error messages for common issues.
