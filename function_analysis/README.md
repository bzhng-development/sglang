# SGLang SRT Function Analysis

This directory contains comprehensive AST-based function analysis for the entire SGLang SRT codebase, automatically generated and regularly maintained.

## 🚀 Quick Start

```bash
# Generate/update analysis (from repository root)
./scripts/generate_function_analysis.sh --minimal-only

# View token-efficient minimal versions
cat function_analysis/minimal/srt_minimal.txt
cat function_analysis/minimal/layers_minimal.txt

# Search across all functions
rg "forward" function_analysis/minimal/*_minimal.txt
```

## 📊 What's Included

### **Detailed Analysis** (`function_analysis/*.txt`)
Complete AST metadata with:
- Full function signatures with type annotations
- Return type information
- Class context for methods
- Documentation summaries
- Line numbers
- **Total**: ~1.5MB, 6,626 functions across 467 files

### **Minimal Analysis** (`function_analysis/minimal/*.txt`)
Token-efficient format with:
- Function names + basic parameters only
- No types, docs, or metadata
- **56% smaller** - perfect for LLM context
- **Total**: ~679KB

## 📁 Directory Coverage

**All SGLang SRT directories analyzed:**
- Core: `layers/` (1,529 functions), `managers/` (486), `model_executor/` (117)
- I/O: `entrypoints/` (265), `multimodal/` (119), `sampling/` (62)
- Advanced: `disaggregation/` (254), `distributed/` (215), `speculative/` (65)
- ML: `models/` (1,649), `configs/` (121), `lora/` (104)
- Infrastructure: `mem_cache/` (540), `metrics/` (20), `tokenizer/` (10)
- Plus root-level files (499 functions)

## 🔄 Regular Generation

### Automated Script
```bash
./scripts/generate_function_analysis.sh [OPTIONS]
```

**Options:**
- `--minimal-only` - Generate only compact versions (faster, recommended for regular use)
- `--clean` - Clean existing output and regenerate everything
- `--help` - Show usage information

### Usage Patterns
```bash
# Daily development (fast)
./scripts/generate_function_analysis.sh --minimal-only

# After major changes (comprehensive)
./scripts/generate_function_analysis.sh --clean

# Basic update
./scripts/generate_function_analysis.sh
```

## 📋 File Reference

### Main Files
| File | Description | Size | Use Case |
|------|-------------|------|----------|
| `srt_functions_index.txt` | Main codebase (detailed) | 528KB | Complete analysis |
| `minimal/srt_minimal.txt` | Main codebase (minimal) | 240KB | LLM context |
| `minimal/layers_minimal.txt` | Layers directory | 125KB | Neural network layers |
| `minimal/models_minimal.txt` | Models directory | 115KB | Model implementations |

### Directory-Specific Files
- `<directory>_functions_index.txt` - Detailed analysis per directory
- `minimal/<directory>_minimal.txt` - Minimal analysis per directory
- `root_functions_index.txt` / `minimal/root_minimal.txt` - Root-level files

## 🎯 Usage Examples

### LLM Context (Token-Efficient)
```bash
# Get overview of neural network layers
head -50 function_analysis/minimal/layers_minimal.txt

# Find all forward passes
rg "\.forward\(" function_analysis/minimal/*_minimal.txt

# Examine model architectures
cat function_analysis/minimal/models_minimal.txt | grep -A5 "__init__"
```

### Detailed Analysis
```bash
# Complete function signatures with types
head -100 function_analysis/layers_functions_index.txt

# Search with full metadata
rg "return.*torch.Tensor" function_analysis/*_functions_index.txt
```

### Development Workflow
```bash
# After modifying SRT code
./scripts/generate_function_analysis.sh --minimal-only

# Check what functions you changed
git diff function_analysis/minimal/

# Use in code review or documentation
cat function_analysis/minimal/relevant_directory_minimal.txt
```

## 🔧 Requirements

- **Python 3.9+** (for AST parsing)
- **ripgrep (rg)** - preferred, falls back to find+grep
- Run from **SGLang repository root**

## 📖 Documentation

- [`GENERATION_GUIDE.md`](GENERATION_GUIDE.md) - Comprehensive generation documentation
- [`minimal/README.md`](minimal/README.md) - Minimal format specific guide
- `srt_complete_directory_summary.txt` - Overview report with statistics

## ⚡ Performance

- **Generation time**: 10-90 seconds depending on options
- **Minimal vs Detailed**: 56% size reduction (679KB vs 1.5MB)
- **Coverage**: 100% of SGLang SRT Python codebase

## 🎨 Integration

The script integrates easily with:
- **Git hooks** for automatic updates
- **CI/CD pipelines** for continuous analysis
- **Development workflows** for regular updates
- **LLM tools** for code understanding

---

**Generated**: Automatically maintained by `scripts/generate_function_analysis.sh`
**Last updated**: Run the generation script to update this analysis
**Coverage**: Complete SGLang SRT codebase (python/sglang/srt/)

For detailed generation instructions, see [`GENERATION_GUIDE.md`](GENERATION_GUIDE.md).
