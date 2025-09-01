# SGLang SRT Function Analysis

This directory contains comprehensive AST-based function analysis for the entire SGLang SRT codebase.

## Overview

- **Total**: 467 Python files with 6,626 functions
- **Complete coverage** of all directories with detailed function signatures
- **Generated on**: August 31, 2025

## Files

### Summary Files
- `srt_complete_directory_summary.txt` - Complete overview with usage examples
- `srt_directories_summary.txt` - Legacy summary (older format)

### Main Analysis
- `srt_functions_index.txt` (528KB) - Main codebase (3,544 functions)
- `root_functions_index.txt` (51KB) - Root-level files (499 functions)

### Directory-Specific Analysis

| Directory | Functions | Size | File |
|-----------|-----------|------|------|
| models/ | 1,649 | 275KB | `models_functions_index.txt` |
| layers/ | 1,529 | 273KB | `layers_functions_index.txt` |
| mem_cache/ | 540 | 67KB | `mem_cache_functions_index.txt` |
| managers/ | 486 | 65KB | `managers_functions_index.txt` |
| entrypoints/ | 265 | 44KB | `entrypoints_functions_index.txt` |
| disaggregation/ | 254 | 33KB | `disaggregation_functions_index.txt` |
| distributed/ | 215 | 29KB | `distributed_functions_index.txt` |
| eplb/ | 182 | 23KB | `eplb_functions_index.txt` |
| configs/ | 121 | 23KB | `configs_functions_index.txt` |
| multimodal/ | 119 | 19KB | `multimodal_functions_index.txt` |
| model_executor/ | 117 | 14KB | `model_executor_functions_index.txt` |
| function_call/ | 113 | 18KB | `function_call_functions_index.txt` |
| model_loader/ | 105 | 16KB | `model_loader_functions_index.txt` |
| lora/ | 104 | 18KB | `lora_functions_index.txt` |
| constrained/ | 103 | 13KB | `constrained_functions_index.txt` |
| speculative/ | 65 | 10KB | `speculative_functions_index.txt` |
| sampling/ | 62 | 8.8KB | `sampling_functions_index.txt` |
| connector/ | 46 | 5.5KB | `connector_functions_index.txt` |
| metrics/ | 20 | 2.7KB | `metrics_functions_index.txt` |
| debug_utils/ | 16 | 1.3KB | `debug_utils_functions_index.txt` |
| tokenizer/ | 10 | 1.3KB | `tokenizer_functions_index.txt` |
| weight_sync/ | 6 | 1.4KB | `weight_sync_functions_index.txt` |

## Usage

```bash
# View functions in a specific directory
cat function_analysis/layers_functions_index.txt

# Preview first few functions
head -50 function_analysis/models_functions_index.txt

# Search for specific functions
rg "forward" function_analysis/layers_functions_index.txt

# Search across all directories
rg "async def" function_analysis/*_functions_index.txt
```

## Function Information

Each index file contains:
- **Function name** and complete signature
- **Parameter types** and default values
- **Return type annotations**
- **Class context** for methods
- **Documentation** (first line of docstring)
- **Line numbers** in source files

## Tools

- `extract_functions_ast.py` - The AST extraction script used to generate these files

## Notes

- Generated using Python's `ast` module for accurate parsing
- Includes both regular functions (`def`) and async functions (`async def`)
- Covers all Python files in the SGLang SRT codebase
- Function signatures preserve original formatting and annotations
