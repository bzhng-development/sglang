#!/bin/bash
# =============================================================================
# SGLang SRT Function Analysis Generator
# =============================================================================
#
# This script generates comprehensive AST-based function analysis for the entire
# SGLang SRT codebase, creating both detailed and minimal token-efficient versions.
#
# Author: Generated for SGLang analysis
# Usage: ./scripts/generate_function_analysis.sh [--minimal-only] [--clean]
#
# Requirements:
# - Python 3.9+ (for ast.unparse)
# - ripgrep (rg) - preferred over grep
# - Run from SGLang repository root
#
# Outputs:
# - function_analysis/ - Detailed AST analysis (full metadata)
# - function_analysis/minimal/ - Token-efficient minimal versions
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRT_DIR="${REPO_ROOT}/python/sglang/srt"
OUTPUT_DIR="${REPO_ROOT}/function_analysis"
MINIMAL_DIR="${OUTPUT_DIR}/minimal"
TMP_DIR="${TMPDIR:-/tmp}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Parse arguments
MINIMAL_ONLY=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal-only)
            MINIMAL_ONLY=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --minimal-only    Generate only minimal versions (faster)"
            echo "  --clean          Clean existing output before generating"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validation
validate_environment() {
    log_info "Validating environment..."

    # Check if running from correct directory
    if [[ ! -d "${SRT_DIR}" ]]; then
        log_error "SGLang SRT directory not found: ${SRT_DIR}"
        log_error "Please run this script from the SGLang repository root"
        exit 1
    fi

    # Check Python version
    if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
        log_error "Python 3.9+ required for ast.unparse support"
        exit 1
    fi

    # Check for ripgrep (preferred) or fallback to find+grep
    if ! command -v rg &> /dev/null; then
        log_warn "ripgrep (rg) not found, falling back to find+grep (slower)"
        if ! command -v find &> /dev/null || ! command -v grep &> /dev/null; then
            log_error "Neither ripgrep nor find+grep available"
            exit 1
        fi
    fi

    log_success "Environment validation passed"
}

# Clean existing output
clean_output() {
    if [[ "${CLEAN}" == "true" ]]; then
        log_info "Cleaning existing output..."
        rm -rf "${OUTPUT_DIR}"
    fi
}

# Create output directories
setup_directories() {
    log_info "Setting up output directories..."
    mkdir -p "${OUTPUT_DIR}" "${MINIMAL_DIR}"
}

# Create AST extraction scripts
create_ast_scripts() {
    log_info "Creating AST extraction scripts..."

    # Detailed AST extractor
    cat > "${TMP_DIR}/extract_functions_detailed.py" << 'EOF'
import ast
import sys
from pathlib import Path
import argparse
from typing import List, Dict, Optional, Any

def unparse(node):
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except AttributeError:
        # Fallback for Python < 3.9
        return str(node)

def format_arg(arg: ast.arg) -> str:
    s = arg.arg
    if getattr(arg, "annotation", None):
        ann = unparse(arg.annotation)
        if ann:
            s += f": {ann}"
    return s

def format_arguments(args: ast.arguments) -> str:
    parts = []

    posonly = list(getattr(args, "posonlyargs", [])) or []
    normal = list(getattr(args, "args", [])) or []
    positional = posonly + normal

    defaults = list(getattr(args, "defaults", [])) or []
    n_pos = len(positional)
    n_def = len(defaults)

    for i, a in enumerate(positional):
        s = format_arg(a)
        if i >= (n_pos - n_def):
            d = defaults[i - (n_pos - n_def)]
            if d is not None:
                s += f" = {unparse(d)}"
        parts.append(s)

    if posonly:
        parts.append("/")

    if args.vararg:
        parts.append("*" + format_arg(args.vararg))

    if not args.vararg and args.kwonlyargs:
        parts.append("*")

    for a, d in zip(args.kwonlyargs or [], args.kw_defaults or []):
        s = format_arg(a)
        if d is not None:
            s += f" = {unparse(d)}"
        parts.append(s)

    if args.kwarg:
        parts.append("**" + format_arg(args.kwarg))

    return ", ".join(parts)

def summarize_docstring(doc: Optional[str]) -> Optional[str]:
    if not doc:
        return None
    lines = [ln.strip() for ln in doc.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    s = lines[0]
    if len(s) > 240:
        s = s[:240] + "..."
    return s

class FunctionCollector(ast.NodeVisitor):
    def __init__(self):
        self.class_stack: List[str] = []
        self.collected: List[Dict[str, Any]] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def _collect_func(self, node, is_async: bool):
        func = {
            "name": node.name,
            "is_async": is_async,
            "parameters": format_arguments(node.args),
            "return": unparse(node.returns) if getattr(node, "returns", None) else None,
            "doc": summarize_docstring(ast.get_docstring(node, clean=True)),
            "class_context": ".".join(self.class_stack) if self.class_stack else None,
            "lineno": getattr(node, "lineno", None),
        }
        self.collected.append(func)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._collect_func(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._collect_func(node, is_async=True)

def scan_file(path: Path) -> List[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        collector = FunctionCollector()
        collector.visit(tree)
        return collector.collected
    except Exception as e:
        print(f"[WARN] Could not process {path}: {e}", file=sys.stderr)
        return []

def main():
    ap = argparse.ArgumentParser(description="Extract detailed function information via AST.")
    ap.add_argument("--root", default=".", help="Root directory to scan")
    ap.add_argument("--exclude", nargs="*", default=[], help="Directory names to exclude")
    ap.add_argument("--files-from", help="File containing list of files to process")
    ap.add_argument("--output", default="-", help="Output path")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    excluded = set(args.exclude or [])

    if args.files_from:
        files = []
        with open(args.files_from, "r") as f:
            for line in f:
                line = line.strip()
                if line and line.endswith('.py'):
                    p = root / line if not line.startswith('/') else Path(line)
                    if p.exists():
                        files.append(p)
    else:
        files = []
        for p in root.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            parts = p.relative_to(root).parts
            if any(part in excluded for part in parts):
                continue
            files.append(p)

    files.sort()

    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")

    try:
        print("AST Function Index", file=out)
        print(f"Root: {root}", file=out)
        print("Excluded directories: " + ", ".join(sorted(excluded)), file=out)
        print("", file=out)

        for fp in files:
            rel = fp.relative_to(root)
            functions = scan_file(fp)
            functions.sort(key=lambda f: (f.get("lineno") or 0, f["name"]))
            print(f"File: {rel}", file=out)
            if not functions:
                print("  (no function definitions found)", file=out)
                continue
            for f in functions:
                name = f["name"]
                params = f["parameters"]
                ret = f.get("return")
                cls = f.get("class_context")
                doc = f.get("doc") or ""
                sig = f"({params})"
                print(f"  - name: {name}", file=out)
                print(f"    signature: {sig}", file=out)
                if ret:
                    print(f"    return: {ret}", file=out)
                if cls:
                    print(f"    class: {cls}", file=out)
                if doc:
                    print(f"    doc: {doc}", file=out)
            print("", file=out)
    finally:
        if out != sys.stdout:
            out.close()

if __name__ == "__main__":
    main()
EOF

    # Minimal AST extractor
    cat > "${TMP_DIR}/extract_functions_minimal.py" << 'EOF'
import ast
from pathlib import Path
import sys

class MinimalCollector(ast.NodeVisitor):
    def __init__(self):
        self.class_stack = []
        self.funcs = []

    def visit_ClassDef(self, node):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        args = [a.arg for a in node.args.args]
        if node.args.vararg:
            args.append('*' + node.args.vararg.arg)
        for a in getattr(node.args, 'kwonlyargs', []):
            args.append(a.arg)
        if node.args.kwarg:
            args.append('**' + node.args.kwarg.arg)

        cls = '.'.join(self.class_stack) + '.' if self.class_stack else ''
        self.funcs.append(cls + node.name + '(' + ','.join(args) + ')')
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        args = [a.arg for a in node.args.args]
        if node.args.vararg:
            args.append('*' + node.args.vararg.arg)
        for a in getattr(node.args, 'kwonlyargs', []):
            args.append(a.arg)
        if node.args.kwarg:
            args.append('**' + node.args.kwarg.arg)

        cls = '.'.join(self.class_stack) + '.' if self.class_stack else ''
        self.funcs.append('async ' + cls + node.name + '(' + ','.join(args) + ')')
        self.generic_visit(node)

def process_directory(dir_path, output_file, base_path):
    root = Path(dir_path)
    if not root.exists():
        return 0

    with open(output_file, 'w') as f:
        for py_file in sorted(root.rglob('*.py')):
            if '__pycache__' in str(py_file):
                continue
            try:
                tree = ast.parse(py_file.read_text(encoding='utf-8'))
                collector = MinimalCollector()
                collector.visit(tree)
                if collector.funcs:
                    rel_path = py_file.relative_to(Path(base_path))
                    print(str(rel_path) + ':', file=f)
                    for func in sorted(collector.funcs):
                        print('  ' + func, file=f)
            except:
                pass

    return Path(output_file).stat().st_size if Path(output_file).exists() else 0

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 extract_functions_minimal.py <dir_path> <output_file> <base_path>")
        sys.exit(1)

    dir_path, output_file, base_path = sys.argv[1:4]
    size = process_directory(dir_path, output_file, base_path)
    print(f"Generated: {size} bytes")
EOF

    log_success "AST extraction scripts created"
}

# Generate file lists using ripgrep or find
generate_file_lists() {
    local target_dir="$1"
    local exclude_args="$2"
    local output_file="$3"

    if command -v rg &> /dev/null; then
        # Use ripgrep (faster)
        eval "rg --files -t py ${exclude_args} \"${target_dir}/\" > \"${output_file}\" 2>/dev/null || true"
    else
        # Fallback to find+grep
        local exclude_pattern=""
        if [[ -n "${exclude_args}" ]]; then
            # Convert rg exclude args to find exclude pattern
            exclude_pattern=$(echo "${exclude_args}" | sed "s/-g '!\([^/]*\)\/\*\*'/-not -path '*\/\1\/*'/g")
        fi
        eval "find \"${target_dir}\" -name '*.py' -type f ${exclude_pattern} > \"${output_file}\" 2>/dev/null || true"
    fi
}

# Generate detailed analysis
generate_detailed_analysis() {
    log_info "Generating detailed AST analysis..."

    # Directories to exclude from main analysis
    local excluded_dirs=("configs" "connector" "constrained" "debug_utils" "entrypoints" "function_call" "lora" "mem_cache" "metrics" "model_loader" "models")
    local all_dirs=("configs" "connector" "constrained" "debug_utils" "disaggregation" "distributed" "entrypoints" "eplb" "function_call" "layers" "lora" "managers" "mem_cache" "metrics" "model_executor" "model_loader" "models" "multimodal" "sampling" "speculative" "tokenizer" "weight_sync")

    # Generate main SRT analysis (excluding specified directories)
    local rg_excludes=""
    for dir in "${excluded_dirs[@]}"; do
        rg_excludes+=" -g '!${dir}/**'"
    done

    generate_file_lists "${SRT_DIR}" "${rg_excludes}" "${TMP_DIR}/srt_files.txt"

    python3 "${TMP_DIR}/extract_functions_detailed.py" \
        --root "${SRT_DIR}" \
        --files-from "${TMP_DIR}/srt_files.txt" \
        --output "${OUTPUT_DIR}/srt_functions_index.txt"

    local srt_size=$(wc -c < "${OUTPUT_DIR}/srt_functions_index.txt")
    log_info "Main SRT analysis: ${srt_size} bytes"

    # Generate individual directory analyses
    for dir in "${all_dirs[@]}"; do
        if [[ -d "${SRT_DIR}/${dir}" ]]; then
            generate_file_lists "${SRT_DIR}/${dir}" "" "${TMP_DIR}/${dir}_files.txt"

            if [[ -s "${TMP_DIR}/${dir}_files.txt" ]]; then
                python3 "${TMP_DIR}/extract_functions_detailed.py" \
                    --root "${SRT_DIR}" \
                    --files-from "${TMP_DIR}/${dir}_files.txt" \
                    --output "${OUTPUT_DIR}/${dir}_functions_index.txt"

                local dir_size=$(wc -c < "${OUTPUT_DIR}/${dir}_functions_index.txt")
                log_info "Directory ${dir}: ${dir_size} bytes"
            fi
        fi
    done

    # Generate root-level analysis
    find "${SRT_DIR}" -maxdepth 1 -name "*.py" -type f > "${TMP_DIR}/root_files.txt"
    if [[ -s "${TMP_DIR}/root_files.txt" ]]; then
        python3 "${TMP_DIR}/extract_functions_detailed.py" \
            --root "${SRT_DIR}" \
            --files-from "${TMP_DIR}/root_files.txt" \
            --output "${OUTPUT_DIR}/root_functions_index.txt"

        local root_size=$(wc -c < "${OUTPUT_DIR}/root_functions_index.txt")
        log_info "Root level: ${root_size} bytes"
    fi
}

# Generate minimal analysis
generate_minimal_analysis() {
    log_info "Generating minimal token-efficient analysis..."

    local all_dirs=("configs" "connector" "constrained" "debug_utils" "disaggregation" "distributed" "entrypoints" "eplb" "function_call" "layers" "lora" "managers" "mem_cache" "metrics" "model_executor" "model_loader" "models" "multimodal" "sampling" "speculative" "tokenizer" "weight_sync")

    # Generate minimal versions for all directories
    for dir in "${all_dirs[@]}"; do
        if [[ -d "${SRT_DIR}/${dir}" ]]; then
            python3 "${TMP_DIR}/extract_functions_minimal.py" \
                "${SRT_DIR}/${dir}" \
                "${MINIMAL_DIR}/${dir}_minimal.txt" \
                "${SRT_DIR}" > /dev/null 2>&1

            local size=$(wc -c < "${MINIMAL_DIR}/${dir}_minimal.txt" 2>/dev/null || echo 0)
            if [[ $size -gt 0 ]]; then
                log_info "Minimal ${dir}: ${size} bytes"
            fi
        fi
    done

    # Generate root-level minimal
    python3 -c "
import ast
from pathlib import Path

class MinimalCollector(ast.NodeVisitor):
    def __init__(self):
        self.class_stack = []
        self.funcs = []

    def visit_ClassDef(self, node):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        args = [a.arg for a in node.args.args]
        if node.args.vararg: args.append('*' + node.args.vararg.arg)
        for a in getattr(node.args, 'kwonlyargs', []): args.append(a.arg)
        if node.args.kwarg: args.append('**' + node.args.kwarg.arg)
        cls = '.'.join(self.class_stack) + '.' if self.class_stack else ''
        self.funcs.append(cls + node.name + '(' + ','.join(args) + ')')
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        args = [a.arg for a in node.args.args]
        if node.args.vararg: args.append('*' + node.args.vararg.arg)
        for a in getattr(node.args, 'kwonlyargs', []): args.append(a.arg)
        if node.args.kwarg: args.append('**' + node.args.kwarg.arg)
        cls = '.'.join(self.class_stack) + '.' if self.class_stack else ''
        self.funcs.append('async ' + cls + node.name + '(' + ','.join(args) + ')')
        self.generic_visit(node)

with open('${MINIMAL_DIR}/root_minimal.txt', 'w') as f:
    for py_file in sorted(Path('${SRT_DIR}').glob('*.py')):
        try:
            tree = ast.parse(py_file.read_text(encoding='utf-8'))
            collector = MinimalCollector()
            collector.visit(tree)
            if collector.funcs:
                print(py_file.name + ':', file=f)
                for func in sorted(collector.funcs):
                    print('  ' + func, file=f)
        except: pass
"

    # Generate main SRT minimal (excluding specified directories)
    python3 -c "
import ast
from pathlib import Path

class MinimalCollector(ast.NodeVisitor):
    def __init__(self):
        self.class_stack = []
        self.funcs = []

    def visit_ClassDef(self, node):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        args = [a.arg for a in node.args.args]
        if node.args.vararg: args.append('*' + node.args.vararg.arg)
        for a in getattr(node.args, 'kwonlyargs', []): args.append(a.arg)
        if node.args.kwarg: args.append('**' + node.args.kwarg.arg)
        cls = '.'.join(self.class_stack) + '.' if self.class_stack else ''
        self.funcs.append(cls + node.name + '(' + ','.join(args) + ')')
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        args = [a.arg for a in node.args.args]
        if node.args.vararg: args.append('*' + node.args.vararg.arg)
        for a in getattr(node.args, 'kwonlyargs', []): args.append(a.arg)
        if node.args.kwarg: args.append('**' + node.args.kwarg.arg)
        cls = '.'.join(self.class_stack) + '.' if self.class_stack else ''
        self.funcs.append('async ' + cls + node.name + '(' + ','.join(args) + ')')
        self.generic_visit(node)

excluded_dirs = {'configs', 'connector', 'constrained', 'debug_utils', 'entrypoints', 'function_call', 'lora', 'mem_cache', 'metrics', 'model_loader', 'models'}

with open('${MINIMAL_DIR}/srt_minimal.txt', 'w') as f:
    for py_file in sorted(Path('${SRT_DIR}').rglob('*.py')):
        parts = py_file.relative_to(Path('${SRT_DIR}')).parts
        if any(part in excluded_dirs for part in parts):
            continue
        if '__pycache__' in str(py_file):
            continue

        try:
            tree = ast.parse(py_file.read_text(encoding='utf-8'))
            collector = MinimalCollector()
            collector.visit(tree)
            if collector.funcs:
                rel_path = py_file.relative_to(Path('${SRT_DIR}'))
                print(str(rel_path) + ':', file=f)
                for func in sorted(collector.funcs):
                    print('  ' + func, file=f)
        except: pass
"
}

# Generate summary reports
generate_summaries() {
    log_info "Generating summary reports..."

    # Count files and functions
    local total_files=0
    local total_functions=0

    # Create detailed summary
    cat > "${OUTPUT_DIR}/srt_complete_directory_summary.txt" << EOF
SGLang SRT Function Analysis Summary
===================================

Generated: $(date)
Script: $0

OVERVIEW:
---------
Complete AST-based function analysis for SGLang SRT codebase.

MAIN CODEBASE (non-excluded directories):
=========================================
File: ./function_analysis/srt_functions_index.txt
$(wc -l < "${SRT_DIR}"/../../../function_analysis/srt_functions_index.txt) lines

USAGE:
======
# View functions in specific directory
cat function_analysis/layers_functions_index.txt

# Search for functions
rg "function_name" function_analysis/*_functions_index.txt

# Minimal token-efficient versions
cat function_analysis/minimal/layers_minimal.txt

REGENERATION:
=============
Run this script regularly to update analysis:
./scripts/generate_function_analysis.sh

For minimal-only (faster):
./scripts/generate_function_analysis.sh --minimal-only

For clean regeneration:
./scripts/generate_function_analysis.sh --clean
EOF

    # Create minimal README
    cat > "${MINIMAL_DIR}/README.md" << EOF
# SGLang SRT Minimal Function Index

Ultra-compact function listings optimized for token efficiency.

## Format

Each entry contains only:
- **File path**
- **Function name with basic parameters** (no types, no docs, no metadata)

## Usage

Perfect for LLM context when you need just function names and signatures:

\`\`\`bash
# View main SRT functions (token-efficient)
cat srt_minimal.txt

# View specific directory
cat layers_minimal.txt

# Search across all
rg "forward" *_minimal.txt
\`\`\`

## Regeneration

\`\`\`bash
# From repository root
./scripts/generate_function_analysis.sh --minimal-only
\`\`\`

Generated: $(date)
EOF

    log_success "Summary reports generated"
}

# Cleanup temporary files
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f "${TMP_DIR}"/extract_functions_*.py
    rm -f "${TMP_DIR}"/*_files.txt
}

# Display results
show_results() {
    log_success "Function analysis generation complete!"
    echo ""
    echo "📁 Output Locations:"
    echo "==================="
    echo "• Detailed analysis: ${OUTPUT_DIR}/"
    echo "• Minimal analysis:  ${MINIMAL_DIR}/"
    echo ""

    if [[ "${MINIMAL_ONLY}" == "false" ]]; then
        echo "📊 Detailed Files:"
        ls -lah "${OUTPUT_DIR}"/*_functions_index.txt 2>/dev/null | head -5 || true
        echo "..."
        echo ""
    fi

    echo "🔥 Minimal Files (token-efficient):"
    ls -lah "${MINIMAL_DIR}"/*_minimal.txt 2>/dev/null | head -5 || true
    echo "..."
    echo ""
    echo "📋 Quick Start:"
    echo "• cat function_analysis/minimal/srt_minimal.txt"
    echo "• cat function_analysis/minimal/layers_minimal.txt"
    echo "• rg 'forward' function_analysis/minimal/*_minimal.txt"
    echo ""
    echo "🔄 To regenerate: ./scripts/generate_function_analysis.sh"
}

# Main execution
main() {
    log_info "Starting SGLang SRT function analysis generation..."

    # Change to repository root
    cd "${REPO_ROOT}"

    # Execute pipeline
    validate_environment
    clean_output
    setup_directories
    create_ast_scripts

    if [[ "${MINIMAL_ONLY}" == "false" ]]; then
        generate_detailed_analysis
    fi

    generate_minimal_analysis
    generate_summaries
    cleanup
    show_results

    log_success "All done! 🎉"
}

# Error handling
trap cleanup EXIT

# Run main function
main "$@"
