#!/bin/bash
set -e

################################
# SGLang API Documentation
################################

echo "[INFO] Checking dependencies..."

# Check if we're in the docs directory
if [[ ! -f "conf.py" ]]; then
    echo "[ERROR] Please run this script from the docs/ directory"
    exit 1
fi

# Check if pdoc3 is installed
if ! command -v pdoc3 &> /dev/null; then
    echo "[ERROR] pdoc3 is not installed"
    echo "Please install it with: uv pip install pdoc3"
    exit 1
fi

echo "[INFO] Dependencies OK"

# Configuration
OUTPUT_DIR="dev/refs/api"

# Clean existing docs if requested
if [[ "$1" == "--clean" ]]; then
    echo "[INFO] Cleaning existing API docs..."
    rm -rf "${OUTPUT_DIR}"
fi

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

# Generate API documentation with pdoc3
echo "[INFO] Generating API documentation..."
pdoc3 --html --output-dir "${OUTPUT_DIR}" --skip-errors sglang 2>&1

if [[ $? -eq 0 ]]; then
    echo "[INFO] Documentation generated successfully"
    echo "[INFO] Output directory: ${OUTPUT_DIR}/"
    echo ""
    echo "================================"
    echo " Generation Complete!"
    echo "================================"
    echo ""
    echo "API documentation has been generated in: ${OUTPUT_DIR}/"
    echo ""
    echo "To view the documentation:"
    echo "  1. Open ${OUTPUT_DIR}/sglang/index.html in your browser"
    echo "  2. Or run: python3 -m http.server 8000 --directory ${OUTPUT_DIR}"
    echo ""
    echo "Main entry points:"
    echo "  - Module index: ${OUTPUT_DIR}/sglang/index.html"
    echo "  - Language API: ${OUTPUT_DIR}/sglang/lang/index.html"
    echo "  - Runtime API: ${OUTPUT_DIR}/sglang/srt/index.html"
else
    echo "[ERROR] Documentation generation failed"
    exit 1
fi
