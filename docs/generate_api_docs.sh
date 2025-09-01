#!/bin/bash
#
# SGLang API Documentation Generator
#
# This script generates comprehensive API documentation using pdoc
# Usage: ./generate_api_docs.sh [--clean] [--serve]
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OUTPUT_DIR="dev/refs/api"
PYTHON_MODULE="sglang"

# Functions
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE} SGLang API Documentation${NC}"
    echo -e "${BLUE}================================${NC}"
    echo
}

print_step() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    print_step "Checking dependencies..."

    # Check if we're in the docs directory
    if [[ ! -f "conf.py" ]]; then
        print_error "Please run this script from the docs/ directory"
        exit 1
    fi

    # Check if sglang is installed
    if ! python3 -c "import sglang" 2>/dev/null; then
        print_warning "SGLang not found. Installing from source..."
        pip install -e ../python/ || {
            print_error "Failed to install SGLang from source"
            exit 1
        }
    fi

    # Check if pdoc is installed
    if ! command -v pdoc &> /dev/null; then
        print_warning "pdoc not found. Installing..."
        pip install pdoc || {
            print_error "Failed to install pdoc"
            exit 1
        }
    fi

    print_step "Dependencies OK"
}

clean_output() {
    if [[ -d "$OUTPUT_DIR" ]]; then
        print_step "Cleaning existing documentation..."
        rm -rf "$OUTPUT_DIR"
    fi
}

generate_docs() {
    print_step "Generating API documentation..."

    # Create output directory
    mkdir -p "$(dirname "$OUTPUT_DIR")"

    # Generate documentation
    pdoc "$PYTHON_MODULE" -o "$OUTPUT_DIR" || {
        print_error "Failed to generate documentation"
        exit 1
    }

    print_step "Documentation generated successfully"
    print_step "Output directory: $OUTPUT_DIR/"
}

show_summary() {
    echo
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN} Generation Complete!${NC}"
    echo -e "${GREEN}================================${NC}"
    echo
    echo "API documentation has been generated in: $OUTPUT_DIR/"
    echo
    echo "To view the documentation:"
    echo "  1. Open $OUTPUT_DIR/sglang/index.html in your browser"
    echo "  2. Or run: python3 -m http.server 8000 --directory $OUTPUT_DIR"
    echo
    echo "Main entry points:"
    echo "  - Module index: $OUTPUT_DIR/sglang/index.html"
    echo "  - Language API: $OUTPUT_DIR/sglang/lang/index.html"
    echo "  - Runtime API: $OUTPUT_DIR/sglang/srt/index.html"
    echo
}

serve_docs() {
    if [[ -d "$OUTPUT_DIR" ]]; then
        print_step "Serving documentation at http://localhost:8000"
        print_step "Press Ctrl+C to stop the server"
        echo
        cd "$OUTPUT_DIR"
        python3 -m http.server 8000
    else
        print_error "No documentation found. Run without --serve first."
        exit 1
    fi
}

# Main execution
main() {
    local clean_flag=false
    local serve_flag=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean)
                clean_flag=true
                shift
                ;;
            --serve)
                serve_flag=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [--clean] [--serve]"
                echo ""
                echo "Options:"
                echo "  --clean  Clean existing documentation before generation"
                echo "  --serve  Serve documentation after generation"
                echo "  --help   Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    print_header

    # Only serve if flag is set and docs exist
    if [[ "$serve_flag" == true ]] && [[ "$clean_flag" == false ]] && [[ -d "$OUTPUT_DIR" ]]; then
        serve_docs
        exit 0
    fi

    # Normal generation workflow
    check_dependencies

    if [[ "$clean_flag" == true ]]; then
        clean_output
    fi

    generate_docs
    show_summary

    # Serve if requested
    if [[ "$serve_flag" == true ]]; then
        echo
        serve_docs
    fi
}

# Run main function with all arguments
main "$@"
