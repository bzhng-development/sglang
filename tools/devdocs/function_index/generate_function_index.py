#!/usr/bin/env python3
"""
Unified function index generator for SGLang codebase.
Extracts function signatures and metadata via AST parsing.
"""

import argparse
import ast
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


class FunctionExtractor(ast.NodeVisitor):
    """AST visitor to extract function definitions with metadata."""

    def __init__(self, filepath: Path, module_prefix: str = ""):
        self.filepath = filepath
        self.module_prefix = module_prefix
        self.functions = []
        self.current_class = None
        self.source_lines = []

    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node):
        self._process_function(node, is_async=True)

    def _process_function(self, node, is_async=False):
        # Skip private functions unless they're special methods
        if node.name.startswith("_") and not node.name.startswith("__"):
            return

        # Build qualified name
        qualname = node.name
        if self.current_class:
            qualname = f"{self.current_class}.{node.name}"

        # Extract signature
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                try:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                except:
                    pass
            args.append(arg_str)

        signature = f"{'async ' if is_async else ''}def {node.name}({', '.join(args)})"

        # Extract return type
        return_type = None
        if node.returns:
            try:
                return_type = ast.unparse(node.returns)
            except:
                pass

        # Extract docstring
        docstring = ast.get_docstring(node)
        doc_summary = None
        if docstring:
            lines = docstring.split("\n")
            doc_summary = lines[0][:200] if lines else None

        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            try:
                decorators.append(ast.unparse(decorator))
            except:
                pass

        # Determine visibility
        visibility = "private" if node.name.startswith("_") else "public"

        self.functions.append(
            {
                "module": self.module_prefix,
                "file": str(self.filepath),
                "line": node.lineno,
                "qualname": qualname,
                "signature": signature,
                "return_type": return_type,
                "doc_summary": doc_summary,
                "doc_full": docstring,
                "decorators": decorators,
                "visibility": visibility,
                "is_async": is_async,
                "is_method": self.current_class is not None,
                "class": self.current_class,
            }
        )


def extract_functions_from_file(filepath: Path, module_prefix: str = "") -> List[Dict]:
    """Extract all functions from a Python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        extractor = FunctionExtractor(filepath, module_prefix)
        extractor.visit(tree)
        return extractor.functions
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return []


def compute_module_from_path(filepath: Path, root: Path) -> str:
    """Compute Python module name from file path."""
    try:
        relative = filepath.relative_to(root)
        parts = list(relative.parts[:-1]) + [relative.stem]
        return ".".join(parts)
    except:
        return ""


def enrich_with_urls(
    functions: List[Dict], repo_url: str = "https://github.com/sgl-project/sglang"
) -> List[Dict]:
    """Add source URLs and pdoc anchors to function entries."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except:
        commit = "main"

    for func in functions:
        # Add source URL
        filepath = func["file"]
        if filepath.startswith("/"):
            # Make relative from repo root
            if "/sglang/" in filepath:
                filepath = filepath.split("/sglang/", 1)[1]
                filepath = filepath if not filepath.startswith("sglang/") else filepath

        func["source_url"] = f"{repo_url}/blob/{commit}/{filepath}#L{func['line']}"

        # Add pdoc anchor
        module = func.get("module", "")
        qualname = func.get("qualname", "")
        if module and qualname:
            func["pdoc_anchor"] = f"{module}.{qualname}"
            func["pdoc_url"] = f"docs/dev/refs/api/{module}.html#{module}.{qualname}"

    return functions


def generate_minimal_format(functions: List[Dict]) -> List[str]:
    """Generate token-efficient minimal format with types."""
    lines = []
    current_file = None

    for func in sorted(functions, key=lambda x: (x["file"], x["line"])):
        if func["file"] != current_file:
            current_file = func["file"]
            lines.append(f"\n# {current_file}")

        # Compact format: name, params with types, and return type
        sig = func["signature"]
        # Extract function name and parameters
        if "(" in sig:
            name_part = sig.split("(")[0].replace("async def ", "").replace("def ", "")
            params_part = sig.split("(")[1].rstrip(")")

            # Keep type annotations but make them concise
            params = []
            for param in params_part.split(","):
                param = param.strip()
                if param and param != "self":
                    # Keep the full parameter with type annotation
                    params.append(param)

            # Build the function signature with types
            func_line = f"{name_part}({', '.join(params)})"

            # Add return type if present
            if func.get("return_type"):
                func_line += f" -> {func['return_type']}"

            # Add class prefix if it's a method
            if func.get("class"):
                lines.append(f"  {func['class']}.{func_line}")
            else:
                lines.append(f"{func_line}")

    return lines


def generate_human_readable_format(
    functions: List[Dict], module_name: str
) -> List[str]:
    """Generate human-readable format with better formatting."""
    lines = []
    lines.append(f"=" * 80)
    lines.append(f"FUNCTION INDEX: {module_name}")
    lines.append(f"=" * 80)
    lines.append(f"Total Functions: {len(functions)}")
    lines.append(f"Documented: {sum(1 for f in functions if f.get('doc_summary'))}")
    lines.append("")

    # Group by file
    by_file = defaultdict(list)
    for func in functions:
        by_file[func["file"]].append(func)

    for filepath in sorted(by_file.keys()):
        funcs = by_file[filepath]
        lines.append(f"\n{'='*60}")
        lines.append(f"FILE: {filepath}")
        lines.append(f"Functions: {len(funcs)}")
        lines.append(f"{'='*60}\n")

        # Group by class
        class_funcs = defaultdict(list)
        module_funcs = []

        for func in sorted(funcs, key=lambda x: x["line"]):
            if func.get("class"):
                class_funcs[func["class"]].append(func)
            else:
                module_funcs.append(func)

        # Module-level functions first
        if module_funcs:
            lines.append("MODULE FUNCTIONS:")
            lines.append("-" * 40)
            for func in module_funcs:
                sig = func["signature"]
                # Clean up signature for readability
                if len(sig) > 80:
                    # Break long signatures
                    sig = sig.replace(", ", ",\n        ")

                lines.append(f"  L{func['line']:4d}: {sig}")
                if func.get("return_type"):
                    lines.append(f"         → {func['return_type']}")
                if func.get("doc_full"):
                    # Use full docstring, properly formatted for multi-line
                    doc_lines = func["doc_full"].strip().split("\n")
                    lines.append(f"         📝 {doc_lines[0]}")
                    for doc_line in doc_lines[1:]:
                        if doc_line.strip():  # Skip empty lines
                            lines.append(f"            {doc_line.strip()}")
                if func.get("decorators"):
                    for dec in func["decorators"]:
                        lines.append(f"         @{dec}")
                lines.append("")

        # Class methods
        for class_name in sorted(class_funcs.keys()):
            lines.append(f"\nCLASS: {class_name}")
            lines.append("-" * 40)
            for func in class_funcs[class_name]:
                method_name = func["qualname"].split(".")[-1]
                sig = func["signature"]

                lines.append(f"  L{func['line']:4d}: {method_name}({sig.split('(')[1]}")
                if func.get("return_type"):
                    lines.append(f"         → {func['return_type']}")
                if func.get("doc_full"):
                    # Use full docstring, properly formatted for multi-line
                    doc_lines = func["doc_full"].strip().split("\n")
                    lines.append(f"         📝 {doc_lines[0]}")
                    for doc_line in doc_lines[1:]:
                        if doc_line.strip():  # Skip empty lines
                            lines.append(f"            {doc_line.strip()}")
                lines.append("")

    return lines


def compute_coverage_metrics(functions: List[Dict], root: Path) -> Dict:
    """Compute coverage metrics for the codebase."""
    # Count functions with docs
    with_docs = sum(1 for f in functions if f.get("doc_summary"))

    # Count by visibility
    public_count = sum(1 for f in functions if f["visibility"] == "public")
    private_count = sum(1 for f in functions if f["visibility"] == "private")

    # Count by type
    method_count = sum(1 for f in functions if f.get("is_method"))
    async_count = sum(1 for f in functions if f.get("is_async"))

    # Files analyzed
    files = set(f["file"] for f in functions)

    return {
        "total_functions": len(functions),
        "documented": with_docs,
        "undocumented": len(functions) - with_docs,
        "documentation_rate": (
            f"{(with_docs/len(functions)*100):.1f}%" if functions else "0%"
        ),
        "public_functions": public_count,
        "private_functions": private_count,
        "methods": method_count,
        "async_functions": async_count,
        "files_analyzed": len(files),
        "functions_per_file": f"{len(functions)/len(files):.1f}" if files else "0",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate function index for SGLang codebase"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("python/sglang"),
        help="Root directory to scan",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/sglang/llm-txt-ref"),
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--format",
        choices=["detailed", "minimal", "both"],
        default="both",
        help="Output format",
    )
    parser.add_argument(
        "--modules",
        type=str,
        default="",
        help="Comma-separated module prefixes to limit scope",
    )
    parser.add_argument(
        "--per-directory",
        action="store_true",
        help="Generate separate index files per directory",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.out.mkdir(parents=True, exist_ok=True)

    # Collect all Python files
    print(f"Scanning {args.root}...")
    py_files = list(args.root.rglob("*.py"))
    print(f"Found {len(py_files)} Python files")

    # Group files by directory for per-directory output
    by_directory = defaultdict(list)
    all_functions = []

    for filepath in py_files:
        module = compute_module_from_path(filepath, args.root.parent)

        # Filter by module prefix if specified
        if args.modules:
            prefixes = [p.strip() for p in args.modules.split(",")]
            if not any(module.startswith(p) for p in prefixes):
                continue

        functions = extract_functions_from_file(filepath, module)
        all_functions.extend(functions)

        # Group by directory
        dir_path = filepath.parent
        dir_name = dir_path.name if dir_path != args.root else "root"
        by_directory[dir_name].extend(functions)

    print(f"Extracted {len(all_functions)} functions")

    # Enrich with URLs
    all_functions = enrich_with_urls(all_functions)

    # Generate main outputs
    if args.format in ["detailed", "both"]:
        output_file = args.out / "function-index.json"
        with open(output_file, "w") as f:
            json.dump(all_functions, f, indent=2)
        print(f"Wrote detailed index to {output_file}")

    if args.format in ["minimal", "both"]:
        # Minimal JSON (subset of fields)
        minimal_functions = [
            {
                k: v
                for k, v in func.items()
                if k in ["module", "file", "line", "qualname", "signature"]
            }
            for func in all_functions
        ]
        output_file = args.out / "function-index-min.json"
        with open(output_file, "w") as f:
            json.dump(minimal_functions, f, indent=2)
        print(f"Wrote minimal JSON to {output_file}")

        # Ultra-minimal text format
        minimal_lines = generate_minimal_format(all_functions)
        output_file = args.out / "function-index-min.txt"
        with open(output_file, "w") as f:
            f.write("\n".join(minimal_lines))
        print(f"Wrote minimal text to {output_file}")

        # Human-readable format
        readable_lines = generate_human_readable_format(
            all_functions, "SGLang Complete Codebase"
        )
        output_file = args.out / "function-index-readable.txt"
        with open(output_file, "w") as f:
            f.write("\n".join(readable_lines))
        print(f"Wrote human-readable index to {output_file}")

    # Generate per-directory indexes if requested
    if args.per_directory:
        for dir_name, funcs in by_directory.items():
            if not funcs:
                continue

            # Enrich directory-specific functions
            funcs = enrich_with_urls(funcs)

            # Create directory for per-directory outputs
            dir_out = args.out / "by_directory"
            dir_out.mkdir(exist_ok=True)

            # Minimal text for this directory
            minimal_lines = generate_minimal_format(funcs)
            output_file = dir_out / f"{dir_name}_minimal.txt"
            with open(output_file, "w") as f:
                f.write("\n".join(minimal_lines))

            # Human-readable for this directory
            readable_lines = generate_human_readable_format(funcs, f"{dir_name} module")
            output_file = dir_out / f"{dir_name}_readable.txt"
            with open(output_file, "w") as f:
                f.write("\n".join(readable_lines))

        print(f"Generated per-directory indexes in {dir_out}")

    # Generate coverage metrics
    metrics = compute_coverage_metrics(all_functions, args.root)

    # Write coverage JSON
    output_file = args.out / "coverage.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Write coverage markdown
    output_file = args.out / "coverage.md"
    with open(output_file, "w") as f:
        f.write("# SGLang Function Coverage Report\n\n")
        f.write(f"**Total Functions**: {metrics['total_functions']}\n")
        f.write(f"**Files Analyzed**: {metrics['files_analyzed']}\n")
        f.write(f"**Average Functions per File**: {metrics['functions_per_file']}\n\n")
        f.write("## Documentation Coverage\n")
        f.write(
            f"- **Documented**: {metrics['documented']} ({metrics['documentation_rate']})\n"
        )
        f.write(f"- **Undocumented**: {metrics['undocumented']}\n\n")
        f.write("## Function Types\n")
        f.write(f"- **Public**: {metrics['public_functions']}\n")
        f.write(f"- **Private**: {metrics['private_functions']}\n")
        f.write(f"- **Methods**: {metrics['methods']}\n")
        f.write(f"- **Async**: {metrics['async_functions']}\n\n")
        f.write("## Known Exclusions\n")
        f.write("- Generated code (protobuf, thrift)\n")
        f.write("- Vendored dependencies\n")
        f.write("- C++ extension modules\n")
        f.write("- Dynamic factory functions\n")

    print(f"Wrote coverage report to {output_file}")
    print(
        f"\nSummary: {metrics['total_functions']} functions across {metrics['files_analyzed']} files"
    )
    print(f"Documentation rate: {metrics['documentation_rate']}")


if __name__ == "__main__":
    main()
