import argparse
import ast
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def unparse(node):
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except AttributeError:
        # Fallback for Python < 3.9
        return ast.dump(node, annotate_fields=False, include_attributes=False)


def format_arg(arg: ast.arg) -> str:
    s = arg.arg
    if getattr(arg, "annotation", None):
        ann = unparse(arg.annotation)
        if ann:
            s += f": {ann}"
    elif getattr(arg, "type_comment", None):
        s += f": {arg.type_comment}"
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


def is_excluded_path(path: Path, excluded_dirs: List[str], root: Path) -> bool:
    parts = path.relative_to(root).parts
    return any(part in excluded_dirs for part in parts)


def scan_file(path: Path) -> List[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}", file=sys.stderr)
        return []
    try:
        tree = ast.parse(text, filename=str(path))
    except Exception as e:
        print(f"[WARN] AST parse failed for {path}: {e}", file=sys.stderr)
        return []
    collector = FunctionCollector()
    collector.visit(tree)
    return collector.collected


def main():
    ap = argparse.ArgumentParser(description="Extract function definitions via AST.")
    ap.add_argument("--root", default=".", help="Root directory to scan")
    ap.add_argument(
        "--exclude", nargs="*", default=[], help="Directory names to exclude"
    )
    ap.add_argument(
        "--files-from",
        default=None,
        help="Optional path to a file list to scan (one path per line)",
    )
    ap.add_argument("--output", default="-", help="Output path (default stdout)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    excluded = set(args.exclude or [])

    if args.files_from:
        file_list = []
        with open(args.files_from, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                p = (
                    (root / line).resolve()
                    if not line.startswith("/")
                    else Path(line).resolve()
                )
                if p.suffix == ".py" and not is_excluded_path(p, list(excluded), root):
                    file_list.append(p)
    else:
        file_list = []
        for p in root.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            if is_excluded_path(p, list(excluded), root):
                continue
            file_list.append(p)

    file_list = sorted(file_list, key=lambda p: str(p.relative_to(root)))

    outfh = (
        sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    )

    try:
        print("AST Function Index", file=outfh)
        print(f"Root: {root}", file=outfh)
        print("Excluded directories: " + ", ".join(sorted(excluded)), file=outfh)
        print("", file=outfh)

        for fp in file_list:
            rel = fp.relative_to(root)
            functions = scan_file(fp)
            functions.sort(key=lambda f: (f.get("lineno") or 0, f["name"]))
            print(f"File: {rel}", file=outfh)
            if not functions:
                print("  (no function definitions found)", file=outfh)
                continue
            for f in functions:
                name = f["name"]
                params = f["parameters"]
                ret = f.get("return")
                cls = f.get("class_context")
                doc = f.get("doc") or ""
                sig = f"({params})"
                print(f"  - name: {name}", file=outfh)
                print(f"    signature: {sig}", file=outfh)
                if ret:
                    print(f"    return: {ret}", file=outfh)
                if cls:
                    print(f"    class: {cls}", file=outfh)
                if doc:
                    print(f"    doc: {doc}", file=outfh)
            print("", file=outfh)
    finally:
        if outfh is not sys.stdout:
            outfh.close()


if __name__ == "__main__":
    main()
