# SGLang Documentation

**📖 For the complete documentation guide, see: [DOCUMENTATION_GUIDE.md](DOCUMENTATION_GUIDE.md)**

This README provides a quick overview. For comprehensive workflows, troubleshooting, and detailed instructions, use the consolidated guide above.

## Quick Overview

SGLang uses three documentation systems:

1. **Sphinx Documentation** (Primary) - Main docs at https://docs.sglang.ai
2. **API Documentation** - Developer reference (`docs/dev/refs/api/`)
3. **Function Index** - Code analysis tools (`docs/dev/refs/code-index/`)

## Docs Workflow

### Install Dependency

```bash
apt-get update && apt-get install -y pandoc parallel retry
pip install -r requirements.txt
```

### Update Documentation

Update your Jupyter notebooks in the appropriate subdirectories under `docs/`. If you add new files, remember to update `index.rst` (or relevant `.rst` files) accordingly.

- **`pre-commit run --all-files`** manually runs all configured checks, applying fixes if possible. If it fails the first time, re-run it to ensure lint errors are fully resolved. Make sure your code passes all checks **before** creating a Pull Request.

```bash
# 1) Compile all Jupyter notebooks
make compile  # This step can take a long time (10+ mins). You can consider skipping this step if you can make sure your added files are correct.
make html

# 2) Compile and Preview documentation locally with auto-build
# This will automatically rebuild docs when files change
# Open your browser at the displayed port to view the docs
bash serve.sh

# 2a) Alternative ways to serve documentation
# Directly use make serve
make serve
# With custom port
PORT=8080 make serve

# 3) Clean notebook outputs
# nbstripout removes notebook outputs so your PR stays clean
pip install nbstripout
find . -name '*.ipynb' -exec nbstripout {} \;

# 4) Pre-commit checks and create a PR
# After these checks pass, push your changes and open a PR on your branch
pre-commit run --all-files
```

## API Documentation Generation

### Generate API Documentation (pdoc)

For developers who want to generate comprehensive API documentation:

```bash
# Install sglang from source (required)
pip install -e python/

# Install pdoc
pip install pdoc

# Generate API documentation
pdoc sglang -o dev/refs/api

# View the generated documentation
# Open dev/refs/api/sglang/index.html in your browser
```

### Generate Function Index

For code analysis and function coverage:

```bash
# Generate function index with improved minimal and readable formats
python3 ../tools/devdocs/function_index/generate_function_index.py

# Output files:
# - dev/refs/code-index/function-index.json (detailed metadata)
# - dev/refs/code-index/function-index-min.txt (minimal with types)
# - dev/refs/code-index/function-index-readable.txt (human-readable)
# - dev/refs/code-index/coverage.md (documentation coverage)
```

### Clean API Documentation

```bash
# Clean before regenerating to avoid duplication
rm -rf dev/refs/api
pdoc sglang -o dev/refs/api
```

---

## Documentation Style Guidelines

- For common functionalities, we prefer **Jupyter Notebooks** over Markdown so that all examples can be executed and validated by our docs CI pipeline. For complex features (e.g., distributed serving), Markdown is preferred.
- Keep in mind the documentation execution time when writing interactive Jupyter notebooks. Each interactive notebook will be run and compiled against every commit to ensure they are runnable, so it is important to apply some tips to reduce the documentation compilation time:
  - Use small models (e.g., `qwen/qwen2.5-0.5b-instruct`) for most cases to reduce server launch time.
  - Reuse the launched server as much as possible to reduce server launch time.
- Do not use absolute links (e.g., `https://docs.sglang.ai/get_started/install.html`). Always prefer relative links (e.g., `../get_started/install.md`).
- Follow the existing examples to learn how to launch a server, send a query and other common styles.
