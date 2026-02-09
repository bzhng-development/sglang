---
name: check-model-support
description: Check whether a HuggingFace model architecture is natively supported by SGLang. Searches the model registry, compares with HuggingFace config, and reports support status with details. Use when evaluating model compatibility, triaging feature requests, or updating the model support tracking issue.
---

# Check Model Support in SGLang

Determine whether a given HuggingFace model architecture has native SGLang support, is only available via the Transformers fallback, or is completely unsupported.

## Parameters

- `architecture_name` (required): The HuggingFace architecture class name (e.g., `"OPTForCausalLM"`, `"BloomForCausalLM"`, `"LlamaForCausalLM"`). This is the value found in a model's `config.json` under the `"architectures"` field.
- `hf_model_id` (optional): A HuggingFace model repo ID (e.g., `"facebook/opt-125m"`) to fetch and cross-reference the config.json automatically.

## How SGLang Model Registration Works

SGLang uses a **registry pattern** to discover and load model implementations:

1. **Model files** live in `python/sglang/srt/models/` — each `.py` file implements one or more architectures.
2. Each file must define an **`EntryClass`** variable at module level that maps to the model class(es):
   ```python
   # Single architecture
   EntryClass = OPTForCausalLM

   # Multiple architectures in one file
   EntryClass = [LlamaForCausalLM, Phi3ForCausalLM, InternLM3ForCausalLM]
   ```
3. The **registry** (`python/sglang/srt/models/registry.py`) auto-discovers all `EntryClass` entries at import time via `import_model_classes()`.
4. The **model loader** (`python/sglang/srt/model_loader/utils.py`) reads the `"architectures"` field from HuggingFace `config.json` and resolves it against the registry.
5. If no native match is found, SGLang falls back to `TransformersForCausalLM` — a generic wrapper around HuggingFace Transformers.

## Step-by-Step: How to Check Support

### Step 1: Get the Architecture Name

If you already know the architecture name (e.g., `BloomForCausalLM`), skip to Step 2.

If you only have a HuggingFace model ID, fetch its `config.json`:

```bash
# Option A: Use the HuggingFace Hub API
curl -s "https://huggingface.co/<org>/<model>/raw/main/config.json" | python3 -c "
import json, sys
config = json.load(sys.stdin)
print('Architectures:', config.get('architectures', ['NOT FOUND']))
print('Model type:', config.get('model_type', 'unknown'))
"

# Option B: Use huggingface_hub Python package
python3 -c "
from huggingface_hub import hf_hub_download
import json
path = hf_hub_download(repo_id='facebook/opt-125m', filename='config.json')
config = json.load(open(path))
print('Architectures:', config.get('architectures'))
"
```

The `"architectures"` field contains the class name(s) to check. Example:
```json
{
  "architectures": ["OPTForCausalLM"],
  "model_type": "opt"
}
```

### Step 2: Search the SGLang Model Registry

Search for the architecture name in the `EntryClass` definitions:

```bash
# Search for the architecture name across all model files
grep -r "ARCHITECTURE_NAME" python/sglang/srt/models/ --include="*.py"
```

**What to look for:**

| Search Result | Meaning |
|---|---|
| Found in an `EntryClass = ...` line | **Natively supported** — SGLang has an optimized implementation |
| Found in class definition but NOT in `EntryClass` | **Not registered** — implementation exists but isn't wired up (possible bug or WIP) |
| Not found anywhere | **Not natively supported** — will fall back to Transformers or fail |

### Step 3: Verify the EntryClass Registration

If you found the architecture in a model file, confirm it's properly registered:

```bash
# Check the specific file's EntryClass
grep "EntryClass" python/sglang/srt/models/<model_file>.py
```

The `EntryClass` must include the architecture name either as:
- A direct assignment: `EntryClass = OPTForCausalLM`
- A list element: `EntryClass = [OPTForCausalLM]`
- One of multiple: `EntryClass = [LlamaForCausalLM, Phi3ForCausalLM]`

### Step 4: Check Transformers Fallback Compatibility

If the architecture is NOT natively supported, SGLang attempts a Transformers fallback. This happens automatically in `get_model_architecture()` (`python/sglang/srt/model_loader/utils.py`):

```python
if not is_native_supported or model_config.model_impl == ModelImpl.TRANSFORMERS:
    architectures = resolve_transformers_arch(model_config, architectures)
```

The fallback works if:
1. The architecture exists in the `transformers` library (`getattr(transformers, arch, None)`)
2. The model passes `is_backend_compatible()` check (if defined)

**To test manually:**
```python
import transformers
arch = "BloomForCausalLM"
model_cls = getattr(transformers, arch, None)
if model_cls:
    print(f"{arch} exists in transformers — fallback possible")
    if hasattr(model_cls, 'is_backend_compatible'):
        print(f"Backend compatible: {model_cls.is_backend_compatible()}")
    else:
        print("No compatibility check — assumed compatible")
else:
    print(f"{arch} NOT in transformers — fallback NOT possible")
```

### Step 5: Compare with HuggingFace Model Card

Cross-reference the architecture with the HuggingFace model page to ensure alignment:

1. Visit `https://huggingface.co/<org>/<model>`
2. Check the **"Model card"** for architecture details
3. Download `config.json` and verify `"architectures"` matches what you searched for
4. Check `"auto_map"` field — if present, the model may define custom classes that override the architecture name

**Important:** Some models use custom architecture names in `auto_map` that differ from the `architectures` field. SGLang handles this via `get_class_from_dynamic_module()` in the Transformers fallback path.

## Interpreting Results

### Result: Natively Supported

**Indicators:**
- Architecture name found in `EntryClass` of a model file
- `grep` returns a match in `python/sglang/srt/models/*.py`

**Example — OPTForCausalLM:**
```bash
$ grep -rn "OPTForCausalLM" python/sglang/srt/models/ --include="*.py"
python/sglang/srt/models/opt.py:387:class OPTForCausalLM(nn.Module):
python/sglang/srt/models/opt.py:637:EntryClass = [OPTForCausalLM]
```
- File: `python/sglang/srt/models/opt.py`
- Class defined at line 387
- Registered via `EntryClass` at line 637
- **Status: NATIVELY SUPPORTED** — SGLang has a dedicated, optimized implementation

### Result: NOT Natively Supported (Transformers Fallback Only)

**Indicators:**
- Architecture name NOT found in any `EntryClass`
- Architecture name NOT found in any model file at all (or only in comments/docs)
- But the architecture EXISTS in the `transformers` library

**Example — BloomForCausalLM:**
```bash
$ grep -rn "BloomForCausalLM" python/sglang/srt/models/ --include="*.py"
# (no output — not found)
```
- Not in any SGLang model file
- BUT `BloomForCausalLM` exists in `transformers` → may work via `TransformersForCausalLM` fallback
- **Status: NOT NATIVELY SUPPORTED** — no optimized implementation; relies on generic Transformers wrapper
- Performance may be suboptimal; some SGLang features (tensor parallelism, custom kernels) may not apply

### Result: Completely Unsupported

**Indicators:**
- Not found in SGLang model files
- Not found in `transformers` library
- Not in `auto_map` of config.json

**Status: UNSUPPORTED** — cannot be loaded by SGLang at all.

## Quick Reference: One-Liner Check

```bash
# Check if architecture is natively supported in SGLang
grep -rn "EntryClass.*ARCHITECTURE_NAME\|ARCHITECTURE_NAME.*EntryClass" python/sglang/srt/models/ --include="*.py"
```

If this returns results → **natively supported**.
If no results → check Transformers fallback or it's **unsupported**.

## List All Supported Architectures

```bash
# Extract all registered architecture names from EntryClass lines
grep -h "EntryClass" python/sglang/srt/models/*.py | \
  sed 's/EntryClass = //' | \
  tr '[],' '\n' | \
  sed 's/^ *//;s/ *$//' | \
  grep -v '^$' | \
  sort
```

Or programmatically:
```python
from sglang.srt.models.registry import ModelRegistry
for arch in sorted(ModelRegistry.get_supported_archs()):
    print(arch)
```

## Environment Variables That Affect Model Loading

| Variable | Effect |
|---|---|
| `SGLANG_DISABLED_MODEL_ARCHS` | Comma-separated list of model file names to skip during registration (e.g., `"minimax_m2,opt"`) |
| `SGLANG_EXTERNAL_MODEL_PACKAGE` | Python package name to load additional models from (registered with `overwrite=True`) |

## Worked Examples

### Example A: Checking OPTForCausalLM (Natively Supported)

1. **Architecture name**: `OPTForCausalLM` (from `facebook/opt-125m` config.json)
2. **Search**:
   ```bash
   $ grep -rn "OPTForCausalLM" python/sglang/srt/models/ --include="*.py"
   python/sglang/srt/models/opt.py:387:class OPTForCausalLM(nn.Module):
   python/sglang/srt/models/opt.py:603:  f"Error getting weights by name {name} in OPTForCausalLM..."
   python/sglang/srt/models/opt.py:637:EntryClass = [OPTForCausalLM]
   ```
3. **EntryClass confirmed** at `opt.py:637`
4. **Result**: Natively supported with dedicated implementation in `opt.py`

### Example B: Checking BloomForCausalLM (NOT Natively Supported)

1. **Architecture name**: `BloomForCausalLM` (from `bigscience/bloom-560m` config.json)
2. **Search**:
   ```bash
   $ grep -rn "BloomForCausalLM" python/sglang/srt/models/ --include="*.py"
   # (no output)
   ```
3. **Not found** in any SGLang model file
4. **Transformers check**:
   ```python
   import transformers
   print(getattr(transformers, "BloomForCausalLM", None))
   # <class 'transformers.models.bloom.modeling_bloom.BloomForCausalLM'>
   ```
5. **Result**: NOT natively supported. May work via `TransformersForCausalLM` fallback with degraded performance. Candidate for native implementation.

### Example C: Checking MiniMaxM1ForCausalLM (Does Not Exist)

1. **Architecture name**: `MiniMaxM1ForCausalLM`
2. **Search**:
   ```bash
   $ grep -rn "MiniMaxM1" python/sglang/srt/models/ --include="*.py"
   # (no output)
   ```
3. **Not found** — only `MiniMaxM2ForCausalLM` exists (in `minimax_m2.py`)
4. **Transformers check**: Not in `transformers` library either
5. **Result**: Does not exist as an architecture. The correct architecture name may be `MiniMaxM2ForCausalLM`. Always verify against the HuggingFace `config.json` for the actual model.

## Key Files Reference

| File | Purpose |
|---|---|
| `python/sglang/srt/models/registry.py` | Model registry — discovers and stores all supported architectures |
| `python/sglang/srt/model_loader/utils.py` | Model loading — resolves architecture to class, handles Transformers fallback |
| `python/sglang/srt/models/*.py` | Individual model implementations — each defines `EntryClass` |
| `python/sglang/srt/models/transformers.py` | Generic Transformers fallback wrapper (`TransformersForCausalLM`) |
| `python/sglang/srt/environ.py` | Environment variable definitions (`SGLANG_DISABLED_MODEL_ARCHS`, `SGLANG_EXTERNAL_MODEL_PACKAGE`) |

## Updating the Tracking Issue

After checking support status, update the tracking issue with:
- Architecture name and support status
- File path if natively supported
- Whether Transformers fallback works
- Link to HuggingFace model page
- Any related issues or PRs
