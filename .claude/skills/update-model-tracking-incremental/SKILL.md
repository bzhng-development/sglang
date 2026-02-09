---
name: update-model-tracking-incremental
description: Incrementally update MODEL_SUPPORT_TRACKING.md by detecting only NEW architectures (new EntryClass in SGLang or new classes in transformers) and appending them. Skips all existing entries. Use this after merging a PR that adds a new model.
---

# Update MODEL_SUPPORT_TRACKING.md (Incremental)

Detects new architectures not yet in the tracking doc and appends them to the correct section. Existing rows are never modified.

For a full rebuild from scratch, use the `update-model-tracking` skill instead.

## When to Run

- After merging a PR that adds a new model file to `python/sglang/srt/models/`
- After bumping `transformers` version (new HF classes may appear)
- Quick check: "did we miss anything?"

## Single Command

Run from repo root. Prints what's new, appends rows, recalculates counts.

```bash
python3 << 'PYEOF'
import os, re, glob, importlib, json
from datetime import date

TRACKING_FILE = 'MODEL_SUPPORT_TRACKING.md'

# =====================================================================
# 1. Parse existing architectures from the tracking doc
# =====================================================================
existing = set()
with open(TRACKING_FILE) as f:
    for line in f:
        m = re.match(r'\| \*\*(\w+)\*\*', line)
        if m:
            existing.add(m.group(1))

print(f"Existing architectures in doc: {len(existing)}")

# =====================================================================
# 2. Scan current SGLang native architectures
# =====================================================================
models_dir = 'python/sglang/srt/models'
native = {}  # arch -> file

for fname in sorted(os.listdir(models_dir)):
    if not fname.endswith('.py') or fname.startswith('_'):
        continue
    with open(os.path.join(models_dir, fname)) as f:
        content = f.read()
    for m in re.finditer(r'EntryClass\s*=\s*(.+)', content):
        val = m.group(1).strip()
        if val.startswith('['):
            bm = re.search(r'EntryClass\s*=\s*\[(.*?)\]', content[m.start():], re.DOTALL)
            if bm:
                for item in bm.group(1).replace('\n', ',').split(','):
                    cls = item.strip()
                    if cls and re.match(r'^[A-Z]\w+$', cls) and cls not in native:
                        native[cls] = fname
        else:
            for cls in val.split('#')[0].strip().rstrip(',').strip('[]').split(','):
                cls = cls.strip()
                if cls and re.match(r'^[A-Z]\w+$', cls) and cls not in native:
                    native[cls] = fname

# =====================================================================
# 3. Scan current transformers architectures
# =====================================================================
hf_archs = set()
try:
    tf_path = os.path.dirname(importlib.import_module('transformers').__file__)
    tf_models = os.path.join(tf_path, 'models')
    patterns = [
        r'class\s+(\w+ForCausalLM)\s*\(',
        r'class\s+(\w+ForConditionalGeneration)\s*\(',
        r'class\s+(\w+ForSequenceClassification)\s*\(',
        r'class\s+(\w+LMHeadModel)\s*\(',
        r'class\s+(\w+Model)\s*\(',
    ]
    combined = '|'.join(patterns)
    for mf in glob.glob(f"{tf_models}/*/modeling_*.py"):
        if 'flax' in mf or 'tf_' in mf:
            continue
        with open(mf) as f:
            content = f.read()
        for m in re.finditer(combined, content):
            cls = next(g for g in m.groups() if g is not None)
            if not cls.startswith('_') and 'PreTrained' not in cls:
                hf_archs.add(cls)
except Exception as e:
    print(f"Warning: could not scan transformers: {e}")

# =====================================================================
# 4. Find NEW architectures (not in existing doc)
# =====================================================================
all_current = set(native.keys()) | hf_archs
new_archs = all_current - existing

if not new_archs:
    print("No new architectures found. Tracking doc is up to date.")
    raise SystemExit(0)

print(f"\nFound {len(new_archs)} new architecture(s):")
for a in sorted(new_archs):
    src = f"native ({native[a]})" if a in native else "HF transformers"
    print(f"  + {a}  [{src}]")

# =====================================================================
# 5. Try to resolve HF example model IDs for new archs
# =====================================================================
# Strategy: _CHECKPOINT_FOR_DOC and from_pretrained from transformers
checkpoint_map = {}
pretrained_map = {}
class_to_type = {}

try:
    for pattern in [f"{tf_models}/*/configuration_*.py", f"{tf_models}/*/modeling_*.py"]:
        for fpath in glob.glob(pattern):
            if 'flax' in fpath or 'tf_' in fpath:
                continue
            model_type = fpath.split('/')[-2]
            with open(fpath) as f:
                content = f.read()
            if model_type not in checkpoint_map:
                m = re.search(r'_CHECKPOINT_FOR_DOC\s*=\s*["\']([^"\']+)["\']', content)
                if m:
                    checkpoint_map[model_type] = m.group(1)

    for fpath in glob.glob(f"{tf_models}/*/modeling_*.py"):
        if 'flax' in fpath or 'tf_' in fpath or 'auto' in fpath:
            continue
        model_type = fpath.split('/')[-2]
        with open(fpath) as f:
            content = f.read()
        if model_type not in pretrained_map:
            m = re.search(r'from_pretrained\(\s*["\']([a-zA-Z0-9_\-]+/[a-zA-Z0-9_.\-]+)["\']', content)
            if m:
                pretrained_map[model_type] = m.group(1)
        for m in re.finditer(r'^class\s+(\w+)\s*\(', content, re.MULTILINE):
            cls = m.group(1)
            if cls not in class_to_type:
                class_to_type[cls] = model_type
except:
    pass

def resolve_example(arch):
    mt = class_to_type.get(arch)
    if mt:
        hf_id = checkpoint_map.get(mt) or pretrained_map.get(mt)
        if hf_id:
            return hf_id
    # Fuzzy match
    lower = arch.lower().replace('forcausallm', '').replace('forconditionalgeneration', '') \
                       .replace('forsequenceclassification', '').replace('lmheadmodel', '') \
                       .replace('model', '')
    for mt2 in checkpoint_map:
        if mt2.replace('_', '') == lower or lower.startswith(mt2.replace('_', '')):
            return checkpoint_map[mt2]
    return ''

# =====================================================================
# 6. Categorize new architectures
#    Uses heuristic rules. Print a notice so user can verify.
# =====================================================================
def auto_categorize(arch):
    """Heuristic categorization. Returns 'text', 'embedding', or 'multimodal'."""
    a = arch
    # Embedding/Classification/Reward
    if any(k in a for k in ['ForSequenceClassification', 'ForRewardModel',
                             'EmbeddingModel', 'VisionModel']):
        return 'embedding'
    if a.endswith('Model') and 'ForCausalLM' not in a and 'ForConditional' not in a:
        # Bare "Model" suffix often means embedding/base model
        return 'embedding'
    # Multimodal
    if 'ForConditionalGeneration' in a:
        return 'multimodal'
    if any(k in a for k in ['VL', 'Vision', 'Audio', 'OCR', 'MM', 'Omni',
                             'CPMO', 'CPMV', 'Ovis', 'Molmo', 'Fuyu',
                             'Llava', 'Janus', 'NVILA', 'InternVL',
                             'MultiModality']):
        return 'multimodal'
    # Default: text-only
    return 'text'

SECTION_HEADERS = {
    'text': '## Text-only Language Models (Generative)',
    'embedding': '## Embedding / Classification / Reward / Retrieval Models',
    'multimodal': '## Multimodal Models (Vision / Audio / OCR)',
}

# =====================================================================
# 7. Insert new rows into the tracking doc
# =====================================================================
with open(TRACKING_FILE) as f:
    lines = f.readlines()

# Group new archs by category
by_cat = {'text': [], 'embedding': [], 'multimodal': []}
for arch in sorted(new_archs):
    cat = auto_categorize(arch)
    is_native = arch in native
    is_hf = arch in hf_archs
    sfile = native.get(arch, '')
    example = resolve_example(arch)
    n = 'Y' if is_native else ''
    h = 'Y' if is_hf else ''
    f_col = f'`{sfile}`' if sfile else ''
    row = f'| **{arch}** | {n} | {h} | {f_col} | {example} |'
    by_cat[cat].append((arch, row))

# For each category, find the last table row in that section and insert
# new rows in sorted position
output = []
current_section = None
section_rows = {}  # section -> list of (arch_name, line_index) for sorting

for i, line in enumerate(lines):
    stripped = line.strip()
    # Detect section headers
    for cat, header in SECTION_HEADERS.items():
        if stripped == header:
            current_section = cat
            break

    output.append(line)

    # If this is a table row in a section, and next line is blank or stats,
    # we're at the end of a table — insert new rows here
    if current_section and by_cat.get(current_section) and stripped.startswith('| **'):
        # Peek ahead: is the next line NOT a table row?
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
        if not next_line.startswith('| **'):
            # Insert new rows for this section, sorted alphabetically
            # But we need to find the right sorted position
            # Simpler: just append at end of table, then we'll re-sort
            for arch, row in by_cat[current_section]:
                output.append(row + '\n')
            by_cat[current_section] = []  # consumed
            current_section = None

# Now re-sort table rows within each section and recount stats
final_lines = []
in_table = False
table_rows = []
section_cat = None

i = 0
while i < len(output):
    line = output[i]
    stripped = line.strip()

    # Detect section
    for cat, header in SECTION_HEADERS.items():
        if stripped == header:
            section_cat = cat
            break

    if stripped.startswith('| **') and section_cat:
        # Collect all table rows
        if not in_table:
            in_table = True
            table_rows = []
        table_rows.append(line)
        i += 1
        continue

    if in_table and not stripped.startswith('| **'):
        # End of table — sort and flush
        table_rows.sort(key=lambda l: re.search(r'\*\*(\w+)\*\*', l).group(1) if re.search(r'\*\*(\w+)\*\*', l) else '')
        final_lines.extend(table_rows)

        # Recount stats for this section
        n_count = sum(1 for r in table_rows if '| Y |' in r.split('|')[2] or r.split('|')[2].strip() == 'Y')
        total = len(table_rows)
        native_count = 0
        hf_only_count = 0
        no_fb_count = 0
        for r in table_rows:
            parts = [p.strip() for p in r.split('|')]
            is_n = parts[2] == 'Y'
            is_h = parts[3] == 'Y'
            if is_n:
                native_count += 1
            elif is_h:
                hf_only_count += 1
            else:
                no_fb_count += 1

        # Check if next line is a stats line
        if stripped.startswith('**') and 'native' in stripped:
            # Replace stats line
            final_lines.append(f'**{native_count}** native / **{hf_only_count}** HF-only / **{no_fb_count}** no fallback / **{total}** total\n')
            in_table = False
            table_rows = []
            i += 1
            continue
        else:
            in_table = False
            table_rows = []
            # Fall through to add current line

    # Stats lines — recalculate
    if stripped.startswith('**') and 'native /' in stripped and section_cat:
        # Already handled above in most cases, but handle standalone
        final_lines.append(line)
        i += 1
        continue

    final_lines.append(line)
    i += 1

# =====================================================================
# 8. Recalculate the Overall Summary table
# =====================================================================
# Re-parse the final output to count per-section stats
section_stats = {'text': [0,0,0,0], 'embedding': [0,0,0,0], 'multimodal': [0,0,0,0]}
cur = None
for line in final_lines:
    s = line.strip()
    for cat, header in SECTION_HEADERS.items():
        if s == header:
            cur = cat
            break
    if s == '## Overall Summary':
        cur = None
    if cur and s.startswith('| **'):
        parts = [p.strip() for p in s.split('|')]
        is_n = parts[2] == 'Y'
        is_h = parts[3] == 'Y'
        section_stats[cur][3] += 1
        if is_n:
            section_stats[cur][0] += 1
        elif is_h:
            section_stats[cur][1] += 1
        else:
            section_stats[cur][2] += 1

# Rewrite summary table and stats lines
result = []
in_summary = False
summary_done = False
for line in final_lines:
    s = line.strip()

    # Update per-section stats lines
    if s.startswith('**') and 'native /' in s and not in_summary:
        # Find which section we just finished
        for cat in ['text', 'embedding', 'multimodal']:
            st = section_stats[cat]
            if st[3] > 0:
                # Check if this count line is for this section (rough match)
                pass
        result.append(line)
        continue

    if s == '## Overall Summary':
        in_summary = True

    if in_summary and s.startswith('| Text-only'):
        t = section_stats['text']
        result.append(f'| Text-only | {t[0]} | {t[1]} | {t[2]} | {t[3]} |\n')
        continue
    if in_summary and s.startswith('| Embedding'):
        e = section_stats['embedding']
        result.append(f'| Embedding/Classification | {e[0]} | {e[1]} | {e[2]} | {e[3]} |\n')
        continue
    if in_summary and s.startswith('| Multimodal'):
        m = section_stats['multimodal']
        result.append(f'| Multimodal | {m[0]} | {m[1]} | {m[2]} | {m[3]} |\n')
        continue
    if in_summary and s.startswith('| **Total**'):
        tt = [sum(section_stats[c][i] for c in section_stats) for i in range(4)]
        result.append(f'| **Total** | **{tt[0]}** | **{tt[1]}** | **{tt[2]}** | **{tt[3]}** |\n')
        continue
    if in_summary and s.startswith('>') and 'natively supported' in s:
        tt = [sum(section_stats[c][i] for c in section_stats) for i in range(4)]
        result.append(f'> **{tt[0]}** natively supported ({tt[0]*100//tt[3]}%) — optimized SGLang implementations\n')
        continue
    if in_summary and s.startswith('>') and 'HF fallback' in s:
        tt = [sum(section_stats[c][i] for c in section_stats) for i in range(4)]
        result.append(f'> **{tt[1]}** HF fallback only ({tt[1]*100//tt[3]}%) — may work via `TransformersForCausalLM`\n')
        continue
    if in_summary and s.startswith('>') and 'no fallback' in s:
        tt = [sum(section_stats[c][i] for c in section_stats) for i in range(4)]
        result.append(f'> **{tt[2]}** no fallback ({tt[2]*100//tt[3]}%) — need full implementation\n')
        continue
    # Update the "Last updated" date
    if '**Last updated:**' in s:
        result.append(f'> **Last updated:** {date.today().isoformat()} | **Source:** `EntryClass` in `python/sglang/srt/models/*.py` + `transformers` library classes + HuggingFace `config.json`\n')
        continue

    result.append(line)

with open(TRACKING_FILE, 'w') as f:
    f.writelines(result)

total_new = sum(len(v) for v in [by_cat_orig for by_cat_orig in [{}]])  # already consumed
print(f"\nDone. Added {len(new_archs)} new architecture(s) to {TRACKING_FILE}")
print(f"Review the new entries and verify categorization is correct.")
print(f"If a new arch is miscategorized, move its row to the correct section manually,")
print(f"or add it to MULTIMODAL/EMBEDDING in the full-rebuild skill and re-run that instead.")
PYEOF
```

**What it does:**
1. Parses existing `MODEL_SUPPORT_TRACKING.md` to get all architecture names already tracked
2. Scans `python/sglang/srt/models/*.py` EntryClass definitions (handles multiline)
3. Scans local `transformers` library for architecture classes
4. Diffs: `(native + transformers) - existing_doc = new architectures`
5. Auto-categorizes new archs (text / embedding / multimodal) via heuristic
6. Inserts new rows into the correct table section, sorted alphabetically
7. Recalculates all counts and the summary table
8. Updates the "Last updated" date

**If nothing is new, it prints "No new architectures found" and exits.**

## Output Example

```
Existing architectures in doc: 225
Found 3 new architecture(s):
  + NewModelForCausalLM  [native (new_model.py)]
  + AnotherVLForConditionalGeneration  [native (another_vl.py)]
  + SomeHFModelForCausalLM  [HF transformers]

Done. Added 3 new architecture(s) to MODEL_SUPPORT_TRACKING.md
Review the new entries and verify categorization is correct.
```

## Auto-Categorization Rules

The heuristic categorizer uses these rules (in order):

| Pattern in arch name | Category assigned |
|---|---|
| `ForSequenceClassification`, `ForRewardModel`, `EmbeddingModel`, `VisionModel` | Embedding |
| Ends with `Model` (no `ForCausalLM` / `ForConditional`) | Embedding |
| `ForConditionalGeneration` | Multimodal |
| Contains `VL`, `Vision`, `Audio`, `OCR`, `MM`, `Omni`, `Llava`, `Molmo`, `NVILA`, etc. | Multimodal |
| Everything else | Text-only |

If a new architecture gets miscategorized, either:
- Move its row manually to the correct table section
- Or add it to `MULTIMODAL`/`EMBEDDING` in the full-rebuild skill and run that instead

## Limitations

- **Does not update existing rows.** If an architecture gains native support (new EntryClass added for an arch already in the doc), this script won't update its Native column. Use the full-rebuild skill for that.
- **HF example IDs** are resolved automatically but may be empty for SGLang-only architectures. Fill in manually or add to the `MANUAL` dict in the full-rebuild skill.
