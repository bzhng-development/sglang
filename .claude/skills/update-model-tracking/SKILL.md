---
name: update-model-tracking
description: Regenerate MODEL_SUPPORT_TRACKING.md — the ground truth document listing all 225+ model architectures with their SGLang native support status, HuggingFace fallback availability, source files, and example model IDs. Run this after adding new models, updating transformers, or before release audits.
---

# Regenerate MODEL_SUPPORT_TRACKING.md

End-to-end workflow to rebuild the ground truth model support tracking document from source. Produces a markdown file with every architecture categorized as native, HF-fallback, or unsupported.

## When to Run

- After merging new model implementations (new `EntryClass` in `python/sglang/srt/models/`)
- After upgrading the `transformers` dependency
- Before releases or when auditing model coverage
- When triaging model support issues

## Prerequisites

- Working directory: repo root (where `python/sglang/` lives)
- Local `transformers` package installed (check with `python3 -c "import transformers; print(transformers.__file__)"`)
- Internet access (for HF example model ID extraction, optional — cached results in `/tmp/hf_arch_examples.txt` can be reused)

---

## Step 1: Extract SGLang Native Architectures

Scans all `EntryClass` definitions in `python/sglang/srt/models/*.py`, correctly handling multiline list assignments.

```bash
python3 << 'PYEOF'
import os, re, json

models_dir = 'python/sglang/srt/models'
native = {}  # arch -> source_file

for fname in sorted(os.listdir(models_dir)):
    if not fname.endswith('.py') or fname.startswith('_'):
        continue
    with open(os.path.join(models_dir, fname)) as f:
        content = f.read()

    for m in re.finditer(r'EntryClass\s*=\s*(.+)', content):
        val = m.group(1).strip()
        if val.startswith('['):
            # Multiline list: EntryClass = [\n  Class1,\n  Class2,\n]
            bracket_match = re.search(
                r'EntryClass\s*=\s*\[(.*?)\]', content[m.start():], re.DOTALL
            )
            if bracket_match:
                for item in bracket_match.group(1).replace('\n', ',').split(','):
                    cls = item.strip()
                    if cls and re.match(r'^[A-Z]\w+$', cls) and cls not in native:
                        native[cls] = fname
        else:
            val = val.split('#')[0].strip().rstrip(',').strip('[]')
            for cls in val.split(','):
                cls = cls.strip()
                if cls and re.match(r'^[A-Z]\w+$', cls) and cls not in native:
                    native[cls] = fname

with open('/tmp/sglang_native_archs.json', 'w') as f:
    json.dump(native, f, indent=2, sort_keys=True)
print(f"Extracted {len(native)} native architectures -> /tmp/sglang_native_archs.json")
PYEOF
```

**Gotcha — multiline EntryClass:** Files like `llama.py` define `EntryClass = [LlamaForCausalLM, Phi3ForCausalLM, ...]` across multiple lines. A simple `grep -h "EntryClass"` one-liner will miss these. Always use the Python script above with `re.DOTALL`.

---

## Step 2: Extract Transformers Library Architectures

Extracts all architecture class names from the locally installed `transformers` package.

```bash
python3 << 'PYEOF'
import os, re, glob, importlib

tf_path = os.path.dirname(importlib.import_module('transformers').__file__)
models_dir = os.path.join(tf_path, 'models')

archs = set()
patterns = [
    r'class\s+(\w+ForCausalLM)\s*\(',
    r'class\s+(\w+ForConditionalGeneration)\s*\(',
    r'class\s+(\w+ForSequenceClassification)\s*\(',
    r'class\s+(\w+LMHeadModel)\s*\(',
    r'class\s+(\w+Model)\s*\(',
]
combined = '|'.join(patterns)

for modeling_file in glob.glob(f"{models_dir}/*/modeling_*.py"):
    if 'flax' in modeling_file or 'tf_' in modeling_file:
        continue
    with open(modeling_file) as f:
        content = f.read()
    for m in re.finditer(combined, content):
        cls = next(g for g in m.groups() if g is not None)
        # Skip internal/private classes
        if not cls.startswith('_') and 'PreTrained' not in cls:
            archs.add(cls)

with open('transformers_architectures.txt', 'w') as f:
    for a in sorted(archs):
        f.write(a + '\n')
print(f"Extracted {len(archs)} transformers architectures -> transformers_architectures.txt")
PYEOF
```

---

## Step 3: Extract HuggingFace Example Model IDs

Maps each architecture to a real HuggingFace repo ID using three strategies in priority order:

1. `_CHECKPOINT_FOR_DOC` from transformers config/modeling files (most reliable)
2. `from_pretrained("org/model")` patterns in docstrings
3. Manual overrides for well-known models without doc patterns

```bash
python3 << 'PYEOF'
import os, re, glob, json, importlib

tf_path = os.path.dirname(importlib.import_module('transformers').__file__)
models_dir = os.path.join(tf_path, 'models')

# --- Strategy 1: _CHECKPOINT_FOR_DOC ---
checkpoint_map = {}  # model_type -> hf_id
for pattern in [f"{models_dir}/*/configuration_*.py", f"{models_dir}/*/modeling_*.py"]:
    for fpath in glob.glob(pattern):
        if 'flax' in fpath or 'tf_' in fpath:
            continue
        model_type = fpath.split('/')[-2]
        if model_type in checkpoint_map:
            continue
        with open(fpath) as f:
            content = f.read()
        m = re.search(r'_CHECKPOINT_FOR_DOC\s*=\s*["\']([^"\']+)["\']', content)
        if m:
            checkpoint_map[model_type] = m.group(1)

# --- Strategy 2: from_pretrained in docstrings ---
pretrained_map = {}  # model_type -> hf_id
for fpath in glob.glob(f"{models_dir}/*/modeling_*.py"):
    if 'flax' in fpath or 'tf_' in fpath or 'auto' in fpath:
        continue
    model_type = fpath.split('/')[-2]
    if model_type in pretrained_map:
        continue
    with open(fpath) as f:
        content = f.read()
    m = re.search(r'from_pretrained\(\s*["\']([a-zA-Z0-9_\-]+/[a-zA-Z0-9_.\-]+)["\']', content)
    if m:
        pretrained_map[model_type] = m.group(1)

# --- Map class names -> model_type ---
class_to_type = {}
for fpath in glob.glob(f"{models_dir}/*/modeling_*.py"):
    if 'flax' in fpath or 'tf_' in fpath or 'auto' in fpath:
        continue
    model_type = fpath.split('/')[-2]
    with open(fpath) as f:
        content = f.read()
    for m in re.finditer(r'^class\s+(\w+)\s*\(', content, re.MULTILINE):
        cls = m.group(1)
        if cls not in class_to_type:
            class_to_type[cls] = model_type

# --- Manual overrides for SGLang-only architectures ---
# These architectures exist only in SGLang (no transformers class), so we
# hardcode known HF repos. Update this dict when adding new SGLang models.
MANUAL = {
    'AfmoeForCausalLM': 'arcee-ai/Trinity-Large-Base',
    'BaichuanForCausalLM': 'baichuan-inc/Baichuan2-7B-Chat',
    'BailingMoEForCausalLM': 'inclusiveai/bailing-moe',
    'BailingMoeForCausalLM': 'inclusiveai/bailing-moe',
    'BailingMoeForCausalLMNextN': 'inclusiveai/bailing-moe',
    'BailingMoeV2ForCausalLM': 'inclusiveai/bailing-moe',
    'ChatGLMModel': 'THUDM/chatglm3-6b',
    'Contriever': 'facebook/contriever',
    'DeciLMForCausalLM': 'Deci/DeciLM-7B',
    'DeepseekForCausalLM': 'deepseek-ai/deepseek-llm-7b-base',
    'DeepseekOCRForCausalLM': 'deepseek-ai/deepseek-ocr',
    'DeepseekV2ForCausalLM': 'deepseek-ai/DeepSeek-V2-Lite',
    'DeepseekV3ForCausalLM': 'deepseek-ai/DeepSeek-V3',
    'DeepseekV32ForCausalLM': 'deepseek-ai/DeepSeek-V3',
    'DeepseekV3ForCausalLMNextN': 'deepseek-ai/DeepSeek-V3',
    'DeepseekVL2ForCausalLM': 'deepseek-ai/deepseek-vl2-tiny',
    'DotsOCRForCausalLM': 'rednote-hilab/dots.llm1.inst',
    'DotsVLMForCausalLM': 'rednote-hilab/dots.llm1.inst',
    'Ernie4_5_ForCausalLM': 'baidu/ERNIE-4.5-0.3B-Instruct-PT',
    'Ernie4_5_MoeForCausalLM': 'baidu/ERNIE-4.5-21B-A3B-Instruct-PT',
    'Ernie4_5_MoeForCausalLMMTP': 'baidu/ERNIE-4.5-21B-A3B-Instruct-PT',
    'Ernie4_5_VLMoeForConditionalGeneration': 'baidu/ERNIE-4.5-VL-28B-A3B-Instruct-PT',
    'ExaoneForCausalLM': 'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct',
    'ExaoneMoEForCausalLM': 'LGAI-EXAONE/EXAONE-4.0-32B',
    'ExaoneMoEForCausalLMMTP': 'LGAI-EXAONE/EXAONE-4.0-32B',
    'GlmAsrForConditionalGeneration': 'THUDM/GLM-4-Voice',
    'GlmOcrForConditionalGeneration': 'THUDM/GLM-OCR',
    'GlmOcrForConditionalGenerationNextN': 'THUDM/GLM-OCR',
    'Glm4MoeForCausalLMNextN': 'THUDM/GLM-4-9B-0414',
    'Glm4MoeLiteForCausalLM': 'THUDM/GLM-4-9B-0414',
    'Grok1ForCausalLM': 'xai-org/grok-1',
    'Grok1ModelForCausalLM': 'xai-org/grok-1',
    'IQuestCoderForCausalLM': 'iQuestCoder/iQuestCoder-7B',
    'IQuestLoopCoderForCausalLM': 'iQuestCoder/iQuestLoopCoder-7B',
    'InternLM2ForCausalLM': 'internlm/internlm2-7b',
    'InternLM2ForRewardModel': 'internlm/internlm2-7b-reward',
    'InternLM3ForCausalLM': 'internlm/internlm3-8b-instruct',
    'InternS1ForConditionalGeneration': 'internlm/InternS1',
    'InternS1ProForConditionalGeneration': 'internlm/InternS1-Pro',
    'InternVLChatModel': 'OpenGVLab/InternVL2-8B',
    'JetNemotronForCausalLM': 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
    'JetVLMForConditionalGeneration': 'nvidia/NVLM-D-72B',
    'KimiK25ForConditionalGeneration': 'moonshotai/Kimi-K2-Instruct',
    'KimiLinearForCausalLM': 'moonshotai/Kimi-K2-Instruct',
    'KimiVLForConditionalGeneration': 'moonshotai/Kimi-VL-A3B-Instruct',
    'LLaDA2MoeModelLM': 'GSAI-ML/LLaDA-8B-Instruct',
    'LightOnOCRForConditionalGeneration': 'lightonai/lighton-ocr',
    'LlamaEmbeddingModel': 'meta-llama/Llama-2-7b-hf',
    'LlamaForCausalLMEagle': 'meta-llama/Llama-2-7b-hf',
    'LlamaForCausalLMEagle3': 'meta-llama/Llama-2-7b-hf',
    'LlamaForClassification': 'meta-llama/Llama-2-7b-hf',
    'LlavaLlamaForCausalLM': 'llava-hf/llava-1.5-7b-hf',
    'LlavaMistralForCausalLM': 'llava-hf/llava-v1.6-mistral-7b-hf',
    'LlavaQwenForCausalLM': 'llava-hf/llava-1.5-7b-hf',
    'LlavaVidForCausalLM': 'llava-hf/LLaVA-NeXT-Video-7B-hf',
    'LongcatFlashForCausalLM': 'nvidia/Llama-3_3-Nemotron-Super-49B-v1',
    'LongcatFlashForCausalLMNextN': 'nvidia/Llama-3_3-Nemotron-Super-49B-v1',
    'MiDashengLMModel': 'dasheng-ai/MiDashengLM',
    'MiMoForCausalLM': 'xiaomi/MiMo-7B-RL',
    'MiMoMTP': 'xiaomi/MiMo-7B-RL',
    'MiMoV2FlashForCausalLM': 'xiaomi/MiMo-7B-RL',
    'MiMoV2MTP': 'xiaomi/MiMo-7B-RL',
    'MindSporeForCausalLM': 'mindspore/mindspore-llm',
    'MiniCPM3ForCausalLM': 'openbmb/MiniCPM3-4B',
    'MiniCPMForCausalLM': 'openbmb/MiniCPM-2B-sft-bf16',
    'MiniCPMO': 'openbmb/MiniCPM-o-2_6',
    'MiniCPMV': 'openbmb/MiniCPM-V-2_6',
    'MiniMaxM2ForCausalLM': 'MiniMaxAI/MiniMax-M2',
    'Ministral3ForCausalLM': 'mistralai/Ministral-3-14B-Reasoning-2512',
    'MistralLarge3ForCausalLM': 'mistralai/Mistral-Large-Instruct-2411',
    'MistralLarge3ForCausalLMEagle': 'mistralai/Mistral-Large-Instruct-2411',
    'MultiModalityCausalLM': 'deepseek-ai/Janus-Pro-7B',
    'NVILAForConditionalGeneration': 'nvidia/NVILA-Lite-2B-hf',
    'NVILALiteForConditionalGeneration': 'nvidia/NVILA-Lite-2B-hf',
    'NemotronHForCausalLM': 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
    'NemotronHForCausalLMMTP': 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
    'NemotronH_Nano_VL_V2': 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16',
    'OrionForCausalLM': 'OrionStarAI/Orion-14B-Chat',
    'POINTSV15ChatModel': 'WePOINTS/POINTS-v1.5-7B',
    'PaddleOCRVLForConditionalGeneration': 'PaddlePaddle/PaddleOCR',
    'Phi3SmallForCausalLM': 'microsoft/phi-1',
    'Phi4MMForCausalLM': 'microsoft/Phi-4-multimodal-instruct',
    'PhiMoEForCausalLM': 'microsoft/Phi-3.5-MoE-instruct',
    'PixtralForConditionalGeneration': 'mistralai/Pixtral-12B-2409',
    'QWenLMHeadModel': 'Qwen/Qwen-7B',
    'QuantMixtralForCausalLM': 'mistralai/Mixtral-8x7B-v0.1',
    'Qwen2ForCausalLMEagle': 'Qwen/Qwen2-7B',
    'Qwen2ForRewardModel': 'Qwen/Qwen2-7B',
    'Qwen3NextForCausalLM': 'Qwen/Qwen3-Next-80B-A3B-Instruct',
    'Qwen3NextForCausalLMMTP': 'Qwen/Qwen3-Next-80B-A3B-Instruct',
    'Sarashina2VisionForCausalLM': 'sbintuitions/sarashina2-vision-14b',
    'SolarForCausalLM': 'upstage/SOLAR-10.7B-v1.0',
    'Step3VLForConditionalGeneration': 'stepfun-ai/Step-3-VL',
    'Step3p5ForCausalLM': 'stepfun-ai/Step-3.5',
    'Step3p5MTP': 'stepfun-ai/Step-3.5',
    'StepVLForConditionalGeneration': 'stepfun-ai/Step-VL-10B',
    'TeleFLMForCausalLM': 'CofeAI/FLM-2-52B',
    'TorchNativeLlamaForCausalLM': 'meta-llama/Llama-2-7b-hf',
    'TorchNativePhi3ForCausalLM': 'microsoft/Phi-3-mini-4k-instruct',
    'TransformersForCausalLM': '(generic fallback)',
    'XverseForCausalLM': 'xverse/XVERSE-7B-Chat',
    'XverseMoeForCausalLM': 'xverse/XVERSE-MoE-A36B',
    'YiVLForCausalLM': '01-ai/Yi-VL-6B',
    # --- Non-native but known ---
    'AquilaForCausalLM': 'BAAI/Aquila-7B',
    'ArcticForCausalLM': 'Snowflake/snowflake-arctic-instruct',
    'GritLM': 'GritLM/GritLM-7B',
    'GteModel': 'thenlper/gte-base',
    'GteNewModel': 'Alibaba-NLP/gte-large-en-v1.5',
    'H2OVLChatModel': 'h2oai/h2ovl-mississippi-2b',
    'InternLMForCausalLM': 'internlm/internlm-7b',
    'JAISLMHeadModel': 'inceptionai/jais-13b',
    'LlamaNemotronVLModel': 'nvidia/llama-nemotron-colembed-vl-3b-v2',
    'MPTForCausalLM': 'mosaicml/mpt-7b',
    'MiniMaxM1ForCausalLM': 'MiniMaxAI/MiniMax-M1-80k',
    'MiniMaxText01ForCausalLM': 'MiniMaxAI/MiniMax-Text-01',
    'MiniMaxVL01ForConditionalGeneration': 'MiniMaxAI/MiniMax-VL-01',
    'MolmoForCausalLM': 'allenai/Molmo-7B-D-0924',
    'Molmo2ForConditionalGeneration': 'allenai/Molmo2-8B',
    'NVLM_D_Model': 'nvidia/NVLM-D-72B',
    'NomicBertModel': 'nomic-ai/nomic-embed-text-v1.5',
    'Ovis': 'AIDC-AI/Ovis2-1B',
    'Phi3VForCausalLM': 'microsoft/Phi-3-vision-128k-instruct',
    'Plamo2ForCausalLM': 'pfnet/plamo-2-1b',
    'Qwen3VLNemotronEmbedModel': 'nvidia/nemotron-colembed-vl-8b-v2',
    'SkyworkR1VChatModel': 'Skywork/Skywork-R1V-38B',
    'SolarOpenForCausalLM': 'upstage/Solar-Open-100B',
    'Tarsier2ForConditionalGeneration': 'ByteDance/Tarsier2-7B',
    'TarsierForConditionalGeneration': 'ByteDance/Tarsier-34b',
    'TeleChat2ForCausalLM': 'Tele-AI/TeleChat2-7B',
    'llama_NemoRetrieverColEmbed': 'nvidia/llama-nemoretriever-colembed-3b-v1',
}

# --- Resolve: class -> model_type -> hf_id ---
all_archs = set()
with open('/tmp/sglang_native_archs.json') as f:
    native = json.load(f)
all_archs.update(native.keys())

with open('transformers_architectures.txt') as f:
    for line in f:
        all_archs.add(line.strip())

# Also include any manual-override-only archs
all_archs.update(MANUAL.keys())

results = {}
for arch in sorted(all_archs):
    # Priority: manual override > checkpoint_for_doc > from_pretrained
    if arch in MANUAL:
        results[arch] = MANUAL[arch]
        continue
    mt = class_to_type.get(arch)
    hf_id = None
    if mt:
        hf_id = checkpoint_map.get(mt) or pretrained_map.get(mt)
    if not hf_id:
        # Fuzzy: strip suffixes and try matching model_type
        lower = arch.lower().replace('forcausallm', '').replace('forconditionalgeneration', '') \
                           .replace('forsequenceclassification', '').replace('lmheadmodel', '') \
                           .replace('model', '')
        for mt2 in checkpoint_map:
            if mt2.replace('_', '') == lower or lower.startswith(mt2.replace('_', '')):
                hf_id = checkpoint_map[mt2]
                break
    results[arch] = hf_id or ''

with open('/tmp/hf_arch_examples.txt', 'w') as f:
    for arch in sorted(results):
        f.write(f"{arch}|{results[arch]}\n")

found = sum(1 for v in results.values() if v)
print(f"Mapped {found}/{len(results)} architectures to HF examples -> /tmp/hf_arch_examples.txt")
missing = [a for a, v in sorted(results.items()) if not v]
if missing:
    print(f"Missing examples ({len(missing)}): {', '.join(missing[:20])}")
    print("Add these to the MANUAL dict in this script.")
PYEOF
```

**Gotcha — missing examples:** If a new architecture has no `_CHECKPOINT_FOR_DOC` and no `from_pretrained` in docstrings, it won't get an example automatically. The script prints missing architectures at the end — add them to the `MANUAL` dict.

---

## Step 4: (Optional) Verify New Models via HuggingFace config.json

When adding specific models from an issue or PR, download their `config.json` to confirm the architecture name.

```bash
# Download config.json for one or more HF repos
for repo in "nvidia/Some-New-Model" "mistralai/Another-Model"; do
  echo "=== $repo ==="
  curl -sL "https://huggingface.co/${repo}/raw/main/config.json" | \
    python3 -c "import json,sys; c=json.load(sys.stdin); print('architectures:', c.get('architectures')); print('model_type:', c.get('model_type'))"
done
```

Add any new architectures found to the `MANUAL` dict in Step 3 and to the categorization sets in Step 5.

---

## Step 5: Generate MODEL_SUPPORT_TRACKING.md

Combines all extracted data, categorizes architectures, and writes the final markdown document.

```bash
python3 << 'PYEOF'
import json

# --- Load data from previous steps ---
with open('/tmp/sglang_native_archs.json') as f:
    native = json.load(f)  # arch -> file

hf_examples = {}
with open('/tmp/hf_arch_examples.txt') as f:
    for line in f:
        if '|' in line:
            arch, ex = line.strip().split('|', 1)
            hf_examples[arch] = ex

hf_archs = set()
with open('transformers_architectures.txt') as f:
    for line in f:
        if line.strip():
            hf_archs.add(line.strip())

all_archs = set(native.keys()) | set(hf_examples.keys())

# =====================================================================
# CATEGORIZATION SETS
# Maintain these when adding new architectures.
# Rule of thumb:
#   - Multimodal: processes images/audio/video (VL, Vision, Audio, OCR,
#     ForConditionalGeneration for vision/audio models, multi-modal chat)
#   - Embedding: produces embeddings/scores, not generative text
#     (ForSequenceClassification, Reward, Embedding, Model-only classes)
#   - Text-only: everything else (ForCausalLM generative text models,
#     including MTP/Eagle speculative-decoding variants)
# =====================================================================

MULTIMODAL = {
    'AriaForConditionalGeneration', 'AyaVisionForConditionalGeneration',
    'Blip2ForConditionalGeneration', 'ChameleonForConditionalGeneration',
    'DeepseekOCRForCausalLM', 'DeepseekVL2ForCausalLM',
    'DotsOCRForCausalLM', 'DotsVLMForCausalLM',
    'Ernie4_5_VLMoeForConditionalGeneration',
    'Florence2ForConditionalGeneration', 'FuyuForCausalLM',
    'Gemma3ForConditionalGeneration', 'Gemma3nForConditionalGeneration',
    'Glm4vForConditionalGeneration', 'Glm4vMoeForConditionalGeneration',
    'GlmAsrForConditionalGeneration', 'GlmOcrForConditionalGeneration',
    'GlmOcrForConditionalGenerationNextN',
    'GraniteSpeechForConditionalGeneration',
    'H2OVLChatModel',
    'Idefics3ForConditionalGeneration',
    'InternS1ForConditionalGeneration', 'InternS1ProForConditionalGeneration',
    'InternVLChatModel',
    'JetVLMForConditionalGeneration',
    'KimiK25ForConditionalGeneration', 'KimiVLForConditionalGeneration',
    'LightOnOCRForConditionalGeneration',
    'Llama4ForConditionalGeneration',
    'LlavaForConditionalGeneration', 'LlavaLlamaForCausalLM',
    'LlavaMistralForCausalLM', 'LlavaQwenForCausalLM',
    'LlavaVidForCausalLM', 'LlavaNextVideoForConditionalGeneration',
    'LlamaNemotronVLModel',
    'MiniCPMO', 'MiniCPMV',
    'MiniMaxVL01ForConditionalGeneration',
    'Mistral3ForConditionalGeneration',
    'MllamaForConditionalGeneration',
    'MolmoForCausalLM', 'Molmo2ForConditionalGeneration',
    'MultiModalityCausalLM',
    'NVILAForConditionalGeneration', 'NVILALiteForConditionalGeneration',
    'NVLM_D_Model',
    'NemotronH_Nano_VL_V2',
    'Ovis',
    'PaddleOCRVLForConditionalGeneration',
    'PaliGemmaForConditionalGeneration',
    'Phi3VForCausalLM', 'Phi4MMForCausalLM',
    'PixtralForConditionalGeneration',
    'Qwen2AudioForConditionalGeneration',
    'Qwen2VLForConditionalGeneration', 'Qwen2_5OmniThinkerForConditionalGeneration',
    'Qwen2_5_VLForConditionalGeneration',
    'Qwen3OmniMoeForConditionalGeneration',
    'Qwen3VLForConditionalGeneration', 'Qwen3VLMoeForConditionalGeneration',
    'Qwen3VLNemotronEmbedModel',
    'Sarashina2VisionForCausalLM',
    'SkyworkR1VChatModel',
    'SmolVLMForConditionalGeneration',
    'Step3VLForConditionalGeneration', 'StepVLForConditionalGeneration',
    'Tarsier2ForConditionalGeneration', 'TarsierForConditionalGeneration',
    'YiVLForCausalLM',
}

EMBEDDING = {
    'BertForSequenceClassification', 'BertModel',
    'CLIPModel',
    'Contriever',
    'Gemma2ForSequenceClassification',
    'GteModel', 'GteNewModel',
    'GritLM',
    'InternLM2ForRewardModel',
    'JAISLMHeadModel',
    'JambaForSequenceClassification',
    'LLaDA2MoeModelLM',
    'LlamaEmbeddingModel',
    'LlamaForClassification',
    'LlamaForSequenceClassification', 'LlamaForSequenceClassificationWithNormal_Weights',
    'MiDashengLMModel',
    'MistralModel',
    'ModernBertModel',
    'NomicBertModel',
    'POINTSV15ChatModel',
    'PixtralVisionModel',
    'QWenLMHeadModel',
    'Qwen2ForRewardModel',
    'Qwen2ForSequenceClassification',
    'Qwen3ForSequenceClassification',
    'RobertaForSequenceClassification', 'RobertaModel',
    'XLMRobertaForSequenceClassification', 'XLMRobertaModel',
    'llama_NemoRetrieverColEmbed',
}

def categorize(arch):
    if arch in MULTIMODAL:
        return 'multimodal'
    if arch in EMBEDDING:
        return 'embedding'
    return 'text'

# --- Build rows ---
rows = []
for arch in sorted(all_archs):
    rows.append({
        'arch': arch,
        'native': arch in native,
        'hf': arch in hf_archs,
        'file': native.get(arch, ''),
        'example': hf_examples.get(arch, ''),
        'category': categorize(arch),
    })

# --- Generate markdown ---
def make_table(cat_rows):
    lines = [
        "| Architecture | Native | HF | SGLang File | Example HF Model |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(cat_rows, key=lambda x: x['arch']):
        n = 'Y' if r['native'] else ''
        h = 'Y' if r['hf'] else ''
        f = f"`{r['file']}`" if r['file'] else ''
        lines.append(f"| **{r['arch']}** | {n} | {h} | {f} | {r['example']} |")
    return '\n'.join(lines)

def stats(cat_rows):
    n = sum(1 for r in cat_rows if r['native'])
    hf = sum(1 for r in cat_rows if not r['native'] and r['hf'])
    no = sum(1 for r in cat_rows if not r['native'] and not r['hf'])
    return n, hf, no, len(cat_rows)

text_rows = [r for r in rows if r['category'] == 'text']
embed_rows = [r for r in rows if r['category'] == 'embedding']
multi_rows = [r for r in rows if r['category'] == 'multimodal']

t = stats(text_rows)
e = stats(embed_rows)
m = stats(multi_rows)
total = (t[0]+e[0]+m[0], t[1]+e[1]+m[1], t[2]+e[2]+m[2], t[3]+e[3]+m[3])

from datetime import date
today = date.today().isoformat()

doc = f"""# [Tracking] Model Support in SGLang — Ground Truth

Complete inventory of all model architectures: natively supported in SGLang, available via HuggingFace Transformers fallback, or unsupported.

If you need help implementing a new model, see https://docs.sglang.ai/supported_models/support_new_models.html

> **How to check support:** Use the `check-model-support` skill in `.claude/skills/check-model-support/SKILL.md`
>
> **Last updated:** {today} | **Source:** `EntryClass` in `python/sglang/srt/models/*.py` + `transformers` library classes + HuggingFace `config.json`

### Legend

| Column | Meaning |
|---|---|
| Native | `Y` = has optimized SGLang implementation via `EntryClass` |
| HF | `Y` = class exists in `transformers` library (may work via `TransformersForCausalLM` fallback) |
| SGLang File | Source file in `python/sglang/srt/models/` |
| Example HF Model | A known HuggingFace model repo using this architecture |

---

## Text-only Language Models (Generative)

{make_table(text_rows)}

**{t[0]}** native / **{t[1]}** HF-only / **{t[2]}** no fallback / **{t[3]}** total

---

## Embedding / Classification / Reward / Retrieval Models

{make_table(embed_rows)}

**{e[0]}** native / **{e[1]}** HF-only / **{e[2]}** no fallback / **{e[3]}** total

---

## Multimodal Models (Vision / Audio / OCR)

{make_table(multi_rows)}

**{m[0]}** native / **{m[1]}** HF-only / **{m[2]}** no fallback / **{m[3]}** total

---

## Overall Summary

| Category | Native | HF Fallback Only | No Fallback | Total |
|---|---|---|---|---|
| Text-only | {t[0]} | {t[1]} | {t[2]} | {t[3]} |
| Embedding/Classification | {e[0]} | {e[1]} | {e[2]} | {e[3]} |
| Multimodal | {m[0]} | {m[1]} | {m[2]} | {m[3]} |
| **Total** | **{total[0]}** | **{total[1]}** | **{total[2]}** | **{total[3]}** |

> **{total[0]}** natively supported ({total[0]*100//total[3]}%) — optimized SGLang implementations
> **{total[1]}** HF fallback only ({total[1]*100//total[3]}%) — may work via `TransformersForCausalLM`
> **{total[2]}** no fallback ({total[2]*100//total[3]}%) — need full implementation
"""

with open('MODEL_SUPPORT_TRACKING.md', 'w') as f:
    f.write(doc)

print(f"Wrote MODEL_SUPPORT_TRACKING.md ({total[3]} architectures)")
print(f"  Text-only:  {t[0]} native / {t[1]} HF-only / {t[2]} no fallback / {t[3]} total")
print(f"  Embedding:  {e[0]} native / {e[1]} HF-only / {e[2]} no fallback / {e[3]} total")
print(f"  Multimodal: {m[0]} native / {m[1]} HF-only / {m[2]} no fallback / {m[3]} total")
print(f"  TOTAL:      {total[0]} native / {total[1]} HF-only / {total[2]} no fallback / {total[3]} total")
PYEOF
```

---

## Quick Run (All Steps)

Run all steps sequentially from repo root:

```bash
# Step 1: Extract SGLang native architectures
python3 -c "
import os, re, json
models_dir = 'python/sglang/srt/models'
native = {}
for fname in sorted(os.listdir(models_dir)):
    if not fname.endswith('.py') or fname.startswith('_'): continue
    with open(os.path.join(models_dir, fname)) as f: content = f.read()
    for m in re.finditer(r'EntryClass\s*=\s*(.+)', content):
        val = m.group(1).strip()
        if val.startswith('['):
            bm = re.search(r'EntryClass\s*=\s*\[(.*?)\]', content[m.start():], re.DOTALL)
            if bm:
                for item in bm.group(1).replace('\n',',').split(','):
                    cls = item.strip()
                    if cls and re.match(r'^[A-Z]\w+$', cls) and cls not in native: native[cls] = fname
        else:
            for cls in val.split('#')[0].strip().rstrip(',').strip('[]').split(','):
                cls = cls.strip()
                if cls and re.match(r'^[A-Z]\w+$', cls) and cls not in native: native[cls] = fname
json.dump(native, open('/tmp/sglang_native_archs.json','w'), indent=2, sort_keys=True)
print(f'{len(native)} native architectures')
"

# Step 2: Extract transformers architectures
# (run the Step 2 script above)

# Step 3: Extract HF examples
# (run the Step 3 script above)

# Step 5: Generate final doc
# (run the Step 5 script above)
```

---

## Maintaining the Categorization Sets

When adding a new architecture, determine its category:

| If the model... | Category | Add to |
|---|---|---|
| Generates text from text input only | Text-only | (default, no set needed) |
| Processes images, audio, video, or OCR | Multimodal | `MULTIMODAL` set in Step 5 |
| Produces embeddings, classifications, or reward scores | Embedding | `EMBEDDING` set in Step 5 |
| Is an MTP/Eagle speculative decoding variant | Same as base model | Same set as base model |

**Naming conventions that help identify category:**
- `ForCausalLM` → usually text-only (except VL/Vision/Audio/OCR variants)
- `ForConditionalGeneration` → usually multimodal (except BART-like seq2seq)
- `ForSequenceClassification` / `ForRewardModel` / `EmbeddingModel` → embedding
- `VL` / `Vision` / `Audio` / `OCR` / `MM` in the name → multimodal

---

## Known Gotchas

1. **Multiline `EntryClass`**: `llama.py`, `llava.py`, `llama_reward.py` use `EntryClass = [\n  Class1,\n  Class2\n]`. Simple `grep -h "EntryClass" | sed` misses these. Always use the Python `re.DOTALL` approach.

2. **ANSI escape codes**: If you pipe grep output to a file and read it in Python, ANSI color codes (`\x1b[...m`) corrupt architecture names. Use `--color=never` or run extraction directly in Python with `subprocess`.

3. **Python 3.14+ regex**: Pattern `(?!\s)` causes `PatternError` in Python 3.14. Avoid lookahead assertions in extraction scripts.

4. **Missing HF examples**: SGLang-only architectures (no transformers class) won't have `_CHECKPOINT_FOR_DOC`. The `MANUAL` dict in Step 3 covers these — update it when adding new models.

5. **Placeholder model IDs**: Some auto-generated examples like `meta-arcee/Arcee-2-7b-hf` may not be real HF repos. Verify important entries with `curl -sI https://huggingface.co/ORG/MODEL`.
