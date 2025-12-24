"""
Search GitHub for SGLang server arguments and model commands,
extract valid commands, and document undocumented features.
"""

import argparse
import asyncio
import csv
import json
import os
import re
import traceback
from datetime import datetime

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from github import Auth, Github
from openai import AsyncOpenAI
from tqdm import tqdm

# Load .env if available
if load_dotenv is not None:
    load_dotenv()

# GitHub token
GH_TOKEN = os.getenv("GITHUB_TOKEN")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Repo
REPO = "sgl-project/sglang"

# Output files
OUTPUT_FILE = "spec_v2_results.jsonl"

# NVFP4 models to test
NVFP4_MODELS = [
    "nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4",
    "nvidia/Kimi-K2-Thinking-NVFP4",
    "nvidia/Qwen2.5-VL-7B-Instruct-NVFP4",
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-NVFP4",
    "nvidia/DeepSeek-V3.1-NVFP4",
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4",
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD",
    "nvidia/Llama-3.1-8B-Instruct-NVFP4",
    "nvidia/Qwen3-30B-A3B-NVFP4",
    "nvidia/Qwen3-32B-NVFP4",
    "nvidia/Qwen3-14B-NVFP4",
    "nvidia/Qwen3-8B-NVFP4",
    "nvidia/Phi-4-reasoning-plus-NVFP4",
    "nvidia/Phi-4-multimodal-instruct-NVFP4",
    "nvidia/DeepSeek-R1-0528-NVFP4-v2",
    "nvidia/DeepSeek-V3-0324-NVFP4",
    "nvidia/DeepSeek-R1-0528-NVFP4",
    "nvidia/Llama-3.3-70B-Instruct-NVFP4",
    "nvidia/DeepSeek-R1-NVFP4-v2",
    "nvidia/Qwen3-235B-A22B-NVFP4",
    "nvidia/DeepSeek-R1-NVFP4",
    "nvidia/Llama-3.1-405B-Instruct-NVFP4",
]

# FP8 models to test (cleaned up)
FP8_MODELS = [
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
    "nvidia/Qwen2.5-VL-7B-Instruct-FP8",
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8",
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8",
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8",
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1-FP8",
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1-FP8",
    "nvidia/Phi-4-reasoning-plus-FP8",
    "nvidia/Qwen3-14B-FP8",
    "nvidia/Qwen3-8B-FP8",
    "nvidia/Phi-4-multimodal-instruct-FP8",
    "nvidia/Llama-3.1-8B-Medusa-FP8",
    "nvidia/Llama-3.3-70B-Instruct-FP8",
    "nvidia/Llama-3.1-70B-Instruct-FP8",
    "nvidia/Llama-3.1-405B-Instruct-FP8",
    "nvidia/Llama-3.1-8B-Instruct-FP8",
    "nvidia/Nemotron-H-8B-Reasoning-128K-FP8",
    "nvidia/Nemotron-H-47B-Reasoning-128K-FP8",
    "nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "nvidia/Qwen3-235B-A22B-FP8",
]

# Format: "arg_name,description,default,type_info"
LIST_OF_ARGS_TO_TEST = [
    "--enable-dynamic-batch-tokenizer,Enable async dynamic batch tokenizer for improved performance when multiple requests arrive concurrently.,False,bool flag (set to enable)",
    "--enable-piecewise-cuda-graph,Optimize the model with piecewise cuda graph for extend/prefill only. Experimental feature.,False,bool flag (set to enable)",
    "--enable-torch-compile,Optimize the model with torch.compile. Experimental feature.,False,bool flag (set to enable)",
    "--enable-two-batch-overlap,Enabling two micro batches to overlap.,False,bool flag (set to enable)",
    "--enable-single-batch-overlap,Let computation and communication overlap within one micro batch.,False,bool flag (set to enable)",
    "--enable-mixed-chunk,Enabling mixing prefill and decode in a batch when using chunked prefill.,False,bool flag (set to enable)",
    "--enable-dp-attention,Enabling data parallelism for attention and tensor parallelism for FFN. The dp size should be equal to the tp size. Currently DeepSeek-V2 and Qwen 2/3 MoE models are supported.,False,bool flag (set to enable)",
    "--enable-dp-lm-head,Enable vocabulary parallel across the attention TP group to avoid all-gather across DP groups optimizing performance under DP attention.,False,bool flag (set to enable)",
    "--enable-hierarchical-cache,Enable hierarchical cache,False,bool flag (set to enable)",
    "--mamba-scheduler-strategy,The strategy to use for mamba scheduler. auto defaults to no_buffer. no_buffer does not support overlap scheduler. extra_buffer supports overlap schedule by allocating extra mamba state buffers.,auto,auto no_buffer extra_buffer",
    "--max-mamba-cache-size,The maximum size of the mamba cache.,None,Type: int",
    "--enable-eplb,Enable EPLB algorithm,False,bool flag (set to enable)",
    "--deepep-mode,Select the mode when enable DeepEP MoE could be normal low_latency or auto. Default is auto which means low_latency for decode batch and normal for prefill batch.,auto,normal low_latency auto",
    "--enable-flashinfer-allreduce-fusion,Enable FlashInfer allreduce fusion with Residual RMSNorm.,False,bool flag (set to enable)",
    "--speculative-moe-runner-backend,MOE backend for EAGLE speculative decoding see --moe-runner-backend for options. Same as moe runner backend if unset.,None,",
    "--speculative-num-steps,The number of steps sampled from draft model in Speculative Decoding.,None,Type: int",
    "--speculative-eagle-topk,The number of tokens sampled from the draft model in eagle2 each step.,None,Type: int",
    "--speculative-num-draft-tokens,The number of tokens sampled from the draft model in Speculative Decoding.,None,Type: int",
    "--page-size,The number of tokens in a page.,1,Type: int",
    "--chunked-prefill-size,The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill.,None,Type: int",
    "--modelopt-quant,The ModelOpt quantization configuration. Supported values: fp8 int4_awq w4a8_awq nvfp4 nvfp4_awq. Requires NVIDIA Model Optimizer library.,None,Type: str",
    "--quantization,The quantization method.,None,modelopt_fp4 petit_nvfp4 modelopt",
]


def parse_arg_info(arg_string):
    """Parse arg string into components."""
    parts = arg_string.split(",", 3)
    return {
        "arg_name": parts[0] if len(parts) > 0 else "",
        "description": parts[1] if len(parts) > 1 else "",
        "default": parts[2] if len(parts) > 2 else "",
        "type_info": parts[3] if len(parts) > 3 else "",
    }


def extract_json_array(text):
    """Extract a JSON array from a model response, handling fenced blocks."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


def search_github(query):
    """Search for issues and PRs matching the query."""
    g = Github(auth=Auth.Token(GH_TOKEN))

    all_results = []

    searches = [
        ("pr", "open"),
        ("pr", "closed"),
        ("issue", "open"),
        ("issue", "closed"),
    ]

    for item_type, state in searches:
        full_query = f"{query} repo:{REPO} is:{item_type} is:{state}"
        print(f"Searching: {full_query}")

        results = g.search_issues(query=full_query)

        for item in results:
            is_pr = (
                item.pull_request is not None and item.pull_request.html_url is not None
            )

            result = {
                "number": item.number,
                "title": item.title,
                "type": "pr" if is_pr else "issue",
                "state": item.state,
                "body": item.body or "",
                "html_url": item.html_url,
                "author": item.user.login if item.user else "unknown",
                "created_at": item.created_at.isoformat() if item.created_at else "",
                "updated_at": item.updated_at.isoformat() if item.updated_at else "",
                "labels": [label.name for label in item.labels],
                "comments_count": item.comments,
            }

            repo = g.get_repo(REPO)
            if is_pr:
                pr = repo.get_pull(item.number)
                result["merged"] = pr.merged
                result["merge_commit_sha"] = pr.merge_commit_sha
                comments = []
                for c in pr.get_issue_comments():
                    comments.append(
                        {
                            "author": c.user.login if c.user else "unknown",
                            "body": c.body,
                            "created_at": (
                                c.created_at.isoformat() if c.created_at else ""
                            ),
                        }
                    )
                result["comments"] = comments

                review_comments = []
                for c in pr.get_review_comments():
                    review_comments.append(
                        {
                            "author": c.user.login if c.user else "unknown",
                            "body": c.body,
                            "path": c.path,
                            "created_at": (
                                c.created_at.isoformat() if c.created_at else ""
                            ),
                        }
                    )
                result["review_comments"] = review_comments
            else:
                issue = repo.get_issue(item.number)
                comments = []
                for c in issue.get_comments():
                    comments.append(
                        {
                            "author": c.user.login if c.user else "unknown",
                            "body": c.body,
                            "created_at": (
                                c.created_at.isoformat() if c.created_at else ""
                            ),
                        }
                    )
                result["comments"] = comments

            all_results.append(result)
            print(f"  Found: #{item.number} - {item.title[:50]}...")

    return all_results


def collect_search_results(queries):
    """Search multiple queries and deduplicate results by URL."""
    dedup = {}
    for q in queries:
        print("=" * 60)
        print(f"Searching GitHub for: {q}")
        print("=" * 60)
        results = search_github(q)
        for item in results:
            dedup[item["html_url"]] = item
    return list(dedup.values())


def save_jsonl(results, filename):
    """Save results to JSONL format."""
    with open(filename, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"\nSaved {len(results)} results to {filename}")


def _slugify(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "query"


def _build_arg_info(arg_name):
    arg_info = None
    for arg_str in LIST_OF_ARGS_TO_TEST:
        info = parse_arg_info(arg_str)
        if info["arg_name"] == arg_name or info["arg_name"].lstrip(
            "-"
        ) == arg_name.lstrip("-"):
            arg_info = info
            break
    if arg_info is None:
        arg_info = {
            "arg_name": arg_name if arg_name.startswith("-") else f"--{arg_name}",
            "description": "Unknown argument",
            "default": "Unknown",
            "type_info": "Unknown",
        }
    return arg_info


def run_query_list(queries, query_mode, output_prefix, concurrency):
    """For each query, search GitHub and run the LLM extraction pipeline."""
    for term in queries:
        mode = query_mode
        print("=" * 60)
        print(f"Searching GitHub for: {term}")
        print("=" * 60)
        results = search_github(term)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        search_output = f"{output_prefix}_search_{_slugify(term)}_{timestamp}.jsonl"
        save_jsonl(results, search_output)

        if mode == "arg":
            arg_info = _build_arg_info(term)
            process_entries_for_arg(
                search_output, arg_info, output_prefix, concurrency=concurrency
            )
        else:
            process_entries_for_model(
                search_output, term, output_prefix, concurrency=concurrency
            )


def format_for_llm(result):
    """Format a single result as context for LLM."""
    lines = [
        f"## {result['type'].upper()} #{result['number']}: {result['title']}",
        f"**State:** {result['state']} | **Author:** {result['author']}",
        f"**Created:** {result['created_at']} | **Updated:** {result['updated_at']}",
        f"**Labels:** {', '.join(result['labels']) if result['labels'] else 'None'}",
        f"**URL:** {result['html_url']}",
        "",
        "### Description",
        result["body"] or "(No description)",
    ]

    if result.get("comments"):
        lines.append("")
        lines.append(f"### Comments ({len(result['comments'])})")
        for c in result["comments"]:
            lines.append(f"\n**{c['author']}** ({c['created_at']}):")
            lines.append(c["body"])

    if result.get("review_comments"):
        lines.append("")
        lines.append(f"### Review Comments ({len(result['review_comments'])})")
        for c in result["review_comments"]:
            lines.append(f"\n**{c['author']}** on `{c.get('path', 'unknown')}`:")
            lines.append(c["body"])

    return "\n".join(lines)


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

PROMPT_TEMPLATE_SERVER_ARG_GENERIC = """Below is a GitHub issue/PR that may contain usage of the SGLang server argument `{arg_name}`.

{context}

---

## Background

You are helping document the SGLang server argument `{arg_name}`.

**Argument info:**
- Name: `{arg_name}`
- Description: {arg_description}
- Default: {arg_default}
- Type: {arg_type}

Many SGLang arguments are undocumented or poorly documented. By finding real-world commands and understanding the context (debugging sessions, performance tests, bug reports), we can create better documentation.

## Your Task

Extract ALL commands from the above GitHub issue/PR that use `{arg_name}`. There may be MULTIPLE commands - extract each one separately.

## Output Format

Return a JSON array. Each command gets its own object:
- `command`: Full executable command (env vars inline, no `export`)
- `arg_value`: The value used for `{arg_name}` (or "flag" if boolean flag)
- `context`: Why this command was run - include relevant details about what was being tested or debugged
- `outcome`: What happened when this was run - include specific errors, performance numbers, or success details mentioned
- `related_args`: List of other notable args used alongside `{arg_name}` that may interact with it
- `model`: The model being used (if identifiable)
- `status`: "working", "broken", or "experimental"

## Example Output

```json
[
  {{
    "command": "python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B --{arg_name_stripped} --tp 2 --attention-backend triton",
    "arg_value": "flag",
    "context": "User was benchmarking throughput on 2xA100-80GB setup, comparing performance with and without the flag to measure overhead. This was part of a larger performance investigation for production deployment.",
    "outcome": "Working. Achieved 1850 tok/s output throughput with flag enabled vs 1620 tok/s without. The 14% improvement was consistent across multiple runs with batch size 32.",
    "related_args": ["--tp 2", "--attention-backend triton"],
    "model": "meta-llama/Llama-3.1-8B",
    "status": "working"
  }},
  {{
    "command": "python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --{arg_name_stripped} --tp 8 --enable-dp-attention",
    "arg_value": "flag",
    "context": "Bug report: User attempting to run DeepSeek-V3 on 8xH100 with DP attention enabled. The flag was suggested by maintainers as a potential fix for memory fragmentation issues seen in earlier runs.",
    "outcome": "BROKEN: Server crashes during CUDA graph capture phase with 'RuntimeError: CUDA error: invalid configuration argument at line 342 in attention_kernel.cu'. The combination of this flag with --enable-dp-attention appears incompatible on multi-node setups.",
    "related_args": ["--tp 8", "--enable-dp-attention"],
    "model": "deepseek-ai/DeepSeek-V3",
    "status": "broken"
  }},
  {{
    "command": "SGLANG_ENABLE_SPEC_V2=1 python -m sglang.launch_server --model-path Qwen/Qwen2.5-72B-Instruct --{arg_name_stripped} --tp 4 --speculative-algorithm EAGLE",
    "arg_value": "flag",
    "context": "Developer testing experimental integration of this flag with speculative decoding. Part of PR implementing overlap between draft and verify phases.",
    "outcome": "Experimental - runs without crash but acceptance rate dropped from 0.85 to 0.72. Developer noted this needs further investigation before the combination can be recommended.",
    "related_args": ["--tp 4", "--speculative-algorithm EAGLE", "SGLANG_ENABLE_SPEC_V2=1"],
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "status": "experimental"
  }}
]
```

## Rules

1. Extract EACH command separately with its own context/outcome
2. Convert `export VAR=value` to inline `VAR=value command` format
3. Only include commands that actually use `{arg_name}`
4. Include meaningful detail in context and outcome - specific numbers, error messages, hardware info when available
5. If no commands use `{arg_name}`, return empty array: `[]`
6. Return ONLY the JSON array, no additional text
"""


PROMPT_TEMPLATE_MODEL_RUN_CMD = """Below is a GitHub issue/PR that may contain commands for running the model `{model_id}`.

{context}

---

## Background

You are helping find optimal run configurations for the model `{model_id}`.

By collecting real-world commands from issues/PRs, we can identify:
- Recommended configurations that work well
- Configurations that cause problems
- Performance tuning tips
- Hardware requirements

## Your Task

Extract ALL commands from the above GitHub issue/PR that run `{model_id}`. There may be MULTIPLE commands - extract each one separately.

## Output Format

Return a JSON array. Each command gets its own object:
- `command`: Full executable command (env vars inline, no `export`)
- `hardware`: GPU info if mentioned (e.g., "8xH100", "4xA100-80GB")
- `purpose`: Why was this command run - include details about what was being tested, benchmarked, or debugged
- `result`: What happened - include throughput/latency numbers if mentioned, errors if it failed, or specific observations about behavior
- `status`: "working", "broken", or "experimental"

## Example Output

```json
[
  {{
    "command": "python -m sglang.launch_server --model-path {model_id} --tp 8 --attention-backend fa3 --enable-dp-attention --mem-fraction-static 0.9",
    "hardware": "8xH100-80GB NVLink",
    "purpose": "Production deployment benchmark for high-throughput serving. Testing DP attention with FA3 backend to maximize throughput while maintaining low latency for a chatbot application with ~50 concurrent users.",
    "result": "Working well. Achieved 2850 output tok/s at batch size 64, p50 latency 45ms, p99 latency 120ms. Memory usage stable at 76GB per GPU. Ran for 48 hours without issues.",
    "status": "working"
  }},
  {{
    "command": "python -m sglang.launch_server --model-path {model_id} --tp 4 --quantization fp8",
    "hardware": "4xA100-40GB",
    "purpose": "Testing FP8 quantization to fit model on smaller GPUs. User migrating from 8xA100-80GB setup and needs to reduce memory footprint while maintaining acceptable quality.",
    "result": "BROKEN: OOM during model loading at default settings. Adding --mem-fraction-static 0.8 allowed loading but inference failed with 'CUDA out of memory' during first batch. Needs 80GB GPUs or further memory optimizations.",
    "status": "broken"
  }},
  {{
    "command": "SGLANG_ENABLE_SPEC_V2=1 python -m sglang.launch_server --model-path {model_id} --tp 8 --speculative-algorithm EAGLE --speculative-draft-model-path {model_id}-EAGLE",
    "hardware": "8xH100-80GB",
    "purpose": "Testing speculative decoding with EAGLE to improve single-request latency. Developer evaluating whether spec decoding provides meaningful speedup for this model architecture.",
    "result": "Experimental. Server starts and serves requests. Seeing 1.4x speedup on average but acceptance rate varies significantly (0.65-0.85) depending on prompt type. Code generation prompts show better acceptance than creative writing.",
    "status": "experimental"
  }}
]
```

## Rules

1. Extract EACH command separately
2. Convert `export VAR=value` to inline `VAR=value command` format
3. Only include commands that run `{model_id}` (or close variants like quantized versions)
4. Include meaningful detail in purpose and result - specific numbers, error messages, observations when available
5. If no commands for this model, return empty array: `[]`
6. Return ONLY the JSON array, no additional text
"""


def process_entries_for_arg(jsonl_file, arg_info, output_prefix, concurrency=8):
    """Process entries looking for a specific server argument."""

    async def _run():
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        results = []
        with open(jsonl_file, "r") as f:
            for line in f:
                results.append(json.loads(line))

        arg_name = arg_info["arg_name"]
        arg_name_stripped = arg_name.lstrip("-")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_jsonl = f"{output_prefix}_{arg_name_stripped}_{timestamp}.jsonl"
        output_csv = f"{output_prefix}_{arg_name_stripped}_{timestamp}.csv"

        all_commands = []
        outputs = [None] * len(results)
        sem = asyncio.Semaphore(concurrency)

        async def handle(idx, result):
            async with sem:
                context = format_for_llm(result)
                prompt = PROMPT_TEMPLATE_SERVER_ARG_GENERIC.format(
                    context=context,
                    arg_name=arg_name,
                    arg_name_stripped=arg_name_stripped,
                    arg_description=arg_info["description"],
                    arg_default=arg_info["default"],
                    arg_type=arg_info["type_info"],
                )

                try:
                    completion = await client.chat.completions.create(
                        model="google/gemini-3-flash-preview",
                        messages=[
                            {
                                "role": "system",
                                "content": "You extract commands from GitHub issues/PRs. Return only valid JSON arrays.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                    )

                    response = completion.choices[0].message.content
                    output = {
                        "number": result["number"],
                        "title": result["title"],
                        "type": result["type"],
                        "arg_analyzed": arg_name,
                        "extracted_commands": response,
                    }

                    commands = extract_json_array(response)
                    parsed_commands = []
                    if isinstance(commands, list):
                        for cmd in commands:
                            cmd["arg_analyzed"] = arg_name
                            cmd["issue_number"] = result["html_url"]
                            cmd["issue_title"] = result["title"]
                            parsed_commands.append(cmd)
                    elif commands is None:
                        tqdm.write(
                            f"Parse failed for #{result['number']} ({result['title'][:80]})."
                        )

                    return idx, output, parsed_commands
                except Exception as e:
                    traceback.print_exc()
                    tqdm.write(
                        f"Error on #{result['number']} ({result['title'][:80]}): {e}"
                    )
                    tqdm.write(
                        f"Arg: {arg_name} | Input: {jsonl_file} | Output: {output_jsonl}"
                    )
                    tqdm.write(
                        "Hint: OpenRouter connection errors are often transient or due to rate limits."
                    )
                    return idx, None, []

        tasks = [asyncio.create_task(handle(i, r)) for i, r in enumerate(results)]
        pbar = tqdm(total=len(tasks), desc=f"Processing for {arg_name}")
        for coro in asyncio.as_completed(tasks):
            idx, output, parsed_commands = await coro
            outputs[idx] = output
            all_commands.extend(parsed_commands)
            pbar.update(1)
        pbar.close()

        with open(output_jsonl, "w") as out_f:
            for output in outputs:
                if output is None:
                    continue
                out_f.write(json.dumps(output) + "\n")

        # Write CSV
        if all_commands:
            csv_fields = [
                "arg_analyzed",
                "command",
                "arg_value",
                "context",
                "outcome",
                "related_args",
                "model",
                "status",
                "issue_number",
                "issue_title",
            ]
            with open(output_csv, "w", newline="") as csvfile:
                writer = csv.DictWriter(
                    csvfile, fieldnames=csv_fields, extrasaction="ignore"
                )
                writer.writeheader()
                for cmd in all_commands:
                    # Convert lists to strings for CSV
                    if isinstance(cmd.get("related_args"), list):
                        cmd["related_args"] = "; ".join(cmd["related_args"])
                    writer.writerow(cmd)
            print(f"Saved CSV to {output_csv}")

        print(f"Saved JSONL to {output_jsonl}")
        return all_commands

    return asyncio.run(_run())


def process_entries_for_model(jsonl_file, model_id, output_prefix, concurrency=32):
    """Process entries looking for a specific model."""

    async def _run():
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY
        )

        results = []
        with open(jsonl_file, "r") as f:
            for line in f:
                results.append(json.loads(line))

        model_name = model_id.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_jsonl = f"{output_prefix}_model_{model_name}_{timestamp}.jsonl"
        output_csv = f"{output_prefix}_model_{model_name}_{timestamp}.csv"

        all_commands = []
        outputs = [None] * len(results)
        sem = asyncio.Semaphore(concurrency)

        async def handle(idx, result):
            async with sem:
                context = format_for_llm(result)
                prompt = PROMPT_TEMPLATE_MODEL_RUN_CMD.format(
                    context=context,
                    model_id=model_id,
                )

                try:
                    completion = await client.chat.completions.create(
                        model="google/gemini-3-flash-preview",
                        messages=[
                            {
                                "role": "system",
                                "content": "You extract commands from GitHub issues/PRs. Return only valid JSON arrays.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                    )

                    response = completion.choices[0].message.content
                    output = {
                        "number": result["number"],
                        "title": result["title"],
                        "type": result["type"],
                        "model_analyzed": model_id,
                        "extracted_commands": response,
                    }

                    commands = extract_json_array(response)
                    parsed_commands = []
                    if isinstance(commands, list):
                        for cmd in commands:
                            cmd["model_analyzed"] = model_id
                            cmd["issue_number"] = result["html_url"]
                            cmd["issue_title"] = result["title"]
                            parsed_commands.append(cmd)
                    elif commands is None:
                        tqdm.write(
                            f"Parse failed for #{result['number']} ({result['title'][:80]})."
                        )

                    return idx, output, parsed_commands

                except Exception as e:
                    traceback.print_exc()
                    tqdm.write(
                        f"Error on #{result['number']} ({result['title'][:80]}): {e}"
                    )
                    tqdm.write(
                        f"Model: {model_id} | Input: {jsonl_file} | Output: {output_jsonl}"
                    )
                    tqdm.write(
                        "Hint: OpenRouter connection errors are often transient or due to rate limits."
                    )
                    return idx, None, []

        tasks = [asyncio.create_task(handle(i, r)) for i, r in enumerate(results)]
        pbar = tqdm(total=len(tasks), desc=f"Processing for {model_id}")
        for coro in asyncio.as_completed(tasks):
            idx, output, parsed_commands = await coro
            outputs[idx] = output
            all_commands.extend(parsed_commands)
            pbar.update(1)
        pbar.close()

        with open(output_jsonl, "w") as out_f:
            for output in outputs:
                if output is None:
                    continue
                out_f.write(json.dumps(output) + "\n")

        # Write CSV
        if all_commands:
            csv_fields = [
                "model_analyzed",
                "command",
                "hardware",
                "purpose",
                "result",
                "status",
                "issue_number",
                "issue_title",
            ]
            with open(output_csv, "w", newline="") as csvfile:
                writer = csv.DictWriter(
                    csvfile, fieldnames=csv_fields, extrasaction="ignore"
                )
                writer.writeheader()
                for cmd in all_commands:
                    writer.writerow(cmd)
            print(f"Saved CSV to {output_csv}")

        print(f"Saved JSONL to {output_jsonl}")
        return all_commands

    return asyncio.run(_run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search GitHub and extract SGLang commands"
    )
    parser.add_argument(
        "--search", type=str, help="Search query to fetch new issues/PRs"
    )
    parser.add_argument(
        "--process-arg",
        type=str,
        help="Process JSONL for a specific server argument (e.g., --enable-dp-attention)",
    )
    parser.add_argument(
        "--process-model",
        type=str,
        help="Process JSONL for a specific model ID (e.g., deepseek-ai/DeepSeek-V3)",
    )
    parser.add_argument(
        "--test-args",
        action="store_true",
        help="Test first half of LIST_OF_ARGS_TO_TEST",
    )
    parser.add_argument(
        "--all-args",
        action="store_true",
        help="Process all args in LIST_OF_ARGS_TO_TEST",
    )
    parser.add_argument(
        "--all-nvfp4",
        action="store_true",
        help="Process all models in NVFP4_MODELS",
    )
    parser.add_argument(
        "--all-fp8",
        action="store_true",
        help="Process all models in FP8_MODELS",
    )
    parser.add_argument(
        "--input", type=str, default=OUTPUT_FILE, help="Input JSONL file"
    )
    parser.add_argument(
        "--output-prefix", type=str, default="arg_analysis", help="Output file prefix"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max concurrent LLM requests when processing",
    )
    parser.add_argument(
        "--list-args", action="store_true", help="List all available args to test"
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Run search and then process results in one run (requires --search)",
    )
    parser.add_argument(
        "--query-list",
        type=str,
        help="Comma-separated list of queries; each will be searched and processed",
    )
    parser.add_argument(
        "--query-file",
        type=str,
        help="File with one query per line; each will be searched and processed",
    )
    parser.add_argument(
        "--query-mode",
        type=str,
        choices=["model", "arg"],
        help="How to interpret query items (model/arg). Required with --query-list/--query-file.",
    )

    args = parser.parse_args()

    if args.list_args:
        print("Available arguments to test:")
        for i, arg_str in enumerate(LIST_OF_ARGS_TO_TEST):
            info = parse_arg_info(arg_str)
            print(f"  {i + 1}. {info['arg_name']}: {info['description'][:60]}...")
        exit(0)

    did_action = False
    query_items = []
    if args.query_list:
        query_items.extend([q.strip() for q in args.query_list.split(",") if q.strip()])
    if args.query_file:
        with open(args.query_file, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                query_items.append(s)

    if args.e2e and args.all_nvfp4:
        if args.search:
            print(
                "[e2e] --all-nvfp4 provided; ignoring --search and querying each model."
            )
        run_query_list(
            NVFP4_MODELS, "model", args.output_prefix, concurrency=args.concurrency
        )
        did_action = True
    elif args.e2e and args.all_fp8:
        if args.search:
            print(
                "[e2e] --all-fp8 provided; ignoring --search and querying each model."
            )
        run_query_list(
            FP8_MODELS, "model", args.output_prefix, concurrency=args.concurrency
        )
        did_action = True
    elif query_items:
        if not args.query_mode:
            raise SystemExit(
                "--query-mode is required with --query-list/--query-file. Use 'model' or 'arg'."
            )
        run_query_list(
            query_items,
            args.query_mode,
            args.output_prefix,
            concurrency=args.concurrency,
        )
        did_action = True

    if args.e2e and not did_action:
        if not args.search:
            if args.process_arg:
                args.search = args.process_arg
                print(f"[e2e] No --search provided; using --process-arg {args.search}")
            elif args.process_model:
                args.search = args.process_model
                print(
                    f"[e2e] No --search provided; using --process-model {args.search}"
                )
            else:
                raise SystemExit(
                    "--e2e requires --search, --process-arg, or --process-model."
                )
        else:
            if args.process_arg and args.search != args.process_arg:
                print(
                    f"[e2e] Warning: --search '{args.search}' != --process-arg '{args.process_arg}'. Using --search."
                )
            if args.process_model and args.search != args.process_model:
                print(
                    f"[e2e] Warning: --search '{args.search}' != --process-model '{args.process_model}'. Using --search."
                )
        print("=" * 60)
        print(f"Searching GitHub for: {args.search}")
        print("=" * 60)
        results = search_github(args.search)
        save_jsonl(results, args.input)
        if not (
            args.process_arg
            or args.process_model
            or args.test_args
            or args.all_args
            or args.all_nvfp4
            or args.all_fp8
        ):
            args.process_model = args.search
            print(
                f"[e2e] No processing flag provided; using --process-model {args.process_model}"
            )
        # Run the processing step directly for e2e
        if args.process_arg:
            arg_info = _build_arg_info(args.process_arg)
            process_entries_for_arg(
                args.input,
                arg_info,
                args.output_prefix,
                concurrency=args.concurrency,
            )
        elif args.process_model:
            process_entries_for_model(
                args.input,
                args.process_model,
                args.output_prefix,
                concurrency=args.concurrency,
            )
        elif args.test_args:
            half = len(LIST_OF_ARGS_TO_TEST) // 2
            print(f"Testing first {half} arguments...")
            for arg_str in LIST_OF_ARGS_TO_TEST[:half]:
                arg_info = parse_arg_info(arg_str)
                print(f"\n{'=' * 60}")
                print(f"Processing: {arg_info['arg_name']}")
                print(f"{'=' * 60}")
                process_entries_for_arg(
                    args.input,
                    arg_info,
                    args.output_prefix,
                    concurrency=args.concurrency,
                )
        elif args.all_args:
            print(f"Processing all {len(LIST_OF_ARGS_TO_TEST)} arguments...")
            for arg_str in LIST_OF_ARGS_TO_TEST:
                arg_info = parse_arg_info(arg_str)
                print(f"\n{'=' * 60}")
                print(f"Processing: {arg_info['arg_name']}")
                print(f"{'=' * 60}")
                process_entries_for_arg(
                    args.input,
                    arg_info,
                    args.output_prefix,
                    concurrency=args.concurrency,
                )
        did_action = True

    if args.search and not args.e2e:
        print("=" * 60)
        print(f"Searching GitHub for: {args.search}")
        print("=" * 60)
        results = search_github(args.search)
        save_jsonl(results, args.input)
        did_action = True

    if args.process_arg and not did_action:
        # Find the arg info
        arg_info = None
        for arg_str in LIST_OF_ARGS_TO_TEST:
            info = parse_arg_info(arg_str)
            if info["arg_name"] == args.process_arg or info["arg_name"].lstrip(
                "-"
            ) == args.process_arg.lstrip("-"):
                arg_info = info
                break

        if arg_info is None:
            # Create basic info if not in list
            arg_info = {
                "arg_name": (
                    args.process_arg
                    if args.process_arg.startswith("-")
                    else f"--{args.process_arg}"
                ),
                "description": "Unknown argument",
                "default": "Unknown",
                "type_info": "Unknown",
            }

        process_entries_for_arg(
            args.input, arg_info, args.output_prefix, concurrency=args.concurrency
        )
        did_action = True

    elif args.process_model and not did_action:
        process_entries_for_model(
            args.input,
            args.process_model,
            args.output_prefix,
            concurrency=args.concurrency,
        )
        did_action = True
    elif args.all_nvfp4 and not did_action:
        print(f"Processing all {len(NVFP4_MODELS)} NVFP4 models...")
        for model_id in NVFP4_MODELS:
            print(f"\n{'=' * 60}")
            print(f"Processing model: {model_id}")
            print(f"{'=' * 60}")
            process_entries_for_model(
                args.input,
                model_id,
                args.output_prefix,
                concurrency=args.concurrency,
            )
        did_action = True
    elif args.all_fp8 and not did_action:
        print(f"Processing all {len(FP8_MODELS)} FP8 models...")
        for model_id in FP8_MODELS:
            print(f"\n{'=' * 60}")
            print(f"Processing model: {model_id}")
            print(f"{'=' * 60}")
            process_entries_for_model(
                args.input,
                model_id,
                args.output_prefix,
                concurrency=args.concurrency,
            )
        did_action = True

    elif args.test_args:
        # Test first half of args
        half = len(LIST_OF_ARGS_TO_TEST) // 2
        print(f"Testing first {half} arguments...")
        for arg_str in LIST_OF_ARGS_TO_TEST[:half]:
            arg_info = parse_arg_info(arg_str)
            print(f"\n{'=' * 60}")
            print(f"Processing: {arg_info['arg_name']}")
            print(f"{'=' * 60}")
            process_entries_for_arg(
                args.input, arg_info, args.output_prefix, concurrency=args.concurrency
            )
        did_action = True

    elif args.all_args:
        print(f"Processing all {len(LIST_OF_ARGS_TO_TEST)} arguments...")
        for arg_str in LIST_OF_ARGS_TO_TEST:
            arg_info = parse_arg_info(arg_str)
            print(f"\n{'=' * 60}")
            print(f"Processing: {arg_info['arg_name']}")
            print(f"{'=' * 60}")
            process_entries_for_arg(
                args.input, arg_info, args.output_prefix, concurrency=args.concurrency
            )
        did_action = True

    if not did_action:
        parser.print_help()
        print(
            """
Paths (pick one):

1) Search only (collect issues/PRs into JSONL):
   python search_spec_v2.py --search SGLANG_ENABLE_SPEC_V2=1

2) Process only (use an existing JSONL, no GitHub search):
   python search_spec_v2.py --process-arg --enable-dp-attention
   python search_spec_v2.py --process-model deepseek-ai/DeepSeek-V3

3) E2E (search + process in one run):
   python search_spec_v2.py --e2e --search SGLANG_ENABLE_SPEC_V2=1 --process-arg --enable-dp-attention
   python search_spec_v2.py --e2e --search deepseek-ai/DeepSeek-V3
   python search_spec_v2.py --e2e --process-arg --enable-dp-attention
   python search_spec_v2.py --e2e --process-model deepseek-ai/DeepSeek-V3
   python search_spec_v2.py --e2e --all-nvfp4
   python search_spec_v2.py --e2e --all-fp8

4) Query list (each entry is searched and processed):
   python search_spec_v2.py --query-list "deepseek-ai/DeepSeek-V3,--enable-dp-attention" --query-mode model
   python search_spec_v2.py --query-file queries.txt --query-mode arg

Why search vs process can look similar:
- --search is a GitHub query (can be broad or different from the arg/model you want).
- --process-arg/--process-model is the exact thing you want extracted.
Use the same string when you want the search to be tightly scoped; use different strings when you want a broader search but still extract a specific arg/model.
"""
        )
