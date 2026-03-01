"""
Run one test prompt using the /generate endpoint with chat template applied client-side.

This gives access to rich meta_info (spec decoding stats, server-side latency, etc.)
while still using the chat message format. Results are appended to send_1_results.jsonl.

Usage:
python3 -m sglang.test.send_1
python3 -m sglang.test.send_1 --stream
python3 -m sglang.test.send_1 --output results.jsonl
python3 -m sglang.test.send_1 --profile --profile-steps 5
python3 -m sglang.test.send_1 --profile --profile-by-stage
"""

import argparse
import dataclasses
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests
from rich.console import Console
from rich.table import Table

from sglang.profiler import run_profile

EST = timezone(timedelta(hours=-5), "EST")
PST = timezone(timedelta(hours=-8), "PST")
TIME_FMT = "%Y-%m-%d %H:%M:%S %Z"


def get_timestamps() -> dict:
    """Return human-readable timestamps for EST, PST, and UTC."""
    now = datetime.now(timezone.utc)
    return {
        "utc": now.strftime(TIME_FMT),
        "est": now.astimezone(EST).strftime(TIME_FMT),
        "pst": now.astimezone(PST).strftime(TIME_FMT),
    }


@dataclasses.dataclass
class BenchArgs:
    host: str = "localhost"
    port: int = 30000
    batch_size: int = 1
    different_prompts: bool = False
    seed: Optional[int] = None
    temperature: float = 0.0
    max_new_tokens: int = 32000
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    json: bool = False
    return_logprob: bool = False
    prompt: str = "Give me a fully functional FastAPI server. Show the python code."
    image: bool = False
    many_images: bool = False
    stream: bool = False
    profile: bool = False
    profile_steps: int = 3
    profile_by_stage: bool = False
    profile_prefix: Optional[str] = None
    output: str = "send_1_results.jsonl"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--host", type=str, default=BenchArgs.host)
        parser.add_argument("--port", type=int, default=BenchArgs.port)
        parser.add_argument("--batch-size", type=int, default=BenchArgs.batch_size)
        parser.add_argument(
            "--different-prompts",
            action="store_true",
            default=BenchArgs.different_prompts,
        )
        parser.add_argument("--seed", type=int, default=BenchArgs.seed)
        parser.add_argument("--temperature", type=float, default=BenchArgs.temperature)
        parser.add_argument(
            "--max-new-tokens", type=int, default=BenchArgs.max_new_tokens
        )
        parser.add_argument(
            "--frequency-penalty", type=float, default=BenchArgs.frequency_penalty
        )
        parser.add_argument(
            "--presence-penalty", type=float, default=BenchArgs.presence_penalty
        )
        parser.add_argument("--json", action="store_true")
        parser.add_argument("--return-logprob", action="store_true")
        parser.add_argument("--prompt", type=str, default=BenchArgs.prompt)
        parser.add_argument("--image", action="store_true")
        parser.add_argument("--many-images", action="store_true")
        parser.add_argument("--stream", action="store_true")
        parser.add_argument("--profile", action="store_true")
        parser.add_argument(
            "--profile-steps", type=int, default=BenchArgs.profile_steps
        )
        parser.add_argument("--profile-by-stage", action="store_true")
        parser.add_argument(
            "--profile-prefix", type=str, default=BenchArgs.profile_prefix
        )
        parser.add_argument("--output", type=str, default=BenchArgs.output)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def get_tokenizer(base_url: str):
    """Fetch the model's tokenizer path from the server and load it."""
    from transformers import AutoTokenizer

    resp = requests.get(f"{base_url}/model_info")
    model_path = resp.json()["tokenizer_path"]
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def apply_chat_template(tokenizer, prompt: str, image_data=None) -> str:
    """Apply the model's chat template to a user prompt."""
    if image_data:
        content = [{"type": "text", "text": prompt}]
        if isinstance(image_data, list):
            for url in image_data:
                content.append({"type": "image_url", "image_url": {"url": url}})
        else:
            content.append(
                {"type": "image_url", "image_url": {"url": image_data}}
            )
    else:
        content = prompt

    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def send_one_prompt(args: BenchArgs):
    base_url = f"http://{args.host}:{args.port}"

    # Load tokenizer for chat template
    tokenizer = get_tokenizer(base_url)

    # Get model info for logging
    model_info = requests.get(f"{base_url}/model_info").json()

    # Construct the prompt
    image_url = "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"
    image_data = None

    if args.image:
        args.prompt = "Describe this image in a very short sentence."
        image_data = image_url
    elif args.many_images:
        args.prompt = (
            "I have one reference image and many images. "
            "Describe their relationship in a very short sentence."
        )
        image_data = [image_url] * 4
    elif args.json:
        args.prompt = (
            "What is the capital of France and how is that city like. "
            "Give me 3 trivial information about that city. "
            "Write in a format of json."
        )

    if args.batch_size > 1 and args.different_prompts:
        prompts = [
            apply_chat_template(tokenizer, f"Test case {i+1}: {args.prompt}")
            for i in range(args.batch_size)
        ]
    elif args.batch_size > 1:
        prompts = [apply_chat_template(tokenizer, args.prompt)] * args.batch_size
    else:
        prompts = apply_chat_template(tokenizer, args.prompt, image_data)

    json_data = {
        "text": prompts,
        "image_data": image_data,
        "sampling_params": {
            "sampling_seed": args.seed,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
            "stop": ["Question", "Assistant:", "<|separator|>", "<|eos|>"],
        },
        "return_logprob": args.return_logprob,
        "stream": args.stream,
    }

    if args.json:
        json_data["sampling_params"]["json_schema"] = "$$ANY$$"

    # Run profiler if requested
    if args.profile:
        print(f"Running profiler with {args.profile_steps} steps...")
        run_profile(
            url=base_url,
            num_steps=args.profile_steps,
            activities=["CPU", "GPU"],
            profile_by_stage=args.profile_by_stage,
            profile_prefix=args.profile_prefix,
        )

    # Send the request
    response = requests.post(
        f"{base_url}/generate",
        json=json_data,
        stream=args.stream,
    )

    if args.stream:
        last_len = 0
        ret = None
        for chunk in response.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                ret = json.loads(chunk[5:].strip("\n"))
                chunk_str = ret["text"][last_len:]
                last_len = len(ret["text"])
                print(chunk_str, end="", flush=True)
        print()
    else:
        ret = response.json()
        if args.batch_size > 1:
            ret = ret[0]

        if response.status_code != 200:
            print(ret)
            return 0, 0

        print(ret["text"])

    if ret is None:
        print("No response received")
        return 0, 0

    # Extract and display all meta_info
    meta = ret["meta_info"]
    console = Console()
    timestamps = get_timestamps()

    # Compute derived metrics
    latency = meta.get("e2e_latency", 0)
    tokens = meta.get("completion_tokens", 0)
    prompt_tokens = meta.get("prompt_tokens", 0)
    speed = tokens / latency if latency > 0 else 0

    spec_verify_ct = meta.get("spec_verify_ct", 0)
    if spec_verify_ct > 0:
        acc_length = tokens / spec_verify_ct
    else:
        acc_length = 1.0

    # -- Timestamp --
    console.print()
    console.print(
        f"[bold yellow]{timestamps['est']}[/]  /  [bold yellow]{timestamps['pst']}[/]"
    )

    # -- Main performance table --
    perf_table = Table(title="Performance", show_header=True, header_style="bold cyan")
    perf_table.add_column("Metric", style="bold")
    perf_table.add_column("Value", justify="right")

    perf_table.add_row("E2E Latency", f"{latency:.3f} s")
    perf_table.add_row("Prompt Tokens", str(prompt_tokens))
    perf_table.add_row("Completion Tokens", str(tokens))
    perf_table.add_row("Speed", f"{speed:.2f} tok/s")
    perf_table.add_row("Cached Tokens", str(meta.get("cached_tokens", 0)))

    cached_details = meta.get("cached_tokens_details")
    if cached_details:
        perf_table.add_row("Cached Details", str(cached_details))

    finish = meta.get("finish_reason", {})
    if isinstance(finish, dict):
        perf_table.add_row("Finish Reason", f"{finish.get('type', '?')} ({finish.get('length', finish.get('matched', ''))})")
    else:
        perf_table.add_row("Finish Reason", str(finish))

    console.print(perf_table)

    # -- Speculative decoding table --
    if spec_verify_ct > 0:
        spec_table = Table(title="Speculative Decoding", show_header=True, header_style="bold magenta")
        spec_table.add_column("Metric", style="bold")
        spec_table.add_column("Value", justify="right")

        spec_table.add_row("Accept Length", f"{acc_length:.3f}")
        spec_table.add_row("Accept Rate", f"{meta.get('spec_accept_rate', 0):.2%}")
        spec_table.add_row("Verify Steps", str(spec_verify_ct))
        spec_table.add_row("Accepted Tokens", str(meta.get("spec_accept_token_num", 0)))
        spec_table.add_row("Draft Tokens", str(meta.get("spec_draft_token_num", 0)))
        spec_table.add_row("Total Retractions", str(meta.get("total_retractions", 0)))

        histogram = meta.get("spec_accept_histogram")
        if histogram:
            hist_parts = [f"{i}:{n}" for i, n in enumerate(histogram)]
            spec_table.add_row("Accept Histogram", " | ".join(hist_parts))

        console.print(spec_table)

    # -- Request info table --
    info_table = Table(title="Request Info", show_header=True, header_style="bold green")
    info_table.add_column("Metric", style="bold")
    info_table.add_column("Value", justify="right")

    info_table.add_row("Request ID", meta.get("id", "?"))
    info_table.add_row("Weight Version", str(meta.get("weight_version", "?")))

    dp_rank = meta.get("dp_rank")
    if dp_rank is not None:
        info_table.add_row("DP Rank", str(dp_rank))

    console.print(info_table)

    # -- Write JSONL (full response + timestamps) --
    record = ret.copy()
    record["timestamps"] = timestamps

    output_path = Path(args.output)
    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    console.print(f"\n[dim]Results appended to {output_path}[/]")

    return acc_length, speed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BenchArgs.add_cli_args(parser)
    args = BenchArgs.from_cli_args(parser.parse_args())

    send_one_prompt(args)
