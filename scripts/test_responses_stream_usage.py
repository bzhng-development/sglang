#!/usr/bin/env python3
"""
Test script to stream from /v1/responses and print token usage.

Goal:
- Verify that for multi-chunk responses, prompt_tokens and cached_tokens remain
  fixed across chunks while completion/output tokens accumulate correctly.
- Compare server behavior before vs. after the HarmonyContext patch.

Usage:
  export OPENAI_API_KEY=sk-123456  # or leave unset if your server has no API key
  python3 scripts/test_responses_stream_usage.py \
    --base-url http://localhost:30000/v1 \
    --model openai/gpt-oss-120b \
    --prompt "Explain the concept of attention in transformers in at least 400 words." \
    --max-output-tokens 512
"""

import argparse
import json
import os
import sys
from typing import Any, Optional

try:
    from openai import OpenAI
except Exception as e:
    print(
        "Missing dependency: openai. Install via: pip install openai>=1.30",
        file=sys.stderr,
    )
    raise


def safe_getattr(obj: Any, path: str, default=None):
    """Safely traverse attributes (dot-separated)."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        try:
            cur = getattr(cur, part)
        except Exception:
            try:
                # fall back to dict-like
                cur = cur.get(part)  # type: ignore[attr-defined]
            except Exception:
                return default
    return cur if cur is not None else default


def main():
    parser = argparse.ArgumentParser(
        description="Stream test for /v1/responses usage accounting."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SGLANG_BASE_URL", "http://localhost:30000/v1"),
        help="OpenAI-compatible base URL, e.g. http://localhost:30000/v1",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "sk-xxx"),
        help="API key (ignored by server if auth not enabled)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("SGLANG_MODEL", "openai/gpt-oss-120b"),
        help="Model name served by sglang",
    )
    parser.add_argument(
        "--prompt",
        default="Write a detailed explanation of how transformers work with examples. Target 500 words.",
        help="Prompt to ensure multi-chunk streaming output",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
        help="Max output tokens to encourage multi-chunk responses",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Also print raw JSON of final responses usage fields",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    chunk_count = 0
    completed_usage = None  # usage mapping from response.completed event
    print("=== Streaming output (response.output_text.delta) ===")
    try:
        with client.responses.stream(
            model=args.model,
            input=args.prompt,
            max_output_tokens=args.max_output_tokens,
        ) as stream:
            for event in stream:
                etype = getattr(event, "type", None)

                if etype == "response.output_text.delta":
                    # Print streamed text
                    delta = safe_getattr(event, "delta", "")
                    if delta:
                        chunk_count += 1
                        sys.stdout.write(delta)
                        sys.stdout.flush()

                elif etype == "response.completed":
                    # Capture usage mapped by server for Responses API:
                    #   input_tokens, input_tokens_details.cached_tokens,
                    #   output_tokens, total_tokens
                    completed_usage = safe_getattr(event, "response.usage")
                    print("\n\n=== response.completed usage (mapped) ===")
                    if completed_usage is None:
                        print("No usage present on response.completed")
                    else:
                        # Try attribute-style, then dict fallback
                        try:
                            input_tokens = safe_getattr(completed_usage, "input_tokens")
                            cached_tokens = safe_getattr(
                                completed_usage, "input_tokens_details.cached_tokens"
                            )
                            output_tokens = safe_getattr(
                                completed_usage, "output_tokens"
                            )
                            total_tokens = safe_getattr(completed_usage, "total_tokens")
                            print(
                                f"input_tokens={input_tokens}, cached_tokens={cached_tokens}, "
                                f"output_tokens={output_tokens}, total_tokens={total_tokens}"
                            )
                        except Exception:
                            print(str(completed_usage))

            # After stream ends, inspect final_response (server-native UsageInfo style)
            final = getattr(stream, "final_response", None)
    except Exception as e:
        print(f"\nStream error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n=== final_response.usage (native) ===")
    if final is None:
        print("No final_response found.")
        sys.exit(2)

    final_usage = safe_getattr(final, "usage")
    if final_usage is None:
        print("No usage on final_response.")
    else:
        prompt_tokens = safe_getattr(final_usage, "prompt_tokens")
        completion_tokens = safe_getattr(final_usage, "completion_tokens")
        total_tokens = safe_getattr(final_usage, "total_tokens")
        # Try prompt_tokens_details.cached_tokens if present
        cached_tokens_native = safe_getattr(
            final_usage, "prompt_tokens_details.cached_tokens"
        )
        print(
            f"prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens_native}, "
            f"completion_tokens={completion_tokens}, total_tokens={total_tokens}"
        )

        if args.print_json:
            try:
                # Prefer pydantic model_dump if available
                if hasattr(final_usage, "model_dump"):
                    print("\n[final_response.usage JSON]")
                    print(json.dumps(final_usage.model_dump(), indent=2))
                else:
                    # Fallback for dict-like
                    print("\n[final_response.usage JSON]")
                    print(json.dumps(final_usage, indent=2))
            except Exception:
                pass

    print(f"\n=== chunks observed ===\n{chunk_count} response.output_text.delta chunks")

    print("\n=== How to compare before/after the patch ===")
    print(
        "1) Run this script against the server BEFORE applying the HarmonyContext patch and note:"
    )
    print("   - response.completed usage: input_tokens, cached_tokens, output_tokens")
    print(
        "   - final_response.usage: prompt_tokens, cached_tokens (if reported), completion_tokens"
    )
    print("2) Apply the patch and restart the server.")
    print("3) Run the same command again and compare. Expected:")
    print("   - input_tokens/prompt_tokens remain constant across chunks")
    print("   - cached_tokens remains constant")
    print("   - output_tokens/completion_tokens grows with longer outputs")


if __name__ == "__main__":
    main()
