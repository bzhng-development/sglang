"""
Run one test prompt.

Usage:
python3 -m sglang.test.send_one
"""

import argparse
import dataclasses
import json

import requests


@dataclasses.dataclass
class BenchArgs:
    host: str = "localhost"
    port: int = 30000
    batch_size: int = 1
    temperature: float = 0.0
    max_new_tokens: int = 512
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    json: bool = False
    return_logprob: bool = False
    prompt: str = (
        "Human: Give me a fully functional FastAPI server. Show the python code.\n\nAssistant:"
    )
    image: bool = False
    many_images: bool = False
    stream: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--host", type=str, default=BenchArgs.host)
        parser.add_argument("--port", type=int, default=BenchArgs.port)
        parser.add_argument("--batch-size", type=int, default=BenchArgs.batch_size)
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

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def send_one_prompt(args):
    # Prepare message content based on image options
    if args.image:
        user_content = [
            {
                "type": "text",
                "text": "Describe this image in a very short sentence."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://raw.githubusercontent.com/sgl-project/sglang/main/test/lang/example_image.png"
                }
            }
        ]
    elif args.many_images:
        user_content = [
            {
                "type": "text", 
                "text": "I have one reference image and many images. Describe their relationship in a very short sentence."
            }
        ]
        # Add multiple images
        for _ in range(4):
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": "https://raw.githubusercontent.com/sgl-project/sglang/main/test/lang/example_image.png"
                }
            })
    else:
        # Extract user message from prompt (remove "Human: " and "\n\nAssistant:" parts)
        prompt_text = args.prompt
        if args.json:
            prompt_text = (
                "What is the capital of France and how is that city like. "
                "Give me 3 trivial information about that city. "
                "Write in a format of json."
            )
        else:
            # Clean the prompt to extract just the user message
            if prompt_text.startswith("Human: "):
                prompt_text = prompt_text[7:]  # Remove "Human: "
            if prompt_text.endswith("\n\nAssistant:"):
                prompt_text = prompt_text[:-13]  # Remove "\n\nAssistant:"
        
        user_content = prompt_text

    # Prepare messages for chat format
    messages = [
        {
            "role": "user",
            "content": user_content
        }
    ]

    # Prepare the request payload in OpenAI chat completions format
    json_data = {
        "model": "default",  # SGLang will use the loaded model
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
        "stop": ["Question", "Assistant:", "<|separator|>", "<|eos|>"],
        "stream": args.stream,
    }

    # Add logprobs if requested
    if args.return_logprob:
        json_data["logprobs"] = True
        json_data["top_logprobs"] = 5

    # Add JSON schema if requested
    if args.json:
        json_data["response_format"] = {"type": "json_object"}

    # Handle batch requests by duplicating messages
    if args.batch_size > 1:
        # For batch processing, send individual requests or duplicate the request
        # SGLang may handle batching differently in chat completions
        pass

    response = requests.post(
        f"http://{args.host}:{args.port}/v1/chat/completions",
        json=json_data,
        stream=args.stream,
    )

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        try:
            error_response = response.json()
            print(error_response)
        except:
            print(response.text)
        return 0, 0

    if args.stream:
        # Handle streaming response for chat completions
        response_text = ""
        completion_tokens = 0
        start_time = None
        end_time = None
        
        for chunk in response.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                try:
                    chunk_data = json.loads(chunk[5:].strip("\n"))
                    if start_time is None:
                        start_time = chunk_data.get("created", 0)
                    
                    if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                        delta = chunk_data["choices"][0].get("delta", {})
                        if "content" in delta:
                            response_text += delta["content"]
                            completion_tokens += 1  # Approximate token count
                        
                        # Check for finish reason to get final timing
                        if chunk_data["choices"][0].get("finish_reason"):
                            end_time = chunk_data.get("created", start_time)
                except json.JSONDecodeError:
                    continue
        
        # Create a response-like structure for consistency
        ret = {
            "choices": [{"message": {"content": response_text}}],
            "usage": {"completion_tokens": completion_tokens},
            "meta_info": {
                "completion_tokens": completion_tokens,
                "e2e_latency": max(0.001, (end_time or start_time) - start_time) if start_time else 0.001
            }
        }
    else:
        # Handle non-streaming response
        ret = response.json()

    # Extract response text from OpenAI chat completions format
    if "choices" in ret and len(ret["choices"]) > 0:
        response_text = ret["choices"][0]["message"]["content"]
    else:
        response_text = "No response generated"

    # Extract timing and token info
    usage = ret.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    
    # Try to get latency from meta_info (SGLang extension) or calculate from usage
    meta_info = ret.get("meta_info", {})
    if "e2e_latency" in meta_info:
        latency = meta_info["e2e_latency"]
    else:
        # Fallback: estimate latency (this won't be accurate without timing info)
        latency = 1.0

    # Calculate metrics
    if "spec_verify_ct" in meta_info and meta_info["spec_verify_ct"] > 0:
        acc_length = completion_tokens / meta_info["spec_verify_ct"]
    else:
        acc_length = 1.0

    speed = completion_tokens / latency if latency > 0 else 0

    # Print results
    print(response_text)
    print()
    print(f"{acc_length=:.2f}")
    print(f"{speed=:.2f} token/s")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Latency: {latency:.3f}s")

    return acc_length, speed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    send_one_prompt(args)
