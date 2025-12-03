#!/bin/bash
set -x
if [[ "$1" == "server" ]]; then
model_str=deepseek-ai/DeepSeek-R1-0528
export SGLANG_ENABLE_FLASHINFER_GEMM=true
export SGLANG_SUPPORT_CUTLASS_BLOCK_FP8=true

  python3 -m sglang.launch_server \
    --trust-remote-code \
    --disable-radix-cache \
    --max-running-requests 512 \
    --chunked-prefill-size 8192 \
    --mem-fraction-static 0.9 \
    --cuda-graph-max-bs 128 \
    --max-prefill-tokens 8192 \
    --kv-cache-dtype fp8_e4m3 \
    --quantization fp8 \
    --model-path ${model_str} \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size=4 \
    --data-parallel-size=1 \
    --expert-parallel-size=1 \
    --scheduler-recv-interval 10 \
    --stream-interval 10 \
    --tokenizer-path ${model_str} \
    --attention-backend trtllm_mla \
    --moe-runner-backend flashinfer_trtllm \
    --enable-symm-mem \

fi

if [[ "$1" == "bench" ]]; then
python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
  --num-questions 2000 \
  --parallel 2000 \
  --num-shots 8 \
  --port 8000
fi

if [[ "$1" == "chat" ]]; then
  curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "$model_str",
      "messages": [{"role": "assistant", "content": "What is 37 * 42?"}],
      "extra_body": {
        "chat_template_kwargs": {
          "thinking": true
        }
      }
    }'
fi