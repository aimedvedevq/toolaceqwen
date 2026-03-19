#!/bin/bash
# Production serving of the fine-tuned tool-calling model via vLLM
#
# Includes production optimizations:
#   - Prefix caching (reuses KV cache for repeated system prompts / tool defs)
#   - Chunked prefill (better scheduling under concurrent load)
#   - GPU memory utilization 0.92 (leave headroom for peaks)
#   - Max 64 concurrent sequences (tuned for 16-32 client concurrency)
#
# Usage:
#   bash scripts/serve.sh                                # BF16
#   bash scripts/serve.sh --quantization fp8             # FP8 (recommended)
#   MODEL=./output_grpo/w4a16 bash scripts/serve.sh --quantization compressed-tensors

MODEL="${MODEL:-kenkaneki/Qwen3-8B-ToolACE}"
PORT="${PORT:-8100}"
EXTRA_ARGS="$@"

echo "========================================="
echo "  Serving: $MODEL"
echo "  Port: $PORT"
echo "  Extra: $EXTRA_ARGS"
echo "========================================="

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --dtype auto \
  --trust-remote-code \
  --max-model-len 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 64 \
  $EXTRA_ARGS
