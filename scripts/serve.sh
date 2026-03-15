#!/bin/bash
# Deploy model for production serving on Nebius VM
#
# Usage:
#   bash scripts/serve.sh                          # BF16
#   bash scripts/serve.sh --quantization fp8       # FP8
#   MODEL=./output_grpo/awq bash scripts/serve.sh  # Custom model path

MODEL="${MODEL:-./output_grpo/merged}"
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
  $EXTRA_ARGS
