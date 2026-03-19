#!/bin/bash
# Deploy model with EAGLE-3 speculative decoding via vLLM
#
# This uses the fine-tuned EAGLE-3 draft head trained on ToolACE data,
# starting from the official RedHatAI/Qwen3-8B-speculator.eagle3 checkpoint.
#
# Gives ~1.8x E2EL speedup and ~1.7x throughput improvement at c=1.
#
# Usage:
#   bash scripts/serve_eagle.sh                     # default: BF16 + EAGLE3
#   bash scripts/serve_eagle.sh --quantization fp8  # FP8 + EAGLE3

MODEL="${MODEL:-kenkaneki/Qwen3-8B-ToolACE}"
EAGLE_CKPT="${EAGLE_CKPT:-kenkaneki/Qwen3-8B-ToolACE-speculator.eagle3}"
PORT="${PORT:-8100}"
EXTRA_ARGS="$@"

SPEC_CONFIG="{\"model\":\"${EAGLE_CKPT}\",\"num_speculative_tokens\":3,\"method\":\"eagle3\"}"

echo "========================================="
echo "  Serving: $MODEL"
echo "  EAGLE-3 draft: $EAGLE_CKPT"
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
  --speculative-config "$SPEC_CONFIG" \
  $EXTRA_ARGS
