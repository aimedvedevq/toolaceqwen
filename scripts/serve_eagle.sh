#!/bin/bash
# Production serving with EAGLE-3 speculative decoding via vLLM
#
# Uses the fine-tuned EAGLE-3 draft head for ~1.8x E2EL speedup.
# EAGLE-3 is lossless: output identical to target model by construction.
#
# Includes production optimizations:
#   - Prefix caching
#   - Chunked prefill
#   - GPU memory utilization 0.92
#   - Max 64 concurrent sequences
#
# Usage:
#   bash scripts/serve_eagle.sh                     # BF16 + EAGLE3
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
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 64 \
  --speculative-config "$SPEC_CONFIG" \
  $EXTRA_ARGS
