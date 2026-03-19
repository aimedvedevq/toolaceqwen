#!/bin/bash
# Production deployment via official vLLM Docker image
#
# Pulls models from HuggingFace, no local checkpoints needed.
# Includes prefix caching, chunked prefill, and EAGLE-3 speculative decoding.
#
# Usage:
#   bash scripts/docker_serve.sh                    # FP8 + EAGLE3 (recommended)
#   bash scripts/docker_serve.sh --no-eagle         # FP8 only
#   bash scripts/docker_serve.sh --bf16             # BF16 baseline
#   bash scripts/docker_serve.sh --w4a16            # W4A16 quantized
#
# Requirements:
#   - Docker with NVIDIA Container Toolkit
#   - GPU with >= 40GB VRAM (H100/A100/A6000)

set -e

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.17.1}"
MODEL="kenkaneki/Qwen3-8B-ToolACE"
EAGLE_CKPT="kenkaneki/Qwen3-8B-ToolACE-speculator.eagle3"
W4A16_MODEL="kenkaneki/Qwen3-8B-ToolACE-W4A16"
PORT="${PORT:-8100}"
HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || echo '')}"

USE_EAGLE=true
QUANTIZATION="fp8"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-eagle)   USE_EAGLE=false; shift ;;
        --bf16)       QUANTIZATION=""; shift ;;
        --fp8)        QUANTIZATION="fp8"; shift ;;
        --w4a16)      QUANTIZATION="compressed-tensors"; MODEL="$W4A16_MODEL"; USE_EAGLE=false; shift ;;
        --port)       PORT="$2"; shift 2 ;;
        *)            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

VLLM_ARGS=(
    --model "$MODEL"
    --port 8000
    --dtype auto
    --trust-remote-code
    --max-model-len 4096
    --enable-auto-tool-choice
    --tool-call-parser hermes
    --enable-prefix-caching
    --enable-chunked-prefill
    --gpu-memory-utilization 0.92
    --max-num-seqs 64
)

if [ -n "$QUANTIZATION" ]; then
    VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

if [ "$USE_EAGLE" = true ]; then
    SPEC_CONFIG="{\"model\":\"${EAGLE_CKPT}\",\"num_speculative_tokens\":3,\"method\":\"eagle3\"}"
    VLLM_ARGS+=(--speculative-config "$SPEC_CONFIG")
fi

echo "========================================="
echo "  Docker vLLM Deployment"
echo "  Image: $VLLM_IMAGE"
echo "  Model: $MODEL"
echo "  Quantization: ${QUANTIZATION:-none}"
echo "  EAGLE-3: $USE_EAGLE"
echo "  Port: $PORT → container:8000"
echo "========================================="

docker run --rm -it \
    --gpus all \
    --shm-size=16g \
    -p "${PORT}:8000" \
    -e HF_TOKEN="$HF_TOKEN" \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    "$VLLM_IMAGE" \
    "${VLLM_ARGS[@]}" \
    $EXTRA_ARGS
