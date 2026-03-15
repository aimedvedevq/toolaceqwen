#!/bin/bash
# Deploy fine-tuned Qwen3-8B on Nebius AI endpoint with vLLM + hermes tool calling
#
# Uses nebius CLI to create a GPU endpoint with vLLM container
# serving our model from HuggingFace Hub.
#
# Prerequisites:
#   - nebius CLI configured (nebius iam whoami)
#   - Model pushed to HF Hub (kenkaneki/Qwen3-8B-ToolACE)
#
# Usage:
#   bash scripts/deploy_nebius.sh                    # BF16
#   bash scripts/deploy_nebius.sh --fp8              # FP8 dynamic quantization

set -e

MODEL_ID="${MODEL_ID:-kenkaneki/Qwen3-8B-ToolACE}"
ENDPOINT_NAME="${ENDPOINT_NAME:-qwen3-8b-toolace}"
HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}"
PLATFORM="${PLATFORM:-gpu-h100-sxm}"
PRESET="${PRESET:-1gpu-16vcpu-200gb}"

# Check if --fp8 flag
EXTRA_ARGS=""
if [[ "$1" == "--fp8" ]]; then
    EXTRA_ARGS="--quantization fp8"
    ENDPOINT_NAME="${ENDPOINT_NAME}-fp8"
fi

echo "========================================="
echo "  Deploying to Nebius AI"
echo "  Model: $MODEL_ID"
echo "  Endpoint: $ENDPOINT_NAME"
echo "  Platform: $PLATFORM"
echo "  Preset: $PRESET"
echo "========================================="

# Create endpoint with vLLM container
nebius ai endpoint create \
  --name "$ENDPOINT_NAME" \
  --platform "$PLATFORM" \
  --preset "$PRESET" \
  --image "vllm/vllm-openai:latest" \
  --container-port 8000 \
  --env "HF_TOKEN=$HF_TOKEN" \
  --args "--model $MODEL_ID --dtype auto --trust-remote-code --max-model-len 4096 --enable-auto-tool-choice --tool-call-parser hermes $EXTRA_ARGS" \
  --disk-size 100Gi \
  --public \
  --auth token \
  --token "$(openssl rand -hex 16)"

echo ""
echo "Endpoint creating... Check status with:"
echo "  nebius ai endpoint list"
echo "  nebius ai endpoint get-by-name --name $ENDPOINT_NAME"
echo ""
echo "Once ready, test with:"
echo '  curl https://<endpoint-url>/v1/chat/completions \'
echo '    -H "Authorization: Bearer <token>" \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"model":"'$MODEL_ID'","messages":[{"role":"user","content":"Weather in Tokyo?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}'"'"
