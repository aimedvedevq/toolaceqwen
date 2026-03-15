#!/bin/bash
# Deploy model to Nebius AI endpoint and benchmark it
#
# Prerequisites:
#   nebius profile create   # configure auth
#   export HF_TOKEN=...     # HuggingFace token for model download
#
# Usage:
#   bash scripts/deploy_and_bench.sh
#   QUANTIZATION=fp8 bash scripts/deploy_and_bench.sh

set -e

MODEL_ID="${MODEL_ID:-kenkaneki/Qwen3-8B-ToolACE}"
ENDPOINT_NAME="${ENDPOINT_NAME:-qwen3-toolace}"
HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}"
QUANTIZATION="${QUANTIZATION:-}"
AUTH_TOKEN="$(openssl rand -hex 16)"

VLLM_ARGS="--model $MODEL_ID --dtype auto --trust-remote-code --max-model-len 4096 --enable-auto-tool-choice --tool-call-parser hermes"
if [ -n "$QUANTIZATION" ]; then
    VLLM_ARGS="$VLLM_ARGS --quantization $QUANTIZATION"
    ENDPOINT_NAME="${ENDPOINT_NAME}-${QUANTIZATION}"
fi

echo "========================================="
echo "  Deploying: $MODEL_ID"
echo "  Endpoint:  $ENDPOINT_NAME"
echo "  Quant:     ${QUANTIZATION:-none}"
echo "  Auth:      $AUTH_TOKEN"
echo "========================================="

# Create endpoint
nebius ai endpoint create \
  --name "$ENDPOINT_NAME" \
  --platform gpu-h100-sxm \
  --preset 1gpu-16vcpu-200gb \
  --image "vllm/vllm-openai:latest" \
  --container-port 8000 \
  --env "HF_TOKEN=$HF_TOKEN" \
  --env "VLLM_ARGS=$VLLM_ARGS" \
  --container-command "python -m vllm.entrypoints.openai.api_server $VLLM_ARGS" \
  --disk-size 100Gi \
  --public \
  --auth token \
  --token "$AUTH_TOKEN"

echo "Waiting for endpoint to be ready..."
for i in $(seq 1 60); do
    sleep 10
    STATUS=$(nebius ai endpoint get-by-name --name "$ENDPOINT_NAME" --format json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',{}).get('phase',''))" 2>/dev/null)
    echo "  [$i] Status: $STATUS"
    if [ "$STATUS" == "Running" ] || [ "$STATUS" == "RUNNING" ]; then
        break
    fi
done

# Get endpoint URL
ENDPOINT_URL=$(nebius ai endpoint get-by-name --name "$ENDPOINT_NAME" --format json | python3 -c "
import sys, json
data = json.load(sys.stdin)
urls = data.get('status', {}).get('urls', [])
print(urls[0] if urls else 'NOT_READY')
")

echo ""
echo "Endpoint URL: $ENDPOINT_URL"
echo "Auth Token: $AUTH_TOKEN"
echo ""

if [ "$ENDPOINT_URL" == "NOT_READY" ]; then
    echo "Endpoint not ready yet. Check: nebius ai endpoint get-by-name --name $ENDPOINT_NAME"
    exit 1
fi

# Quick test
echo "Testing endpoint..."
curl -s "${ENDPOINT_URL}/v1/chat/completions" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Weather in Tokyo?\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}}],\"max_tokens\":128,\"temperature\":0}" | python3 -m json.tool

echo ""
echo "Running benchmark..."
python scripts/bench.py \
  --url "${ENDPOINT_URL}/v1/chat/completions" \
  --concurrency 1,4,8,16,32 \
  --num-prompts 100 \
  --output "results/bench_nebius_${ENDPOINT_NAME}.json"

echo ""
echo "========================================="
echo "  Deployment complete!"
echo "  Endpoint: $ENDPOINT_URL"
echo "  Results:  results/bench_nebius_${ENDPOINT_NAME}.json"
echo "========================================="
echo ""
echo "To clean up:"
echo "  nebius ai endpoint delete --name $ENDPOINT_NAME"
