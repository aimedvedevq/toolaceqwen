# Financial Workflow Automation: Tool-Calling LLM

Fine-tuned Qwen3-8B for function calling, optimized for accuracy → latency → cost.

## Results

| Stage | BFCL Overall | Key Improvements |
|---|---|---|
| Baseline | 82.2% | — |
| + SFT (1 epoch) | 82.6% | irrelevance +10%, live_parallel +12.5% |
| + GRPO (250 steps) | **82.9%** | parallel_multiple +5.5%, live_relevance +12.5% |

| Quantization | BFCL | Model Size | Throughput (c=32) |
|---|---|---|---|
| BF16 | 82.9% | 16.4 GB | 2,485 tok/s |
| FP8 | 82.9% | 8.8 GB | 7,411 tok/s |
| W4A16 | 82.9% | 6.1 GB | 2,288 tok/s |

## Quick Start

```bash
# Install
conda activate ai
pip install -r requirements.txt

# Train SFT
python scripts/sft.py --config configs/sft.yaml

# Train GRPO
python scripts/grpo.py --config configs/grpo.yaml model.name=./output/merged

# Evaluate on BFCL
python scripts/eval.py --model Qwen/Qwen3-8B --local-model-path ./output_grpo/merged \
  --test-category single_turn

# Benchmark latency
python scripts/bench.py --model ./output_grpo/merged \
  --url http://localhost:8100/v1/chat/completions \
  --concurrency 1,4,8,16,32

# Quantize
python scripts/quantize.py --model ./output_grpo/merged --method fp8
python scripts/quantize.py --model ./output_grpo/merged --method w4a16

# EAGLE-3 speculative decoding
python scripts/train_eagle.py --model ./output_grpo/merged
```

## Repository Structure

```
├── configs/
│   ├── sft.yaml          # SFT training config (full SFTConfig)
│   └── grpo.yaml         # GRPO training config
├── scripts/
│   ├── sft.py            # SFT training (LoRA, assistant-only masking)
│   ├── grpo.py           # GRPO with custom reward functions (Unsloth)
│   ├── eval.py           # BFCL evaluation wrapper (official gorilla eval)
│   ├── bench.py          # Concurrent inference benchmark (vLLM)
│   ├── data_utils.py     # ToolACE → Qwen3 tool calling format converter
│   ├── train_eagle.py    # EAGLE-3 speculative decoding training
│   └── bench_*.py        # Various benchmark scripts
├── results/              # Benchmark JSON results
├── logs/                 # Training and benchmark logs
├── report.ipynb          # Full analysis notebook with visualizations
├── report.html           # HTML version of the notebook
├── REPORT.md             # Text summary of all experiments
└── output*/              # Model checkpoints (merged, fp8, awq, eagle)
```

## Key Design Decisions

1. **Qwen3-8B** — Native tool calling, strong BFCL baseline, fits single H100
2. **LoRA r=64, all-linear** — Full coverage with 2% trainable params
3. **Assistant-only loss masking** — Train only on tool call responses
4. **nothink mode** — Disable reasoning for deterministic tool calling
5. **SFT warmup → GRPO** — SFT for format learning, GRPO for quality refinement
6. **FP8 quantization** — Zero quality loss, 4.9x throughput, 47% memory reduction

## Serving

```bash
# vLLM with hermes tool calling
python -m vllm.entrypoints.openai.api_server \
  --model ./output_grpo/merged \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --dtype bfloat16 --max-model-len 4096

# SGLang (higher throughput)
python -m sglang.launch_server \
  --model-path ./output_grpo/merged \
  --port 8100
```

## Hardware
- GPU: NVIDIA H100 80GB HBM3
- Training: ~30 min SFT + ~15 min GRPO
- Full pipeline (train + eval): ~2 hours
