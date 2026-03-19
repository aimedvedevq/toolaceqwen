# Tool-Calling LLM: Qwen3-8B Fine-Tuned on ToolACE

Fine-tuned Qwen3-8B for precise function calling, optimized for **accuracy → latency → cost**.

## Pipeline

```
ToolACE (11,300 samples) → 70/30 split
    ├── SFT (1 epoch, LoRA r=64)        → format learning
    └── GRPO (400 steps, DAPO loss)     → quality refinement
                │
                ├── BF16                 (baseline)
                ├── FP8 dynamic          (1.5x throughput, no quality loss)
                ├── W4A16                (2.7x compression)
                │
                └── + EAGLE-3 FT         (1.8x E2EL speedup, lossless)
                      fine-tuned from RedHatAI/Qwen3-8B-speculator.eagle3
```

### BFCL v4 Accuracy (`simple_python`, n=400)

| Config | Accuracy |
|--------|:--------:|
| Baseline Qwen3-8B | 95.00% |
| + SFT | 95.75% |
| + GRPO | 96.5% |
| FP8 dynamic | 96.50% |
| W4A16 | 96.50% |

### BFCL v4 Full Core (13 categories, weighted average)

| Config | Weighted Accuracy |
|--------|:-----------------:|
| Baseline | 77.34% |
| + SFT | 83.05% |
| + GRPO | 82.86% |

### Inference Latency (H100, vLLM, `vllm bench serve`, ToolACE prompts, 100 requests)

| Config | c | TTFT p50 | E2EL p50 | Output tok/s |
|--------|:-:|:--------:|:--------:|:------------:|
| BF16 | 1 | 15.5 ms | 323.9 ms | 150.4 |
| FP8 dynamic | 1 | 13.2 ms | 222.9 ms | 217.7 |
| W4A16 | 1 | 13.7 ms | 222.3 ms | 208.9 |
| **EAGLE3 FT** | **1** | **18.7 ms** | **175.4 ms** | **271.3** |
| BF16 | 16 | 29.4 ms | 375.6 ms | 1623.0 |
| FP8 dynamic | 16 | 24.1 ms | 284.4 ms | 2207.6 |
| W4A16 | 16 | 19.8 ms | 239.1 ms | 2235.1 |
| **EAGLE3 FT** | **16** | **30.5 ms** | **204.4 ms** | **3280.0** |
| BF16 | 32 | 33.9 ms | 447.5 ms | 2293.1 |
| FP8 dynamic | 32 | 33.0 ms | 247.3 ms | 3268.1 |
| W4A16 | 32 | 23.8 ms | 276.5 ms | 3096.7 |
| **EAGLE3 FT** | **32** | **40.7 ms** | **232.7 ms** | **4378.3** |

## How to Reproduce

### Setup
```bash
pip install -r requirements.txt            # or: pip install -r requirements.lock
cd ~/gorilla/berkeley-function-call-leaderboard && pip install -e .
```

### 1. Training (~47 min total)
```bash
python scripts/sft.py                       # SFT: 1 epoch LoRA on ToolACE (~27 min)
python scripts/grpo.py                      # GRPO: 400 steps with decomposed rewards (~20 min)
```

### 2. Quantization (~5 min)
```bash
python scripts/quantize.py --model ./output_grpo/merged --method fp8 --output ./output_grpo/fp8
python scripts/quantize.py --model ./output_grpo/merged --method w4a16 --output ./output_grpo/w4a16
```

### 3. EAGLE-3 Speculative Decoding (~15 min)
```bash
python scripts/finetune_eagle.py            # Fine-tune official draft on ToolACE
```

### 4. BFCL Evaluation
```bash
# Full evaluation (all 13 categories, ~30 min per config)
python scripts/eval.py --all

# Quick check (Python subset only, ~2 min per config)
python scripts/eval.py --configs baseline sft grpo fp8 w4a16 --test-category simple_python
```
> EAGLE3 is lossless speculative decoding — accuracy is identical to BF16 by construction.

### 5. Latency Benchmarks
```bash
# Full suite: BF16 / FP8 / W4A16 / EAGLE3 × concurrency 1,4,8,16,32
python scripts/bench.py --suite

# Quick check at production concurrency
python scripts/bench.py --suite --concurrency 1,16,32

# Benchmark a running server
python scripts/bench.py --port 8100 --concurrency 1,16,32
```

### 6. Serving
```bash
# Recommended: FP8 dynamic (best quality/latency tradeoff)
bash scripts/serve.sh --quantization fp8

# Maximum performance: EAGLE3 speculative decoding
bash scripts/serve_eagle.sh

# BF16 baseline
bash scripts/serve.sh

# Python launcher with all options
python scripts/run_inference_vm.py --quantization fp8
```

### 7. Report
```bash
make report     # Execute notebook + render HTML
```

### Shortcut (full pipeline via Make)
```bash
make train && make quantize && make eval && make bench && make report
```

## Repository Structure

```
configs/
    sft.yaml              SFT config (LoRA r=64, cosine LR, assistant-only loss)
    grpo.yaml             GRPO config (DAPO loss, decomposed rewards)
scripts/
    data_utils.py         ToolACE → Qwen3 tool calling format
    sft.py                SFT training — LoRA + assistant-only loss masking
    grpo.py               GRPO with decomposed rewards (format/name/args)
    quantize.py           FP8 dynamic + calibrated W4A16 via llm-compressor
    eval.py               BFCL evaluation (multi-config, auto server management)
    bench.py              Inference benchmark via vllm bench serve
    serve.sh              vLLM production serving with hermes tool calling
    serve_eagle.sh        vLLM serving with EAGLE-3 speculative decoding
    run_inference_vm.py   Recommended VM inference launcher
    train_eagle.py        EAGLE-3 training (experimental)
    finetune_eagle.py     EAGLE-3 fine-tuning from official checkpoint
results/
    bfcl_full_core.json   BFCL v4 full 13-category results (canonical)
    bfcl_results.json     BFCL summary (derived from bfcl_full_core.json)
    bench_inference.json  Latency benchmarks BF16/FP8/W4A16 × c=1,4,8,16,32
    engine_compare.json   vLLM vs SGLang BF16 comparison
    grpo_rewards.json     GRPO training reward curves
    bench/                Raw vllm bench serve output files
report.ipynb              Full analysis notebook
Makefile                  Reproducibility commands
requirements.lock         Pinned dependency versions
```

## Design Decisions

1. **Qwen3-8B** — Native tool calling, strong BFCL baseline, fits H100 with KV cache headroom
2. **nothink mode** — Disables chain-of-thought for deterministic, low-latency tool calls
3. **LoRA r=64 all-linear** — Full coverage of attention + MLP layers without full fine-tuning cost
4. **Assistant-only loss masking** — Only supervise tool call outputs, not prompts/system messages
5. **SFT → GRPO** — SFT teaches format, GRPO refines quality via decomposed rewards
6. **DAPO loss** — No KL penalty for more aggressive exploration
7. **Decomposed rewards** — Format (0.1), Tool Name (0.5), Tool Args (0.4) with format decay
8. **FP8 dynamic** — Best accuracy-latency tradeoff; ~1.44x throughput with no quality loss
9. **W4A16 + ToolACE calibration** — Domain-matched calibration for better quantization
10. **EAGLE-3 fine-tuned** — Official `RedHatAI/Qwen3-8B-speculator.eagle3` fine-tuned on ToolACE, deployed via vLLM native speculative-config. Gives **1.8x E2EL speedup** at c=1.


## Hardware

- NVIDIA H100 80GB HBM3
- Training: ~27 min SFT + ~20 min GRPO + ~15 min EAGLE3 FT ≈ 62 min total

## Models on HuggingFace

| Model | Description |
|-------|-------------|
| [kenkaneki/Qwen3-8B-ToolACE](https://huggingface.co/kenkaneki/Qwen3-8B-ToolACE) | Post-GRPO merged model (BF16) |
| [kenkaneki/Qwen3-8B-ToolACE-W4A16](https://huggingface.co/kenkaneki/Qwen3-8B-ToolACE-W4A16) | W4A16 quantized (ToolACE-calibrated) |
| [kenkaneki/Qwen3-8B-ToolACE-speculator.eagle3](https://huggingface.co/kenkaneki/Qwen3-8B-ToolACE-speculator.eagle3) | EAGLE-3 draft head (1.8x speedup) |

All scripts default to HF model paths — no local checkpoints needed to run eval, bench, or serve.

## Benchmarking Tool

Latency measured with [`vllm bench serve`](https://docs.vllm.ai/en/latest/cli/bench/serve.html) — the standard vLLM benchmarking CLI. Prompts auto-generated from ToolACE held-out split (100 requests per concurrency level, 5 warmup rounds).
