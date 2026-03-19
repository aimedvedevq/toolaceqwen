# Tool-Calling LLM: Qwen3-8B Fine-Tuned on ToolACE

Fine-tuned Qwen3-8B for precise function calling, optimized for **accuracy → latency → cost**.

## Pipeline

```
ToolACE (11,300 samples) → 70/30 split
    ├── SFT (1 epoch, LoRA r=64)     → format learning
    └── GRPO (400 steps, DAPO loss)  → quality refinement via reward decomposition
                ├── BF16   (baseline serving)
                ├── FP8    (recommended: ~1.44x throughput, no quality loss)
                └── W4A16  (2.7x compression)
```

## Results

> All numbers below are from fresh runs on 2026-03-18. Source of truth: `results/bfcl_full_core.json` and `results/bench_inference.json`.

### BFCL v4 Accuracy (`simple_python`, n=400)

| Config | Accuracy |
|--------|:--------:|
| Baseline Qwen3-8B | 95.00% |
| + SFT | 96.00% |
| + GRPO | 96.25% |
| FP8 dynamic | 96.50% |
| W4A16 | 96.50% |

### BFCL v4 Full Core (13 categories, weighted average)

| Config | Weighted Accuracy |
|--------|:-----------------:|
| Baseline | 77.34% |
| + SFT | 83.05% |
| + GRPO | 82.86% |

### Inference Latency (H100, vLLM, `vllm bench serve`, 100 requests)

| Config | c | TTFT p50 | E2EL p50 | Output tok/s |
|--------|:-:|:--------:|:--------:|:------------:|
| BF16 | 1 | 14.6 ms | 197.8 ms | 146.7 |
| FP8 dynamic | 1 | 11.7 ms | 139.5 ms | 211.9 |
| W4A16 | 1 | 12.2 ms | 142.8 ms | 203.0 |
| BF16 | 16 | 29.6 ms | 233.0 ms | 1861.1 |
| FP8 dynamic | 16 | 26.1 ms | 177.4 ms | 2474.0 |
| W4A16 | 16 | 27.7 ms | 185.0 ms | 2384.1 |
| BF16 | 32 | 33.6 ms | 271.8 ms | 2933.4 |
| FP8 dynamic | 32 | 33.0 ms | 247.3 ms | 3268.1 |
| W4A16 | 32 | 29.8 ms | 237.1 ms | 3362.0 |

## Quick Start

```bash
pip install -r requirements.txt
pip freeze > requirements.lock  # pin exact versions
cd ~/gorilla/berkeley-function-call-leaderboard && pip install -e .

# Full pipeline
make train      # SFT + GRPO
make quantize   # FP8 + W4A16
make eval       # BFCL all categories
make bench      # Latency benchmarks
make report     # Execute notebook + render HTML
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
10. **EAGLE-3** — Investigated; official checkpoint + domain fine-tune approach documented

## Known Limitations

- No ablation study (LoRA rank, SFT-only vs GRPO-only)
- FP8/W4A16 evaluated only on `simple_python`, not full BFCL categories
- EAGLE-3 speculative decoding did not outperform baseline in our tests
- SGLang showed worse performance than vLLM on this workload

## Hardware

- NVIDIA H100 80GB HBM3
- Training: ~27 min SFT + ~20 min GRPO ≈ 47 min total
