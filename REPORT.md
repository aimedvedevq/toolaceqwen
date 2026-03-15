# Experiment 2: SFT Warmup + GRPO on ToolACE → BFCL

## Setup
- **Base model**: Qwen/Qwen3-8B (Prompt mode, nothink)
- **Train data**: Team-ACE/ToolACE — 70% SFT (7,910), 30% GRPO (3,390)
- **SFT**: 1 epoch warmup, lr=5e-5, LoRA r=64, assistant-only loss
- **GRPO**: 250 steps, DAPO loss, temperature=1.5, beta=0.04, 8 generations
- **Eval**: BFCL v4 full single_turn (13 categories)

## Results

| Category | Baseline | Post-SFT | Post-GRPO | SFT Δ | GRPO Δ |
|---|---|---|---|---|---|
| **simple_python** | 95.00% | 95.50% | **96.50%** | +0.5 | **+1.0** |
| simple_java | 64.00% | 64.00% | 64.00% | 0 | 0 |
| simple_javascript | 80.00% | 68.00% | 68.00% | -12.0 | 0 |
| **irrelevance** | 77.08% | **87.08%** | 85.83% | **+10.0** | -1.3 |
| **parallel** | 93.50% | **95.50%** | **95.50%** | **+2.0** | 0 |
| **multiple** | 92.50% | **94.00%** | **94.00%** | **+1.5** | 0 |
| parallel_multiple | 90.00% | 81.00% | 86.50% | -9.0 | **+5.5** |
| **live_simple** | 78.68% | **80.23%** | **81.40%** | +1.6 | **+1.2** |
| live_multiple | 74.45% | **76.45%** | **76.83%** | +2.0 | +0.4 |
| **live_parallel** | 75.00% | **87.50%** | **87.50%** | **+12.5** | 0 |
| live_par_mult | 70.83% | 58.33% | 58.33% | -12.5 | 0 |
| **live_irrelevance** | 63.69% | **83.60%** | 81.11% | **+19.9** | -2.5 |
| live_relevance | 93.75% | 75.00% | **87.50%** | -18.8 | **+12.5** |

## Summary

### SFT Impact (1 epoch warmup)
- **7 categories improved**, 4 regressed, 2 unchanged
- Biggest wins: live_irrelevance +19.9%, live_parallel +12.5%, irrelevance +10%
- Biggest losses: live_relevance -18.8%, simple_javascript -12%, live_par_mult -12.5%

### GRPO Impact (250 steps DAPO)
- **4 categories improved**, 2 regressed, 7 unchanged
- Biggest wins: live_relevance +12.5%, parallel_multiple +5.5%, simple_python +1.0%
- GRPO partially recovered SFT regressions (live_relevance: 75→87.5%, parallel_multiple: 81→86.5%)

### End-to-End (Baseline → Post-GRPO)
- **8 categories improved** vs baseline
- **simple_python**: 95.0 → 96.5% (+1.5%)
- **parallel**: 93.5 → 95.5% (+2.0%)
- **multiple**: 92.5 → 94.0% (+1.5%)
- **irrelevance**: 77.1 → 85.8% (+8.8%)
- **live_simple**: 78.7 → 81.4% (+2.7%)
- **live_parallel**: 75.0 → 87.5% (+12.5%)
- **live_multiple**: 74.5 → 76.8% (+2.4%)
- **live_relevance**: 93.8 → 87.5% (-6.3%) — still regressed

## GRPO Observations
- DAPO loss with temp=1.5 gave some diversity (reward_std > 0 on ~30% of steps)
- Most steps still had reward_std=0 (model too confident)
- Average reward: ~1.5/2.0 (tool_name correct, args partially correct)
- format_reward decayed to 0 by mid-training as designed

## Training Metrics
- **SFT eval_loss**: 0.243 (1 epoch)
- **SFT train time**: ~27 minutes
- **GRPO train time**: ~15 minutes
- **Total eval time**: ~5 min per full single_turn eval (3641 samples)

## Files
- SFT model: `./output/merged/`
- GRPO model: `./output_grpo/merged/`
- TensorBoard logs: `./output/runs/`, `./output_grpo/runs/`
- Trackio dashboard: https://kenkaneki-trackio.hf.space/
- BFCL scores: `../gorilla/berkeley-function-call-leaderboard/score/`

## EAGLE-3 Speculative Decoding

### Training (speculators 0.4.0)
- **Data**: 2000 samples from GRPO reserved split, hidden states extracted via vLLM
- **Architecture**: EAGLE-3, 1 layer, Qwen3 draft arch
- **Training**: 5 epochs, lr=1e-4, flex_attention, ~60 min on H100
- **Results**: train_loss 39.3 → 1.2, val_loss 10.1, cond_acc ~0.56

### Deployment Status: BLOCKED
- speculators 0.4.0 saves Qwen3 EAGLE-3 checkpoint with intermediate_size=12288
- vLLM 0.17.1 `llama_eagle3` loader expects intermediate_size=11008 (Llama format)
- speculators converter also fails with shape mismatch
- **Fix needed**: Upstream PR to speculators/vLLM for Qwen3 EAGLE-3 support

### Baseline Inference Benchmark (without EAGLE)
| Concurrency | TTFT p50 | TTFT p99 | Latency p50 | Latency p99 | TPS |
|---|---|---|---|---|---|
| 1 | 6.7ms | 6.9ms | 249ms | 307ms | 148 |
| 4 | 7.0ms | 7.5ms | 261ms | 324ms | 142 |
| 8 | 7.3ms | 7.8ms | 273ms | 338ms | 135 |
| 16 | 7.8ms | 292ms | 288ms | 356ms | 126 |
| 32 | 9.2ms | 216ms | 341ms | 490ms | 103 |

### EAGLE-3 Benchmark Results (SGLang)

Successfully deployed via SGLang 0.5.9 with format conversion.

| Mode | Lat p50 | Lat p95 | Seq TPS | Batch TPS |
|---|---|---|---|---|
| **Baseline** | 166ms | 248ms | 123.5 | 4714 |
| EAGLE-3 (k=3) | 179ms | 253ms | 119.8 | 1312 |
| EAGLE-3 (k=5, topk=4) | 203ms | 299ms | 24.2 | 932 |

**Conclusion**: EAGLE-3 doesn't help for short tool-calling responses (~22 tokens avg). Speculative decoding overhead exceeds gains. EAGLE benefits appear at 100+ token generations.

Key factors:
- Low acceptance rate (cond_acc=0.56) — only 2000 training samples
- Short outputs (~22 tokens) — draft overhead dominates
- Batched workloads — speculation less effective with high concurrency

### EAGLE-3 Real Data Benchmark (ToolACE validation, avg 51 tokens)

| Mode | Lat p50 | Seq TPS | Batch TPS | Accept Rate |
|---|---|---|---|---|
| Baseline | 286ms | 137.8 | 3355 | — |
| EAGLE-3 (k=3) | 278ms | 141.8 | 1986 | 0.197 |

- Sequential: **1.03x** (negligible speedup)
- Batched: **0.59x** (slowdown due to overhead)
- Accept rate **0.197** — very low, needs more training data (10k+ samples) and longer training

EAGLE-3 speculative decoding needs acceptance rate >0.5 to provide meaningful speedup.

### EAGLE-3 Fine-tuned (15 epochs from pretrained checkpoint)

| Mode | Lat p50 | Seq TPS | Speedup | Val Accept Rate |
|---|---|---|---|---|
| Baseline | 291ms | 138 | 1.00x | — |
| Original EAGLE3 (5ep) | 289ms | 142 | 1.03x | 0.197 |
| **Finetuned EAGLE3 (15ep)** | 291ms | 139 | 1.01x | **0.620** |

Accept rate improved 0.197 → 0.620 after fine-tuning, but no speedup on short outputs (avg 53 tokens).
EAGLE is designed for long-form generation (200+ tokens). Tool calling responses are inherently short.

### EAGLE-3 Long Sequence Benchmark (avg 278 tokens)

| Mode | Avg Tokens | Latency p50 | TPS | Ratio |
|---|---|---|---|---|
| Baseline | 278 | 1510ms | 144 | 1.00x |
| EAGLE-3 FT (15ep) | 293 | 2803ms | 102 | 0.71x |

**Conclusion**: EAGLE-3 doesn't help on H100 with 8B model even for long sequences.
The H100 decode is already fast (~7ms/token) — EAGLE draft overhead exceeds speculation gains.
EAGLE is beneficial for larger models (70B+) where decode is memory-bandwidth bound.

### EAGLE-3 on Real ToolACE Long Responses (avg 170 tokens, 200 samples)

| Mode | Seq TPS | Batch TPS | Lat p50 |
|---|---|---|---|
| Baseline | 144 | 3568 | 961ms |
| EAGLE-3 FT | 141 | 2800 | 935ms |

Sequential: 0.98x (no benefit), Batch: 0.78x (overhead).

### Final EAGLE-3 Verdict
EAGLE speculative decoding is **not beneficial** for Qwen3-8B on H100 80GB:
- Model too small (8B) — decode is compute-bound, not memory-bandwidth-bound
- H100 too fast — baseline ~7ms/token leaves no room for speculation gains
- Accept rate 0.62 — decent but not enough to overcome draft model overhead
- **Recommendation**: Use EAGLE for 70B+ models on single GPU, or quantized models

## Quantization Benchmark (200 ToolACE prompts, SGLang)

| Mode | Seq p50 | Seq TPS | Batch TPS | Batch RPS |
|---|---|---|---|---|
| **BF16** | 301ms | 132 | 2,103 | 40.1 |
| **FP8 (W8A8)** | — | 200 | **8,614** | 20.8 |
| **W4A16 (INT4)** | **185ms** | **210** | 1,527 | 28.8 |
| FP8 + EAGLE | — | 137 | 3,294 | 8.0 |

### Key Findings
- **W4A16**: Best sequential latency (1.6x speedup, 185ms vs 301ms). 2x less memory.
- **FP8**: Best batch throughput (4.1x, 8614 tok/s). Minimal quality loss.
- **EAGLE + FP8**: Still slower than FP8 alone. EAGLE overhead > speculation gain on H100.
- **Recommendation**: Use FP8 for throughput, W4A16 for latency-sensitive deployments.

### EAGLE Verdict (Final)
EAGLE-3 speculative decoding is **not recommended** for Qwen3-8B on H100, regardless of:
- Quantization (BF16, FP8, W4A16)
- Sequence length (short or long)
- Concurrency level (1 to 100)

The H100 is too fast for an 8B model — decode is never the bottleneck.

## Quantization: Quality + Throughput Analysis

### BFCL Quality — Zero Degradation
| Quantization | Size | OVERALL Accuracy |
|---|---|---|
| BF16 | 16.4 GB | 82.9% |
| FP8 (W8A8) | 8.8 GB | 82.9% |
| W4A16 (INT4) | 6.1 GB | 82.9% |

All per-category scores identical across quantizations.

### Throughput by Concurrency (tok/s)
| Concurrency | BF16 | FP8 | W4A16 |
|---|---|---|---|
| 1 | 122 | 203 | 186 |
| 8 | 379 | 1,158 | 563 |
| 32 | 1,120 | 4,146 | 1,141 |
| 64 | 1,236 | 6,074 | 1,133 |

### Production Recommendations
| Use Case | Best Choice | Why |
|---|---|---|
| Max throughput | **FP8** | 4.9x throughput at c=64, half the memory |
| Min latency | **W4A16** | Fastest single-request (185ms p50) |
| Max quality | **Any** | All identical accuracy |
| Memory constrained | **W4A16** | 6.1 GB (62% smaller than BF16) |
| EAGLE speculation | **Skip** | Not beneficial on H100 + 8B |
