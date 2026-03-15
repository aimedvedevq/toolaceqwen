#!/usr/bin/env python3
"""Benchmark BF16 vs FP8 vs W4A16, with and without EAGLE-3."""

import json, time, numpy as np, sglang, gc
from datasets import load_from_disk
from transformers import AutoTokenizer

ROLE_MAP = {"user": "user", "assistant": "assistant", "tool": "tool"}

def build_prompts(tok, n=200):
    ds = load_from_disk("./output/grpo_data")
    prompts = []
    for ex in ds:
        msgs = []
        if ex.get("system"):
            msgs.append({"role": "system", "content": ex["system"]})
        last_idx = -1
        for i, turn in enumerate(ex["conversations"]):
            if turn["from"] == "assistant": last_idx = i
        if last_idx < 1: continue
        for turn in ex["conversations"][:last_idx]:
            msgs.append({"role": ROLE_MAP.get(turn["from"], "user"), "content": turn["value"]})
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if len(tok.encode(text)) > 3000: continue
        prompts.append(text)
        if len(prompts) >= n: break
    return prompts

def bench(engine, prompts, label):
    # Sequential (first 50)
    seq_prompts = prompts[:50]
    lats, toks = [], []
    for p in seq_prompts:
        t0 = time.perf_counter()
        r = engine.generate(prompt=p, sampling_params={"max_new_tokens": 512, "temperature": 0})
        lats.append((time.perf_counter() - t0) * 1000)
        toks.append(r["meta_info"]["completion_tokens"])
    lats, toks = np.array(lats), np.array(toks)
    seq_tps = toks.sum() / (lats.sum() / 1000)

    # Batched (all 200)
    t0 = time.perf_counter()
    br = engine.generate(prompt=prompts, sampling_params={"max_new_tokens": 512, "temperature": 0})
    bt = time.perf_counter() - t0
    btok = sum(r["meta_info"]["completion_tokens"] for r in br)
    batch_tps = btok / bt
    batch_rps = len(prompts) / bt

    print(f"  {label}:")
    print(f"    seq:   p50={np.median(lats):.0f}ms  p95={np.percentile(lats,95):.0f}ms  tps={seq_tps:.0f}  avg={toks.mean():.0f}tok")
    print(f"    batch: {batch_tps:.0f} tok/s  {batch_rps:.1f} req/s  avg={btok/len(prompts):.0f}tok  ({bt:.1f}s)")
    return {
        "seq_tps": float(seq_tps), "seq_p50": float(np.median(lats)), "seq_p95": float(np.percentile(lats, 95)),
        "batch_tps": float(batch_tps), "batch_rps": float(batch_rps),
        "avg_tok": float(toks.mean()), "batch_avg_tok": btok / len(prompts),
    }

def main():
    tok = AutoTokenizer.from_pretrained("./output_grpo/merged", trust_remote_code=True)
    prompts = build_prompts(tok)
    print(f"{len(prompts)} prompts\n")

    configs = [
        ("BF16", "./output_grpo/merged", None, {}),
        ("FP8", "./output_grpo/fp8", None, {}),
        ("W4A16", "./output_grpo/w4a16", None, {"quantization": "compressed-tensors"}),
        ("FP8 + EAGLE", "./output_grpo/fp8", "./output_eagle/finetuned_sglang",
         {"speculative_algorithm": "EAGLE3", "speculative_num_steps": 3,
          "speculative_eagle_topk": 1, "speculative_num_draft_tokens": 4}),
        ("W4A16 + EAGLE", "./output_grpo/w4a16", "./output_eagle/finetuned_sglang",
         {"quantization": "compressed-tensors", "speculative_algorithm": "EAGLE3",
          "speculative_num_steps": 3, "speculative_eagle_topk": 1, "speculative_num_draft_tokens": 4}),
    ]

    results = {}
    for label, model, eagle, extra in configs:
        print(f"{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")
        kwargs = {"model_path": model, "mem_fraction_static": 0.7}
        if eagle:
            kwargs["speculative_draft_model_path"] = eagle
        kwargs.update(extra)
        try:
            e = sglang.Engine(**kwargs)
            results[label] = bench(e, prompts, label)
            e.shutdown()
        except Exception as ex:
            print(f"  FAILED: {ex}")
            results[label] = {"error": str(ex)}
        gc.collect()
        time.sleep(5)

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Mode':20s} {'Seq p50':>10s} {'Seq TPS':>10s} {'Batch TPS':>10s} {'Batch RPS':>10s} {'Avg Tok':>8s}")
    print(f"{'-'*90}")
    for label, r in results.items():
        if "error" in r:
            print(f"  {label:18s} {'FAILED':>10s}")
            continue
        print(f"  {label:18s} {r['seq_p50']:>9.0f}ms {r['seq_tps']:>9.0f} {r['batch_tps']:>9.0f} {r['batch_rps']:>9.1f} {r['avg_tok']:>7.0f}")

    # Speedup vs BF16
    if "BF16" in results and "error" not in results["BF16"]:
        bf16 = results["BF16"]
        print(f"\n  Speedup vs BF16:")
        for label, r in results.items():
            if "error" in r or label == "BF16": continue
            print(f"    {label:18s} seq={r['seq_tps']/bf16['seq_tps']:.2f}x  batch={r['batch_tps']/bf16['batch_tps']:.2f}x")
    print(f"{'='*90}")

    with open("bench_quant.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved to bench_quant.json")

if __name__ == "__main__":
    main()
