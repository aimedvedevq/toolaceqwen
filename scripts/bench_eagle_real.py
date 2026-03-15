#!/usr/bin/env python3
"""Benchmark EAGLE-3 on real ToolACE validation prompts (long responses)."""

import json
import time
import numpy as np
import sglang
from datasets import load_from_disk
from transformers import AutoTokenizer

ROLE_MAP = {"user": "user", "assistant": "assistant", "tool": "tool"}


def build_prompts(ds, tokenizer, max_prompts=200):
    """Build prompts from validation data — only first user turn."""
    prompts = []
    for ex in ds:
        system = ex.get("system", "")
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        # Add turns up to first assistant response
        for turn in ex["conversations"]:
            role = ROLE_MAP.get(turn["from"], "user")
            if role == "assistant":
                break
            msgs.append({"role": role, "content": turn["value"]})

        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        # Skip very long prompts
        if len(tokenizer.encode(text)) > 2048:
            continue
        prompts.append(text)
        if len(prompts) >= max_prompts:
            break
    return prompts


def run_bench(engine, prompts, label, max_tokens=512):
    """Run benchmark — sequential."""
    latencies = []
    tokens_list = []
    accept_rates = []

    for i, p in enumerate(prompts):
        t0 = time.perf_counter()
        result = engine.generate(
            prompt=p,
            sampling_params={"max_new_tokens": max_tokens, "temperature": 0},
        )
        t1 = time.perf_counter()
        lat = (t1 - t0) * 1000
        ntok = result["meta_info"]["completion_tokens"]
        latencies.append(lat)
        tokens_list.append(ntok)

        # Try to get acceptance rate from meta
        meta = result.get("meta_info", {})
        if "spec_accept_rate" in meta:
            accept_rates.append(meta["spec_accept_rate"])

        if i < 3:
            print(f"  [{i}] {ntok} tokens, {lat:.0f}ms, {ntok/(lat/1000):.0f} tok/s")

    latencies = np.array(latencies)
    tokens_arr = np.array(tokens_list)
    total_tokens = tokens_arr.sum()
    total_time = latencies.sum() / 1000

    stats = {
        "label": label,
        "n": len(prompts),
        "avg_tokens": float(tokens_arr.mean()),
        "total_tokens": int(total_tokens),
        "latency_p50": float(np.percentile(latencies, 50)),
        "latency_p95": float(np.percentile(latencies, 95)),
        "latency_p99": float(np.percentile(latencies, 99)),
        "tps": float(total_tokens / total_time),
        "time_per_token_ms": float(total_time / total_tokens * 1000) if total_tokens > 0 else 0,
    }
    if accept_rates:
        stats["accept_rate"] = float(np.mean(accept_rates))

    print(f"\n--- {label} (n={len(prompts)}, avg_tokens={tokens_arr.mean():.0f}) ---")
    print(f"  Latency: p50={stats['latency_p50']:.1f}ms  p95={stats['latency_p95']:.1f}ms")
    print(f"  TPS: {stats['tps']:.1f}  time/token: {stats['time_per_token_ms']:.2f}ms")
    if accept_rates:
        print(f"  Accept rate: {np.mean(accept_rates):.3f}")

    # Batched
    t0 = time.perf_counter()
    results = engine.generate(
        prompt=prompts,
        sampling_params={"max_new_tokens": max_tokens, "temperature": 0},
    )
    t1 = time.perf_counter()
    batch_tokens = sum(r["meta_info"]["completion_tokens"] for r in results)
    batch_time = t1 - t0
    stats["batch_tps"] = float(batch_tokens / batch_time)
    stats["batch_rps"] = float(len(prompts) / batch_time)
    print(f"  Batched: {stats['batch_tps']:.0f} tok/s, {stats['batch_rps']:.1f} req/s")

    return stats


def main():
    tokenizer = AutoTokenizer.from_pretrained("./output_grpo/merged", trust_remote_code=True)
    ds = load_from_disk("./output/grpo_data")
    prompts = build_prompts(ds, tokenizer, max_prompts=200)
    print(f"Built {len(prompts)} prompts")

    results = {}

    # Baseline
    print("\n" + "=" * 60)
    print("BASELINE")
    print("=" * 60)
    engine = sglang.Engine(model_path="./output_grpo/merged", mem_fraction_static=0.7)
    results["baseline"] = run_bench(engine, prompts, "Baseline")
    engine.shutdown()
    time.sleep(5)

    # EAGLE-3 conservative
    print("\n" + "=" * 60)
    print("EAGLE-3 (k=3)")
    print("=" * 60)
    engine = sglang.Engine(
        model_path="./output_grpo/merged",
        mem_fraction_static=0.7,
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path="./output_eagle/sglang_ckpt",
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
    )
    results["eagle3"] = run_bench(engine, prompts, "EAGLE-3 (k=3)")
    engine.shutdown()

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Mode':25s} {'Avg Tok':>8s} {'Lat p50':>10s} {'Lat p95':>10s} {'Seq TPS':>10s} {'Batch TPS':>10s}")
    print("-" * 80)
    for v in results.values():
        print(f"  {v['label']:23s} {v['avg_tokens']:>7.0f} {v['latency_p50']:>9.1f}ms {v['latency_p95']:>9.1f}ms {v['tps']:>9.1f} {v['batch_tps']:>9.0f}")

    if "baseline" in results and "eagle3" in results:
        seq_speedup = results["eagle3"]["tps"] / results["baseline"]["tps"]
        batch_speedup = results["eagle3"]["batch_tps"] / results["baseline"]["batch_tps"]
        print(f"\n  Speedup: seq={seq_speedup:.2f}x, batch={batch_speedup:.2f}x")

    print("=" * 80)

    with open("bench_eagle_real.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved to bench_eagle_real.json")


if __name__ == "__main__":
    main()
