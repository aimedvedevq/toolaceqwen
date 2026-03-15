#!/usr/bin/env python3
"""Benchmark EAGLE-3 on ToolACE validation — full conversations, long sequences."""

import json
import time
import numpy as np
import sglang
from datasets import load_from_disk
from transformers import AutoTokenizer

ROLE_MAP = {"user": "user", "assistant": "assistant", "tool": "tool"}


def build_prompts(ds, tokenizer, max_prompts=200):
    """Build prompts: full dialog up to LAST assistant turn → generate that turn."""
    prompts = []
    gt_lengths = []

    for ex in ds:
        system = ex.get("system", "")
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})

        last_assistant_content = None
        last_assistant_idx = -1

        for i, turn in enumerate(ex["conversations"]):
            role = ROLE_MAP.get(turn["from"], "user")
            if role == "assistant":
                last_assistant_content = turn["value"]
                last_assistant_idx = i
            msgs.append({"role": role, "content": turn["value"]})

        if last_assistant_content is None or last_assistant_idx < 1:
            continue

        # Build prompt = everything up to last assistant
        prompt_msgs = []
        if system:
            prompt_msgs.append({"role": "system", "content": system})
        for turn in ex["conversations"][:last_assistant_idx]:
            role = ROLE_MAP.get(turn["from"], "user")
            prompt_msgs.append({"role": role, "content": turn["value"]})

        text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompt_len = len(tokenizer.encode(text))
        gt_len = len(tokenizer.encode(last_assistant_content))

        if prompt_len > 3000 or gt_len < 10:
            continue

        prompts.append({"text": text, "prompt_len": prompt_len, "gt_len": gt_len})
        if len(prompts) >= max_prompts:
            break

    return prompts


def run_bench(engine, prompts, label, max_tokens=512):
    """Run sequential benchmark with detailed stats."""
    latencies = []
    tokens_list = []
    ttfts = []

    for i, p in enumerate(prompts):
        t0 = time.perf_counter()
        result = engine.generate(
            prompt=p["text"],
            sampling_params={"max_new_tokens": max_tokens, "temperature": 0},
        )
        t1 = time.perf_counter()
        lat = (t1 - t0) * 1000
        ntok = result["meta_info"]["completion_tokens"]
        latencies.append(lat)
        tokens_list.append(ntok)
        if ntok > 0:
            ttfts.append(lat / ntok)  # approx time per token

    latencies = np.array(latencies)
    tokens_arr = np.array(tokens_list)
    total_tokens = int(tokens_arr.sum())
    total_time = latencies.sum() / 1000

    stats = {
        "label": label,
        "n": len(prompts),
        "avg_gen_tokens": float(tokens_arr.mean()),
        "avg_prompt_tokens": float(np.mean([p["prompt_len"] for p in prompts])),
        "total_tokens": total_tokens,
        "latency_ms": {
            "mean": float(latencies.mean()),
            "p50": float(np.percentile(latencies, 50)),
            "p90": float(np.percentile(latencies, 90)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        },
        "tps": float(total_tokens / total_time) if total_time > 0 else 0,
        "ms_per_token": float(total_time / total_tokens * 1000) if total_tokens > 0 else 0,
    }

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"  n={len(prompts)}, avg prompt={stats['avg_prompt_tokens']:.0f} tok, avg gen={stats['avg_gen_tokens']:.0f} tok")
    print(f"  Latency: p50={stats['latency_ms']['p50']:.0f}ms  p90={stats['latency_ms']['p90']:.0f}ms  p95={stats['latency_ms']['p95']:.0f}ms")
    print(f"  TPS: {stats['tps']:.1f}  ms/token: {stats['ms_per_token']:.2f}")

    # Batched throughput
    t0 = time.perf_counter()
    batch_results = engine.generate(
        prompt=[p["text"] for p in prompts],
        sampling_params={"max_new_tokens": max_tokens, "temperature": 0},
    )
    t1 = time.perf_counter()
    batch_tokens = sum(r["meta_info"]["completion_tokens"] for r in batch_results)
    batch_time = t1 - t0
    stats["batch_tps"] = float(batch_tokens / batch_time)
    stats["batch_rps"] = float(len(prompts) / batch_time)
    print(f"  Batch: {stats['batch_tps']:.0f} tok/s, {stats['batch_rps']:.1f} req/s ({batch_time:.1f}s)")

    return stats


def main():
    tokenizer = AutoTokenizer.from_pretrained("./output_grpo/merged", trust_remote_code=True)
    ds = load_from_disk("./output/grpo_data")
    prompts = build_prompts(ds, tokenizer, max_prompts=200)
    print(f"Built {len(prompts)} prompts")
    prompt_lens = [p["prompt_len"] for p in prompts]
    gt_lens = [p["gt_len"] for p in prompts]
    print(f"Prompt lengths: mean={np.mean(prompt_lens):.0f}, p50={np.median(prompt_lens):.0f}")
    print(f"GT response lengths: mean={np.mean(gt_lens):.0f}, p50={np.median(gt_lens):.0f}, p90={np.percentile(gt_lens,90):.0f}")

    results = {}

    # Baseline
    engine = sglang.Engine(model_path="./output_grpo/merged", mem_fraction_static=0.7)
    results["baseline"] = run_bench(engine, prompts, "BASELINE (no speculation)")
    engine.shutdown()
    time.sleep(5)

    # EAGLE-3
    engine = sglang.Engine(
        model_path="./output_grpo/merged",
        mem_fraction_static=0.7,
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path="./output_eagle/sglang_ckpt",
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
    )
    results["eagle3"] = run_bench(engine, prompts, "EAGLE-3 (steps=3, topk=1)")
    engine.shutdown()

    # Summary
    b = results["baseline"]
    e = results["eagle3"]
    seq_speedup = e["tps"] / b["tps"] if b["tps"] > 0 else 0
    batch_speedup = e["batch_tps"] / b["batch_tps"] if b["batch_tps"] > 0 else 0

    print(f"\n{'='*70}")
    print(f"{'':25s} {'Baseline':>15s} {'EAGLE-3':>15s} {'Speedup':>10s}")
    print(f"{'-'*70}")
    print(f"  {'Latency p50':25s} {b['latency_ms']['p50']:>14.0f}ms {e['latency_ms']['p50']:>14.0f}ms {b['latency_ms']['p50']/e['latency_ms']['p50']:>9.2f}x")
    print(f"  {'Latency p95':25s} {b['latency_ms']['p95']:>14.0f}ms {e['latency_ms']['p95']:>14.0f}ms {b['latency_ms']['p95']/e['latency_ms']['p95']:>9.2f}x")
    print(f"  {'Sequential TPS':25s} {b['tps']:>14.1f} {e['tps']:>14.1f} {seq_speedup:>9.2f}x")
    print(f"  {'ms/token':25s} {b['ms_per_token']:>13.2f}ms {e['ms_per_token']:>13.2f}ms")
    print(f"  {'Batch TPS':25s} {b['batch_tps']:>14.0f} {e['batch_tps']:>14.0f} {batch_speedup:>9.2f}x")
    print(f"  {'Batch RPS':25s} {b['batch_rps']:>14.1f} {e['batch_rps']:>14.1f}")
    print(f"{'='*70}")

    with open("bench_eagle_toolace.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved to bench_eagle_toolace.json")


if __name__ == "__main__":
    main()
