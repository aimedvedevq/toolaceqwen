#!/usr/bin/env python3
"""Benchmark baseline vs EAGLE-3 via SGLang."""

import json
import time
import sglang
import numpy as np

QUERIES = [
    "What's the weather like in Tokyo right now?",
    "Find me Italian restaurants in San Francisco",
    "Calculate the compound interest on $10000 at 5% for 3 years",
    "Book a flight from NYC to London on March 25th for 2 passengers",
    "Translate 'Hello, how are you?' to Spanish",
    "What's the weather in Paris in celsius?",
    "Find cheap Mexican food near downtown LA",
    "What is 15% of 847.50?",
    "I need a flight from Tokyo to Seoul on April 1st",
    "Translate 'Good morning' to French",
] * 10  # 100 queries

SYSTEM = (
    "You have these functions: get_weather(city,units), search_restaurants(cuisine,location,price_range), "
    "calculate(expression), book_flight(origin,destination,date,passengers), translate_text(text,target_lang). "
    "Call the right function. Reply ONLY with [func(args)]."
)

def run_bench(engine, label, num_prompts=100):
    """Run sequential + batched benchmark."""
    prompts = QUERIES[:num_prompts]

    # Sequential (concurrency=1)
    latencies = []
    ttfts = []
    tokens_list = []
    for q in prompts:
        t0 = time.perf_counter()
        result = engine.generate(
            prompt=f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n",
            sampling_params={"max_new_tokens": 128, "temperature": 0},
        )
        t1 = time.perf_counter()
        lat = (t1 - t0) * 1000
        ntok = result["meta_info"]["completion_tokens"]
        latencies.append(lat)
        tokens_list.append(ntok)
        if ntok > 0:
            ttfts.append(lat / ntok)

    latencies = np.array(latencies)
    tokens_arr = np.array(tokens_list)
    tps = tokens_arr.sum() / (latencies.sum() / 1000)

    print(f"\n--- {label} (sequential, n={num_prompts}) ---")
    print(f"  Latency: p50={np.percentile(latencies,50):.1f}ms  p95={np.percentile(latencies,95):.1f}ms  p99={np.percentile(latencies,99):.1f}ms")
    print(f"  TPS: {tps:.1f}  avg_tokens: {tokens_arr.mean():.1f}")

    # Batched
    t0 = time.perf_counter()
    batch_prompts = [
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        for q in prompts
    ]
    results = engine.generate(
        prompt=batch_prompts,
        sampling_params={"max_new_tokens": 128, "temperature": 0},
    )
    t1 = time.perf_counter()
    total_time = t1 - t0
    total_tokens = sum(r["meta_info"]["completion_tokens"] for r in results)

    print(f"\n--- {label} (batched, n={num_prompts}) ---")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {num_prompts/total_time:.1f} req/s, {total_tokens/total_time:.1f} tok/s")
    print(f"  Avg tokens: {total_tokens/num_prompts:.1f}")

    return {
        "label": label,
        "sequential": {
            "latency_p50": float(np.percentile(latencies, 50)),
            "latency_p95": float(np.percentile(latencies, 95)),
            "latency_p99": float(np.percentile(latencies, 99)),
            "tps": float(tps),
            "avg_tokens": float(tokens_arr.mean()),
        },
        "batched": {
            "total_time_s": total_time,
            "rps": num_prompts / total_time,
            "tps": total_tokens / total_time,
            "avg_tokens": total_tokens / num_prompts,
        },
    }


def main():
    results = {}

    # Baseline (no EAGLE)
    print("=" * 60)
    print("Starting BASELINE engine...")
    print("=" * 60)
    engine = sglang.Engine(
        model_path="./output_grpo/merged",
        mem_fraction_static=0.7,
    )
    results["baseline"] = run_bench(engine, "BASELINE")
    engine.shutdown()
    time.sleep(5)

    # EAGLE-3
    print("\n" + "=" * 60)
    print("Starting EAGLE-3 engine...")
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
    results["eagle3_k3"] = run_bench(engine, "EAGLE-3 (k=3)")
    engine.shutdown()
    time.sleep(5)

    # EAGLE-3 aggressive
    print("\n" + "=" * 60)
    print("Starting EAGLE-3 aggressive engine...")
    print("=" * 60)
    engine = sglang.Engine(
        model_path="./output_grpo/merged",
        mem_fraction_static=0.7,
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path="./output_eagle/sglang_ckpt",
        speculative_num_steps=5,
        speculative_eagle_topk=4,
        speculative_num_draft_tokens=16,
    )
    results["eagle3_k5"] = run_bench(engine, "EAGLE-3 (k=5, topk=4)")
    engine.shutdown()

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Mode':30s} {'Lat p50':>10s} {'Lat p95':>10s} {'Seq TPS':>10s} {'Batch RPS':>10s} {'Batch TPS':>10s}")
    print("-" * 80)
    for k, v in results.items():
        s = v["sequential"]
        b = v["batched"]
        print(f"  {v['label']:28s} {s['latency_p50']:>9.1f}ms {s['latency_p95']:>9.1f}ms {s['tps']:>9.1f} {b['rps']:>9.1f} {b['tps']:>9.1f}")

    # Speedup
    if "baseline" in results and "eagle3_k3" in results:
        base_tps = results["baseline"]["batched"]["tps"]
        eagle_tps = results["eagle3_k3"]["batched"]["tps"]
        print(f"\n  EAGLE-3 speedup: {eagle_tps/base_tps:.2f}x throughput")

    print("=" * 80)

    with open("bench_eagle_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Results saved to bench_eagle_comparison.json")


if __name__ == "__main__":
    main()
