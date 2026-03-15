#!/usr/bin/env python3
"""
Concurrent inference benchmark via vLLM OpenAI API.

Measures per-request TTFT, latency, throughput at various concurrency levels.
Reports distributions: mean, std, SEM, p25/p50/p75/p90/p95/p99.

Usage:
    # Start vLLM server first:
    python -m vllm.entrypoints.openai.api_server --model ./output_grpo/merged --port 8100

    # Then benchmark:
    python scripts/bench.py --url http://localhost:8100/v1/chat/completions
    python scripts/bench.py --url http://localhost:8100/v1/chat/completions --concurrency 1,8,16,32
"""

import argparse
import asyncio
import json
import os
import random
import time
import urllib.request

import aiohttp
import numpy as np

SEED = 42

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
    "Weather forecast for Berlin?",
    "Any good sushi places in Manhattan?",
    "Calculate sqrt(144) + 3^4",
    "Book flight San Francisco to Chicago, May 10, 1 passenger",
    "Translate 'Thank you very much' to Japanese",
    "What's the temperature in Sydney?",
    "Find Thai restaurants in Seattle, mid-range price",
    "What is 2^10 * 3?",
    "Flight from Boston to Miami, June 15, 3 passengers",
    "Translate 'Where is the train station?' to German",
]

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "units": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_restaurants", "description": "Search restaurants",
        "parameters": {"type": "object", "properties": {"cuisine": {"type": "string"}, "location": {"type": "string"}}, "required": ["cuisine", "location"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "Calculate math",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "book_flight", "description": "Book flight",
        "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}, "required": ["origin", "destination", "date"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "Translate",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}}, "required": ["text", "target_lang"]}}},
]

_model_name = None


def build_request(idx):
    return {
        "model": _model_name,
        "messages": [{"role": "user", "content": QUERIES[idx % len(QUERIES)]}],
        "tools": TOOLS,
        "max_tokens": 256,
        "temperature": 0,
        "stream": False,
    }


async def send_request(session, url, request):
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=request) as resp:
            body = await resp.json()
        t_total = time.perf_counter() - t0
        usage = body.get("usage", {})
        comp_tok = usage.get("completion_tokens", 0)
        return {
            "success": True,
            "latency": t_total,
            "ttft": t_total / max(comp_tok, 1),
            "completion_tokens": comp_tok,
            "prompt_tokens": usage.get("prompt_tokens", 0),
        }
    except Exception as e:
        return {"success": False, "latency": time.perf_counter() - t0, "error": str(e)}


async def run_batch(url, concurrency, n):
    sem = asyncio.Semaphore(concurrency)
    async def limited(idx):
        async with sem:
            async with aiohttp.ClientSession() as s:
                return await send_request(s, url, build_request(idx))
    tasks = [limited(i) for i in range(n)]
    return await asyncio.gather(*tasks)


def compute_stats(results):
    ok = [r for r in results if r.get("success")]
    if not ok:
        return {"error": "all failed", "n": 0}
    lats = np.array([r["latency"] for r in ok]) * 1000
    ttfts = np.array([r["ttft"] for r in ok]) * 1000
    toks = np.array([r["completion_tokens"] for r in ok])
    n = len(lats)
    def dist(arr):
        return {
            "mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "sem": float(np.std(arr) / np.sqrt(n)),
            "min": float(np.min(arr)), "max": float(np.max(arr)),
            "p25": float(np.percentile(arr, 25)), "p50": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)), "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)),
        }
    return {
        "n": n, "failed": len(results) - n,
        "latency_ms": dist(lats), "ttft_ms": dist(ttfts),
        "tps": float(toks.sum() / (lats.sum() / 1000)),
        "avg_tokens": float(np.mean(toks)),
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    parser = argparse.ArgumentParser(description="Concurrent Inference Benchmark")
    parser.add_argument("--url", required=True, help="vLLM chat completions URL")
    parser.add_argument("--concurrency", default="1,4,8,16,32")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", default="results/bench_latency.json")
    args = parser.parse_args()

    levels = [int(c) for c in args.concurrency.split(",")]

    # Get model name
    global _model_name
    models_url = args.url.replace("/chat/completions", "/models")
    models = json.loads(urllib.request.urlopen(models_url).read())
    _model_name = models["data"][0]["id"]
    print(f"Model: {_model_name}")

    # Warmup
    print(f"Warmup ({args.warmup} requests)...")
    asyncio.run(run_batch(args.url, 1, args.warmup))

    all_stats = {}
    for c in levels:
        print(f"\n--- Concurrency: {c} ({args.num_prompts} requests) ---")
        results = asyncio.run(run_batch(args.url, c, args.num_prompts))
        stats = compute_stats(results)
        all_stats[c] = stats
        if "error" not in stats:
            l, t = stats["latency_ms"], stats["ttft_ms"]
            print(f"  Latency: mean={l['mean']:.0f}±{l['std']:.0f}ms  p50={l['p50']:.0f}  p95={l['p95']:.0f}  p99={l['p99']:.0f}")
            print(f"  TTFT:    mean={t['mean']:.1f}±{t['std']:.1f}ms  p50={t['p50']:.1f}  p95={t['p95']:.1f}  p99={t['p99']:.1f}")
            print(f"  TPS: {stats['tps']:.0f}  avg_tokens: {stats['avg_tokens']:.0f}")

    # Summary table
    print(f"\n{'='*110}")
    print(f"{'C':>3s} {'n':>5s} {'Lat mean±std':>16s} {'Lat p50':>9s} {'Lat p95':>9s} {'Lat p99':>9s} "
          f"{'TTFT p50':>9s} {'TTFT p95':>9s} {'TPS':>7s}")
    print(f"{'-'*110}")
    for c, s in sorted(all_stats.items()):
        if "error" in s: continue
        l, t = s["latency_ms"], s["ttft_ms"]
        print(f"{c:>3d} {s['n']:>5d} {l['mean']:>7.0f}±{l['std']:<6.0f}ms {l['p50']:>8.0f}ms {l['p95']:>8.0f}ms "
              f"{l['p99']:>8.0f}ms {t['p50']:>8.1f}ms {t['p95']:>8.1f}ms {s['tps']:>6.0f}")
    print(f"{'='*110}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"model": _model_name, "seed": SEED, "num_prompts": args.num_prompts,
                   "stats": {str(k): v for k, v in all_stats.items()}}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
