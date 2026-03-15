#!/usr/bin/env python3
"""
Concurrent inference benchmark for tool-calling models.

Measures TTFT, generation latency, throughput under 1/4/8/16/32 concurrent requests.
Uses BFCL-style tool calling prompts.

Usage:
  python bench.py --model ./output_grpo/merged
  python bench.py --model Qwen/Qwen3-8B --concurrency 1,4,8,16,32
  python bench.py --model ./output_grpo/merged --num-prompts 200 --output bench_results.json
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
import numpy as np
import subprocess
import sys
import signal
import os


# Sample tool-calling prompts (BFCL-style)
SAMPLE_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_restaurants", "description": "Search for restaurants by cuisine and location", "parameters": {"type": "object", "properties": {"cuisine": {"type": "string"}, "location": {"type": "string"}, "price_range": {"type": "string", "enum": ["$", "$$", "$$$"]}}, "required": ["cuisine", "location"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "Perform mathematical calculations", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression to evaluate"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "book_flight", "description": "Book a flight between two cities", "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}, "passengers": {"type": "integer"}}, "required": ["origin", "destination", "date"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "Translate text between languages", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "source_lang": {"type": "string"}, "target_lang": {"type": "string"}}, "required": ["text", "target_lang"]}}},
]

SAMPLE_QUERIES = [
    "What's the weather like in Tokyo right now?",
    "Find me Italian restaurants in San Francisco under $$$",
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


_model_name = None  # set at runtime

def build_request(query_idx: int) -> dict:
    """Build an OpenAI-compatible chat completion request."""
    query = SAMPLE_QUERIES[query_idx % len(SAMPLE_QUERIES)]
    return {
        "model": _model_name,
        "messages": [{"role": "user", "content": query}],
        "tools": SAMPLE_TOOLS,
        "max_tokens": 128,
        "temperature": 0,
        "stream": False,
    }


async def send_request(session: aiohttp.ClientSession, url: str, request: dict) -> dict:
    """Send request and measure TTFT + total latency."""
    t_start = time.perf_counter()

    try:
        async with session.post(url, json=request) as resp:
            body = await resp.json()

        t_end = time.perf_counter()
        total_latency = t_end - t_start

        choice = body.get("choices", [{}])[0]
        usage = body.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls") or []
        has_tool_call = len(tool_calls) > 0

        ttft = total_latency / max(completion_tokens, 1)

        return {
            "success": True,
            "ttft": ttft,
            "total_latency": total_latency,
            "tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "tps": completion_tokens / total_latency if total_latency > 0 else 0,
            "has_tool_call": has_tool_call,
            "content": (json.dumps(tool_calls[0]["function"]) if has_tool_call else content)[:200],
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "ttft": None,
            "total_latency": time.perf_counter() - t_start,
            "tokens": 0,
            "tps": 0,
        }


async def run_concurrent_batch(url: str, concurrency: int, num_prompts: int) -> list[dict]:
    """Run num_prompts requests with given concurrency level."""
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def limited_request(idx):
        async with semaphore:
            req = build_request(idx)
            async with aiohttp.ClientSession() as session:
                result = await send_request(session, url, req)
                result["request_idx"] = idx
                return result

    tasks = [limited_request(i) for i in range(num_prompts)]
    results = await asyncio.gather(*tasks)
    return list(results)


def compute_stats(results: list[dict]) -> dict:
    """Compute percentile stats from benchmark results."""
    successful = [r for r in results if r["success"]]
    if not successful:
        return {"error": "all requests failed", "n": 0}

    ttfts = np.array([r["ttft"] for r in successful]) * 1000  # ms
    latencies = np.array([r["total_latency"] for r in successful]) * 1000  # ms
    tps_values = np.array([r["tps"] for r in successful])
    tokens = np.array([r["tokens"] for r in successful])

    return {
        "n": len(successful),
        "failed": len(results) - len(successful),
        "ttft_ms": {
            "mean": float(np.mean(ttfts)),
            "std": float(np.std(ttfts)),
            "p50": float(np.percentile(ttfts, 50)),
            "p90": float(np.percentile(ttfts, 90)),
            "p95": float(np.percentile(ttfts, 95)),
            "p99": float(np.percentile(ttfts, 99)),
            "min": float(np.min(ttfts)),
            "max": float(np.max(ttfts)),
        },
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p90": float(np.percentile(latencies, 90)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
        },
        "tokens_per_second": {
            "mean": float(np.mean(tps_values)),
            "p50": float(np.percentile(tps_values, 50)),
        },
        "throughput_rps": len(successful) / (np.max(latencies) / 1000) if np.max(latencies) > 0 else 0,
        "avg_tokens": float(np.mean(tokens)),
    }


def print_table(all_stats: dict):
    """Print a nice comparison table."""
    print(f"\n{'='*100}")
    print(f"{'Concurrency':>12s} {'Requests':>9s} {'TTFT p50':>10s} {'TTFT p95':>10s} {'TTFT p99':>10s} "
          f"{'Lat p50':>10s} {'Lat p95':>10s} {'Lat p99':>10s} {'TPS':>8s} {'RPS':>8s}")
    print(f"{'-'*100}")

    for c, stats in sorted(all_stats.items()):
        if "error" in stats:
            print(f"{c:>12d} {'FAILED':>9s}")
            continue
        t = stats["ttft_ms"]
        l = stats["latency_ms"]
        print(f"{c:>12d} {stats['n']:>9d} "
              f"{t['p50']:>9.1f}ms {t['p95']:>9.1f}ms {t['p99']:>9.1f}ms "
              f"{l['p50']:>9.1f}ms {l['p95']:>9.1f}ms {l['p99']:>9.1f}ms "
              f"{stats['tokens_per_second']['mean']:>7.1f} "
              f"{stats['throughput_rps']:>7.1f}")

    print(f"{'='*100}")


def start_vllm_server(model_path: str, port: int = 8100) -> subprocess.Popen:
    """Start a vLLM server in the background."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--max-model-len", "4096",
    ]
    print(f"Starting vLLM server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for server to be ready
    import urllib.request
    for i in range(120):
        time.sleep(2)
        try:
            urllib.request.urlopen(f"http://localhost:{port}/v1/models")
            print(f"vLLM server ready on port {port}")
            return proc
        except Exception:
            pass
    raise RuntimeError("vLLM server failed to start")


def main():
    parser = argparse.ArgumentParser(description="Concurrent inference benchmark")
    parser.add_argument("--model", required=True, help="Model path or HF ID")
    parser.add_argument("--concurrency", default="1,4,8,16,32",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--num-prompts", type=int, default=100,
                        help="Number of prompts per concurrency level")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--url", default=None, help="Use existing server URL (skip starting vLLM)")
    parser.add_argument("--output", default="bench_results.json")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup requests")
    args = parser.parse_args()

    concurrency_levels = [int(c) for c in args.concurrency.split(",")]
    base_url = args.url or f"http://localhost:{args.port}/v1/chat/completions"

    # Start server if needed
    server_proc = None
    if args.url is None:
        server_proc = start_vllm_server(args.model, args.port)

    try:
        # Get model name from server
        global _model_name
        import urllib.request
        models_url = base_url.replace("/chat/completions", "/models")
        models = json.loads(urllib.request.urlopen(models_url).read())
        _model_name = models["data"][0]["id"]
        print(f"Model: {_model_name}")

        # Warmup
        print(f"Warmup ({args.warmup} requests)...")
        asyncio.run(run_concurrent_batch(base_url, 1, args.warmup))

        all_stats = {}
        all_raw = {}

        for c in concurrency_levels:
            print(f"\n--- Concurrency: {c} ({args.num_prompts} requests) ---")
            results = asyncio.run(run_concurrent_batch(base_url, c, args.num_prompts))
            stats = compute_stats(results)
            all_stats[c] = stats
            all_raw[c] = results

            t = stats.get("ttft_ms", {})
            l = stats.get("latency_ms", {})
            if "error" not in stats:
                print(f"  TTFT:    p50={t['p50']:.1f}ms  p95={t['p95']:.1f}ms  p99={t['p99']:.1f}ms")
                print(f"  Latency: p50={l['p50']:.1f}ms  p95={l['p95']:.1f}ms  p99={l['p99']:.1f}ms")
                print(f"  TPS: {stats['tokens_per_second']['mean']:.1f}  RPS: {stats['throughput_rps']:.1f}")

        # Print table
        print_table(all_stats)

        # Save results
        report = {
            "model": args.model,
            "num_prompts": args.num_prompts,
            "concurrency_levels": concurrency_levels,
            "stats": {str(k): v for k, v in all_stats.items()},
            "raw": {str(k): v for k, v in all_raw.items()},
        }

        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    finally:
        if server_proc:
            print("Stopping vLLM server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
