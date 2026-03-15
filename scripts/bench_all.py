#!/usr/bin/env python3
"""
Full inference benchmark: vLLM vs SGLang × BF16/FP8/W4A16 × concurrency 1/8/16/32
All through HTTP API (fair comparison). Per-request distributions.
"""

import asyncio
import json
import os
import random
import signal
import subprocess
import sys
import time

import aiohttp
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SYSTEM = ("You have functions: get_weather(city), search_restaurants(cuisine, location), "
          "calculate(expression), book_flight(origin, destination, date), translate(text, lang). "
          "Call the right function. Reply ONLY with [func(args)].")

QUERIES = [
    "What's the weather in Tokyo?",
    "Find Italian restaurants in San Francisco",
    "Calculate compound interest on $10000 at 5% for 3 years",
    "Book a flight from NYC to London on March 25",
    "Translate 'Hello, how are you?' to Spanish",
    "Weather in Paris celsius?",
    "Find cheap Mexican food downtown LA",
    "What is 15% of 847.50?",
    "Flight Tokyo to Seoul April 1st",
    "Translate 'Good morning' to French",
    "Temperature in Berlin?",
    "Sushi places in Manhattan?",
    "Calculate sqrt(144) + 3^4",
    "Flight SF to Chicago May 10",
    "Translate 'Thank you' to Japanese",
    "Weather in Sydney?",
    "Thai restaurants Seattle mid-range",
    "What is 2^10 * 3?",
    "Flight Boston to Miami June 15 3 passengers",
    "Translate 'Where is the station?' to German",
]


async def http_request(session, url, model, query):
    t0 = time.perf_counter()
    try:
        async with session.post(url, json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": query},
            ],
            "max_tokens": 128, "temperature": 0, "stream": False,
        }, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            body = await resp.json()
        lat = time.perf_counter() - t0
        u = body.get("usage", {})
        ct = u.get("completion_tokens", 0)
        return {"ok": True, "lat": lat, "tok": ct, "ttft": lat / max(ct, 1)}
    except Exception as e:
        return {"ok": False, "lat": time.perf_counter() - t0, "err": str(e)[:80]}


async def bench_concurrent(url, model, concurrency, n=100):
    sem = asyncio.Semaphore(concurrency)
    async def lim(i):
        async with sem:
            async with aiohttp.ClientSession() as s:
                return await http_request(s, url, model, QUERIES[i % len(QUERIES)])
    results = await asyncio.gather(*[lim(i) for i in range(n)])
    return [r for r in results if r["ok"]]


def stats(results):
    if not results:
        return None
    lats = np.array([r["lat"] for r in results]) * 1000
    ttfts = np.array([r["ttft"] for r in results]) * 1000
    toks = np.array([r["tok"] for r in results])
    n = len(lats)
    return {
        "n": n,
        "lat_mean": float(np.mean(lats)), "lat_std": float(np.std(lats)),
        "lat_sem": float(np.std(lats) / np.sqrt(n)),
        "lat_p50": float(np.median(lats)), "lat_p95": float(np.percentile(lats, 95)),
        "lat_p99": float(np.percentile(lats, 99)),
        "ttft_mean": float(np.mean(ttfts)), "ttft_std": float(np.std(ttfts)),
        "ttft_p50": float(np.median(ttfts)), "ttft_p95": float(np.percentile(ttfts, 95)),
        "ttft_p99": float(np.percentile(ttfts, 99)),
        "tps": float(toks.sum() / (lats.sum() / 1000)),
        "avg_tok": float(np.mean(toks)),
        "lat_raw": lats.tolist(),
        "ttft_raw": ttfts.tolist(),
    }


def wait_server(port, timeout=180):
    import urllib.request
    for _ in range(timeout // 3):
        time.sleep(3)
        try:
            r = urllib.request.urlopen(f"http://localhost:{port}/v1/models")
            data = json.loads(r.read())
            return data["data"][0]["id"]
        except:
            pass
    return None


def kill_port(port):
    os.system(f"fuser -k {port}/tcp 2>/dev/null")
    time.sleep(3)


def main():
    CONCURRENCY = [1, 8, 16, 32]
    N = 100
    PORT = 8100

    model_path = "./output_grpo/merged"
    fp8_path = "./output_grpo/fp8"
    awq_path = "./output_grpo/awq"

    configs = [
        ("vLLM BF16", "vllm", model_path, []),
        ("vLLM FP8", "vllm", fp8_path, []),
        ("vLLM W4A16", "vllm", awq_path, ["--quantization", "compressed-tensors"]),
        ("SGLang BF16", "sglang", model_path, []),
        ("SGLang FP8", "sglang", fp8_path, []),
        ("SGLang W4A16", "sglang", awq_path, ["--quantization", "compressed-tensors"]),
    ]

    all_results = {}

    for label, engine, path, extra in configs:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        kill_port(PORT)

        if engine == "vllm":
            cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                   "--model", path, "--port", str(PORT), "--dtype", "auto",
                   "--trust-remote-code", "--max-model-len", "4096"] + extra
        else:
            cmd = [sys.executable, "-m", "sglang.launch_server",
                   "--model-path", path, "--port", str(PORT),
                   "--mem-fraction-static", "0.7"] + extra

        print(f"  Starting {engine}...")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        model_name = wait_server(PORT)

        if not model_name:
            print(f"  FAILED to start {label}")
            proc.kill()
            all_results[label] = {"error": "server failed"}
            continue

        print(f"  Server ready: {model_name}")
        url = f"http://localhost:{PORT}/v1/chat/completions"

        # Warmup
        asyncio.run(bench_concurrent(url, model_name, 1, 5))

        cfg_results = {}
        for c in CONCURRENCY:
            raw = asyncio.run(bench_concurrent(url, model_name, c, N))
            s = stats(raw)
            if s:
                cfg_results[c] = s
                print(f"  c={c:>2d}: lat_p50={s['lat_p50']:>7.0f}ms  ttft_p50={s['ttft_p50']:>5.1f}ms  "
                      f"tps={s['tps']:>6.0f}  avg_tok={s['avg_tok']:.0f}")
            else:
                print(f"  c={c:>2d}: all requests failed")
                cfg_results[c] = {"error": "all failed"}

        all_results[label] = cfg_results
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except:
            proc.kill()
        time.sleep(5)

    # ═══════ Summary Table ═══════
    print(f"\n{'='*120}")
    print(f"{'Config':<20s} {'c':>3s} {'Lat p50':>9s} {'Lat p95':>9s} {'Lat p99':>9s} "
          f"{'Lat mean±std':>16s} {'TTFT p50':>9s} {'TTFT p95':>9s} {'TPS':>7s} {'AvgTok':>7s}")
    print(f"{'-'*120}")
    for label, data in all_results.items():
        if "error" in data:
            print(f"  {label:<18s} FAILED: {data['error']}")
            continue
        for c in CONCURRENCY:
            s = data.get(c, {})
            if "error" in s or not s:
                continue
            print(f"  {label:<18s} {c:>3d} {s['lat_p50']:>8.0f}ms {s['lat_p95']:>8.0f}ms "
                  f"{s['lat_p99']:>8.0f}ms {s['lat_mean']:>7.0f}±{s['lat_std']:<6.0f}ms "
                  f"{s['ttft_p50']:>8.1f}ms {s['ttft_p95']:>8.1f}ms {s['tps']:>6.0f} {s['avg_tok']:>6.0f}")
    print(f"{'='*120}")

    # Save (strip raw arrays for size)
    save = {}
    for label, data in all_results.items():
        if "error" in data:
            save[label] = data
            continue
        save[label] = {}
        for c, s in data.items():
            if isinstance(s, dict) and "lat_raw" in s:
                save[label][c] = {k: v for k, v in s.items() if k not in ("lat_raw", "ttft_raw")}
            else:
                save[label][c] = s

    with open("results/bench_inference.json", "w") as f:
        json.dump(save, f, indent=2)

    # Save raw for distributions
    with open("results/bench_inference_raw.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved to results/bench_inference.json ({os.path.getsize('results/bench_inference.json')/1024:.0f}KB)")
    print(f"Raw data: results/bench_inference_raw.json ({os.path.getsize('results/bench_inference_raw.json')/1024:.0f}KB)")


if __name__ == "__main__":
    main()
