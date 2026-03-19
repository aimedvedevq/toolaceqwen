#!/usr/bin/env python3
"""
Inference benchmark using vllm bench serve (standard vLLM benchmarking tool).

Measures TTFT, TPOT, ITL, E2EL with full percentile distributions
at each concurrency level. Uses tool-calling requests via --extra-body.

Usage:
    # Benchmark a running server:
    python scripts/bench.py --port 8100

    # Specific concurrency levels:
    python scripts/bench.py --port 8100 --concurrency 1,16,32

    # Full suite — starts/stops vLLM for BF16, FP8, W4A16:
    python scripts/bench.py --suite
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

PROMPTS = [
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
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get current weather for a city",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        }, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search_restaurants", "description": "Search for restaurants",
        "parameters": {"type": "object", "properties": {
            "cuisine": {"type": "string"}, "location": {"type": "string"},
            "price_range": {"type": "string"}
        }, "required": ["cuisine", "location"]}}},
    {"type": "function", "function": {
        "name": "calculate", "description": "Evaluate a math expression",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}
        }, "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "book_flight", "description": "Book a flight",
        "parameters": {"type": "object", "properties": {
            "origin": {"type": "string"}, "destination": {"type": "string"},
            "date": {"type": "string"}, "passengers": {"type": "integer"}
        }, "required": ["origin", "destination", "date"]}}},
    {"type": "function", "function": {
        "name": "translate_text", "description": "Translate text to another language",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}, "target_language": {"type": "string"}
        }, "required": ["text", "target_language"]}}},
]

SUITE_CONFIGS = [
    ("BF16",         "./output_grpo/merged", []),
    ("FP8_dynamic",  "./output_grpo/merged", ["--quantization", "fp8"]),
    ("W4A16",        "./output_grpo/w4a16",  ["--quantization", "compressed-tensors"]),
]


def make_prompts_file():
    """Write benchmark prompts to a temporary JSONL file."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for p in PROMPTS:
        f.write(json.dumps({"prompt": p}) + "\n")
    f.close()
    return f.name


def run_vllm_bench(port, concurrency, num_prompts, result_dir, label,
                   prompts_path, model=None):
    """Run vllm bench serve for one concurrency level."""
    extra = json.dumps({
        "tools": TOOLS,
        "tool_choice": "auto",
        "chat_template_kwargs": {"enable_thinking": False},
    })

    result_file = f"{label}_c{concurrency}.json"
    cmd = [
        sys.executable, "-m", "vllm", "bench", "serve",
        "--backend", "openai-chat",
        "--port", str(port),
        "--endpoint", "/v1/chat/completions",
        "--dataset-name", "custom",
        "--dataset-path", prompts_path,
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--request-rate", "inf",
        "--extra-body", extra,
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "25,50,75,90,95,99",
        "--custom-output-len", "256",
        "--save-result",
        "--result-dir", result_dir,
        "--result-filename", result_file,
        "--label", f"{label}_c{concurrency}",
        "--num-warmups", "5",
        "--seed", "42",
    ]
    if model:
        cmd += ["--model", model]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    output = result.stdout + result.stderr
    for line in output.splitlines():
        if any(k in line.lower() for k in ["throughput", "ttft", "tpot", "e2el"]):
            print(f"    {line.strip()}")

    if result.returncode != 0:
        err = result.stderr[-300:] if result.stderr else "unknown error"
        print(f"    FAILED: {err}")
        return None

    result_path = os.path.join(result_dir, result_file)
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)
    return None


def wait_server(port, timeout=240):
    for _ in range(timeout // 3):
        time.sleep(3)
        try:
            r = urllib.request.urlopen(f"http://localhost:{port}/v1/models")
            data = json.loads(r.read())
            return data["data"][0]["id"]
        except Exception:
            pass
    return None


def kill_port(port):
    os.system(f"fuser -k {port}/tcp 2>/dev/null")
    os.system("pkill -9 -f 'vllm.entrypoints' 2>/dev/null")
    os.system("pkill -9 -f 'vllm' 2>/dev/null")
    time.sleep(5)
    for _ in range(10):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            mem = int(result.stdout.strip().split("\n")[0])
            if mem < 1000:
                return
        time.sleep(3)
    print("  WARNING: GPU memory not fully freed")


def bench_config(port, label, concurrency_levels, num_prompts, result_dir,
                 prompts_path, model=None):
    """Benchmark one model config across all concurrency levels."""
    results = {}
    for c in concurrency_levels:
        print(f"  c={c:>2d} ({num_prompts} prompts)...")
        data = run_vllm_bench(
            port, c, num_prompts, result_dir, label, prompts_path, model,
        )
        results[c] = data
    return results


def run_suite(concurrency_levels, num_prompts, result_dir, port=8100):
    """Start/stop vLLM servers for each config and benchmark."""
    prompts_path = make_prompts_file()
    all_results = {}

    for label, model_path, extra_args in SUITE_CONFIGS:
        if not os.path.exists(model_path):
            print(f"\n  Skipping {label}: {model_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        kill_port(port)
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(port),
            "--dtype", "auto",
            "--trust-remote-code",
            "--max-model-len", "4096",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ] + extra_args

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        model_name = wait_server(port)

        if not model_name:
            print(f"  FAILED to start server")
            proc.kill()
            all_results[label] = {"error": "server failed to start"}
            continue

        print(f"  Server ready: {model_name}")
        all_results[label] = bench_config(
            port, label, concurrency_levels, num_prompts,
            result_dir, prompts_path, model_name,
        )

        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        kill_port(port)

    os.unlink(prompts_path)
    return all_results


def run_single(port, concurrency_levels, num_prompts, result_dir):
    """Benchmark an already-running server."""
    models_url = f"http://localhost:{port}/v1/models"
    data = json.loads(urllib.request.urlopen(models_url).read())
    model_name = data["data"][0]["id"]
    print(f"Model: {model_name}")

    prompts_path = make_prompts_file()
    label = model_name.replace("/", "_")
    results = bench_config(
        port, label, concurrency_levels, num_prompts,
        result_dir, prompts_path, model_name,
    )
    os.unlink(prompts_path)
    return {label: results}


def main():
    parser = argparse.ArgumentParser(
        description="Inference benchmark using vllm bench serve",
    )
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--suite", action="store_true",
                        help="Full suite: BF16/FP8/W4A16 with server management")
    parser.add_argument("--concurrency", default="1,4,8,16,32")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--result-dir", default="results/bench")
    args = parser.parse_args()

    levels = [int(c) for c in args.concurrency.split(",")]
    os.makedirs(args.result_dir, exist_ok=True)

    if args.suite:
        results = run_suite(levels, args.num_prompts, args.result_dir, args.port)
    else:
        results = run_single(
            args.port, levels, args.num_prompts, args.result_dir,
        )

    summary_path = os.path.join(args.result_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.result_dir}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
