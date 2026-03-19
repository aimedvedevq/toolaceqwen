#!/usr/bin/env python3
"""
Reproducible inference benchmark using vllm bench serve.

Generates prompts from the ToolACE held-out split (same data the model was
designed for), then benchmarks each serving configuration across concurrency
levels. All results are saved as JSON for the report notebook.

Configurations benchmarked:
    BF16, FP8 dynamic, W4A16, EAGLE3 fine-tuned

Usage:
    python scripts/bench.py --suite                    # full matrix (recommended)
    python scripts/bench.py --suite --concurrency 1,16,32  # fewer levels
    python scripts/bench.py --port 8100                # benchmark running server
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

SEED = 42
PORT = 8200

# Models: use local paths if available, else HuggingFace
MODEL_PATH = os.environ.get("MODEL_PATH", "./output_grpo/merged")
W4A16_PATH = os.environ.get("W4A16_PATH", "./output_grpo/w4a16")
EAGLE_CKPT = os.environ.get("EAGLE_CKPT", "./output_eagle_ft/checkpoints/0")

# HuggingFace fallbacks
HF_MODEL = "kenkaneki/Qwen3-8B-ToolACE"
HF_W4A16 = "kenkaneki/Qwen3-8B-ToolACE-W4A16"
HF_EAGLE = "kenkaneki/Qwen3-8B-ToolACE-speculator.eagle3"


def resolve(local: str, hf: str) -> str:
    return local if os.path.exists(local) else hf


SUITE_CONFIGS = [
    ("bf16",      lambda: resolve(MODEL_PATH, HF_MODEL), []),
    ("fp8",       lambda: resolve(MODEL_PATH, HF_MODEL), ["--quantization", "fp8"]),
    ("w4a16",     lambda: resolve(W4A16_PATH, HF_W4A16), ["--quantization", "compressed-tensors"]),
    ("eagle3ft",  lambda: resolve(MODEL_PATH, HF_MODEL), [
        "--speculative-config",
        json.dumps({
            "model": resolve(EAGLE_CKPT, HF_EAGLE),
            "num_speculative_tokens": 3,
            "method": "eagle3",
        }),
    ]),
]

TOOLS_JSON = json.dumps({
    "tools": [
        {"type": "function", "function": {
            "name": "get_weather", "description": "Get weather",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string"}}, "required": ["city"]}}},
        {"type": "function", "function": {
            "name": "search_restaurants", "description": "Search restaurants",
            "parameters": {"type": "object", "properties": {
                "cuisine": {"type": "string"}, "location": {"type": "string"}},
                "required": ["cuisine", "location"]}}},
        {"type": "function", "function": {
            "name": "calculate", "description": "Calculate math expression",
            "parameters": {"type": "object", "properties": {
                "expression": {"type": "string"}}, "required": ["expression"]}}},
        {"type": "function", "function": {
            "name": "book_flight", "description": "Book a flight",
            "parameters": {"type": "object", "properties": {
                "origin": {"type": "string"}, "destination": {"type": "string"},
                "date": {"type": "string"}}, "required": ["origin", "destination", "date"]}}},
        {"type": "function", "function": {
            "name": "translate_text", "description": "Translate text",
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string"}, "target_language": {"type": "string"}},
                "required": ["text", "target_language"]}}},
    ],
    "tool_choice": "auto",
    "chat_template_kwargs": {"enable_thinking": False},
})


def prepare_toolace_prompts(path: str, max_prompts: int = 100) -> str:
    """Build benchmark prompts from ToolACE held-out split."""
    grpo_data = Path("./output/grpo_data")
    if grpo_data.exists():
        from datasets import load_from_disk
        ds = load_from_disk(str(grpo_data))
    else:
        from datasets import load_dataset
        ds = load_dataset("Team-ACE/ToolACE", split="train")
        n = len(ds)
        ds = ds.select(range(int(n * 0.7), n))

    prompts = []
    for ex in ds:
        convs = ex.get("conversations", [])
        user_msgs = [t["value"] for t in convs if t.get("from") == "user"]
        if user_msgs:
            prompts.append({"prompt": user_msgs[0]})
            if len(prompts) >= max_prompts:
                break

    with open(path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    print(f"Prepared {len(prompts)} ToolACE prompts → {path}")
    return path


def wait_server(port: int, timeout: int = 300) -> str | None:
    for _ in range(timeout // 3):
        time.sleep(3)
        try:
            r = urllib.request.urlopen(f"http://localhost:{port}/v1/models")
            return json.loads(r.read())["data"][0]["id"]
        except Exception:
            pass
    return None


def kill_port(port: int) -> None:
    os.system(f"fuser -k {port}/tcp 2>/dev/null")
    os.system("pkill -9 -f 'vllm.entrypoints' 2>/dev/null")
    os.system("pkill -9 -f 'VLLM::EngineCore' 2>/dev/null")
    time.sleep(8)
    for _ in range(10):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            if r.returncode == 0 and int(r.stdout.strip().split("\n")[0]) < 1000:
                return
        except Exception:
            pass
        time.sleep(3)


def run_vllm_bench(
    port: int, concurrency: int, num_prompts: int,
    result_dir: str, label: str, prompts_path: str,
) -> dict | None:
    result_file = f"{label}_c{concurrency}.json"
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "openai-chat",
        "--port", str(port),
        "--endpoint", "/v1/chat/completions",
        "--dataset-name", "custom",
        "--dataset-path", prompts_path,
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--request-rate", "inf",
        "--extra-body", TOOLS_JSON,
        "--temperature", "0",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "25,50,75,90,95,99",
        "--custom-output-len", "256",
        "--save-result",
        "--result-dir", result_dir,
        "--result-filename", result_file,
        "--label", f"{label}_c{concurrency}",
        "--num-warmups", "5",
        "--seed", str(SEED),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    for line in (result.stdout + result.stderr).splitlines():
        if any(k in line.lower() for k in ["median e2el", "output token", "median ttft"]):
            print(f"    {line.strip()}")

    if result.returncode != 0:
        print(f"    FAILED: {result.stderr[-200:]}")
        return None

    rp = os.path.join(result_dir, result_file)
    if os.path.exists(rp):
        with open(rp) as f:
            return json.load(f)
    return None


def run_suite(
    concurrency_levels: list[int],
    num_prompts: int,
    result_dir: str,
    prompts_path: str,
    port: int,
) -> dict:
    all_results = {}

    for label, model_fn, extra_args in SUITE_CONFIGS:
        model_path = model_fn() if callable(model_fn) else model_fn
        if not os.path.exists(model_path) and not model_path.startswith("kenkaneki/"):
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
            all_results[label] = {"error": "server failed"}
            continue

        print(f"  Server ready: {model_name}")
        cfg_results = {}
        for c in concurrency_levels:
            print(f"  c={c:>2d}...")
            data = run_vllm_bench(port, c, num_prompts, result_dir, label, prompts_path)
            if data:
                cfg_results[c] = data

        all_results[label] = cfg_results
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        kill_port(port)

    return all_results


def run_single(port: int, concurrency_levels: list[int], num_prompts: int,
               result_dir: str, prompts_path: str) -> dict:
    data = json.loads(urllib.request.urlopen(
        f"http://localhost:{port}/v1/models"
    ).read())
    model_name = data["data"][0]["id"]
    print(f"Model: {model_name}")

    label = model_name.replace("/", "_").replace(".", "_")
    results = {}
    for c in concurrency_levels:
        print(f"  c={c:>2d}...")
        d = run_vllm_bench(port, c, num_prompts, result_dir, label, prompts_path)
        if d:
            results[c] = d
    return {label: results}


def print_summary(all_results: dict) -> None:
    print(f"\n{'='*80}")
    print(f"{'Config':<15s} {'c':>3s} {'TTFT p50':>10s} {'E2EL p50':>10s} {'tok/s':>8s}")
    print(f"{'-'*80}")
    for label, data in all_results.items():
        if isinstance(data, dict) and "error" in data:
            print(f"  {label:<13s} FAILED: {data['error']}")
            continue
        for c in sorted(k for k in data if isinstance(k, int)):
            d = data[c]
            ttft = d.get("median_ttft_ms", d.get("p50_ttft_ms", 0))
            e2el = d.get("median_e2el_ms", d.get("p50_e2el_ms", 0))
            tps = d.get("output_throughput", 0)
            print(f"  {label:<13s} {c:>3d} {ttft:>9.1f}ms {e2el:>9.1f}ms {tps:>7.1f}")
    print(f"{'='*80}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproducible inference benchmark on ToolACE prompts",
    )
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--suite", action="store_true",
                        help="Run full suite: BF16/FP8/W4A16/EAGLE3")
    parser.add_argument("--concurrency", default="1,4,8,16,32")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--result-dir", default="results/bench")
    parser.add_argument("--prompts", default="bench_toolace.jsonl",
                        help="Prompts JSONL (auto-generated from ToolACE if missing)")
    args = parser.parse_args()

    levels = [int(c) for c in args.concurrency.split(",")]
    os.makedirs(args.result_dir, exist_ok=True)

    if not os.path.exists(args.prompts):
        prepare_toolace_prompts(args.prompts, args.num_prompts)

    if args.suite:
        results = run_suite(levels, args.num_prompts, args.result_dir,
                            args.prompts, args.port)
    else:
        results = run_single(args.port, levels, args.num_prompts,
                             args.result_dir, args.prompts)

    print_summary(results)

    summary_path = os.path.join(args.result_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {args.result_dir}/")


if __name__ == "__main__":
    main()
