#!/usr/bin/env python3
"""
BFCL evaluation for tool-calling models.

Manages vLLM server lifecycle and runs the official BFCL eval pipeline
(generate + evaluate) for one or more model configurations.

Usage:
    python scripts/eval.py --configs grpo                  # single config
    python scripts/eval.py --configs baseline sft grpo     # multiple
    python scripts/eval.py --all                           # all configs
    python scripts/eval.py --all --test-category simple_python  # Python subset only
    # NOTE: EAGLE3 is lossless — same accuracy as BF16, use bench.py for latency
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

BFCL_ROOT = os.path.expanduser("~/gorilla/berkeley-function-call-leaderboard")
PORT = 8100
PYTHON = sys.executable

ALL_CONFIGS = {
    "baseline": {
        "model_name": "Qwen/Qwen3-8B",
        "local_model_path": None,
        "vllm_args": [],
        "label": "Baseline Qwen3-8B",
    },
    "sft": {
        "model_name": "Qwen/Qwen3-8B",
        "local_model_path": "./output/merged",
        "vllm_args": [],
        "label": "Post-SFT",
    },
    "grpo": {
        "model_name": "Qwen/Qwen3-8B",
        "local_model_path": "./output_grpo/merged",
        "vllm_args": [],
        "label": "Post-GRPO (BF16)",
    },
    "fp8": {
        "model_name": "Qwen/Qwen3-8B",
        "local_model_path": "./output_grpo/merged",
        "vllm_args": ["--quantization", "fp8"],
        "label": "Post-GRPO (FP8)",
    },
    "w4a16": {
        "model_name": "Qwen/Qwen3-8B",
        "local_model_path": "./output_grpo/w4a16",
        "vllm_args": ["--quantization", "compressed-tensors"],
        "label": "Post-GRPO (W4A16)",
    },
    # NOTE: EAGLE3 is lossless speculative decoding — output is identical to
    # the target model by construction. No separate BFCL eval needed.
    # Use scripts/bench.py to measure its latency advantage instead.
}


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


def kill_server(port):
    os.system(f"fuser -k {port}/tcp 2>/dev/null")
    os.system("pkill -9 -f 'vllm.entrypoints' 2>/dev/null")
    time.sleep(5)


def run_bfcl(config_name, cfg, test_category, result_base_dir):
    """Start vLLM, run BFCL generate + evaluate, return results."""
    kill_server(PORT)

    serve_path = cfg["local_model_path"] or cfg["model_name"]
    cmd = [
        PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--model", serve_path,
        "--port", str(PORT),
        "--dtype", "auto",
        "--trust-remote-code",
        "--max-model-len", "4096",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ] + cfg["vllm_args"]

    print(f"\n{'='*60}")
    print(f"  BFCL: {cfg['label']}")
    print(f"  Model: {serve_path}")
    print(f"{'='*60}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    model = wait_server(PORT)

    if not model:
        print(f"  FAILED: server didn't start")
        proc.kill()
        return None

    print(f"  Server ready: {model}")
    result_dir = os.path.join(result_base_dir, f"bfcl_{config_name}")
    os.makedirs(result_dir, exist_ok=True)

    # Generate
    gen_cmd = [
        PYTHON, "-m", "bfcl_eval", "generate",
        "--model", cfg["model_name"],
        "--test-category", test_category,
        "--backend", "vllm",
        "--num-gpus", "1",
        "--temperature", "0.001",
        "--allow-overwrite",
    ]
    if cfg["local_model_path"]:
        gen_cmd += ["--local-model-path", cfg["local_model_path"]]
    gen_cmd += ["--result-dir", result_dir]

    print(f"  Generating...")
    gen = subprocess.run(
        gen_cmd, cwd=BFCL_ROOT,
        capture_output=True, text=True, timeout=1800,
    )
    if gen.returncode != 0:
        print(f"  Generate failed:\n{gen.stderr[-500:]}")
        proc.terminate()
        kill_server(PORT)
        return None

    # Evaluate
    eval_cmd = [
        PYTHON, "-m", "bfcl_eval", "evaluate",
        "--model", cfg["model_name"],
        "--test-category", test_category,
        "--result-dir", result_dir,
    ]

    print(f"  Evaluating...")
    ev = subprocess.run(
        eval_cmd, cwd=BFCL_ROOT,
        capture_output=True, text=True, timeout=600,
    )

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    kill_server(PORT)

    output = ev.stdout or ""
    print(output[-1500:])
    return output


def main():
    parser = argparse.ArgumentParser(description="BFCL Evaluation")
    parser.add_argument(
        "--configs", nargs="*",
        help=f"Configs to eval: {', '.join(ALL_CONFIGS)}",
    )
    parser.add_argument("--all", action="store_true", help="Run all configs")
    parser.add_argument("--test-category", default="all",
                        help="BFCL test categories (default: all)")
    parser.add_argument("--result-dir", default="results",
                        help="Base directory for results")
    args = parser.parse_args()

    if args.all:
        configs = list(ALL_CONFIGS.keys())
    elif args.configs:
        configs = args.configs
    else:
        parser.error("Specify --configs or --all")

    results = {}
    for name in configs:
        if name not in ALL_CONFIGS:
            print(f"Unknown config: {name}. Available: {list(ALL_CONFIGS)}")
            continue
        output = run_bfcl(name, ALL_CONFIGS[name], args.test_category,
                          args.result_dir)
        results[name] = output

    summary_path = os.path.join(args.result_dir, "bfcl_summary.txt")
    with open(summary_path, "w") as f:
        for name, output in results.items():
            f.write(f"\n{'='*60}\n{name}\n{'='*60}\n")
            f.write(output or "FAILED\n")

    print(f"\n{'='*60}")
    print(f"  All evaluations complete. Summary: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
