#!/usr/bin/env python3
"""
BFCL evaluation wrapper — uses official gorilla eval pipeline.

Usage:
    python scripts/eval.py --model Qwen/Qwen3-8B --test-category single_turn
    python scripts/eval.py --model Qwen/Qwen3-8B --local-model-path ./output/merged
"""

import argparse
import subprocess
import sys


def run(cmd, check=True):
    print(f">>> {' '.join(cmd)}")
    return subprocess.run(cmd, check=check).returncode


def main():
    parser = argparse.ArgumentParser(description="BFCL Evaluation")
    parser.add_argument("--model", required=True, help="BFCL model name")
    parser.add_argument("--local-model-path", default=None)
    parser.add_argument("--test-category", default="simple_python,multiple,parallel,parallel_multiple")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "sglang"])
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    gen_cmd = [
        sys.executable, "-m", "bfcl_eval", "generate",
        "--model", args.model,
        "--test-category", args.test_category,
        "--backend", args.backend,
        "--num-gpus", str(args.num_gpus),
        "--temperature", str(args.temperature),
    ]
    if args.local_model_path:
        gen_cmd += ["--local-model-path", args.local_model_path]
    if args.result_dir:
        gen_cmd += ["--result-dir", args.result_dir]
    if args.overwrite:
        gen_cmd += ["--allow-overwrite"]

    if run(gen_cmd) != 0:
        print("Generation failed")
        sys.exit(1)

    eval_cmd = [
        sys.executable, "-m", "bfcl_eval", "evaluate",
        "--model", args.model,
        "--test-category", args.test_category,
    ]
    if args.result_dir:
        eval_cmd += ["--result-dir", args.result_dir]
    run(eval_cmd, check=False)
    print("Done.")


if __name__ == "__main__":
    main()
