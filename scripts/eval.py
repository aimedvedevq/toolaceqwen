#!/usr/bin/env python3
"""
BFCL evaluation wrapper — uses the official gorilla eval pipeline.

Usage:
  # Full eval (generate + evaluate) on Python subset
  python eval.py --model Qwen/Qwen3-8B --test-category python

  # Eval a merged/local model
  python eval.py --model Qwen/Qwen3-8B --local-model-path ./output/merged

  # Eval with LoRA adapter (vLLM native)
  python eval.py --model Qwen/Qwen3-8B --lora-path ./output

  # Quick smoke test
  python eval.py --model Qwen/Qwen3-8B --local-model-path ./output/merged --dry-run

  # Only run specific categories
  python eval.py --model Qwen/Qwen3-8B --test-category simple_python,multiple,parallel,parallel_multiple

  # Just evaluate existing results (skip generation)
  python eval.py --model Qwen/Qwen3-8B --eval-only

  # Custom result directory
  python eval.py --model Qwen/Qwen3-8B --result-dir /path/to/results
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BFCL_ROOT = Path(__file__).parent

# Python subset categories
PYTHON_CATEGORIES = [
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
]

DEFAULT_RESULT_DIR = BFCL_ROOT / "bfcl_results"


def run_cmd(cmd: list[str], check: bool = True):
    print(f">>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result.returncode


def generate(
    model: str,
    test_category: str,
    result_dir: str,
    backend: str = "vllm",
    num_gpus: int = 1,
    temperature: float = 0.001,
    local_model_path: str = None,
    lora_path: str = None,
    overwrite: bool = False,
):
    cmd = [
        sys.executable, "-m", "bfcl_eval", "generate",
        "--model", model,
        "--test-category", test_category,
        "--backend", backend,
        "--num-gpus", str(num_gpus),
        "--temperature", str(temperature),
        "--result-dir", str(result_dir),
    ]
    if local_model_path:
        cmd += ["--local-model-path", local_model_path]
    if lora_path:
        cmd += ["--enable-lora", "--lora-modules", f"default={lora_path}"]
    if overwrite:
        cmd += ["--allow-overwrite"]
    return run_cmd(cmd)


def evaluate(model: str, test_category: str, result_dir: str, score_dir: str = None):
    cmd = [
        sys.executable, "-m", "bfcl_eval", "evaluate",
        "--model", model,
        "--test-category", test_category,
        "--result-dir", str(result_dir),
    ]
    if score_dir:
        cmd += ["--score-dir", str(score_dir)]
    return run_cmd(cmd, check=False)


def scores():
    cmd = [sys.executable, "-m", "bfcl_eval", "scores"]
    try:
        return run_cmd(cmd, check=False)
    except Exception:
        print("(scores display failed, non-critical)")
        return 1


def collect_scores(score_dir: Path, model: str, categories: list[str]) -> dict:
    """Read score JSONs and aggregate results."""
    results = {}
    model_score_dir = score_dir / model

    if not model_score_dir.exists():
        return results

    for cat in categories:
        score_file = model_score_dir / f"BFCL_v4_{cat}_score.json"
        if not score_file.exists():
            # Try v3 naming
            score_file = model_score_dir / f"BFCL_v3_{cat}_score.json"
        if score_file.exists():
            with open(score_file) as f:
                data = json.load(f)
            results[cat] = data

    return results


def main():
    parser = argparse.ArgumentParser(description="BFCL Evaluation (official)")
    parser.add_argument("--model", required=True, help="Model name (as registered in BFCL)")
    parser.add_argument("--local-model-path", default=None, help="Path to local model weights")
    parser.add_argument("--lora-path", default=None, help="Path to LoRA adapter")
    parser.add_argument("--test-category", default="python",
                        help="Comma-separated categories or group name (default: python)")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "sglang"])
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--score-dir", default=None)
    parser.add_argument("--eval-only", action="store_true", help="Skip generation, only evaluate")
    parser.add_argument("--generate-only", action="store_true", help="Only generate, skip evaluation")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    # Resolve categories
    categories = args.test_category

    print(f"Model: {args.model}")
    if args.local_model_path:
        print(f"Local model path: {args.local_model_path}")
    if args.lora_path:
        print(f"LoRA path: {args.lora_path}")
    print(f"Categories: {categories}")
    print(f"Result dir: {args.result_dir}")
    print()

    if args.dry_run:
        print("DRY RUN — commands that would be executed:")
        if not args.eval_only:
            cmd = [
                "python", "-m", "bfcl_eval", "generate",
                "--model", args.model,
                "--test-category", categories,
                "--backend", args.backend,
                "--result-dir", args.result_dir,
            ]
            if args.local_model_path:
                cmd += ["--local-model-path", args.local_model_path]
            print(f"  {' '.join(cmd)}")
        if not args.generate_only:
            print(f"  python -m bfcl_eval evaluate --model {args.model} --test-category {categories}")
            print(f"  python -m bfcl_eval scores --model {args.model}")
        return

    # Step 1: Generate
    if not args.eval_only:
        print("=" * 60)
        print("STEP 1: Generating model responses")
        print("=" * 60)
        rc = generate(
            model=args.model,
            test_category=categories,
            result_dir=args.result_dir,
            backend=args.backend,
            num_gpus=args.num_gpus,
            temperature=args.temperature,
            local_model_path=args.local_model_path,
            lora_path=args.lora_path,
            overwrite=args.overwrite,
        )
        if rc != 0:
            print(f"Generation failed with code {rc}")
            sys.exit(rc)

    # Step 2: Evaluate
    if not args.generate_only:
        print()
        print("=" * 60)
        print("STEP 2: Evaluating results")
        print("=" * 60)
        evaluate(
            model=args.model,
            test_category=categories,
            result_dir=args.result_dir,
            score_dir=args.score_dir,
        )

        # Step 3: Show scores
        print()
        print("=" * 60)
        print("SCORES")
        print("=" * 60)
        scores()

    print("\nDone.")


if __name__ == "__main__":
    main()
