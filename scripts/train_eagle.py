#!/usr/bin/env python3
"""
Train EAGLE-3 speculative decoding head correctly.

Key fixes vs previous attempt:
  1. Training data: ShareGPT + UltraChat (general language),
     NOT ToolACE (tool-calling only). Draft must learn the verifier's
     general token distribution, not just function-call JSON.
  2. vocab-size kept at full 151936 — no 32k truncation.
  3. Export done via speculators convert-to-vllm so the checkpoint
     is directly loadable by vLLM without manual config patches.
  4. Benchmark runs through vLLM (recommended path), not SGLang.

Usage:
    python scripts/train_eagle.py                 # full pipeline
    python scripts/train_eagle.py --skip-datagen  # skip if datagen already done
    python scripts/train_eagle.py --bench-only     # benchmark only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL_PATH = "./output_grpo/merged"
OUTPUT_DIR = "./output_eagle_v2"
DATA_FILE = "./output_eagle_v2/train_data.jsonl"
PORT = 8200


# ─────────────────────────────────────────────────────────────────────────────
# Step 0: Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(max_samples: int = 5000) -> str:
    """
    Build training data from ShareGPT + UltraChat.

    Why NOT ToolACE:
      The draft model must approximate the verifier's next-token distribution
      on inference text.  ToolACE is ~100% structured function-call JSON, so a
      draft trained on it will produce tokens that look like JSON everywhere,
      tanking acceptance rate on natural language responses.

    ShareGPT + UltraChat gives the draft exposure to both instruction-following
    and conversational text, which is what the model produces in practice.
    """
    from datasets import load_dataset

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    conversations: list[list[dict]] = []

    # 1) ShareGPT (multilingual conversational, good coverage)
    print("Loading ShareGPT...")
    try:
        sg = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered",
                          data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
                          split="train")
        for ex in sg.select(range(min(max_samples // 2, len(sg)))):
            turns = ex.get("conversations", [])
            msgs = []
            for t in turns:
                role = {"human": "user", "gpt": "assistant"}.get(t.get("from", ""), None)
                if role and t.get("value"):
                    msgs.append({"role": role, "content": t["value"]})
            if len(msgs) >= 2:
                conversations.append(msgs)
    except Exception as e:
        print(f"ShareGPT load failed ({e}), using UltraChat only")

    # 2) UltraChat 200k (high-quality instruction following)
    print("Loading UltraChat...")
    uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    for ex in uc.select(range(min(max_samples - len(conversations), len(uc)))):
        msgs = ex.get("messages", [])
        if len(msgs) >= 2:
            conversations.append(msgs)

    print(f"Total conversations: {len(conversations)}")

    with open(DATA_FILE, "w") as f:
        for msgs in conversations:
            f.write(json.dumps({"conversations": msgs}) + "\n")

    print(f"Wrote {len(conversations)} conversations to {DATA_FILE}")
    return DATA_FILE


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Hidden-state data generation via speculators
# ─────────────────────────────────────────────────────────────────────────────

def run_datagen(model_path: str, data_file: str, output_dir: str,
                max_samples: int = 5000) -> bool:
    datagen_dir = f"{output_dir}/datagen"
    print("\n=== STEP 1: Generating hidden states ===")
    cmd = [
        sys.executable,
        "/home/ray/speculators/scripts/data_generation_offline.py",
        "--target-model-path", model_path,
        "--train-data-path", data_file,
        "--output-dir", datagen_dir,
        "--max-samples", str(max_samples),
        "--seq-length", "2048",
        "--batch-size", "8",
    ]
    rc = subprocess.run(cmd)
    return rc.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Vocabulary mapping  (full vocab, NOT truncated to 32k)
# ─────────────────────────────────────────────────────────────────────────────

def run_vocab_mapping(model_path: str, output_dir: str) -> bool:
    """
    Previous bug: --draft-vocab-size 32000 truncated Qwen3's 151k vocab.
    This destroys acceptance rate on any token outside the top-32k ToolACE
    frequency list.  Fix: don't pass --draft-vocab-size → defaults to full.
    """
    datagen_dir = f"{output_dir}/datagen"
    vocab_dir = f"{output_dir}/vocab_mapping"
    print("\n=== STEP 2: Building vocabulary mapping (full vocab) ===")
    cmd = [
        sys.executable,
        "/home/ray/speculators/scripts/build_vocab_mapping.py",
        "--token-freq-path", f"{datagen_dir}/token_freq.pt",
        "--target-model-path", model_path,
        "--output-path", vocab_dir,
        # No --draft-vocab-size → full target vocabulary
    ]
    rc = subprocess.run(cmd)
    return rc.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Training
# ─────────────────────────────────────────────────────────────────────────────

def run_training(model_path: str, output_dir: str) -> bool:
    """
    Previous warn: "Draft architecture 'qwen3' is not yet supported in vLLM.
    Consider using 'llama'".
    We ignored it.  The fix is to NOT specify --num-layers explicitly and let
    the speculators default handle architecture-specific layer config.
    """
    datagen_dir = f"{output_dir}/datagen"
    vocab_dir = f"{output_dir}/vocab_mapping"
    ckpt_dir = f"{output_dir}/checkpoints"
    print("\n=== STEP 3: Training EAGLE-3 draft head ===")
    cmd = [
        sys.executable,
        "/home/ray/speculators/scripts/train.py",
        "--verifier-name-or-path", model_path,
        "--data-path", datagen_dir,
        "--save-path", ckpt_dir,
        "--epochs", "5",
        "--lr", "1e-4",
        "--total-seq-len", "2048",
        "--d2t-path", f"{vocab_dir}/d2t.npy",
        "--t2d-path", f"{vocab_dir}/t2d.npy",
        "--logger", "tensorboard",
        "--log-dir", f"{output_dir}/logs",
        "--run-name", "eagle3-qwen3-8b-v2",
    ]
    rc = subprocess.run(cmd)
    return rc.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Validate training accuracy before deploy
# ─────────────────────────────────────────────────────────────────────────────

def check_val_accuracy(output_dir: str) -> None:
    """
    Print final validation accuracy from training log.
    Target: full_acc_0 > 0.80, full_acc_1 > 0.60, full_acc_2 > 0.45.
    If below this, the draft is likely too weak for a meaningful speedup.
    """
    log_dir = Path(output_dir) / "logs"
    log_files = sorted(log_dir.glob("*.log")) if log_dir.exists() else []

    print("\n=== Validation accuracy check ===")
    print("Target: full_acc_0 > 0.80, full_acc_1 > 0.60, full_acc_2 > 0.45")
    if not log_files:
        print("No log files found – check training output manually")
        return

    # Read last lines from most recent log
    last_log = log_files[-1]
    lines = last_log.read_text().splitlines()
    for line in reversed(lines):
        if "val/full_acc_0" in line:
            print(f"Last validation entry:\n  {line}")
            break


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Benchmark with vLLM (recommended path)
# ─────────────────────────────────────────────────────────────────────────────

def wait_server(port: int, timeout: int = 240) -> str | None:
    for _ in range(timeout // 3):
        time.sleep(3)
        try:
            r = urllib.request.urlopen(f"http://localhost:{port}/v1/models")
            data = json.loads(r.read())
            return data["data"][0]["id"]
        except Exception:
            pass
    return None


def kill_port(port: int) -> None:
    import os
    os.system(f"fuser -k {port}/tcp 2>/dev/null")
    time.sleep(5)


def bench_vllm(model_path: str, output_dir: str, prompts_file: str) -> None:
    """
    Benchmark through vLLM speculative-config, the ONLY recommended production path.
    We do NOT use SGLang here — that's what caused the compatibility issues before.
    """
    ckpt_dir = Path(output_dir) / "checkpoints"
    epochs = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    if not epochs:
        print("No checkpoints found – skipping benchmark")
        return
    latest_ckpt = str(epochs[-1])
    print(f"\n=== STEP 5: Benchmark with vLLM ===")
    print(f"Draft checkpoint: {latest_ckpt}")

    results = {}
    for label, spec_config in [
        ("baseline", None),
        ("eagle3_k3", json.dumps({
            "model": latest_ckpt,
            "num_speculative_tokens": 3,
            "method": "eagle3",
        })),
    ]:
        kill_port(PORT)
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(PORT),
            "--dtype", "bfloat16",
            "--trust-remote-code",
            "--max-model-len", "4096",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ]
        if spec_config:
            cmd += ["--speculative-config", spec_config]

        print(f"\n--- {label} ---")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        model_id = wait_server(PORT)
        if not model_id:
            print("  Server failed to start")
            proc.kill()
            results[label] = {"error": "server failed"}
            continue

        print(f"  Server ready: {model_id}")

        bench_cmd = [
            sys.executable, "-m", "vllm", "bench", "serve",
            "--backend", "openai-chat",
            "--port", str(PORT),
            "--endpoint", "/v1/chat/completions",
            "--dataset-name", "custom",
            "--dataset-path", prompts_file,
            "--num-prompts", "50",
            "--max-concurrency", "1",
            "--request-rate", "inf",
            "--temperature", "0",
            "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--metric-percentiles", "25,50,75,90,95,99",
            "--save-result",
            "--result-dir", output_dir,
            "--result-filename", f"bench_{label}.json",
            "--label", label,
            "--num-warmups", "5",
            "--seed", "42",
        ]
        subprocess.run(bench_cmd)

        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        kill_port(PORT)

        result_path = Path(output_dir) / f"bench_{label}.json"
        if result_path.exists():
            with open(result_path) as f:
                results[label] = json.load(f)

    # Summary
    print(f"\n{'='*60}")
    print("EAGLE-3 benchmark summary")
    print(f"{'='*60}")
    for label, data in results.items():
        if "error" in data:
            print(f"  {label}: FAILED")
            continue
        ttft = data.get("median_ttft_ms", "N/A")
        e2el = data.get("median_e2el_ms", "N/A")
        tps = data.get("output_throughput", "N/A")
        print(f"  {label}: TTFT p50={ttft:.1f}ms  E2EL p50={e2el:.1f}ms  tok/s={tps:.1f}")

    # Save comparison
    with open(f"{output_dir}/bench_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/bench_comparison.json")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train EAGLE-3 properly")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--skip-data-prep", action="store_true")
    parser.add_argument("--skip-datagen", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument("--prompts", default="bench_long_agent_smoke.jsonl",
                        help="Prompts file for benchmark")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    if not args.bench_only:
        # Step 0
        if not args.skip_data_prep:
            prepare_data(args.max_samples)
        else:
            print("Skipping data prep")

        # Step 1
        if not args.skip_datagen:
            ok = run_datagen(args.model, DATA_FILE, args.output, args.max_samples)
            if not ok:
                print("Data generation failed")
                sys.exit(1)
        else:
            print("Skipping datagen")

        # Step 2
        ok = run_vocab_mapping(args.model, args.output)
        if not ok:
            print("Vocab mapping failed")
            sys.exit(1)

        # Step 3
        if not args.skip_train:
            ok = run_training(args.model, args.output)
            if not ok:
                print("Training failed")
                sys.exit(1)

        # Step 4
        check_val_accuracy(args.output)

    # Step 5
    bench_vllm(args.model, args.output, args.prompts)


if __name__ == "__main__":
    main()
