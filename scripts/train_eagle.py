#!/usr/bin/env python3
"""Train EAGLE-3 speculative decoding head on ToolACE validation data."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MODEL_PATH = "./output_grpo/merged"
OUTPUT_DIR = "./output_eagle"
DATA_FILE = "./output_eagle/train_data.jsonl"


def prepare_data():
    """Convert reserved GRPO data (or val split) to JSONL conversations."""
    from datasets import load_from_disk, load_dataset

    grpo_path = Path("./output/grpo_data")
    if grpo_path.exists():
        from datasets import load_from_disk
        ds = load_from_disk(str(grpo_path))
        print(f"Using GRPO reserved data: {len(ds)} samples")
    else:
        ds = load_dataset("Team-ACE/ToolACE", split="train")
        # Use last 30%
        n = len(ds)
        ds = ds.select(range(int(n * 0.7), n))
        print(f"Using last 30% of ToolACE: {len(ds)} samples")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    role_map = {"user": "user", "assistant": "assistant", "tool": "tool"}
    count = 0
    with open(DATA_FILE, "w") as f:
        for ex in ds:
            messages = []
            if ex.get("system"):
                messages.append({"role": "system", "content": ex["system"]})
            for turn in ex["conversations"]:
                role = role_map.get(turn["from"], "user")
                messages.append({"role": role, "content": turn["value"]})
            f.write(json.dumps({"conversations": messages}) + "\n")
            count += 1

    print(f"Wrote {count} conversations to {DATA_FILE}")
    return DATA_FILE


def run_e2e(model_path: str, data_file: str, output_dir: str, max_samples: int = 3000):
    """Run speculators E2E pipeline: datagen → vocab mapping → train."""

    # Step 1: Data generation (extract hidden states)
    datagen_dir = f"{output_dir}/datagen"
    print(f"\n{'='*60}")
    print("STEP 1: Generating hidden states from target model")
    print(f"{'='*60}")
    cmd = [
        sys.executable, "/home/ray/speculators/scripts/data_generation_offline.py",
        "--target-model-path", model_path,
        "--train-data-path", data_file,
        "--output-dir", datagen_dir,
        "--max-samples", str(max_samples),
        "--seq-length", "2048",
        "--batch-size", "8",
    ]
    print(f">>> {' '.join(cmd)}")
    rc = subprocess.run(cmd)
    if rc.returncode != 0:
        print("Data generation failed")
        return False

    # Step 2: Vocabulary mapping
    print(f"\n{'='*60}")
    print("STEP 2: Building vocabulary mapping")
    print(f"{'='*60}")
    vocab_dir = f"{output_dir}/vocab_mapping"
    cmd = [
        sys.executable, "/home/ray/speculators/scripts/build_vocab_mapping.py",
        "--token-freq-path", f"{datagen_dir}/token_freq.pt",
        "--target-model-path", model_path,
        "--draft-vocab-size", "32000",
        "--output-path", vocab_dir,
    ]
    print(f">>> {' '.join(cmd)}")
    rc = subprocess.run(cmd)
    if rc.returncode != 0:
        print("Vocab mapping failed")
        return False

    # Step 3: Training
    print(f"\n{'='*60}")
    print("STEP 3: Training EAGLE-3 head")
    print(f"{'='*60}")
    ckpt_dir = f"{output_dir}/checkpoints"
    cmd = [
        sys.executable, "/home/ray/speculators/scripts/train.py",
        "--verifier-name-or-path", model_path,
        "--data-path", datagen_dir,
        "--save-path", ckpt_dir,
        "--epochs", "5",
        "--lr", "1e-4",
        "--total-seq-len", "2048",
        "--num-layers", "1",
        "--d2t-path", f"{vocab_dir}/d2t.npy",
        "--t2d-path", f"{vocab_dir}/t2d.npy",
        "--logger", "tensorboard",
        "--log-dir", f"{output_dir}/logs",
        "--run-name", "eagle3-qwen3-8b-toolace",
    ]
    print(f">>> {' '.join(cmd)}")
    rc = subprocess.run(cmd)
    if rc.returncode != 0:
        print("Training failed")
        return False

    print(f"\nEAGLE-3 training complete. Checkpoints in {ckpt_dir}")
    return True


def bench_with_eagle(model_path: str, eagle_path: str, port: int = 8200):
    """Benchmark with and without EAGLE using vLLM."""
    print(f"\n{'='*60}")
    print("Benchmarking with EAGLE-3 speculative decoding")
    print(f"{'='*60}")

    # Find latest checkpoint
    ckpt_dir = Path(eagle_path)
    epochs = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    if not epochs:
        print(f"No checkpoints found in {ckpt_dir}")
        return
    latest = str(epochs[-1])
    print(f"Using checkpoint: {latest}")

    for mode, spec_config in [
        ("baseline", None),
        ("eagle3_k3", f'{{"model": "{latest}", "num_speculative_tokens": 3, "method": "eagle3"}}'),
        ("eagle3_k5", f'{{"model": "{latest}", "num_speculative_tokens": 5, "method": "eagle3"}}'),
    ]:
        print(f"\n--- Mode: {mode} ---")

        # Start vLLM
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(port),
            "--dtype", "bfloat16",
            "--trust-remote-code",
            "--max-model-len", "4096",
        ]
        if spec_config:
            cmd += ["--speculative-config", spec_config]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for ready
        import urllib.request
        ready = False
        for i in range(60):
            time.sleep(3)
            try:
                urllib.request.urlopen(f"http://localhost:{port}/v1/models")
                ready = True
                break
            except Exception:
                pass

        if not ready:
            print(f"  Server failed to start for {mode}")
            proc.kill()
            continue

        # Run bench
        bench_cmd = [
            sys.executable, "bench.py",
            "--model", model_path,
            "--url", f"http://localhost:{port}/v1/chat/completions",
            "--num-prompts", "100",
            "--concurrency", "1,8,32",
            "--output", f"bench_{mode}.json",
        ]
        subprocess.run(bench_cmd)

        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(3)

    # Summary
    print(f"\n{'='*60}")
    print("EAGLE-3 Benchmark Summary")
    print(f"{'='*60}")
    for mode in ["baseline", "eagle3_k3", "eagle3_k5"]:
        bench_file = f"bench_{mode}.json"
        if not Path(bench_file).exists():
            continue
        with open(bench_file) as f:
            data = json.load(f)
        print(f"\n  {mode}:")
        for c, stats in sorted(data["stats"].items(), key=lambda x: int(x[0])):
            if "error" in stats:
                continue
            t = stats["ttft_ms"]
            l = stats["latency_ms"]
            print(f"    c={c:>2s}: TTFT p50={t['p50']:>7.1f}ms  Lat p50={l['p50']:>7.1f}ms  TPS={stats['tokens_per_second']['mean']:>6.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--bench-only", action="store_true")
    args = parser.parse_args()

    if not args.bench_only:
        if not args.skip_train:
            data_file = prepare_data()
            success = run_e2e(args.model, data_file, args.output, args.max_samples)
            if not success:
                print("EAGLE training failed")
                sys.exit(1)

    bench_with_eagle(args.model, f"{args.output}/checkpoints")
    print("\nDone.")


if __name__ == "__main__":
    main()
