from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

VERIFIER_PATH = "./output_grpo/merged"
OFFICIAL_DRAFT = "RedHatAI/Qwen3-8B-speculator.eagle3"
OUTPUT_DIR = "./output_eagle_ft"
DATA_FILE = "./output_eagle_ft/train_data.jsonl"
PORT = 8200

SPECULATORS_SCRIPTS = Path("/home/ray/speculators/scripts")



def prepare_toolace_data(max_samples: int = 3000) -> str:
    """Build fine-tuning data from ToolACE (our target domain)."""
    from datasets import load_dataset

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Loading ToolACE (up to {max_samples} samples)...")

    ds = load_dataset("Team-ACE/ToolACE", split="train")
    n = min(max_samples, len(ds))
    ds = ds.select(range(n))

    role_map = {"user": "user", "assistant": "assistant", "tool": "tool"}
    count = 0
    with open(DATA_FILE, "w") as f:
        for ex in ds:
            msgs = []
            if ex.get("system"):
                msgs.append({"role": "system", "content": ex["system"]})
            for turn in ex["conversations"]:
                role = role_map.get(turn["from"], "user")
                if turn.get("value"):
                    msgs.append({"role": role, "content": turn["value"]})
            if len(msgs) >= 2:
                f.write(json.dumps({"conversations": msgs}) + "\n")
                count += 1

    print(f"Wrote {count} ToolACE conversations → {DATA_FILE}")
    return DATA_FILE



def run_datagen(max_samples: int) -> bool:
    datagen_dir = f"{OUTPUT_DIR}/datagen"
    print("\n=== Step 1: Generating hidden states from verifier ===")
    cmd = [
        sys.executable,
        str(SPECULATORS_SCRIPTS / "data_generation_offline.py"),
        "--target-model-path", VERIFIER_PATH,
        "--train-data-path", DATA_FILE,
        "--output-dir", datagen_dir,
        "--max-samples", str(max_samples),
        "--seq-length", "2048",
        "--batch-size", "8",
    ]
    print(">>> " + " ".join(cmd))
    rc = subprocess.run(cmd)
    if rc.returncode != 0:
        print("Data generation FAILED")
    return rc.returncode == 0



def run_finetune(epochs: int = 3, lr: float = 5e-5) -> bool:
    """
    Fine-tune the official draft head.

    Key flag: --pretrained-speculator-path  loads the official weights
    as the starting point instead of random init.
    """
    datagen_dir = f"{OUTPUT_DIR}/datagen"
    ckpt_dir = f"{OUTPUT_DIR}/checkpoints"
    vocab_dir = f"{OUTPUT_DIR}/vocab_mapping"

    d2t_path = f"{vocab_dir}/d2t.npy"
    t2d_path = f"{vocab_dir}/t2d.npy"
    if not Path(d2t_path).exists():
        print("\n=== Building vocab mapping from datagen token_freq ===")
        Path(vocab_dir).mkdir(parents=True, exist_ok=True)
        vm_cmd = [
            sys.executable,
            str(SPECULATORS_SCRIPTS / "build_vocab_mapping.py"),
            "--token-freq-path", f"{datagen_dir}/token_freq.pt",
            "--target-model-path", VERIFIER_PATH,
            "--output-path", vocab_dir,
        ]
        print(">>> " + " ".join(vm_cmd))
        rc = subprocess.run(vm_cmd)
        if rc.returncode != 0:
            print("Vocab mapping FAILED")
            return False

    print(f"\n=== Step 2: Fine-tuning EAGLE3 draft head (epochs={epochs}) ===")
    cmd = [
        sys.executable,
        str(SPECULATORS_SCRIPTS / "train.py"),
        "--verifier-name-or-path", VERIFIER_PATH,
        "--data-path", datagen_dir,
        "--save-path", ckpt_dir,
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--total-seq-len", "2048",
        "--d2t-path", d2t_path,
        "--t2d-path", t2d_path,
        "--logger", "tensorboard",
        "--log-dir", f"{OUTPUT_DIR}/logs",
        "--run-name", "eagle3-toolace-ft",
        "--pretrained-speculator-path", OFFICIAL_DRAFT,
    ]
    print(">>> " + " ".join(cmd))
    rc = subprocess.run(cmd)
    if rc.returncode != 0:
        # Older speculators may not have --pretrained-speculator-path
        print("Fine-tune with --pretrained-speculator-path failed,")
        print("trying without (will train from official weights via local copy)...")
        return _finetune_manual_init(
            datagen_dir, ckpt_dir, d2t_path, t2d_path, epochs, lr
        )
    return True


def _finetune_manual_init(
    datagen_dir: str, ckpt_dir: str,
    d2t_path: str, t2d_path: str,
    epochs: int, lr: float,
) -> bool:
    """
    Fallback: copy the official checkpoint weights into the first save slot,
    then let speculators trainer continue from there (it auto-resumes).
    """
    import shutil, tempfile
    from huggingface_hub import snapshot_download

    # Download official draft locally so we can inspect & copy
    print("Downloading official draft checkpoint locally...")
    local_official = snapshot_download(OFFICIAL_DRAFT)
    ckpt0 = Path(ckpt_dir) / "0"
    ckpt0.mkdir(parents=True, exist_ok=True)
    for fname in ["config.json", "model.safetensors", "generation_config.json"]:
        src = Path(local_official) / fname
        if src.exists():
            shutil.copy2(src, ckpt0 / fname)
    print(f"Copied official weights to {ckpt0}")

    cmd = [
        sys.executable,
        str(SPECULATORS_SCRIPTS / "train.py"),
        "--verifier-name-or-path", VERIFIER_PATH,
        "--data-path", datagen_dir,
        "--save-path", ckpt_dir,
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--total-seq-len", "2048",
        "--d2t-path", d2t_path,
        "--t2d-path", t2d_path,
        "--logger", "tensorboard",
        "--log-dir", f"{OUTPUT_DIR}/logs",
        "--run-name", "eagle3-toolace-ft",
    ]
    print(">>> " + " ".join(cmd))
    rc = subprocess.run(cmd)
    return rc.returncode == 0



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
    import os
    os.system(f"fuser -k {port}/tcp 2>/dev/null")
    os.system("pkill -9 -f 'vllm.entrypoints' 2>/dev/null")
    time.sleep(8)


def bench_vllm(prompts_file: str) -> None:
    ckpt_dir = Path(OUTPUT_DIR) / "checkpoints"
    epochs = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    if not epochs:
        print("No checkpoints found — skipping benchmark")
        return
    latest = str(epochs[-1])
    print(f"\n=== Step 3: Benchmark via vLLM ===")
    print(f"Fine-tuned draft: {latest}")

    results = {}
    configs = [
        ("no_spec",            None,                          "BF16 baseline (no spec)"),
        ("eagle3_official",    {"model": OFFICIAL_DRAFT, "num_speculative_tokens": 3, "method": "eagle3"}, "Official EAGLE3 (no toolace ft)"),
        ("eagle3_ft_toolace",  {"model": latest,         "num_speculative_tokens": 3, "method": "eagle3"}, "EAGLE3 fine-tuned on ToolACE"),
    ]

    for tag, spec, label in configs:
        kill_port(PORT)
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", VERIFIER_PATH,
            "--port", str(PORT),
            "--dtype", "auto",
            "--trust-remote-code",
            "--max-model-len", "4096",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ]
        if spec:
            cmd += ["--speculative-config", json.dumps(spec)]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        mid = wait_server(PORT)
        if not mid:
            print(f"  {label}: server failed to start")
            proc.kill()
            results[tag] = {"error": "server failed"}
            continue

        print(f"  {label}: server ready ({mid})")
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
            "--result-dir", OUTPUT_DIR,
            "--result-filename", f"bench_{tag}.json",
            "--label", tag,
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

        rp = Path(OUTPUT_DIR) / f"bench_{tag}.json"
        if rp.exists():
            with open(rp) as f:
                results[tag] = json.load(f)

    # Print summary
    print(f"\n{'='*65}")
    print(f"{'Config':<35s} {'TTFT p50':>9s} {'E2EL p50':>9s} {'tok/s':>8s}")
    print(f"{'-'*65}")
    for tag, label in [(c[0], c[2]) for c in configs]:
        d = results.get(tag, {})
        if "error" in d:
            print(f"  {label:<33s} FAILED")
            continue
        print(f"  {label:<33s} {d.get('median_ttft_ms', 0):>8.1f}ms "
              f"{d.get('median_e2el_ms', 0):>8.1f}ms "
              f"{d.get('output_throughput', 0):>7.1f}")
    print(f"{'='*65}")

    with open(f"{OUTPUT_DIR}/bench_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {OUTPUT_DIR}/bench_comparison.json")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--skip-data-prep",  action="store_true")
    parser.add_argument("--skip-datagen",    action="store_true")
    parser.add_argument("--skip-train",      action="store_true")
    parser.add_argument("--bench-only",      action="store_true")
    parser.add_argument("--prompts",
                        default="bench_long_agent_smoke.jsonl",
                        help="Prompts file for the final benchmark")
    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if not args.bench_only:
        if not args.skip_data_prep:
            prepare_toolace_data(args.max_samples)

        if not args.skip_datagen:
            ok = run_datagen(args.max_samples)
            if not ok:
                sys.exit(1)

        if not args.skip_train:
            ok = run_finetune(epochs=args.epochs, lr=args.lr)
            if not ok:
                sys.exit(1)

    bench_vllm(args.prompts)


if __name__ == "__main__":
    main()
