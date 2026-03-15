#!/usr/bin/env python3
"""Full benchmark: BFCL quality + throughput at various concurrency for BF16/FP8/W4A16."""

import json, time, subprocess, sys, os
import numpy as np
import sglang
from datasets import load_from_disk
from transformers import AutoTokenizer
from pathlib import Path

ROLE_MAP = {"user": "user", "assistant": "assistant", "tool": "tool"}
BFCL_CATEGORIES = "simple_python,multiple,parallel,parallel_multiple"


def run_bfcl(model_path, result_dir, label):
    """Run BFCL eval via eval.py."""
    print(f"\n--- BFCL: {label} ---")
    cmd = [
        sys.executable, "eval.py",
        "--model", "Qwen/Qwen3-8B",
        "--local-model-path", model_path,
        "--test-category", BFCL_CATEGORIES,
        "--result-dir", result_dir,
        "--overwrite",
    ]
    subprocess.run(cmd, capture_output=True)

    # Read scores
    bfcl_root = Path("..") / "gorilla" / "berkeley-function-call-leaderboard"
    scores = {}
    for subdir in ["non_live", "live", ""]:
        sd = bfcl_root / "score" / "Qwen_Qwen3-8B" / subdir
        if not sd.exists(): continue
        for f in sd.glob("*_score.json"):
            cat = f.stem.replace("BFCL_v4_", "").replace("_score", "")
            with open(f) as fh:
                scores[cat] = json.loads(fh.readline())
    return scores


def bench_throughput(model_path, prompts, label, concurrency_levels=[1, 8, 32], extra_kwargs={}):
    """Benchmark throughput at different concurrency levels."""
    print(f"\n--- Throughput: {label} ---")

    kwargs = {"model_path": model_path, "mem_fraction_static": 0.7}
    kwargs.update(extra_kwargs)
    engine = sglang.Engine(**kwargs)

    results = {}
    for c in concurrency_levels:
        batch = prompts[:c]
        # Warmup
        engine.generate(prompt=batch[:min(3, c)], sampling_params={"max_new_tokens": 64, "temperature": 0})

        # Timed run
        t0 = time.perf_counter()
        out = engine.generate(prompt=batch, sampling_params={"max_new_tokens": 512, "temperature": 0})
        elapsed = time.perf_counter() - t0
        total_tok = sum(r["meta_info"]["completion_tokens"] for r in out)
        tps = total_tok / elapsed
        rps = len(batch) / elapsed
        print(f"  c={c:>3d}: {tps:>6.0f} tok/s  {rps:>5.1f} req/s  ({elapsed:.1f}s, avg {total_tok/len(batch):.0f} tok)")
        results[c] = {"tps": tps, "rps": rps, "elapsed": elapsed, "avg_tok": total_tok / len(batch)}

    engine.shutdown()
    time.sleep(3)
    return results


def main():
    tok = AutoTokenizer.from_pretrained("./output_grpo/merged", trust_remote_code=True)
    ds = load_from_disk("./output/grpo_data")

    # Build prompts for throughput bench
    prompts = []
    for ex in ds:
        msgs = []
        if ex.get("system"):
            msgs.append({"role": "system", "content": ex["system"]})
        last_idx = -1
        for i, turn in enumerate(ex["conversations"]):
            if turn["from"] == "assistant": last_idx = i
        if last_idx < 1: continue
        for turn in ex["conversations"][:last_idx]:
            msgs.append({"role": ROLE_MAP.get(turn["from"], "user"), "content": turn["value"]})
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if len(tok.encode(text)) > 3000: continue
        prompts.append(text)
        if len(prompts) >= 64: break
    print(f"{len(prompts)} prompts for throughput bench")

    configs = {
        "BF16": {"path": "./output_grpo/merged", "kwargs": {}},
        "FP8": {"path": "./output_grpo/fp8", "kwargs": {}},
        "W4A16": {"path": "./output_grpo/w4a16", "kwargs": {"quantization": "compressed-tensors"}},
    }

    all_results = {}

    for name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"  {name} (model size: {sum(f.stat().st_size for f in Path(cfg['path']).glob('*.safetensors'))/1e9:.1f} GB)")
        print(f"{'='*60}")

        # BFCL quality
        bfcl = run_bfcl(cfg["path"], f"./output/bfcl_eval/quant_{name.lower()}", name)

        # Throughput
        tp = bench_throughput(cfg["path"], prompts, name, [1, 8, 32, 64], cfg["kwargs"])

        all_results[name] = {"bfcl": bfcl, "throughput": tp}

    # ============ Summary ============
    print(f"\n{'='*90}")
    print("BFCL Quality (Accuracy %)")
    print(f"{'='*90}")
    cats = sorted(set(c for r in all_results.values() for c in r["bfcl"].keys()))
    header = f"{'Category':25s}" + "".join(f"{n:>12s}" for n in configs.keys())
    print(header)
    print("-" * len(header))
    for cat in cats:
        row = f"  {cat:23s}"
        for name in configs.keys():
            acc = all_results[name]["bfcl"].get(cat, {}).get("accuracy", 0)
            row += f"{acc*100:>11.1f}%"
        print(row)
    # Overall
    row = f"  {'OVERALL':23s}"
    for name in configs.keys():
        bfcl = all_results[name]["bfcl"]
        total_c = sum(d.get("correct_count", 0) for d in bfcl.values())
        total_n = sum(d.get("total_count", 0) for d in bfcl.values())
        row += f"{total_c/total_n*100 if total_n else 0:>11.1f}%"
    print(row)

    print(f"\n{'='*90}")
    print("Throughput (tok/s) by Concurrency")
    print(f"{'='*90}")
    header = f"{'Concurrency':>12s}" + "".join(f"{n:>15s}" for n in configs.keys())
    print(header)
    print("-" * len(header))
    for c in [1, 8, 32, 64]:
        row = f"{c:>12d}"
        for name in configs.keys():
            tps = all_results[name]["throughput"].get(c, {}).get("tps", 0)
            row += f"{tps:>14.0f}"
        print(row)

    print(f"\n{'='*90}")
    print("Memory & Model Size")
    print(f"{'='*90}")
    for name, cfg in configs.items():
        size = sum(f.stat().st_size for f in Path(cfg["path"]).glob("*.safetensors")) / 1e9
        print(f"  {name:10s}: {size:.1f} GB")
    print(f"{'='*90}")

    with open("bench_quant_full.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved to bench_quant_full.json")


if __name__ == "__main__":
    main()
