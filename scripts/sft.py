#!/usr/bin/env python3
"""SFT training script: LoRA fine-tuning on ToolACE with Qwen3."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

IM_START = 151644
IM_END = 151645
ASSISTANT_TOKEN = 77091
IGNORE_INDEX = -100

ROLE_MAP = {"user": "user", "assistant": "assistant", "tool": "tool"}
BFCL_CATEGORIES = "single_turn"
BFCL_MODEL = "Qwen/Qwen3-8B"  # Prompt mode for eval
BFCL_ROOT = Path(__file__).parent.parent / "gorilla" / "berkeley-function-call-leaderboard"


def tokenize_with_assistant_mask(example, tokenizer, max_length):
    """Tokenize and create labels that only supervise assistant turns."""
    messages = []
    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})
    for turn in example["conversations"]:
        role = ROLE_MAP.get(turn["from"], "user")
        messages.append({"role": role, "content": turn["value"]})

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    except Exception:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    encoding = tokenizer(
        text, truncation=True, max_length=max_length, add_special_tokens=False
    )
    input_ids = encoding["input_ids"]

    labels = [IGNORE_INDEX] * len(input_ids)
    in_assistant = False
    i = 0
    while i < len(input_ids):
        if (
            input_ids[i] == IM_START
            and i + 1 < len(input_ids)
            and input_ids[i + 1] == ASSISTANT_TOKEN
        ):
            i += 3
            in_assistant = True
            continue
        if input_ids[i] == IM_END:
            if in_assistant:
                labels[i] = input_ids[i]
            in_assistant = False
            i += 1
            continue
        if in_assistant:
            labels[i] = input_ids[i]
        i += 1

    return {
        "input_ids": input_ids,
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
    }


def collect_scores(model_name: str, result_dir: str = None) -> dict:
    """Read score + result JSONs and collect per-category accuracy + per-sample details."""
    sanitized_name = model_name.replace("/", "_")
    scores = {}

    for subdir in ["non_live", "live", ""]:
        score_dir = BFCL_ROOT / "score" / sanitized_name / subdir
        if not score_dir.exists():
            continue
        for f in score_dir.glob("*_score.json"):
            cat = f.stem.replace("BFCL_v4_", "").replace("BFCL_v3_", "").replace("_score", "")
            with open(f) as fh:
                lines = [json.loads(line) for line in fh]
            summary = lines[0]
            per_sample = lines[1:]

            error_types = {}
            for entry in per_sample:
                if not entry.get("valid", True):
                    etype = entry.get("error_type", "unknown")
                    error_types[etype] = error_types.get(etype, 0) + 1

            scores[cat] = {**summary, "error_types": error_types, "latencies": []}

    # Latencies from result files
    if result_dir:
        res_dir = Path(result_dir) / sanitized_name
    else:
        res_dir = None

    for subdir in ["non_live", "live", ""]:
        for src in [res_dir / subdir if res_dir else None]:
            if src is None or not src.exists():
                continue
            for f in src.glob("*_result.json"):
                cat = f.stem.replace("BFCL_v4_", "").replace("BFCL_v3_", "").replace("_result", "")
                if cat not in scores:
                    continue
                with open(f) as fh:
                    for line in fh:
                        entry = json.loads(line)
                        if "latency" in entry:
                            scores[cat]["latencies"].append(entry["latency"])

    return scores


def run_bfcl_eval(model_name: str, result_dir: str, local_model_path: str = None, tag: str = "") -> dict | None:
    """Run official BFCL generate + evaluate. Returns scores dict."""
    result_dir = str(Path(result_dir).resolve())

    gen_cmd = [
        sys.executable, "-m", "bfcl_eval", "generate",
        "--model", BFCL_MODEL,
        "--test-category", BFCL_CATEGORIES,
        "--backend", "vllm",
        "--num-gpus", "1",
        "--result-dir", result_dir,
        "--allow-overwrite",
    ]
    if local_model_path:
        gen_cmd += ["--local-model-path", local_model_path]

    print(f"\n{'='*60}")
    print(f"BFCL eval {tag}")
    print(f"{'='*60}")
    print(f">>> {' '.join(gen_cmd)}")
    rc = subprocess.run(gen_cmd)
    if rc.returncode != 0:
        print("BFCL generate failed")
        return None

    eval_cmd = [
        sys.executable, "-m", "bfcl_eval", "evaluate",
        "--model", BFCL_MODEL,
        "--test-category", BFCL_CATEGORIES,
        "--result-dir", result_dir,
    ]
    print(f">>> {' '.join(eval_cmd)}")
    rc = subprocess.run(eval_cmd)
    if rc.returncode != 0:
        print("BFCL evaluate failed")
        return None

    return collect_scores(BFCL_MODEL, result_dir)


def print_scores(scores: dict, tag: str = ""):
    if not scores:
        print(f"No scores for {tag}")
        return
    print(f"\n--- {tag} ---")
    for cat, data in sorted(scores.items()):
        acc = data.get("accuracy", 0)
        correct = data.get("correct_count", "?")
        total = data.get("total_count", "?")
        lats = data.get("latencies", [])
        lat_str = ""
        if lats:
            lat_str = f"  latency: {np.mean(lats):.2f}s ± {np.std(lats):.2f}s"
        print(f"  {cat:30s} {acc*100:.1f}%  ({correct}/{total}){lat_str}")

    # Overall
    all_correct = sum(d.get("correct_count", 0) for d in scores.values())
    all_total = sum(d.get("total_count", 0) for d in scores.values())
    if all_total > 0:
        print(f"  {'OVERALL':30s} {all_correct/all_total*100:.1f}%  ({all_correct}/{all_total})")


def save_report(pre_scores: dict | None, post_scores: dict | None, cfg, output_path: str):
    """Save a full JSON report with all details."""
    categories = sorted(set(
        list(pre_scores.keys() if pre_scores else []) +
        list(post_scores.keys() if post_scores else [])
    ))

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": cfg.model.name,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "categories": {},
        "summary": {},
    }

    for cat in categories:
        entry = {}
        if pre_scores and cat in pre_scores:
            pre = pre_scores[cat]
            lats = pre.get("latencies", [])
            entry["baseline"] = {
                "accuracy": pre["accuracy"],
                "correct": pre.get("correct_count"),
                "total": pre.get("total_count"),
                "latency_mean": float(np.mean(lats)) if lats else None,
                "latency_std": float(np.std(lats)) if lats else None,
                "latency_p50": float(np.median(lats)) if lats else None,
                "latency_p95": float(np.percentile(lats, 95)) if lats else None,
                "error_types": pre.get("error_types", {}),
            }
        if post_scores and cat in post_scores:
            post = post_scores[cat]
            lats = post.get("latencies", [])
            entry["post_training"] = {
                "accuracy": post["accuracy"],
                "correct": post.get("correct_count"),
                "total": post.get("total_count"),
                "latency_mean": float(np.mean(lats)) if lats else None,
                "latency_std": float(np.std(lats)) if lats else None,
                "latency_p50": float(np.median(lats)) if lats else None,
                "latency_p95": float(np.percentile(lats, 95)) if lats else None,
                "error_types": post.get("error_types", {}),
            }
        if "baseline" in entry and "post_training" in entry:
            entry["delta"] = entry["post_training"]["accuracy"] - entry["baseline"]["accuracy"]

        report["categories"][cat] = entry

    # Summary
    if pre_scores:
        pre_total = sum(d.get("total_count", 0) for d in pre_scores.values())
        pre_correct = sum(d.get("correct_count", 0) for d in pre_scores.values())
        report["summary"]["baseline_overall"] = pre_correct / pre_total if pre_total else 0
    if post_scores:
        post_total = sum(d.get("total_count", 0) for d in post_scores.values())
        post_correct = sum(d.get("correct_count", 0) for d in post_scores.values())
        report["summary"]["post_training_overall"] = post_correct / post_total if post_total else 0
    if "baseline_overall" in report["summary"] and "post_training_overall" in report["summary"]:
        report["summary"]["delta_overall"] = (
            report["summary"]["post_training_overall"] - report["summary"]["baseline_overall"]
        )

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {output_path}")

    # Print nice table
    print(f"\n{'='*80}")
    print(f"{'Category':30s} {'Baseline':>10s} {'Post-SFT':>10s} {'Delta':>10s} {'Latency (post)':>16s}")
    print(f"{'-'*80}")
    for cat in categories:
        e = report["categories"][cat]
        pre_acc = e.get("baseline", {}).get("accuracy", 0) * 100
        post_acc = e.get("post_training", {}).get("accuracy", 0) * 100
        delta = e.get("delta", 0) * 100
        lat = e.get("post_training", {})
        lat_str = ""
        if lat.get("latency_mean") is not None:
            lat_str = f"{lat['latency_mean']:.2f}±{lat['latency_std']:.2f}s"
        sign = "+" if delta > 0 else ""
        print(f"  {cat:28s} {pre_acc:9.1f}% {post_acc:9.1f}% {sign}{delta:9.1f}% {lat_str:>16s}")

    s = report["summary"]
    pre_ov = s.get("baseline_overall", 0) * 100
    post_ov = s.get("post_training_overall", 0) * 100
    delta_ov = s.get("delta_overall", 0) * 100
    sign = "+" if delta_ov > 0 else ""
    print(f"{'-'*80}")
    print(f"  {'OVERALL':28s} {pre_ov:9.1f}% {post_ov:9.1f}% {sign}{delta_ov:9.1f}%")
    print(f"{'='*80}")

    # Error breakdown
    for phase_key, phase_name in [("baseline", "Baseline"), ("post_training", "Post-SFT")]:
        errors = {}
        for cat in categories:
            e = report["categories"][cat].get(phase_key, {}).get("error_types", {})
            for etype, cnt in e.items():
                errors[etype] = errors.get(etype, 0) + cnt
        if errors:
            print(f"\n  Error breakdown ({phase_name}):")
            for etype, cnt in sorted(errors.items(), key=lambda x: -x[1]):
                print(f"    {etype:40s} {cnt:5d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cli = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli)
    OmegaConf.resolve(cfg)

    eval_dir = str(Path(cfg.sft.output_dir).resolve() / "bfcl_eval")
    skip_eval = args.skip_eval or args.dry_run
    pre_scores = None
    post_scores = None

    # =========================================================
    # STEP 1: Baseline BFCL eval
    # =========================================================
    if not skip_eval:
        pre_scores = run_bfcl_eval(
            model_name=cfg.model.name,
            result_dir=eval_dir + "/pre",
            tag="BASELINE (pre-training)",
        )
        print_scores(pre_scores, "BASELINE")

    # =========================================================
    # STEP 2: Training
    # =========================================================
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(cfg.data.dataset, split="train")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # Split: 70% SFT, 30% reserved for GRPO
    sft_grpo_split = ds.train_test_split(test_size=0.3, seed=42)
    sft_ds = sft_grpo_split["train"]
    grpo_ds = sft_grpo_split["test"]
    grpo_ds.save_to_disk("./output/grpo_data")
    print(f"SFT: {len(sft_ds)}, GRPO reserved: {len(grpo_ds)}")

    # SFT train/eval split (95/5 of SFT portion)
    split = sft_ds.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"SFT Train: {len(train_ds)}, SFT Eval: {len(eval_ds)}")

    train_ds = train_ds.map(
        lambda ex: tokenize_with_assistant_mask(ex, tokenizer, cfg.sft.max_length),
        remove_columns=train_ds.column_names,
        num_proc=4,
        desc="Tokenizing train",
    )
    train_ds = train_ds.filter(lambda ex: len(ex["input_ids"]) > 0)
    eval_ds = eval_ds.map(
        lambda ex: tokenize_with_assistant_mask(ex, tokenizer, cfg.sft.max_length),
        remove_columns=eval_ds.column_names,
        num_proc=4,
        desc="Tokenizing eval",
    )
    eval_ds = eval_ds.filter(lambda ex: len(ex["input_ids"]) > 0)
    print(f"After filtering: Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        dtype=getattr(torch, cfg.model.torch_dtype),
        attn_implementation=cfg.model.get("attn_implementation"),
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(**OmegaConf.to_container(cfg.lora, resolve=True))

    sft_dict = {k: v for k, v in OmegaConf.to_container(cfg.sft, resolve=True).items() if v is not None}
    if args.dry_run:
        sft_dict["max_steps"] = 3
        sft_dict["logging_steps"] = 1
        sft_dict["save_strategy"] = "no"
        sft_dict["eval_strategy"] = "no"
        sft_dict["load_best_model_at_end"] = False
        sft_dict["report_to"] = "none"

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**sft_dict),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_cfg,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(cfg.sft.output_dir)
    tokenizer.save_pretrained(cfg.sft.output_dir)

    if cfg.merge.enabled:
        print("Merging LoRA weights into base model...")
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(cfg.merge.output_dir)
        tokenizer.save_pretrained(cfg.merge.output_dir)
        print(f"Merged model saved to {cfg.merge.output_dir}")

    del model, trainer
    if "merged" in dir():
        del merged
    torch.cuda.empty_cache()

    # =========================================================
    # STEP 3: Post-training BFCL eval
    # =========================================================
    if not skip_eval:
        post_scores = run_bfcl_eval(
            model_name=cfg.model.name,
            result_dir=eval_dir + "/post",
            local_model_path=str(Path(cfg.merge.output_dir).resolve()),
            tag="POST-TRAINING",
        )
        print_scores(post_scores, "POST-TRAINING")

    # =========================================================
    # STEP 4: Report
    # =========================================================
    if not skip_eval and (pre_scores or post_scores):
        report_path = str(Path(cfg.sft.output_dir) / "eval_report.json")
        save_report(pre_scores, post_scores, cfg, report_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
