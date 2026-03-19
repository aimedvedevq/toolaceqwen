#!/usr/bin/env python3
"""
SFT training: LoRA fine-tuning on ToolACE with Qwen3-8B.

Usage:
    python scripts/sft.py                              # full training
    python scripts/sft.py --dry-run --max-samples=50   # smoke test
    python scripts/sft.py sft.learning_rate=1e-4       # override config
"""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# Qwen3 chat-format token IDs (used for assistant-only loss masking)
IM_START = 151644       # <|im_start|>
IM_END = 151645         # <|im_end|>
ASSISTANT_TOKEN = 77091 # "assistant"
IGNORE_INDEX = -100

ROLE_MAP = {"user": "user", "assistant": "assistant", "tool": "tool"}

_skipped_count = 0


def tokenize_with_assistant_mask(example, tokenizer, max_length):
    """Tokenize and mask labels so loss is computed only on assistant turns."""
    global _skipped_count
    messages = []
    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})
    for turn in example["conversations"]:
        role = ROLE_MAP.get(turn["from"], "user")
        messages.append({"role": role, "content": turn["value"]})

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
    except Exception:
        _skipped_count += 1
        if _skipped_count <= 5:
            log.warning("Failed to apply chat template (skipped %d so far)",
                        _skipped_count)
        return {"input_ids": [], "attention_mask": [], "labels": []}

    encoding = tokenizer(
        text, truncation=True, max_length=max_length, add_special_tokens=False,
    )
    input_ids = encoding["input_ids"]

    labels = [IGNORE_INDEX] * len(input_ids)
    in_assistant = False
    i = 0
    while i < len(input_ids):
        if (input_ids[i] == IM_START and i + 1 < len(input_ids)
                and input_ids[i + 1] == ASSISTANT_TOKEN):
            i += 3  # skip <|im_start|>assistant\n
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


def main():
    seed_everything(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)

    skip_eval = args.skip_eval or args.dry_run

    # ── Data ──
    global _skipped_count
    log.info("Loading %s...", cfg.data.dataset)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(cfg.data.dataset, split="train")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # 70/30 split: SFT / GRPO reserved
    sft_grpo = ds.train_test_split(test_size=0.3, seed=SEED)
    sft_ds = sft_grpo["train"]
    os.makedirs("./output", exist_ok=True)
    sft_grpo["test"].save_to_disk("./output/grpo_data")
    log.info("SFT: %d samples, GRPO reserved: %d samples",
             len(sft_ds), len(sft_grpo["test"]))

    # 95/5 train/eval from SFT portion
    split = sft_ds.train_test_split(test_size=0.05, seed=SEED)
    train_ds, eval_ds = split["train"], split["test"]

    _skipped_count = 0
    train_ds = train_ds.map(
        lambda ex: tokenize_with_assistant_mask(ex, tokenizer, cfg.sft.max_length),
        remove_columns=train_ds.column_names, num_proc=4, desc="Tokenizing train",
    ).filter(lambda ex: len(ex["input_ids"]) > 0)
    if _skipped_count > 0:
        log.warning("Skipped %d examples during train tokenization", _skipped_count)

    _skipped_count = 0
    eval_ds = eval_ds.map(
        lambda ex: tokenize_with_assistant_mask(ex, tokenizer, cfg.sft.max_length),
        remove_columns=eval_ds.column_names, num_proc=4, desc="Tokenizing eval",
    ).filter(lambda ex: len(ex["input_ids"]) > 0)
    if _skipped_count > 0:
        log.warning("Skipped %d examples during eval tokenization", _skipped_count)

    log.info("After filtering: Train=%d, Eval=%d", len(train_ds), len(eval_ds))

    # ── Model ──
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        dtype=getattr(torch, cfg.model.torch_dtype),
        attn_implementation=cfg.model.get("attn_implementation"),
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(**OmegaConf.to_container(cfg.lora, resolve=True))

    # ── Training ──
    sft_dict = {k: v for k, v in OmegaConf.to_container(cfg.sft, resolve=True).items() if v is not None}
    if args.dry_run:
        sft_dict.update({"max_steps": 3, "logging_steps": 1, "save_strategy": "no",
                         "eval_strategy": "no", "load_best_model_at_end": False, "report_to": "none"})

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
        log.info("Merging LoRA weights...")
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(cfg.merge.output_dir)
        tokenizer.save_pretrained(cfg.merge.output_dir)
        log.info("Merged model saved to %s", cfg.merge.output_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
