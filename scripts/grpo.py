#!/usr/bin/env python3
"""GRPO training on ToolACE with Unsloth + custom reward functions."""

import argparse
import ast
import re
import json
from pathlib import Path

from datasets import load_dataset
from omegaconf import OmegaConf
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from data_utils import (
    convert_toolace_example, extract_tools_from_system,
    toolace_to_openai_tools, is_tool_call, parse_bracket_calls,
    ROLE_MAP,
)


# ─────────────────────────────────────────────────────────────
# Dataset: ToolACE → single-step prompt/completion pairs
# ─────────────────────────────────────────────────────────────

def build_grpo_dataset(ds, tokenizer):
    """Convert ToolACE to single-step: prompt with tools, ground truth = tool call.
    Only keeps first assistant tool call per example."""
    samples = []
    for ex in ds:
        tools = extract_tools_from_system(ex.get("system", ""))
        if tools is None:
            continue
        openai_tools = toolace_to_openai_tools(tools)

        messages = []
        for turn in ex["conversations"]:
            role = ROLE_MAP.get(turn["from"], "user")
            content = turn["value"]

            if role == "assistant" and is_tool_call(content):
                # Only keep if last message in prompt is from user
                if messages and messages[-1]["role"] == "user":
                    # Build prompt with tools via chat template
                    prompt_text = tokenizer.apply_chat_template(
                        messages, tools=openai_tools,
                        tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    # Check length
                    if len(tokenizer.encode(prompt_text)) > 3500:
                        continue

                    samples.append({
                        "prompt": prompt_text,
                        "ground_truth": content,
                    })
                # Add to history for multi-turn
                messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                messages.append({"role": role, "content": content})
            else:
                messages.append({"role": role, "content": content})

    return samples


# ─────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────

def extract_tool_calls(text: str) -> list[dict]:
    """Parse tool calls from <tool_call> JSON or [FuncName(args)] format."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    calls = []

    # Try <tool_call> JSON format first
    for match in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL):
        try:
            obj = json.loads(match.group(1))
            name = obj.get("name", "")
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            calls.append({"name": name, "args": args})
        except (json.JSONDecodeError, TypeError):
            pass

    if calls:
        return calls

    # Fallback: bracket format [FuncName(args)]
    for match in re.finditer(r'\[?([A-Za-z_][\w\s]*?\w)\(([^)]*)\)\]?', text):
        func_name = match.group(1).strip()
        args_str = match.group(2).strip()
        kwargs = {}
        if args_str:
            try:
                fake_call = f"f({args_str})"
                tree = ast.parse(fake_call, mode='eval')
                if isinstance(tree.body, ast.Call):
                    for kw in tree.body.keywords:
                        try:
                            kwargs[kw.arg] = ast.literal_eval(kw.value)
                        except (ValueError, TypeError):
                            kwargs[kw.arg] = ast.dump(kw.value)
            except SyntaxError:
                for pair in args_str.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        kwargs[k.strip()] = v.strip().strip("\"'")
        calls.append({"name": func_name, "args": kwargs})

    return calls


# ─────────────────────────────────────────────────────────────
# Reward functions
# ─────────────────────────────────────────────────────────────

def get_completion_text(completion) -> str:
    """Extract text from completion regardless of format."""
    if isinstance(completion, list):
        return completion[0].get("content", "") if completion else ""
    return str(completion)


def format_reward_fn(completions, trainer_state=None, **kwargs) -> list[float]:
    """Reward for correct <tool_call> JSON format.
    Decays from 1.0 to 0 by 50% of training."""
    max_steps = trainer_state.max_steps if trainer_state else 1
    progress = (trainer_state.global_step / max_steps) if trainer_state else 0
    decay = max(0.0, 1.0 - progress / 0.5)

    rewards = []
    for completion in completions:
        text = get_completion_text(completion)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        score = 0.0
        # Check <tool_call>{"name": ..., "arguments": ...}</tool_call> format
        if re.search(r'<tool_call>\s*\{.*?"name".*?"arguments".*?\}\s*</tool_call>', text, re.DOTALL):
            score = 1.0
        # Partial: has tool_call tags but malformed JSON
        elif '<tool_call>' in text and '</tool_call>' in text:
            score = 0.5
        # Also accept bracket format [func(args)] as partial
        elif re.search(r'\[[\w\s]+\(.*?\)\]', text):
            score = 0.3

        rewards.append(score * decay)

    return rewards


def tool_name_reward_fn(completions, ground_truth, **kwargs) -> list[float]:
    """Reward for calling the correct tool name."""
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        text = get_completion_text(completion)

        pred_calls = extract_tool_calls(text)
        gt_calls = extract_tool_calls(gt)

        if not gt_calls:
            rewards.append(0.0)
            continue

        pred_names = set(c["name"] for c in pred_calls)
        gt_names = set(c["name"] for c in gt_calls)

        if not gt_names:
            rewards.append(0.0)
        elif pred_names == gt_names:
            rewards.append(1.0)
        else:
            # Partial credit: intersection / union
            overlap = len(pred_names & gt_names)
            total = len(gt_names)
            rewards.append(overlap / total)

    return rewards


def tool_args_reward_fn(completions, ground_truth, **kwargs) -> list[float]:
    """Reward for matching tool arguments."""
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        text = get_completion_text(completion)

        pred_calls = extract_tool_calls(text)
        gt_calls = extract_tool_calls(gt)

        if not gt_calls:
            rewards.append(0.0)
            continue

        total_score = 0.0
        matched = 0

        for gt_call in gt_calls:
            # Find matching pred call by name
            pred_match = None
            for pc in pred_calls:
                if pc["name"] == gt_call["name"]:
                    pred_match = pc
                    break

            if pred_match is None:
                continue

            matched += 1
            gt_args = gt_call["args"]
            pred_args = pred_match["args"]

            if not gt_args:
                total_score += 1.0
                continue

            # Score: fraction of args that match
            arg_scores = []
            for key, gt_val in gt_args.items():
                if key in pred_args:
                    pred_val = pred_args[key]
                    if str(pred_val).strip() == str(gt_val).strip():
                        arg_scores.append(1.0)
                    else:
                        # Partial: at least the arg name was right
                        arg_scores.append(0.3)
                else:
                    arg_scores.append(0.0)

            total_score += sum(arg_scores) / len(arg_scores) if arg_scores else 0.0

        rewards.append(total_score / len(gt_calls) if gt_calls else 0.0)

    return rewards


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/grpo.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cli = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli)
    OmegaConf.resolve(cfg)

    # --- Model via Unsloth ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        fast_inference=True,
        max_lora_rank=cfg.model.lora_rank,
        gpu_memory_utilization=0.6,
    )

    # Patch chat template to disable thinking by default
    if tokenizer.chat_template and "enable_thinking" in tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "enable_thinking is true",
            "enable_thinking is false",
        )

    target_modules = cfg.lora.target_modules
    if target_modules == "all-linear":
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=target_modules,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # --- Dataset ---
    # Try saved GRPO split first, fallback to full dataset
    grpo_data_path = Path("./output/grpo_data")
    if grpo_data_path.exists():
        from datasets import load_from_disk
        ds = load_from_disk(str(grpo_data_path))
        print(f"Loaded GRPO split from {grpo_data_path}: {len(ds)} examples")
    else:
        ds = load_dataset(cfg.data.dataset, split="train")
        print(f"Using full dataset: {len(ds)} examples")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    print(f"Building single-step GRPO dataset...")
    samples = build_grpo_dataset(ds, tokenizer)
    print(f"Got {len(samples)} prompt/completion pairs")

    from datasets import Dataset
    train_ds = Dataset.from_list(samples)

    # --- GRPOConfig ---
    grpo_dict = {k: v for k, v in OmegaConf.to_container(cfg.grpo, resolve=True).items() if v is not None}

    if args.dry_run:
        grpo_dict["max_steps"] = 3
        grpo_dict["logging_steps"] = 1
        grpo_dict["save_strategy"] = "no"
        grpo_dict["report_to"] = "none"
        grpo_dict["num_generations"] = 2

    training_args = GRPOConfig(**grpo_dict)

    # --- Reward functions with weights ---
    rw = cfg.reward
    reward_funcs = [format_reward_fn, tool_name_reward_fn, tool_args_reward_fn]
    reward_weights = [rw.format_weight, rw.tool_name_weight, rw.tool_args_weight]

    # --- Trainer ---
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
    )

    trainer.train()

    # --- Save ---
    trainer.save_model(cfg.grpo.output_dir)
    tokenizer.save_pretrained(cfg.grpo.output_dir)

    if cfg.merge.enabled:
        print("Merging LoRA weights...")
        model.save_pretrained_merged(
            cfg.merge.output_dir, tokenizer, save_method="merged_16bit"
        )
        print(f"Merged model saved to {cfg.merge.output_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
