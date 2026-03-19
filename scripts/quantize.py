#!/usr/bin/env python3
"""
Quantize model to FP8 or W4A16.

Usage:
    python scripts/quantize.py --model ./output_grpo/merged --method fp8
    python scripts/quantize.py --model ./output_grpo/merged --method w4a16
"""

import argparse
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True, choices=["fp8", "w4a16"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--calibration-samples", type=int, default=256)
    args = parser.parse_args()

    output = args.output or f"{args.model}_{args.method}"

    if args.method == "fp8":
        print(f"Quantizing {args.model} -> FP8 (W8A8 dynamic)")
        oneshot(
            model=args.model,
            output_dir=output,
            recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC"),
        )
    elif args.method == "w4a16":
        print(f"Quantizing {args.model} -> W4A16 (calibrated on ToolACE, {args.calibration_samples} samples)")
        # Use domain-matched calibration data (ToolACE) instead of generic ultrachat
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        # Build calibration data from ToolACE
        ds = load_dataset("Team-ACE/ToolACE", split="train")
        cal_texts = []
        for ex in ds.select(range(min(args.calibration_samples * 2, len(ds)))):
            messages = []
            if ex.get("system"):
                messages.append({"role": "system", "content": ex["system"]})
            role_map = {"user": "user", "assistant": "assistant", "tool": "tool"}
            for turn in ex["conversations"]:
                role = role_map.get(turn["from"], "user")
                messages.append({"role": role, "content": turn["value"]})
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                    enable_thinking=False
                )
                cal_texts.append(text)
            except Exception:
                continue
            if len(cal_texts) >= args.calibration_samples:
                break

        print(f"Built {len(cal_texts)} calibration samples from ToolACE")

        from datasets import Dataset
        cal_ds = Dataset.from_dict({"text": cal_texts})

        oneshot(
            model=args.model,
            output_dir=output,
            dataset=cal_ds,
            recipe=QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
            max_seq_length=2048,
            num_calibration_samples=len(cal_texts),
        )

    print(f"Done: {output}")


if __name__ == "__main__":
    main()
