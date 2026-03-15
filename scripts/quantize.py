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
        print(f"Quantizing {args.model} → FP8 (W8A8 dynamic)")
        oneshot(
            model=args.model,
            output_dir=output,
            recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC"),
        )
    elif args.method == "w4a16":
        print(f"Quantizing {args.model} → W4A16 (calibrated, {args.calibration_samples} samples)")
        oneshot(
            model=args.model,
            output_dir=output,
            dataset="ultrachat_200k",
            recipe=QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
            max_seq_length=2048,
            num_calibration_samples=args.calibration_samples,
            splits={"calibration": f"train_sft[:{args.calibration_samples}]"},
        )

    print(f"Done: {output}")


if __name__ == "__main__":
    main()
