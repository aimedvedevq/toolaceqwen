#!/usr/bin/env python3
"""
Launch the recommended inference server on a GPU VM.

Default recommendation:
    - engine: vLLM
    - model: post-GRPO merged model
    - quantization: FP8 dynamic
    - tool calling: enabled with Hermes parser

Examples:
    python scripts/run_inference_vm.py
    python scripts/run_inference_vm.py --quantization bf16
    python scripts/run_inference_vm.py --engine sglang --quantization bf16
    python scripts/run_inference_vm.py --enable-ngram-spec
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MERGED_MODEL = ROOT / "output_grpo" / "merged"
W4A16_MODEL = ROOT / "output_grpo" / "w4a16"

HF_MODEL = "kenkaneki/Qwen3-8B-ToolACE"
HF_W4A16 = "kenkaneki/Qwen3-8B-ToolACE-W4A16"


def _resolve(local: Path, hf: str) -> str:
    return str(local) if local.exists() else hf


def resolve_model_and_flags(engine: str, quantization: str) -> tuple[str, list[str]]:
    model = _resolve(MERGED_MODEL, HF_MODEL)
    if quantization == "bf16":
        return model, []
    if quantization == "fp8":
        return model, ["--quantization", "fp8"]
    if quantization == "w4a16":
        return _resolve(W4A16_MODEL, HF_W4A16), ["--quantization", "compressed-tensors"]
    raise ValueError(f"Unsupported quantization: {quantization}")


def build_vllm_command(args: argparse.Namespace) -> list[str]:
    model_path, extra_flags = resolve_model_and_flags("vllm", args.quantization)
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--served-model-name",
        str(model_path),
        "--port",
        str(args.port),
        "--host",
        args.host,
        "--dtype",
        "auto",
        "--trust-remote-code",
        "--max-model-len",
        str(args.max_model_len),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        args.tool_call_parser,
    ] + extra_flags

    if args.enable_ngram_spec:
        spec_config = (
            '{"method":"ngram","num_speculative_tokens":5,'
            '"prompt_lookup_max":5,"prompt_lookup_min":2}'
        )
        cmd += ["--speculative-config", spec_config]

    return cmd


def build_sglang_command(args: argparse.Namespace) -> list[str]:
    model_path, extra_flags = resolve_model_and_flags("sglang", args.quantization)
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--served-model-name",
        str(model_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--context-length",
        str(args.max_model_len),
        "--tool-call-parser",
        args.tool_call_parser,
    ] + extra_flags

    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference server on VM")
    parser.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument("--quantization", choices=["bf16", "fp8", "w4a16"], default="fp8")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tool-call-parser", default="hermes")
    parser.add_argument("--enable-ngram-spec", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if args.engine == "vllm":
        cmd = build_vllm_command(args)
    else:
        cmd = build_sglang_command(args)

    print("Launching inference server:")
    print(" ".join(cmd))
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
