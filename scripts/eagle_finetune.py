#!/usr/bin/env python3
"""Fine-tune EAGLE3 — load pretrained weights, then continue training."""

import transformers.models.qwen3.configuration_qwen3 as qwen3_cfg
_orig = qwen3_cfg.Qwen3Config.__init__
def _patched(self, *a, **kw):
    _orig(self, *a, **kw)
    if getattr(self, '_attn_implementation', None) is None:
        self._attn_implementation = 'flex_attention'
        self._attn_implementation_internal = 'flex_attention'
qwen3_cfg.Qwen3Config.__init__ = _patched

# Patch from_pretrained to handle our checkpoint
import speculators.models.eagle3.core as eagle3_core
_orig_from_pretrained = eagle3_core.Eagle3DraftModel.from_pretrained

def patched_from_pretrained(cls_or_self, path, **kwargs):
    """Load weights from checkpoint into freshly created model."""
    import torch
    from safetensors.torch import load_file
    from pathlib import Path

    # Create model from training args first (will be done by train.py)
    # Instead, just skip and let train.py create from scratch, then load weights
    raise AttributeError("use --no-resume flag")

# Don't patch - instead use a different approach: copy pretrained weights to save-path
# so that train.py resumes from them

import shutil, os
src_ckpt = "./output_eagle/checkpoints/4"
dst_dir = "./output_eagle/finetuned_checkpoints/4"
os.makedirs(dst_dir, exist_ok=True)
for f in ["model.safetensors", "config.json", "config.py", "optimizer_state_dict.pt", "scheduler_state_dict.pt"]:
    src = os.path.join(src_ckpt, f)
    if os.path.exists(src):
        shutil.copy2(src, dst_dir)
        print(f"Copied {f}")

import sys
sys.argv = [
    "train.py",
    "--verifier-name-or-path", "./output_grpo/merged",
    "--data-path", "./output_eagle/datagen",
    "--save-path", "./output_eagle/finetuned_checkpoints",
    "--epochs", "15",
    "--lr", "3e-5",
    "--total-seq-len", "2048",
    "--num-layers", "1",
    "--draft-arch", "qwen3",
    "--logger", "tensorboard",
    "--log-dir", "./output_eagle/ft_logs",
    "--run-name", "eagle3-qwen3-8b-continued",
]

exec(open("/home/ray/speculators/scripts/train.py").read())
