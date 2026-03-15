#!/usr/bin/env python3
"""Wrapper to fix attn_implementation=None before running speculators train."""

# Patch Qwen3Config before anything loads
import transformers.models.qwen3.configuration_qwen3 as qwen3_cfg
_orig = qwen3_cfg.Qwen3Config.__init__
def _patched(self, *a, **kw):
    _orig(self, *a, **kw)
    if getattr(self, '_attn_implementation', None) is None:
        self._attn_implementation = 'flex_attention'
        self._attn_implementation_internal = 'flex_attention'
qwen3_cfg.Qwen3Config.__init__ = _patched

# Now run the train script
import sys
sys.argv = [
    "train.py",
    "--verifier-name-or-path", "./output_grpo/merged",
    "--data-path", "./output_eagle/datagen",
    "--save-path", "./output_eagle/checkpoints",
    "--epochs", "5",
    "--lr", "1e-4",
    "--total-seq-len", "2048",
    "--num-layers", "1",
    "--draft-arch", "qwen3",
    "--logger", "tensorboard",
    "--log-dir", "./output_eagle/logs",
    "--run-name", "eagle3-qwen3-8b-toolace",
]

exec(open("/home/ray/speculators/scripts/train.py").read())
