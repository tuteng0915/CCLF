#!/usr/bin/env python3
"""Patch 2: add all_tied_weights_keys compat shim to MDLM class."""
from pathlib import Path

HF_BASE = Path.home() / ".cache/huggingface/hub/models--kuleshov-group--mdlm-owt/snapshots"
snap = next(HF_BASE.iterdir())
fpath = snap / "modeling_mdlm.py"

src = fpath.read_text()

old = (
    'class MDLM(transformers.PreTrainedModel):\n'
    '  """HF-compatible model."""\n'
    '  config_class = MDLMConfig\n'
    '  base_model_prefix = "mdlm"'
)
new = (
    'class MDLM(transformers.PreTrainedModel):\n'
    '  """HF-compatible model."""\n'
    '  config_class = MDLMConfig\n'
    '  base_model_prefix = "mdlm"\n'
    '  # Compat shim: newer transformers (>4.38) expects this dict on the model class\n'
    '  all_tied_weights_keys = {}'
)

assert old in src, "MDLM class header not found"
src = src.replace(old, new)
fpath.write_text(src)
print(f"Patched: {fpath}")
