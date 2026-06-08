#!/usr/bin/env python3
"""Patch 3: fix rotary broadcast shape in apply_rotary_pos_emb."""
import shutil
from pathlib import Path

PATHS = [
    Path.home() / ".cache/huggingface/hub/models--kuleshov-group--mdlm-owt/snapshots/d0958fa851335ece6c15260ce0025f030673c0fb/modeling_mdlm.py",
    Path.home() / ".cache/huggingface/modules/transformers_modules/kuleshov_hyphen_group/mdlm_hyphen_owt/d0958fa851335ece6c15260ce0025f030673c0fb/modeling_mdlm.py",
]

old = (
    "  # Pure-PyTorch rotary: apply to q and k (first 2 of 3 in qkv [B,S,3,H,D])\n"
    "  cos_full = torch.cat([cos, cos], dim=-1)[None, :, None, :]  # [1,S,1,D]\n"
    "  sin_full = torch.cat([sin, sin], dim=-1)[None, :, None, :]\n"
    "  qkv[:, :, :2] = qkv[:, :, :2] * cos_full + rotate_half(qkv[:, :, :2]) * sin_full\n"
    "  return qkv"
)
new = (
    "  # Pure-PyTorch rotary: qkv is [B,S,3,H,D]; apply to q+k (first 2 of 3)\n"
    "  # cos/sin arrive as [S, D//2]; broadcast needs shape [1,S,1,1,D] for 5-D qkv\n"
    "  cos_full = torch.cat([cos, cos], dim=-1)[None, :, None, None, :]  # [1,S,1,1,D]\n"
    "  sin_full = torch.cat([sin, sin], dim=-1)[None, :, None, None, :]\n"
    "  qkv[:, :, :2] = qkv[:, :, :2] * cos_full + rotate_half(qkv[:, :, :2]) * sin_full\n"
    "  return qkv"
)

for fpath in PATHS:
    if not fpath.exists():
        print(f"SKIP (not found): {fpath}")
        continue
    src = fpath.read_text()
    if old not in src:
        print(f"SKIP (pattern not found, already patched?): {fpath.name}")
        continue
    shutil.copy(fpath, fpath.with_suffix(".py.bak"))
    src = src.replace(old, new)
    fpath.write_text(src)
    # Remove stale .pyc
    pyc = fpath.parent / "__pycache__" / (fpath.stem + ".cpython-311.pyc")
    pyc.unlink(missing_ok=True)
    print(f"Patched: {fpath}")
