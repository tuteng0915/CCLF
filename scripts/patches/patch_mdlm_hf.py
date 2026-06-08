#!/usr/bin/env python3
"""Patch modeling_mdlm.py in HF cache to remove flash_attn dependency."""
import shutil
from pathlib import Path

HF_BASE = Path.home() / ".cache/huggingface/hub/models--kuleshov-group--mdlm-owt/snapshots"
snap = next(HF_BASE.iterdir())
fpath = snap / "modeling_mdlm.py"

src = fpath.read_text()
shutil.copy(fpath, fpath.with_suffix(".py.orig"))

# 1. Remove flash_attn imports
src = src.replace(
    "import flash_attn\nimport flash_attn.layers.rotary\n",
    "# flash_attn removed (replaced with pure PyTorch)\n"
)

# 2. Replace apply_rotary_pos_emb body
old_rotary = (
    "  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv,\n"
    "                                                        cos,\n"
    "                                                        sin)"
)
new_rotary = (
    "  # Pure-PyTorch rotary: apply to q and k (first 2 of 3 in qkv [B,S,3,H,D])\n"
    "  cos_full = torch.cat([cos, cos], dim=-1)[None, :, None, :]  # [1,S,1,D]\n"
    "  sin_full = torch.cat([sin, sin], dim=-1)[None, :, None, :]\n"
    "  qkv[:, :, :2] = qkv[:, :, :2] * cos_full + rotate_half(qkv[:, :, :2]) * sin_full\n"
    "  return qkv"
)
assert old_rotary in src, "rotary pattern not found"
src = src.replace(old_rotary, new_rotary)

# 3. Replace flash varlen attention with scaled_dot_product_attention
old_attn = (
    "    x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(\n"
    "      qkv, cu_seqlens, seq_len, 0., causal=False)"
)
new_attn = (
    "    # Pure-PyTorch attention (qkv: [(B*S), 3, H, D])\n"
    "    _qkv_b = rearrange(qkv, '(b s) three h d -> b three h s d', b=batch_size, three=3)\n"
    "    _q, _k, _v = _qkv_b[:, 0], _qkv_b[:, 1], _qkv_b[:, 2]\n"
    "    x = torch.nn.functional.scaled_dot_product_attention(_q, _k, _v, dropout_p=0.)\n"
    "    x = rearrange(x, 'b h s d -> (b s) h d')"
)
assert old_attn in src, "attn pattern not found"
src = src.replace(old_attn, new_attn)

# Verify
remaining = [l for l in src.splitlines()
             if "flash_attn" in l and not l.strip().startswith("#")]
if remaining:
    print("WARNING: remaining flash_attn refs:", remaining)
else:
    print("OK: no remaining flash_attn references")

fpath.write_text(src)
print(f"Patched: {fpath}")
