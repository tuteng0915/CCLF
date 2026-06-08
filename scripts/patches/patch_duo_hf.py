#!/usr/bin/env python3
"""Patch DUO model.py: replace flash_attn with pure PyTorch, add compat shims."""
import shutil
from pathlib import Path

SNAP_BASE = Path.home() / ".cache/huggingface/hub/models--s-sahoo--duo/snapshots"
snap_dir = next(SNAP_BASE.iterdir())
fpath = snap_dir / "model.py"
src = fpath.read_text()
shutil.copy(fpath, fpath.with_suffix(".py.orig"))

# 1. Remove flash_attn imports
src = src.replace(
    "import flash_attn\nimport flash_attn.layers.rotary\n",
    "# flash_attn removed (replaced with pure PyTorch)\n"
)
assert "import flash_attn" not in src, "flash_attn import still present"

# 2. Replace split_and_apply_rotary_pos_emb (q, k separately)
old_split = (
    "  with torch.cuda.amp.autocast(enabled=False):\n"
    "    cos, sin = rotary_cos_sin\n"
    "    cos = cos.to(qkv.dtype)\n"
    "    sin = sin.to(qkv.dtype)\n"
    "    cos = cos[0,:,0,0,:cos.shape[-1]//2]\n"
    "    sin = sin[0,:,0,0,:sin.shape[-1]//2]\n"
    "    q, k, v = qkv.chunk(3, dim=2)\n"
    "    q = flash_attn.layers.rotary.apply_rotary_emb_torch(\n"
    "      q.squeeze(dim=2), cos, sin)\n"
    "    k = flash_attn.layers.rotary.apply_rotary_emb_torch(\n"
    "      k.squeeze(dim=2), cos, sin)\n"
    "    v = v.squeeze(dim=2)\n"
    "  return q, k, v"
)
new_split = (
    "  # Pure-PyTorch split rotary: qkv [B,S,3,H,D]; apply to q and k separately\n"
    "  with torch.cuda.amp.autocast(enabled=False):\n"
    "    cos, sin = rotary_cos_sin\n"
    "    cos = cos[0,:,0,0,:cos.shape[-1]//2].to(qkv.dtype)  # [S, D//2]\n"
    "    sin = sin[0,:,0,0,:sin.shape[-1]//2].to(qkv.dtype)\n"
    "    cos_f = torch.cat([cos, cos], -1)[None, :, None, :]  # [1,S,1,D]\n"
    "    sin_f = torch.cat([sin, sin], -1)[None, :, None, :]\n"
    "    q, k, v = qkv.chunk(3, dim=2)  # each [B,S,1,H,D]\n"
    "    q = q.squeeze(2); k = k.squeeze(2); v = v.squeeze(2)  # [B,S,H,D]\n"
    "    def _rot(x): return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), -1)\n"
    "    q = q * cos_f + _rot(q) * sin_f\n"
    "    k = k * cos_f + _rot(k) * sin_f\n"
    "  return q, k, v"
)
assert old_split in src, f"split_rotary pattern not found"
src = src.replace(old_split, new_split)

# 3. Replace apply_rotary_pos_emb (packed qkv [B,S,3,H,D])
old_rotary = (
    "def apply_rotary_pos_emb(qkv, cos, sin):\n"
    "  cos = cos[0,:,0,0,:cos.shape[-1]//2]\n"
    "  sin = sin[0,:,0,0,:sin.shape[-1]//2]\n"
    "  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)"
)
new_rotary = (
    "def apply_rotary_pos_emb(qkv, cos, sin):\n"
    "  # Pure-PyTorch packed qkv rotary (qkv: [B,S,3,H,D])\n"
    "  cos = cos[0,:,0,0,:cos.shape[-1]//2]  # [S, D//2]\n"
    "  sin = sin[0,:,0,0,:sin.shape[-1]//2]\n"
    "  def _rot(x): return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), -1)\n"
    "  cos_f = torch.cat([cos, cos], -1)[None, :, None, None, :]  # [1,S,1,1,D]\n"
    "  sin_f = torch.cat([sin, sin], -1)[None, :, None, None, :]\n"
    "  qkv[:, :, :2] = qkv[:, :, :2] * cos_f + _rot(qkv[:, :, :2]) * sin_f\n"
    "  return qkv"
)
assert old_rotary in src, "packed rotary pattern not found"
src = src.replace(old_rotary, new_rotary)

# 4. Replace flash varlen attention (causal=True for DUO)
old_attn = (
    "    x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(\n"
    "      qkv, cu_seqlens, seq_len, 0.0, causal=True)"
)
new_attn = (
    "    # Pure-PyTorch causal attention (qkv: [(B*S), 3, H, D])\n"
    "    _qkv_b = einops.rearrange(qkv, '(b s) three h d -> b three h s d', b=batch_size, three=3)\n"
    "    _q, _k, _v = _qkv_b[:, 0], _qkv_b[:, 1], _qkv_b[:, 2]\n"
    "    x = torch.nn.functional.scaled_dot_product_attention(_q, _k, _v, is_causal=True, dropout_p=0.)\n"
    "    x = einops.rearrange(x, 'b h s d -> (b s) h d')"
)
assert old_attn in src, "flash attn pattern not found"
src = src.replace(old_attn, new_attn)

# 5. Add all_tied_weights_keys compat shim to DUO class
old_cls = (
    'class DUO(transformers.PreTrainedModel):\n'
    '  """HF-compatible model."""\n'
    '  config_class = DUOConfig\n'
    "  base_model_prefix = 'duo'"
)
new_cls = (
    'class DUO(transformers.PreTrainedModel):\n'
    '  """HF-compatible model."""\n'
    '  config_class = DUOConfig\n'
    "  base_model_prefix = 'duo'\n"
    '  all_tied_weights_keys = {}  # compat shim for transformers >4.38'
)
assert old_cls in src, "DUO class header not found"
src = src.replace(old_cls, new_cls)

# Verify no remaining flash_attn
remaining = [l for l in src.splitlines() if "flash_attn" in l and not l.strip().startswith("#")]
if remaining:
    print("WARNING: remaining flash_attn refs:", remaining)
else:
    print("OK: no remaining flash_attn references")

fpath.write_text(src)
print(f"Patched hub: {fpath}")

# Also patch modules cache if it exists
modules_base = Path.home() / ".cache/huggingface/modules/transformers_modules/s-sahoo/duo"
if modules_base.exists():
    for mpath in modules_base.glob("*/model.py"):
        mpath.write_text(src)
        pyc = mpath.parent / "__pycache__" / (mpath.stem + ".cpython-311.pyc")
        pyc.unlink(missing_ok=True)
        print(f"Patched modules cache: {mpath}")
else:
    print("No modules cache found (will be created on first load)")
