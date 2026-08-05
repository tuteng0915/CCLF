"""
Convert ELF JAX/Orbax checkpoint → ELF-torch PyTorch checkpoint.

Usage:
    python convert_jax_to_torch.py \
        --jax_ckpt  /path/to/orbax/checkpoint_dir \
        --out        /path/to/output.pt \
        [--validate  /path/to/reference_torch.pt]

The script uses ema_params1 (EMA weights) when available, since that is
what inference uses.  Pass --use_train_params to use raw params instead.
"""

import argparse, re, sys
from pathlib import Path

import numpy as np
import torch

# ── mapping rules ─────────────────────────────────────────────────────────────
# These parameters are stored as raw nn.Parameter (not nn.Linear),
# so they keep the same name AND shape (no transpose).
RAW_PARAMS = {
    "proj_kernel", "proj_bias",
    "unembed_kernel", "unembed_bias",
    "mode_tokens", "t_emb_tokens", "self_cond_cfg_tokens",
}


def jax_key_to_pt(jax_key: str):
    """Return (pt_key, needs_transpose)."""
    if jax_key in RAW_PARAMS:
        return jax_key, False

    # blocks_N.xxx  →  blocks.N.xxx
    pt_key = re.sub(r"^blocks_(\d+)\.", lambda m: f"blocks.{m.group(1)}.", jax_key)

    # xxx.kernel  →  xxx.weight  (Linear: JAX [in,out] → PT [out,in])
    if pt_key.endswith(".kernel"):
        return pt_key[:-7] + ".weight", True

    # everything else (.bias, .weight for norms/raw params) — copy as-is
    return pt_key, False


def convert_params(flat_jax: dict) -> dict:
    pt_state = {}
    skipped = []
    for jax_k, jax_v in flat_jax.items():
        arr = np.array(jax_v, dtype=np.float32)
        pt_k, needs_T = jax_key_to_pt(jax_k)
        tensor = torch.tensor(arr)
        if needs_T:
            tensor = tensor.T
        pt_state[pt_k] = tensor
    return pt_state


def load_jax_flat(ckpt_dir: str, use_ema: bool = True):
    import orbax.checkpoint as ocp
    from flax.traverse_util import flatten_dict

    ckpter = ocp.PyTreeCheckpointer()
    raw = ckpter.restore(str(ckpt_dir))
    key = "ema_params1" if (use_ema and "ema_params1" in raw) else "params"
    print(f"[convert] Using JAX key: '{key}'  (ema_available={('ema_params1' in raw)})")
    params = raw[key]
    flat = flatten_dict(params, sep=".")
    print(f"[convert] JAX flat params: {len(flat)} keys")
    return flat, raw.get("step", 0)


def validate(converted: dict, ref_path: str, device="cuda"):
    print(f"\n[validate] Loading reference checkpoint: {ref_path}")
    ref_ckpt = torch.load(ref_path, map_location="cpu", weights_only=True)
    ref = ref_ckpt["params"]

    # shape check
    mismatches = []
    for k in ref:
        if k not in converted:
            mismatches.append(f"  MISSING in converted: {k}")
        elif converted[k].shape != ref[k].shape:
            mismatches.append(f"  SHAPE MISMATCH {k}: converted={converted[k].shape} ref={ref[k].shape}")
    for k in converted:
        if k not in ref:
            mismatches.append(f"  EXTRA in converted: {k}")
    if mismatches:
        print("[validate] Shape/key issues:")
        for m in mismatches:
            print(m)
    else:
        print("[validate] All 167 keys match in name and shape ✓")

    # numerical diff between converted and official reference
    diffs = {}
    for k in ref:
        if k in converted and converted[k].shape == ref[k].shape:
            d = (converted[k].float() - ref[k].float()).abs().max().item()
            diffs[k] = d
    max_diff = max(diffs.values())
    mean_diff = sum(diffs.values()) / len(diffs)
    worst = max(diffs, key=diffs.get)
    print(f"[validate] Max abs diff:  {max_diff:.6f}  (key: {worst})")
    print(f"[validate] Mean abs diff: {mean_diff:.6f}")
    if max_diff < 0.01:
        print("[validate] ✓ Weights match reference (max diff < 0.01)")
    else:
        print("[validate] ✗ Large weight mismatch — check transpose/raw-param rules")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jax_ckpt",  required=True, help="Path to orbax checkpoint dir")
    ap.add_argument("--out",       required=True, help="Output .pt file path")
    ap.add_argument("--validate",  default=None,  help="Reference torch .pt to diff against")
    ap.add_argument("--use_train_params", action="store_true",
                    help="Use raw params instead of ema_params1")
    args = ap.parse_args()

    print(f"[convert] Loading JAX checkpoint: {args.jax_ckpt}")
    flat, step = load_jax_flat(args.jax_ckpt, use_ema=not args.use_train_params)

    print("[convert] Converting params …")
    pt_params = convert_params(flat)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"params": pt_params, "step": int(step), "epoch": 0}, str(out_path))
    print(f"[convert] Saved → {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")

    if args.validate:
        validate(pt_params, args.validate)


if __name__ == "__main__":
    main()
