"""
EXP-15v2: Parameter-space analysis — module-level decomposition + update direction similarity.

Computes:
  1. Per-block, per-module relative L2 change: R_{l,m} = |Δθ_{l,m}| / |θ_{l,m}|
  2. Update direction similarity between kd_cr and kd2 (cos of Δθ vectors)
  3. Overall block-level R_l for quick comparison with EXP-07b activation transfer

Usage (from ELF-torch root):
  python experiments/probe_elf/analyze_param_distance.py \
    --output_dir results/exp15v2
"""

import argparse
import json
import os

import torch


CKPTS = {
    "baseline": "converted/elf_b-owt-baseline_torch.pt",
    "kd_cr":    "converted/elf_b-owt-kd-cr_torch.pt",
    "kd2":      "converted/elf_b-owt-kd2_torch.pt",
}

# Module name patterns for each block
MODULE_PATTERNS = {
    "attn_qkv":   ["attn.qkv", "attn.in_proj"],
    "attn_out":   ["attn.out", "attn.proj"],
    "mlp_up":     ["mlp.fc1", "mlp.up_proj", "mlp.gate"],
    "mlp_down":   ["mlp.fc2", "mlp.down_proj"],
    "layernorm":  ["norm1", "norm2", "ln_"],
    "timestep":   ["time_", "adaln", "adaLN"],
    "decode_branch": ["proj_kernel", "proj_bias", "unembed_kernel", "unembed_bias"],
    "self_cond":  ["self_cond", "sc_"],
}


def load_params(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("params", ckpt)


def classify_key(key, block_idx):
    """Assign a param key to a module category within a block."""
    block_prefix = f"blocks.{block_idx}."
    if not key.startswith(block_prefix):
        return None
    suffix = key[len(block_prefix):]
    for mod_name, patterns in MODULE_PATTERNS.items():
        if any(p in suffix for p in patterns):
            return mod_name
    return "other"


def rel_l2(delta_sq, base_sq):
    return (delta_sq ** 0.5) / (base_sq ** 0.5 + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="results/exp15v2")
    ap.add_argument("--n_blocks", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    params = {name: load_params(path) for name, path in CKPTS.items()}
    base = params["baseline"]

    results = {}

    for ckpt_name in ["kd_cr", "kd2"]:
        kd = params[ckpt_name]
        print(f"\n=== {ckpt_name} vs baseline ===")

        block_results = {}
        for i in range(args.n_blocks):
            block_prefix = f"blocks.{i}."
            block_keys = [k for k in base if k.startswith(block_prefix)]

            if not block_keys:
                continue

            # Overall block-level R_l
            d_sq = sum((kd[k].float() - base[k].float()).norm().item() ** 2
                       for k in block_keys if k in kd)
            b_sq = sum(base[k].float().norm().item() ** 2 for k in block_keys)
            r_l = rel_l2(d_sq, b_sq)

            # Per-module breakdown
            mod_stats = {}
            for mod_name in list(MODULE_PATTERNS.keys()) + ["other"]:
                mod_keys = [k for k in block_keys
                            if classify_key(k, i) == mod_name and k in kd]
                if not mod_keys:
                    continue
                md = sum((kd[k].float() - base[k].float()).norm().item() ** 2
                         for k in mod_keys)
                mb = sum(base[k].float().norm().item() ** 2 for k in mod_keys)
                n_params = sum(base[k].numel() for k in mod_keys)
                mod_stats[mod_name] = {
                    "rel_l2": rel_l2(md, mb),
                    "n_params": n_params,
                    "n_keys": len(mod_keys),
                }

            block_results[i] = {
                "block_rel_l2": r_l,
                "modules": mod_stats,
            }
            print(f"  Block {i:2d}: R_l={r_l:.4f}  " +
                  "  ".join(f"{m}={v['rel_l2']:.4f}" for m, v in mod_stats.items()))

        results[ckpt_name] = block_results

    # Update direction similarity between kd_cr and kd2 per block
    print("\n=== Update direction similarity: kd_cr vs kd2 (cos of Δθ per block) ===")
    dir_sim = {}
    kd_cr = params["kd_cr"]
    kd2   = params["kd2"]
    for i in range(args.n_blocks):
        block_prefix = f"blocks.{i}."
        block_keys = [k for k in base if k.startswith(block_prefix) and k in kd_cr and k in kd2]
        if not block_keys:
            continue
        # Flatten all param differences for this block
        delta_cr = torch.cat([(kd_cr[k].float() - base[k].float()).flatten() for k in block_keys])
        delta_kd2 = torch.cat([(kd2[k].float() - base[k].float()).flatten() for k in block_keys])
        cosine = torch.nn.functional.cosine_similarity(
            delta_cr.unsqueeze(0), delta_kd2.unsqueeze(0)).item()
        dir_sim[i] = cosine
        print(f"  Block {i:2d}: cos(Δkd_cr, Δkd2) = {cosine:.4f}")

    # Top-level param changes (decode branch, self-cond, etc.)
    print("\n=== Non-block params ===")
    non_block_keys = [k for k in base if not any(f"blocks.{i}." in k for i in range(args.n_blocks))]
    top_level = {}
    for k in non_block_keys:
        if k not in kd_cr:
            continue
        diff = (kd_cr[k].float() - base[k].float()).norm().item()
        base_n = base[k].float().norm().item()
        r = diff / (base_n + 1e-12)
        top_level[k] = {"kd_cr_rel_l2": r}
        if r > 0.05:
            print(f"  {k}: {r:.4f}")

    output = {
        "per_ckpt": results,
        "dir_similarity_kd_cr_vs_kd2": dir_sim,
        "top_level_params": top_level,
    }
    out_path = os.path.join(args.output_dir, "param_distance.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[EXP-15v2] Saved to {out_path}")

    # Summary table
    print("\n=== Block-level summary ===")
    print(f"{'Block':>6}  {'kd_cr R_l':>10}  {'kd2 R_l':>9}  {'cos(Δkd_cr,Δkd2)':>18}")
    for i in range(args.n_blocks):
        r_cr = results.get("kd_cr", {}).get(i, {}).get("block_rel_l2", float("nan"))
        r_k2 = results.get("kd2", {}).get(i, {}).get("block_rel_l2", float("nan"))
        cos  = dir_sim.get(i, float("nan"))
        print(f"  L{i:2d}  {r_cr:>10.4f}  {r_k2:>9.4f}  {cos:>18.4f}")


if __name__ == "__main__":
    main()
