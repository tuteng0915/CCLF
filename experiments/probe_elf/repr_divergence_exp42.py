"""
EXP-42: Residual Stream Divergence (kd_cr vs baseline vs kd2)

Since exp07b_v2 uses the same input sequences for all checkpoints (y_tokens match),
we can directly compare layer-wise hidden states between checkpoints.

Key question: at which layer does the residual stream start to diverge?
If kd_cr and baseline diverge only at block_11 → KD changed only the decode interface.
If they diverge from block_0 → KD also changed the backbone.

Metrics:
1. Relative L2 difference: ||h_A - h_B||_F / ||h_B||_F  (per layer, per t)
2. CKA (Linear Centered Kernel Alignment): measures representation similarity
   independent of linear transformations. CKA = 1 means same linear subspace.
3. Cosine similarity of mean-centered representations (global signal)

Reuses: results/exp07b_v2_{baseline,kd_cr,kd2}/layer_states_t*.pt

Output: results/exp42_repr_divergence/divergence.json
  {comparison: {t_str: {layer_i: {rel_l2, cka, cos_sim}}}

Usage (from ELF-torch root):
  python experiments/probe_elf/repr_divergence_exp42.py
"""

import json
import os

import torch
import numpy as np

DATA_DIRS = {
    "baseline": "results/exp07b_v2_baseline",
    "kd_cr":    "results/exp07b_v2_kd_cr",
    "kd2":      "results/exp07b_v2_kd2",
}

COMPARISONS = [
    ("kd_cr",    "baseline"),
    ("kd2",      "baseline"),
    ("kd_cr",    "kd2"),
]

T_VALUES = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]


def find_state_file(data_dir: str, t: float) -> str | None:
    for fmt in [f"t{t:.3f}", f"t{t:.2f}", f"t{t:.1f}"]:
        p = os.path.join(data_dir, f"layer_states_{fmt}.pt")
        if os.path.exists(p):
            return p
    for fn in os.listdir(data_dir):
        if fn.startswith("layer_states"):
            tv = float(fn.replace("layer_states_t", "").replace(".pt", ""))
            if abs(tv - t) < 1e-4:
                return os.path.join(data_dir, fn)
    return None


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA between X and Y, each [N, d] (N = positions, d = hidden_dim).
    CKA(X,Y) = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)
    Uses the HSIC formulation: efficient for small N.
    """
    X = X.float()
    Y = Y.float()
    # Center
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    # Gram matrices
    K = X @ X.T  # [N, N]
    L = Y @ Y.T  # [N, N]
    # HSIC
    n = X.shape[0]
    hsic_xy = (K * L).sum() / (n - 1) ** 2
    hsic_xx = (K * K).sum() / (n - 1) ** 2
    hsic_yy = (L * L).sum() / (n - 1) ** 2
    denom = (hsic_xx * hsic_yy) ** 0.5
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


def compute_divergence(h_A: torch.Tensor, h_B: torch.Tensor, mask: torch.Tensor) -> dict:
    """
    h_A, h_B: [B, L, D], mask: [B, L] bool
    Returns: {rel_l2, cka, cos_sim}
    """
    # Flatten to [N, D] using mask
    flat_mask = mask.reshape(-1).bool()
    hA_flat = h_A.reshape(-1, h_A.shape[-1])[flat_mask].float()
    hB_flat = h_B.reshape(-1, h_B.shape[-1])[flat_mask].float()

    # Relative L2
    diff = hA_flat - hB_flat
    rel_l2 = float(diff.norm() / (hB_flat.norm() + 1e-8))

    # CKA on a subsample (max 2048 positions for speed)
    N = hA_flat.shape[0]
    if N > 2048:
        idx = torch.randperm(N, generator=torch.Generator().manual_seed(42))[:2048]
        hA_s = hA_flat[idx]
        hB_s = hB_flat[idx]
    else:
        hA_s = hA_flat
        hB_s = hB_flat
    cka = linear_cka(hA_s, hB_s)

    # Global cosine similarity (after mean-centering)
    hA_mc = hA_flat - hA_flat.mean(0)
    hB_mc = hB_flat - hB_flat.mean(0)
    cos = float((hA_mc * hB_mc).sum() / (hA_mc.norm() * hB_mc.norm() + 1e-8))

    return {
        "rel_l2":  round(rel_l2, 6),
        "cka":     round(cka, 6),
        "cos_sim": round(cos, 6),
    }


def main():
    os.makedirs("results/exp42_repr_divergence", exist_ok=True)
    results = {}

    for (name_A, name_B) in COMPARISONS:
        comp_key = f"{name_A}_vs_{name_B}"
        print(f"\n=== {comp_key} ===")
        results[comp_key] = {}

        for t in T_VALUES:
            fA = find_state_file(DATA_DIRS[name_A], t)
            fB = find_state_file(DATA_DIRS[name_B], t)
            if fA is None or fB is None:
                print(f"  WARNING: missing data for t={t}")
                continue

            dA = torch.load(fA, map_location="cpu", weights_only=False)
            dB = torch.load(fB, map_location="cpu", weights_only=False)
            mask = dA["attn_mask"].bool()

            # Verify tokens match
            if not (dA["y_tokens"] == dB["y_tokens"]).all():
                print(f"  WARNING: y_tokens mismatch at t={t}!")

            depth = len(dA["layer_feats"])
            t_str = f"t{t:.3f}"
            results[comp_key][t_str] = {}

            for i in range(depth):
                hA = dA["layer_feats"][i]
                hB = dB["layer_feats"][i]
                metrics = compute_divergence(hA, hB, mask)
                results[comp_key][t_str][f"block_{i:02d}"] = metrics
                print(f"  t={t:.2f} block_{i:02d}: rel_l2={metrics['rel_l2']:.4f}  "
                      f"cka={metrics['cka']:.4f}  cos={metrics['cos_sim']:.4f}")

            del dA, dB

    out_path = "results/exp42_repr_divergence/divergence.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Summary: per comparison, at which block does CKA first drop below 0.95?
    print("\n--- First block where CKA < 0.95 (by t) ---")
    for comp_key, by_t in results.items():
        print(f"\n{comp_key}:")
        for t_str, layers in by_t.items():
            for i in range(12):
                key = f"block_{i:02d}"
                if layers.get(key, {}).get("cka", 1.0) < 0.95:
                    print(f"  {t_str}: first CKA<0.95 at block_{i:02d} "
                          f"(cka={layers[key]['cka']:.4f})")
                    break
            else:
                l11 = layers.get("block_11", {}).get("cka", 1.0)
                print(f"  {t_str}: CKA ≥ 0.95 throughout (L11 cka={l11:.4f})")


if __name__ == "__main__":
    main()
