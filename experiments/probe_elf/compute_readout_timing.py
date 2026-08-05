"""
EXP-16v2: Per-position oracle readout timing (fixed-noise states).

Reads from exp07b_v2_* states (fixed ε across t values, seed=42).
Reports three readout-time definitions:
  T_first  : min{t: ŷ(t) = y}           (first-correct readout time)
  T_stable : min{t_k: ŷ(t_j)=y ∀j≥k, for K consecutive t steps}
  T_margin : T_stable with logit margin > margin_threshold

Also computes paired ΔT between checkpoints for the same positions.

Usage (from ELF-torch root, after exp07b_v2_* states are generated):
  python experiments/probe_elf/compute_readout_timing.py \
    --states_root results \
    --output_dir results/exp16v2 \
    --stable_k 3 \
    --margin_threshold 5.0
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

T_VALUES = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]

CHECKPOINTS = {
    "baseline": "converted/elf_b-owt-baseline_torch.pt",
    "kd_cr":    "converted/elf_b-owt-kd-cr_torch.pt",
    "kd2":      "converted/elf_b-owt-kd2_torch.pt",
}


def load_decode_weights(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    params = ckpt.get("params", ckpt)
    return {
        "proj_kernel":    params["proj_kernel"].float().to(device),
        "proj_bias":      params["proj_bias"].float().to(device),
        "unembed_kernel": params["unembed_kernel"].float().to(device),
        "unembed_bias":   params["unembed_bias"].float().to(device),
    }


def decode_logits_and_top1(x_768, weights, batch_size=512):
    """Returns (preds, margins) both (M,)."""
    pk = weights["proj_kernel"]
    pb = weights["proj_bias"]
    uk = weights["unembed_kernel"]
    ub = weights["unembed_bias"]
    preds, margins = [], []
    for i in range(0, x_768.shape[0], batch_size):
        xb = x_768[i:i + batch_size].to(pk.device)
        h = F.gelu(xb @ pk + pb, approximate="tanh")
        lg = h @ uk + ub                           # (B, V)
        top2 = lg.topk(2, dim=-1)
        pred = top2.indices[:, 0].cpu()
        margin = (top2.values[:, 0] - top2.values[:, 1]).cpu()
        preds.append(pred)
        margins.append(margin)
    return torch.cat(preds), torch.cat(margins)


def compute_readout_timing(states_dir, weights, t_values, device, stable_k=3, margin_threshold=5.0):
    """
    Returns dict with per-position timing arrays.
    Positions where state file not found are excluded.
    """
    # Load reference tokens from first available t
    first_t = t_values[0]
    first_path = os.path.join(states_dir, f"layer_states_t{first_t:.3f}.pt")
    d0 = torch.load(first_path, map_location="cpu", weights_only=False)
    y_tokens  = d0["y_tokens"].reshape(-1).long()
    attn_mask = d0["attn_mask"].reshape(-1).bool()
    y_valid   = y_tokens[attn_mask]   # (M,)
    M = y_valid.shape[0]
    T = len(t_values)
    NEVER = T  # sentinel: not recovered within evaluated range

    # per-t correct and margin matrices: (T, M)
    correct_mat = torch.zeros(T, M, dtype=torch.bool)
    margin_mat  = torch.zeros(T, M)

    for ti, t_val in enumerate(t_values):
        path = os.path.join(states_dir, f"layer_states_t{t_val:.3f}.pt")
        if not os.path.exists(path):
            print(f"    WARNING: {path} not found, skipping t={t_val:.3f}")
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        x_last = data["layer_feats"][-1].reshape(-1, 768)[attn_mask].float()
        preds, marg = decode_logits_and_top1(x_last, weights, batch_size=512)
        correct_mat[ti] = (preds == y_valid)
        margin_mat[ti]  = marg
        frac = correct_mat[ti].float().mean().item()
        print(f"    t={t_val:.3f}: G={frac:.4f}")

    # T_first: min t where correct
    t_first = torch.full((M,), NEVER, dtype=torch.long)
    for ti in range(T):
        new_first = correct_mat[ti] & (t_first == NEVER)
        t_first[new_first] = ti

    # T_stable: min t_k where correct for K consecutive steps starting at k
    t_stable = torch.full((M,), NEVER, dtype=torch.long)
    for ti in range(T - stable_k + 1):
        consec = correct_mat[ti].clone()
        for k in range(1, stable_k):
            if ti + k < T:
                consec &= correct_mat[ti + k]
        new_stable = consec & (t_stable == NEVER)
        t_stable[new_stable] = ti

    # T_margin: T_stable AND logit margin > threshold at each of the K steps
    t_margin = torch.full((M,), NEVER, dtype=torch.long)
    for ti in range(T - stable_k + 1):
        consec = correct_mat[ti].clone()
        high_margin = (margin_mat[ti] > margin_threshold)
        for k in range(1, stable_k):
            if ti + k < T:
                consec &= correct_mat[ti + k]
                high_margin &= (margin_mat[ti + k] > margin_threshold)
        new_margin = consec & high_margin & (t_margin == NEVER)
        t_margin[new_margin] = ti

    def timing_stats(t_idx, label):
        frac_recovered = (t_idx < NEVER).float().mean().item()
        t_vals_at = [t_values[t_idx[i].item()] if t_idx[i] < NEVER else float("inf")
                     for i in range(M)]
        hist = {}
        for ti, tv in enumerate(t_values):
            hist[f"by_t_{tv:.2f}"] = (t_idx <= ti).float().mean().item()
        hist["not_recovered_in_range"] = 1.0 - frac_recovered
        print(f"    [{label}] frac recovered: {frac_recovered:.4f}")
        for k, v in hist.items():
            print(f"      {k}: {v:.4f}")
        return {
            "hist": hist,
            "frac_recovered": frac_recovered,
            "t_idx_array": t_idx.tolist(),
        }

    print(f"  First-correct readout time:")
    first_stats = timing_stats(t_first, "T_first")
    print(f"  Stable readout time (K={stable_k}):")
    stable_stats = timing_stats(t_stable, "T_stable")
    print(f"  High-margin stable time (margin>{margin_threshold}):")
    margin_stats = timing_stats(t_margin, f"T_margin")

    return {
        "M": M,
        "t_values": t_values,
        "stable_k": stable_k,
        "margin_threshold": margin_threshold,
        "T_first":  first_stats,
        "T_stable": stable_stats,
        "T_margin": margin_stats,
        "t_first_idx":  t_first.tolist(),
        "t_stable_idx": t_stable.tolist(),
        "t_margin_idx": t_margin.tolist(),
    }


def paired_analysis(res_a, res_b, name_a, name_b, t_values, NEVER):
    """Compare T_first and T_stable between two checkpoints at same positions."""
    a_first  = torch.tensor(res_a["t_first_idx"])
    b_first  = torch.tensor(res_b["t_first_idx"])
    a_stable = torch.tensor(res_a["t_stable_idx"])
    b_stable = torch.tensor(res_b["t_stable_idx"])

    def delta_stats(a, b, label):
        M = len(a)
        a_rec  = (a < NEVER)
        b_rec  = (b < NEVER)
        both   = a_rec & b_rec
        only_b = (~a_rec) & b_rec  # b recovered, a not
        only_a = a_rec & (~b_rec)  # a recovered, b not
        neither = (~a_rec) & (~b_rec)

        delta = (b.float() - a.float())  # positive = b later than a (b=KD, a=baseline)
        print(f"    [{label}] {name_b} vs {name_a}:")
        print(f"      both recovered: {both.float().mean():.4f}")
        print(f"      only {name_b} recovered: {only_b.float().mean():.4f}")
        print(f"      only {name_a} recovered: {only_a.float().mean():.4f}")
        print(f"      neither: {neither.float().mean():.4f}")
        if both.any():
            d = delta[both]
            print(f"      ΔT={name_b}-{name_a} (among both recovered): "
                  f"mean={d.mean():.2f}, median={d.median():.1f}, "
                  f"frac_{name_b}_earlier={(d < 0).float().mean():.4f}")
        return {
            "frac_both_recovered": both.float().mean().item(),
            f"frac_only_{name_b}": only_b.float().mean().item(),
            f"frac_only_{name_a}": only_a.float().mean().item(),
            "frac_neither": neither.float().mean().item(),
            "delta_mean_both": delta[both].mean().item() if both.any() else None,
            "delta_median_both": delta[both].median().item() if both.any() else None,
            f"frac_{name_b}_earlier_both": (delta[both] < 0).float().mean().item() if both.any() else None,
        }

    print(f"\n[Paired: {name_b} vs {name_a}]")
    r_first  = delta_stats(a_first,  b_first,  "T_first")
    r_stable = delta_stats(a_stable, b_stable, "T_stable")
    return {"T_first": r_first, "T_stable": r_stable}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_root", default="results",
                    help="Root dir; checkpoint states are at <root>/exp07b_v2_<name>")
    ap.add_argument("--output_dir", default="results/exp16v2")
    ap.add_argument("--stable_k", type=int, default=3)
    ap.add_argument("--margin_threshold", type=float, default=5.0)
    ap.add_argument("--t_values", default="0.10,0.20,0.30,0.50,0.70,1.00")
    args = ap.parse_args()

    t_values = [float(x) for x in args.t_values.split(",")]
    NEVER = len(t_values)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EXP-16v2] stable_k={args.stable_k}, margin>{args.margin_threshold}, t_values={t_values}")

    all_results = {}
    for name, ckpt_rel in CHECKPOINTS.items():
        states_dir = os.path.join(args.states_root, f"exp07b_v2_{name}")
        if not os.path.isdir(states_dir):
            print(f"[EXP-16v2] WARNING: {states_dir} not found, skipping {name}")
            continue
        print(f"\n=== {name} ===")
        weights = load_decode_weights(ckpt_rel, device)
        res = compute_readout_timing(
            states_dir, weights, t_values, device,
            stable_k=args.stable_k, margin_threshold=args.margin_threshold,
        )
        all_results[name] = res

    # Paired comparisons
    paired = {}
    ref = "baseline"
    if ref in all_results:
        for name in ["kd_cr", "kd2"]:
            if name in all_results:
                paired[f"{name}_vs_{ref}"] = paired_analysis(
                    all_results[ref], all_results[name], ref, name, t_values, NEVER)

    out = {
        "per_checkpoint": {k: {
            "t_values": v["t_values"],
            "stable_k": v["stable_k"],
            "T_first":  {**v["T_first"],  "t_idx_array": None},
            "T_stable": {**v["T_stable"], "t_idx_array": None},
            "T_margin": {**v["T_margin"], "t_idx_array": None},
        } for k, v in all_results.items()},
        "paired": paired,
    }
    # Save full idx arrays separately for analysis
    for name, res in all_results.items():
        torch.save({
            "t_first_idx":  torch.tensor(res["t_first_idx"]),
            "t_stable_idx": torch.tensor(res["t_stable_idx"]),
            "t_margin_idx": torch.tensor(res["t_margin_idx"]),
            "t_values": t_values,
        }, os.path.join(args.output_dir, f"timing_arrays_{name}.pt"))

    with open(os.path.join(args.output_dir, "readout_timing.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[EXP-16v2] Saved to {args.output_dir}/readout_timing.json")

    # Summary table
    print("\n=== Summary: T_first fraction recovered by t ===")
    for tv in t_values:
        row = f"  by t={tv:.2f}: "
        for name in ["baseline", "kd_cr", "kd2"]:
            if name in all_results:
                v = all_results[name]["T_first"]["hist"].get(f"by_t_{tv:.2f}", float("nan"))
                row += f"{name}={v:.4f}  "
        print(row)

    print("\n=== Summary: T_stable fraction recovered by t ===")
    for tv in t_values:
        row = f"  by t={tv:.2f}: "
        for name in ["baseline", "kd_cr", "kd2"]:
            if name in all_results:
                v = all_results[name]["T_stable"]["hist"].get(f"by_t_{tv:.2f}", float("nan"))
                row += f"{name}={v:.4f}  "
        print(row)


if __name__ == "__main__":
    main()
