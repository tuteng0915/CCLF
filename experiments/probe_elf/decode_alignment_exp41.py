"""
EXP-41: Decode Hidden Alignment

Measures whether the decode intermediate vector (after GELU projection)
points in the direction of the correct token in embedding space.

The decode path is:
  decode_hidden = GELU(h_L11 @ proj_kernel + proj_bias)  ← [512]
  logits        = decode_hidden @ unembed_kernel + unembed_bias

Token v's "direction" in the 512-dim decode space is unembed_kernel[:, v].
We measure:
  cos_align(i) = cos(decode_hidden_i, unembed_kernel[:, y_i])

Key questions:
1. Is cos_align a reliable predictor of oracle accuracy (correct token = top-1)?
2. Does kd_cr have higher cos_align for correct positions than baseline?
3. Does cos_align increase monotonically with t, matching the "commitment cliff"?
4. At fixed t, do positions where the model is correct have higher cos_align than wrong positions?

Reuses: exp07b_v2 layer states (no new GPU forward passes needed)

Output: results/exp41_decode_alignment/alignment.json
  {checkpoint: {t_str: {
    "correct_cos_mean": float,
    "correct_cos_std": float,
    "wrong_cos_mean": float,
    "wrong_cos_std": float,
    "auc": float,       # ROC AUC of using cos_align to predict oracle correctness
    "n_correct": int,
    "n_wrong": int,
  }, ...}, ...}

Usage (from ELF-torch root):
  python experiments/probe_elf/decode_alignment_exp41.py
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

CHECKPOINTS = {
    "baseline": {
        "ckpt": "converted/elf_b-owt-baseline_torch.pt",
        "data": "results/exp07b_v2_baseline",
    },
    "kd_cr": {
        "ckpt": "converted/elf_b-owt-kd-cr_torch.pt",
        "data": "results/exp07b_v2_kd_cr",
    },
    "kd2": {
        "ckpt": "converted/elf_b-owt-kd2_torch.pt",
        "data": "results/exp07b_v2_kd2",
    },
}

T_VALUES = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
BATCH_SIZE = 16


def load_decode_params(ckpt_path: str) -> dict:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    p = raw["params"]
    return {
        "proj_kernel":    p["proj_kernel"].float(),    # [768, 512]
        "proj_bias":      p["proj_bias"].float(),      # [512]
        "unembed_kernel": p["unembed_kernel"].float(), # [512, V]
        "unembed_bias":   p["unembed_bias"].float(),   # [V]
    }


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


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Simple trapezoidal AUC without sklearn."""
    sorted_idx = np.argsort(scores)[::-1]
    labels_sorted = labels[sorted_idx]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    tpr = tp / n_pos
    fpr = fp / n_neg
    # trapezoidal rule
    auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
    return float(abs(auc))


@torch.no_grad()
def analyze_alignment(ckpt_name: str, cfg: dict, params: dict, t: float) -> dict | None:
    fname = find_state_file(cfg["data"], t)
    if fname is None:
        return None

    data = torch.load(fname, map_location="cpu", weights_only=False)
    h_L11     = data["layer_feats"][-1]   # [B, L, 768]
    y_tokens  = data["y_tokens"]          # [B, L]
    attn_mask = data["attn_mask"]         # [B, L]
    B = h_L11.shape[0]

    # Pre-normalize token directions (columns of unembed_kernel)
    unembed_norm = F.normalize(params["unembed_kernel"], dim=0)  # [512, V], cols normalized

    cos_correct_list = []
    cos_wrong_list   = []

    for start in range(0, B, BATCH_SIZE):
        end = min(start + BATCH_SIZE, B)
        h_b    = h_L11[start:end].float()       # [bs, L, 768]
        y_b    = y_tokens[start:end]             # [bs, L]
        mask_b = attn_mask[start:end].bool()     # [bs, L]

        # Compute decode hidden
        decode_h = F.gelu(h_b @ params["proj_kernel"] + params["proj_bias"],
                          approximate="tanh")   # [bs, L, 512]

        # Compute oracle top-1 prediction
        logits = decode_h @ params["unembed_kernel"] + params["unembed_bias"]  # [bs, L, V]
        pred   = logits.argmax(-1)  # [bs, L]
        correct = (pred == y_b)     # [bs, L]

        # Normalize decode_h for cosine
        decode_h_norm = F.normalize(decode_h, dim=-1)  # [bs, L, 512]

        # Cosine similarity to correct token direction
        # For each position (i, j), get unembed_norm[:, y_b[i,j]]
        # Then cos = decode_h_norm[i, j] @ unembed_norm[:, y_b[i,j]]
        bs, L = y_b.shape
        y_flat = y_b.reshape(-1)   # [bs*L]
        token_dirs = unembed_norm[:, y_flat].T  # [bs*L, 512]
        decode_flat = decode_h_norm.reshape(-1, 512)  # [bs*L, 512]
        cos = (decode_flat * token_dirs).sum(-1)  # [bs*L]
        cos = cos.reshape(bs, L)

        # Split by correctness and mask
        mask_flat    = mask_b.reshape(-1)
        correct_flat = correct.reshape(-1)
        cos_flat     = cos.reshape(-1)

        valid = mask_flat
        cos_correct_list.extend(cos_flat[valid & correct_flat].tolist())
        cos_wrong_list.extend(  cos_flat[valid & ~correct_flat].tolist())

        del decode_h, logits, cos, decode_flat

    cos_correct = np.array(cos_correct_list)
    cos_wrong   = np.array(cos_wrong_list)

    all_cos    = np.concatenate([cos_correct, cos_wrong])
    all_labels = np.array([1]*len(cos_correct) + [0]*len(cos_wrong))
    auc = roc_auc(all_cos, all_labels)

    print(f"  t={t:.2f}: correct cos={cos_correct.mean():.4f}±{cos_correct.std():.4f}  "
          f"wrong cos={cos_wrong.mean():.4f}±{cos_wrong.std():.4f}  AUC={auc:.4f}  "
          f"n_correct={len(cos_correct)} n_wrong={len(cos_wrong)}")

    return {
        "correct_cos_mean": round(float(cos_correct.mean()), 6),
        "correct_cos_std":  round(float(cos_correct.std()),  6),
        "wrong_cos_mean":   round(float(cos_wrong.mean()),   6),
        "wrong_cos_std":    round(float(cos_wrong.std()),    6),
        "cos_gap":          round(float(cos_correct.mean() - cos_wrong.mean()), 6),
        "auc":              round(auc, 6),
        "n_correct":        len(cos_correct),
        "n_wrong":          len(cos_wrong),
    }


def main():
    os.makedirs("results/exp41_decode_alignment", exist_ok=True)
    results = {}

    for ckpt_name, cfg in CHECKPOINTS.items():
        print(f"\n=== {ckpt_name} ===")
        params = load_decode_params(cfg["ckpt"])
        results[ckpt_name] = {}
        for t in T_VALUES:
            t_str = f"t{t:.3f}"
            r = analyze_alignment(ckpt_name, cfg, params, t)
            if r:
                results[ckpt_name][t_str] = r

    out_path = "results/exp41_decode_alignment/alignment.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Summary table
    print("\n--- Cos-gap (correct - wrong positions) summary ---")
    print(f"{'ckpt':<10}", end="")
    for t in T_VALUES:
        print(f"  t={t:.2f}", end="")
    print()
    for ckpt_name in results:
        print(f"{ckpt_name:<10}", end="")
        for t in T_VALUES:
            t_str = f"t{t:.3f}"
            gap = results[ckpt_name].get(t_str, {}).get("cos_gap", float("nan"))
            print(f"  {gap:+.4f}", end="")
        print()

    print("\n--- AUC (cos_align predicts oracle accuracy) ---")
    print(f"{'ckpt':<10}", end="")
    for t in T_VALUES:
        print(f"  t={t:.2f}", end="")
    print()
    for ckpt_name in results:
        print(f"{ckpt_name:<10}", end="")
        for t in T_VALUES:
            t_str = f"t{t:.3f}"
            auc = results[ckpt_name].get(t_str, {}).get("auc", float("nan"))
            print(f"  {auc:.4f}", end="")
        print()


if __name__ == "__main__":
    main()
