"""
EXP-12: Residual rank analysis (FIXED).

Uses already-collected layer_states (exp07b) and the FULL DECODE HEAD to
compute, for each (seq, position, t):
  - G(t) = Rec@1 via decode logits  (matches probe_null_vs_oracle baseline)
  - rank of true token in decode logit ranking (1 = top-1 correct)
  - cosine similarity gap (logit_top1 - logit_true)
  - rank distribution for "wrong" positions

Key question: for positions where G(t) misses, is the true token at rank 2-5
(near miss, token is almost committed) or rank 100+ (complete failure)?

DECODE PATH (correct, from model.py):
  x         : layer_feats[-1]  (N, L, 768)  — last transformer block output
  hidden    : GELU(x @ proj_kernel + proj_bias)   (N, L, 512)
  logits    : hidden @ unembed_kernel + unembed_bias  (N, L, V)
  G(t)      : frac where argmax(logits) == true_token

Usage (from ELF-torch root):
  CUDA_VISIBLE_DEVICES=1 python experiments/probe_elf/analyze_residual_rank.py \
    --checkpoint converted/elf_b-owt-baseline_torch.pt \
    --states_dir results/exp07b_baseline \
    --output_dir results/exp12_baseline \
    --t_values 0.10,0.20,0.30,0.50,0.70
"""

import argparse, json, os, sys
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


def load_decode_weights(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    params = ckpt.get("params", ckpt)
    return {
        "proj_kernel":   params["proj_kernel"].float(),    # (768, 512)
        "proj_bias":     params["proj_bias"].float(),      # (512,)
        "unembed_kernel":params["unembed_kernel"].float(), # (512, V)
        "unembed_bias":  params["unembed_bias"].float(),   # (V,)
    }


def analyze_t(x_last, y_tokens, attn_mask, weights, device, batch_size=128):
    """
    x_last    : (N, L, 768)  last transformer block hidden state
    y_tokens  : (N, L) long
    attn_mask : (N, L) float
    weights   : dict of decode head tensors
    """
    N, L, d = x_last.shape
    x_flat  = x_last.reshape(-1, d)          # (N*L, 768)
    y_flat  = y_tokens.reshape(-1).long()    # (N*L,)
    m_flat  = attn_mask.reshape(-1).bool()   # (N*L,)

    xv = x_flat[m_flat].float()    # (M, 768)
    yv = y_flat[m_flat]            # (M,)
    M  = xv.shape[0]

    # Move decode weights to device
    proj_kernel    = weights["proj_kernel"].to(device)    # (768, 512)
    proj_bias      = weights["proj_bias"].to(device)      # (512,)
    unembed_kernel = weights["unembed_kernel"].to(device) # (512, V)
    unembed_bias   = weights["unembed_bias"].to(device)   # (V,)
    vocab = unembed_kernel.shape[1]

    top1_pred  = []
    true_rank  = []
    logit_top1 = []
    logit_true = []

    for bs in range(0, M, batch_size):
        be = min(bs + batch_size, M)
        xb = xv[bs:be].to(device)   # (B, 768)
        yb = yv[bs:be].to(device)   # (B,)

        # Full decode path
        hidden  = F.gelu(xb @ proj_kernel + proj_bias, approximate="tanh")  # (B, 512)
        logits  = hidden @ unembed_kernel + unembed_bias                     # (B, V)

        t1 = logits.argmax(dim=-1)                              # (B,)
        top1_pred.append(t1.cpu())
        logit_top1.append(logits.gather(1, t1.unsqueeze(1)).squeeze(1).cpu())
        logit_true.append(logits.gather(1, yb.unsqueeze(1)).squeeze(1).cpu())

        # Rank of true token (1-indexed)
        ranks = (logits > logits.gather(1, yb.unsqueeze(1))).sum(dim=-1) + 1
        true_rank.append(ranks.cpu())

    top1_pred  = torch.cat(top1_pred)
    true_rank  = torch.cat(true_rank)
    logit_top1 = torch.cat(logit_top1)
    logit_true = torch.cat(logit_true)
    yv_cpu     = yv.cpu()

    correct_mask  = (top1_pred == yv_cpu)
    n_total       = M
    n_correct     = correct_mask.sum().item()
    frac_correct  = n_correct / n_total

    wrong_ranks  = true_rank[~correct_mask]
    all_ranks    = true_rank
    logit_gap    = (logit_top1 - logit_true)

    def rank_hist(ranks, cutoffs=(1, 2, 3, 5, 10, 50, 100)):
        if len(ranks) == 0:
            return {}
        out = {}
        for c in cutoffs:
            out[f"rank_le_{c}"] = (ranks <= c).float().mean().item()
        out["mean_rank"]   = ranks.float().mean().item()
        out["median_rank"] = ranks.float().median().item()
        return out

    stats = {
        "n_total":         n_total,
        "n_correct":       n_correct,
        "frac_correct":    frac_correct,
        "true_rank_all":   rank_hist(all_ranks),
        "true_rank_wrong": rank_hist(wrong_ranks),
        "logit_gap_all_mean":   logit_gap.mean().item(),
        "logit_gap_wrong_mean": logit_gap[~correct_mask].mean().item() if (~correct_mask).any() else 0.0,
        "logit_true_mean":      logit_true.mean().item(),
        "logit_top1_mean":      logit_top1.mean().item(),
    }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--states_dir",  required=True)
    ap.add_argument("--output_dir",  required=True)
    ap.add_argument("--t_values",    default="0.10,0.20,0.30,0.50,0.70")
    ap.add_argument("--layer_idx",   type=int, default=-1,
                    help="Which layer_feats to use (-1 = last = L11)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[EXP-12] Loading decode weights from {args.checkpoint} ...")
    weights = load_decode_weights(args.checkpoint)
    print(f"  proj_kernel: {weights['proj_kernel'].shape}")
    print(f"  unembed_kernel: {weights['unembed_kernel'].shape}")

    t_vals = [float(x) for x in args.t_values.split(",")]
    all_results = {}

    for t_val in t_vals:
        path = os.path.join(args.states_dir, f"layer_states_t{t_val:.3f}.pt")
        print(f"\n[EXP-12] t={t_val:.3f} — loading {path} ...")
        data = torch.load(path, map_location="cpu", weights_only=False)

        layer_feats = data["layer_feats"]           # list of (N, L, 768)
        x_last      = layer_feats[args.layer_idx]   # (N, L, 768)
        y_tokens    = data["y_tokens"]              # (N, L)
        attn_mask   = data["attn_mask"]             # (N, L)

        print(f"  x_last shape: {x_last.shape}")

        stats = analyze_t(x_last, y_tokens, attn_mask, weights, device)

        print(f"  G(t)=frac_correct={stats['frac_correct']:.4f}  "
              f"mean_rank_all={stats['true_rank_all']['mean_rank']:.1f}  "
              f"mean_rank_wrong={stats['true_rank_wrong'].get('mean_rank', 0):.1f}")
        print(f"  rank<=2 (wrong): {stats['true_rank_wrong'].get('rank_le_2', 0):.4f}  "
              f"rank<=5 (wrong): {stats['true_rank_wrong'].get('rank_le_5', 0):.4f}  "
              f"rank<=10 (wrong): {stats['true_rank_wrong'].get('rank_le_10', 0):.4f}")

        all_results[str(t_val)] = stats

    out = os.path.join(args.output_dir, "residual_rank.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[EXP-12] Saved to {out}")


if __name__ == "__main__":
    main()
