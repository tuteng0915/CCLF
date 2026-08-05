"""
EXP-38: ELF Logit Lens

Apply each checkpoint's own decode head (GELU(h @ proj_kernel + proj_bias) @ unembed_kernel + unembed_bias)
to the residual stream at every transformer block depth, using the pre-collected exp07b_v2 layer states.

Key question: at what depth does the decode head first "see" the correct token?
Does kd_cr vs baseline show different depth profiles?

Reuses: results/exp07b_v2_{baseline,kd_cr,kd2}/layer_states_t*.pt (no new GPU forward passes needed)

Output: results/exp38_logit_lens/logit_lens.json
  {checkpoint: {t_str: {layer_i: {top1, top5, mrr, entropy}, ...}, ...}, ...}

Usage (from ELF-torch root):
  python experiments/probe_elf/logit_lens_exp38.py
"""

import json
import os

import torch
import torch.nn.functional as F

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--device", default="cuda:0")
_args, _ = _parser.parse_known_args()
DEVICE = torch.device(_args.device if torch.cuda.is_available() else "cpu")

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


def load_decode_head(ckpt_path: str) -> dict:
    """Load decode head weights from checkpoint."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    p = raw["params"]
    return {
        "proj_kernel":    p["proj_kernel"].float().to(DEVICE),
        "proj_bias":      p["proj_bias"].float().to(DEVICE),
        "unembed_kernel": p["unembed_kernel"].float().to(DEVICE),
        "unembed_bias":   p["unembed_bias"].float().to(DEVICE),
    }


BATCH_SIZE = 16  # process this many sequences at once to cap logit tensor at ~2 GB


@torch.no_grad()
def compute_lens_metrics(h: torch.Tensor, head: dict,
                         y: torch.Tensor, mask: torch.Tensor) -> dict:
    """Compute top-1, top-5, MRR, entropy in batches to avoid OOM.
    h: [B, L, 768], y: [B, L], mask: [B, L]
    """
    B = h.shape[0]
    top1_sum = top5_sum = mrr_sum = ent_sum = 0.0
    n_valid = 0

    for start in range(0, B, BATCH_SIZE):
        end = min(start + BATCH_SIZE, B)
        h_b    = h[start:end].float().to(DEVICE)
        y_b    = y[start:end].to(DEVICE)
        mask_b = mask[start:end].bool().to(DEVICE)

        # Compute logits for this batch
        hidden  = F.gelu(h_b @ head["proj_kernel"] + head["proj_bias"], approximate="tanh")
        logits  = (hidden @ head["unembed_kernel"] + head["unembed_bias"])  # [bs, L, V]

        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_y      = y_b.reshape(-1)
        flat_mask   = mask_b.reshape(-1)
        vl = flat_logits[flat_mask]  # [N, V]
        vy = flat_y[flat_mask]
        N  = vy.shape[0]
        if N == 0:
            continue

        top1_sum += (vl.argmax(-1) == vy).float().sum().item()
        top5_sum += (vl.topk(5, -1).indices == vy.unsqueeze(1)).any(1).float().sum().item()

        correct_logit = vl[torch.arange(N, device=DEVICE), vy].unsqueeze(1)  # [N, 1]
        ranks = (vl > correct_logit).sum(-1).float() + 1.0
        mrr_sum += (1.0 / ranks).sum().item()

        # Entropy
        log_p = F.log_softmax(vl, dim=-1)
        ent_sum += -(log_p.exp() * log_p).sum(-1).sum().item()

        n_valid += N
        del hidden, logits, flat_logits, vl

    if n_valid == 0:
        return {"top1": 0.0, "top5": 0.0, "mrr": 0.0, "entropy": 0.0, "n": 0}

    return {
        "top1":    round(top1_sum / n_valid, 6),
        "top5":    round(top5_sum / n_valid, 6),
        "mrr":     round(mrr_sum  / n_valid, 6),
        "entropy": round(ent_sum  / n_valid, 4),
        "n":       n_valid,
    }


def main():
    os.makedirs("results/exp38_logit_lens", exist_ok=True)
    results = {}

    for ckpt_name, cfg in CHECKPOINTS.items():
        print(f"\n=== {ckpt_name} ===")
        head = load_decode_head(cfg["ckpt"])
        results[ckpt_name] = {}

        for t in T_VALUES:
            t_str = f"t{t:.3f}"
            fname = os.path.join(cfg["data"], f"layer_states_t{t:.3f}.pt")
            if not os.path.exists(fname):
                # Try alternate format t0.10 vs t0.100
                fname2 = os.path.join(cfg["data"], f"layer_states_t{t:.2f}.pt")
                if os.path.exists(fname2):
                    fname = fname2
                else:
                    # Try the stored filenames
                    avail = [f for f in os.listdir(cfg["data"]) if f.startswith("layer_states")]
                    for a in avail:
                        tv = float(a.replace("layer_states_t","").replace(".pt",""))
                        if abs(tv - t) < 1e-4:
                            fname = os.path.join(cfg["data"], a)
                            break
                    else:
                        print(f"  WARNING: no file for t={t}, skipping")
                        continue

            data = torch.load(fname, map_location="cpu", weights_only=False)
            layer_feats = data["layer_feats"]   # list of 12 tensors, each [B, L, 768]
            y_tokens    = data["y_tokens"]      # [B, L]
            attn_mask   = data["attn_mask"]     # [B, L]
            depth = len(layer_feats)

            results[ckpt_name][t_str] = {}
            for i in range(depth):
                h = layer_feats[i]  # [B, L, 768]
                metrics = compute_lens_metrics(h, head, y_tokens, attn_mask)
                results[ckpt_name][t_str][f"block_{i:02d}"] = metrics
                print(f"  t={t:.2f} block_{i:02d}: top1={metrics['top1']:.4f}  "
                      f"mrr={metrics['mrr']:.4f}  H={metrics['entropy']:.2f}")

            # Also store t value for reference
            results[ckpt_name][t_str]["_t"] = t

    out_path = "results/exp38_logit_lens/logit_lens.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Print summary table: for each checkpoint, at each t, which layer first hits 90% of L11 accuracy?
    print("\n--- Summary: layer where top1 first reaches 90% of L11 value ---")
    for ckpt_name in results:
        print(f"\n{ckpt_name}:")
        for t_str, layers in results[ckpt_name].items():
            t = layers.get("_t", "?")
            l11_top1 = layers.get("block_11", {}).get("top1", 0)
            threshold = 0.9 * l11_top1
            for i in range(12):
                key = f"block_{i:02d}"
                if layers.get(key, {}).get("top1", 0) >= threshold:
                    print(f"  t={t:.2f}: first 90% at block_{i:02d} "
                          f"(top1={layers[key]['top1']:.4f}, L11={l11_top1:.4f})")
                    break


if __name__ == "__main__":
    main()
