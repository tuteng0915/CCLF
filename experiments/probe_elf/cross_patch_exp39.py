"""
EXP-39: Decode Head Cross-Patch

Causal test: does swapping the decode head alone explain the oracle accuracy gap?

Protocol:
  For each source checkpoint's h_L11 (last block output), apply each target checkpoint's decode head.
  The 3×3 matrix of (backbone_src × head_tgt) oracle accuracy at each t isolates
  whether the gap is in the backbone representation or the decode interface.

  backbone_src = {baseline, kd_cr, kd2} (which exp07b_v2 layer_feats to load)
  head_tgt     = {baseline, kd_cr, kd2} (whose proj_kernel/unembed_kernel to use)

Reuses: results/exp07b_v2_{...}/layer_states_t*.pt  (pre-collected, no new GPU forward)

Output: results/exp39_cross_patch/cross_patch.json
  {src_ckpt: {tgt_ckpt: {t_str: {top1, top5, mrr}, ...}, ...}, ...}

Usage (from ELF-torch root):
  python experiments/probe_elf/cross_patch_exp39.py
"""

import json
import os

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


def load_decode_head(ckpt_path: str) -> dict:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    p = raw["params"]
    return {
        "proj_kernel":    p["proj_kernel"].float(),
        "proj_bias":      p["proj_bias"].float(),
        "unembed_kernel": p["unembed_kernel"].float(),
        "unembed_bias":   p["unembed_bias"].float(),
    }


BATCH_SIZE = 16


def find_state_file(data_dir: str, t: float) -> str | None:
    for ext in [f"t{t:.3f}", f"t{t:.2f}", f"t{t:.1f}"]:
        p = os.path.join(data_dir, f"layer_states_{ext}.pt")
        if os.path.exists(p):
            return p
    for fn in os.listdir(data_dir):
        if fn.startswith("layer_states"):
            tv = float(fn.replace("layer_states_t", "").replace(".pt", ""))
            if abs(tv - t) < 1e-4:
                return os.path.join(data_dir, fn)
    return None


@torch.no_grad()
def compute_metrics(h_L11: torch.Tensor, head: dict, y: torch.Tensor, mask: torch.Tensor) -> dict:
    """Batched computation to avoid OOM. h_L11: [B, L, 768]"""
    B = h_L11.shape[0]
    top1_sum = top5_sum = mrr_sum = 0.0
    n_valid = 0

    for start in range(0, B, BATCH_SIZE):
        end = min(start + BATCH_SIZE, B)
        h_b    = h_L11[start:end].float()
        y_b    = y[start:end]
        mask_b = mask[start:end].bool()

        hidden = F.gelu(h_b @ head["proj_kernel"] + head["proj_bias"], approximate="tanh")
        logits = hidden @ head["unembed_kernel"] + head["unembed_bias"]

        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_y      = y_b.reshape(-1)
        flat_mask   = mask_b.reshape(-1)
        vl = flat_logits[flat_mask]
        vy = flat_y[flat_mask]
        N  = vy.shape[0]
        if N == 0:
            continue

        top1_sum += (vl.argmax(-1) == vy).float().sum().item()
        top5_sum += (vl.topk(5, -1).indices == vy.unsqueeze(1)).any(1).float().sum().item()
        correct_logit = vl[torch.arange(N), vy].unsqueeze(1)
        ranks = (vl > correct_logit).sum(-1).float() + 1.0
        mrr_sum += (1.0 / ranks).sum().item()
        n_valid += N
        del hidden, logits, vl

    if n_valid == 0:
        return {"top1": 0.0, "top5": 0.0, "mrr": 0.0}
    return {
        "top1": round(top1_sum / n_valid, 6),
        "top5": round(top5_sum / n_valid, 6),
        "mrr":  round(mrr_sum  / n_valid, 6),
    }


def main():
    os.makedirs("results/exp39_cross_patch", exist_ok=True)

    # Pre-load all decode heads
    heads = {name: load_decode_head(cfg["ckpt"]) for name, cfg in CHECKPOINTS.items()}

    results = {}

    for src_name, src_cfg in CHECKPOINTS.items():
        results[src_name] = {}
        for tgt_name in CHECKPOINTS:
            results[src_name][tgt_name] = {}

        for t in T_VALUES:
            fname = find_state_file(src_cfg["data"], t)
            if fname is None:
                print(f"WARNING: no data for {src_name} t={t}")
                continue

            data = torch.load(fname, map_location="cpu", weights_only=False)
            h_L11 = data["layer_feats"][-1].float()  # [B, L, 768]
            y_tokens  = data["y_tokens"]
            attn_mask = data["attn_mask"]

            t_str = f"t{t:.3f}"
            for tgt_name, head in heads.items():
                metrics = compute_metrics(h_L11, head, y_tokens, attn_mask)
                results[src_name][tgt_name][t_str] = metrics
                print(f"  backbone={src_name:<8} head={tgt_name:<8} t={t:.2f}  "
                      f"top1={metrics['top1']:.4f}")

    out_path = "results/exp39_cross_patch/cross_patch.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Print 3×3 matrix at t=0.50 and t=1.00
    for t_show in ["t0.500", "t1.000"]:
        print(f"\n--- Oracle accuracy 3×3 matrix at {t_show} ---")
        print(f"{'':12}", end="")
        for tgt in CHECKPOINTS:
            print(f"head={tgt:<10}", end="")
        print()
        for src in CHECKPOINTS:
            print(f"bb={src:<10}", end="")
            for tgt in CHECKPOINTS:
                val = results[src][tgt].get(t_show, {}).get("top1", float("nan"))
                print(f"  {val:.4f}    ", end="")
            print()


if __name__ == "__main__":
    main()
