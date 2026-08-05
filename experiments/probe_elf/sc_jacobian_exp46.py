#!/usr/bin/env python3
"""
EXP-46: SC Jacobian Analysis

Compute J_SC · δs_t = ∂v_θ(z_t, s_t, t)/∂s_t · (x̂_t_kd_cr - x̂_t_kd2)
and measure alignment with the recovery direction (x̂_t - z_t).

Method: forward-mode AD via torch.func.jvp (exact, O(D_z) memory).
Fallback: central finite differences with ε=1e-3.

Per-t summary across t ∈ {0.10, 0.20, 0.30, 0.50, 0.70, 0.90}.
N_SEQ=64 sequences, BATCH_SIZE=16, seed=42, 128 tokens.

Prediction (from EXP-44/45 findings):
  kd2:   J_SC · δs not aligned with recovery → SC conditioning adds noise
  kd_cr: J_SC · δs positively aligned with recovery → SC conditioning helps
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.model import ELF_B
from utils.sampling_utils import net_out_to_v_x

CHECKPOINTS = {
    "kd_cr":    "converted/elf_b-owt-kd-cr_torch.pt",
    "kd2":      "converted/elf_b-owt-kd2_torch.pt",
    "baseline": "converted/elf_b-owt-baseline_torch.pt",
}
T_VALS     = [0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
N_SEQ      = 64
BATCH_SIZE = 16
SEED       = 42
MAX_LENGTH = 128
T_EPS      = 0.05
OUT_DIR    = Path("results/exp46_sc_jacobian")

MODEL_CFG = dict(
    text_encoder_dim=512, max_length=MAX_LENGTH,
    num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
    vocab_size=32100, bottleneck_dim=128,
)


def load_model(name, device):
    ckpt = torch.load(REPO_ROOT / CHECKPOINTS[name], map_location="cpu", weights_only=False)
    m = ELF_B(**MODEL_CFG)
    m.load_state_dict(ckpt["params"], strict=False)
    return m.eval().to(device)


def _v_from_sc(model, z, t_batch, sc_input, device):
    """Velocity from model given explicit SC input. Returns v (B,L,512)."""
    sc_scale = torch.ones(z.shape[0], dtype=z.dtype, device=device)
    z_in = torch.cat([z, sc_input], dim=-1)
    out  = model(z_in, t_batch, deterministic=True, self_cond_cfg_scale=sc_scale)
    v, _ = net_out_to_v_x(out, z, t_batch, T_EPS)
    return v


def jvp_sc(model, z, t_batch, s0, tangent, device, eps=1e-3):
    """
    Compute J_SC · tangent using central finite differences.
    J_SC = ∂v_θ(z, s, t) / ∂s  evaluated at s=s0.
    Returns jvp (B, L, 512).
    """
    with torch.no_grad():
        v_plus  = _v_from_sc(model, z, t_batch, s0 + eps * tangent, device)
        v_minus = _v_from_sc(model, z, t_batch, s0 - eps * tangent, device)
    return (v_plus - v_minus) / (2 * eps)


def try_jvp_forwardad(model, z, t_batch, s0, tangent, device):
    """
    Try exact forward-mode AD via torch.func.jvp.
    Falls back to finite differences on failure.
    """
    try:
        from torch.func import jvp as func_jvp, functional_call
        params  = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

        def fn(sc_input):
            sc_scale = torch.ones(z.shape[0], dtype=z.dtype, device=device)
            z_in = torch.cat([z, sc_input], dim=-1)
            out  = functional_call(model, (params, buffers), (z_in, t_batch),
                                   kwargs={"deterministic": True,
                                           "self_cond_cfg_scale": sc_scale})
            v, _ = net_out_to_v_x(out, z, t_batch, T_EPS)
            return v

        _, jvp_val = func_jvp(fn, (s0,), (tangent,))
        return jvp_val
    except Exception:
        return jvp_sc(model, z, t_batch, s0, tangent, device)


@torch.no_grad()
def get_xpred(model, z, t_batch, sc_prev, device):
    """Run model forward and return (v, x̂_t)."""
    sc_scale = torch.ones(z.shape[0], dtype=z.dtype, device=device)
    z_in     = torch.cat([z, sc_prev], dim=-1)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out  = model(z_in, t_batch, deterministic=True, self_cond_cfg_scale=sc_scale)
    v, x = net_out_to_v_x(out, z, t_batch, T_EPS)
    return v, x


def run_t(model_kd2, model_kd_cr, z_batch, t_val, device):
    """
    At a single (z_batch, t), compute JVP alignment for both models.
    Returns dict with per-batch cosine alignment values.
    """
    B, L, D = z_batch.shape
    t_batch  = torch.full((B,), t_val, dtype=torch.float32, device=device)
    zeros_sc = torch.zeros_like(z_batch)

    # Get x̂_t from both models on same z_t
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _, x_kd2   = get_xpred(model_kd2,   z_batch, t_batch, zeros_sc, device)
        _, x_kd_cr = get_xpred(model_kd_cr, z_batch, t_batch, zeros_sc, device)

    x_kd2   = x_kd2.float()
    x_kd_cr = x_kd_cr.float()
    z_f     = z_batch.float()

    delta_s = (x_kd_cr - x_kd2).detach()   # (B, L, 512) tangent vector

    # Recovery directions
    recov_kd2   = (x_kd2   - z_f).detach()   # predicted clean - noisy state
    recov_kd_cr = (x_kd_cr - z_f).detach()

    # Normalise tangent (unit vector per batch element) for stable JVP
    delta_norm = delta_s / (delta_s.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8))

    results = {}
    for model_name, model, recov in [
        ("kd2",   model_kd2,   recov_kd2),
        ("kd_cr", model_kd_cr, recov_kd_cr),
    ]:
        # s0: use that model's own x̂_t as the SC base point
        s0 = (x_kd2 if model_name == "kd2" else x_kd_cr).detach()

        # JVP  (finite differences, dtype=float32 for numerical stability)
        z32  = z_f.clone()
        s0_32 = s0.float()
        dn_32 = delta_norm.float()
        t32  = t_batch.float()
        jvp_val = jvp_sc(model, z32, t32, s0_32, dn_32, device)  # (B,L,512)

        # Cosine alignment: per trajectory, flatten L×D
        jvp_flat   = jvp_val.reshape(B, -1)
        recov_flat = recov.reshape(B, -1)

        cos_align = F.cosine_similarity(jvp_flat, recov_flat, dim=-1)  # (B,)

        # δs magnitude vs recovery magnitude ratio
        delta_mag  = delta_s.norm(dim=(1, 2))   # (B,)
        recov_mag  = recov.norm(dim=(1, 2))      # (B,)
        jvp_mag    = jvp_val.norm(dim=(1, 2))    # (B,)

        results[model_name] = {
            "cos_align":       cos_align.cpu().float().tolist(),
            "delta_mag_mean":  delta_mag.mean().item(),
            "recov_mag_mean":  recov_mag.mean().item(),
            "jvp_mag_mean":    jvp_mag.mean().item(),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args   = parser.parse_args()
    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    model_kd2   = load_model("kd2",   device)
    model_kd_cr = load_model("kd_cr", device)

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    all_z = torch.randn(N_SEQ, MAX_LENGTH, 512, generator=gen, device=device)

    all_results = {}

    print(f"\n{'t':>6} | {'model':>8} | {'cos_align mean':>16} | {'cos_align>0 frac':>18} | {'jvp_mag':>10}")
    print("-" * 70)

    for t_val in T_VALS:
        t_res = {"kd2": {"cos_align": []}, "kd_cr": {"cos_align": []}}

        for b in range(0, N_SEQ, BATCH_SIZE):
            z_b   = all_z[b:b+BATCH_SIZE]
            batch_res = run_t(model_kd2, model_kd_cr, z_b, t_val, device)
            for mn in ["kd2", "kd_cr"]:
                t_res[mn]["cos_align"].extend(batch_res[mn]["cos_align"])
                for k in ["delta_mag_mean", "recov_mag_mean", "jvp_mag_mean"]:
                    t_res[mn].setdefault(k, []).append(batch_res[mn][k])

        for mn in ["kd2", "kd_cr"]:
            ca   = np.array(t_res[mn]["cos_align"])
            jmag = np.mean(t_res[mn]["jvp_mag_mean"])
            frac_pos = (ca > 0).mean()
            print(f"{t_val:6.2f} | {mn:>8} | {ca.mean():+16.4f} | {frac_pos:18.3f} | {jmag:10.4f}")
            t_res[mn]["cos_align_mean"] = float(ca.mean())
            t_res[mn]["cos_align_std"]  = float(ca.std())
            t_res[mn]["frac_positive"]  = float(frac_pos)
            t_res[mn]["jvp_mag_mean"]   = float(jmag)

        all_results[str(t_val)] = t_res

    print(f"\n{'='*70}")
    print("VERDICT:")
    ca_kd2   = np.mean([all_results[str(t)]["kd2"]["cos_align_mean"]   for t in T_VALS])
    ca_kd_cr = np.mean([all_results[str(t)]["kd_cr"]["cos_align_mean"] for t in T_VALS])
    print(f"  kd2   mean cos_align across t = {ca_kd2:+.4f}")
    print(f"  kd_cr mean cos_align across t = {ca_kd_cr:+.4f}")
    if ca_kd_cr > ca_kd2 + 0.05:
        print("  >> kd_cr J_SC more recovery-aligned than kd2 (SC compatibility = J_SC direction)")
    elif ca_kd2 > 0 and ca_kd_cr > 0:
        print("  >> Both models' J_SC aligned with recovery; kd2 SC issue may be signal magnitude")
    else:
        print("  >> SC Jacobian does not cleanly explain kd2 vs kd_cr difference")
    print(f"{'='*70}")

    out_path = OUT_DIR / "jacobian.json"
    with open(out_path, "w") as f:
        json.dump({"config": {"N_SEQ": N_SEQ, "T_VALS": T_VALS,
                               "SEED": SEED, "MAX_LENGTH": MAX_LENGTH},
                   "results": all_results}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
