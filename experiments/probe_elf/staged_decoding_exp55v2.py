#!/usr/bin/env python3
"""
EXP-55v2: Two-pass staged decoding with SAME noise for pass 2

EXP-55 used different z0 seed for pass 2 (SEED+1000), causing trajectory
inconsistency. This caused confidence-based staged arms to FAIL despite
left50 working for kd2 (I=−121).

Hypothesis: with the SAME z0 for pass 2, confidence-based arms will benefit
because pass 2 traces the same trajectory as pass 1 but with committed
positions as stable conditioning. The uncommitted positions evolve on the
same trajectory they would have taken anyway, but with anchors.

Design:
  pass 1: ODE-32 on z0 (seed=42) → z_final, x_pred_final
  commitment: pick positions by left50 or confidence from x_pred_final
  pass 2: ODE-32 on SAME z0 (seed=42) with cond_seq=x_pred_p1, cond_mask=committed

If same noise fixes confidence arms → trajectory inconsistency was the root cause
If confidence arms still fail → scattered commitment is inherently problematic

Arms: standard, staged_left50, staged_conf90, staged_conf80, staged_conf70
Models: kd_cr, kd2
N=256, seed=42

Usage:
    cd models/ELF-torch
    conda run -n elf python3 experiments/probe_elf/staged_decoding_exp55v2.py --device cuda:0
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.model import ELF_B
from utils.sampling_utils import restore_cond, _ode_step, get_sampling_steps

CHECKPOINTS = {
    "kd_cr": "converted/elf_b-owt-kd-cr_torch.pt",
    "kd2":   "converted/elf_b-owt-kd2_torch.pt",
}
N_SEQ      = 256
N_STEPS    = 32
SEED       = 42
MAX_LENGTH = 128
BATCH_SIZE = 16
SCCFG      = 1.0
OUT_DIR    = Path("results/exp55v2_staged_samenoise")

CONF_THRESHOLDS = {"staged_conf90": 0.90, "staged_conf80": 0.80, "staged_conf70": 0.70}

MODEL_CFG = dict(
    text_encoder_dim=512, max_length=MAX_LENGTH,
    num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
    vocab_size=32100, bottleneck_dim=128,
)


class _Cfg:
    t_eps = 0.05
    self_cond_prob = 1.0
    num_self_cond_cfg_tokens = 4
    denoiser_noise_scale = 1.0
    use_bf16 = True


@torch.no_grad()
def generate_ode(model, z0, t_steps, sccfg, device):
    """Standard ODE-32. Returns (z_final, x_pred_final)."""
    cfg = _Cfg()
    B, L, D = z0.shape
    cond_seq  = torch.zeros_like(z0)
    cond_mask = torch.zeros(B, L, dtype=z0.dtype, device=device)
    z      = restore_cond(z0.clone(), cond_seq, cond_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
    use_bf16 = cfg.use_bf16 and device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(t_steps.shape[0] - 2):
            z, x_pred = _ode_step(z=z, t=t_steps[i].item(), t_next=t_steps[i+1].item(),
                                  x_pred_prev=x_pred, model=model, config=cfg,
                                  cfg_scale=1.0, self_cond_cfg_scale=sccfg,
                                  cond_seq=cond_seq, cond_seq_mask=cond_mask)
        z, x_pred = _ode_step(z=z, t=t_steps[-2].item(), t_next=t_steps[-1].item(),
                               x_pred_prev=x_pred, model=model, config=cfg,
                               cfg_scale=1.0, self_cond_cfg_scale=sccfg,
                               cond_seq=cond_seq, cond_seq_mask=cond_mask)
    return z, x_pred


@torch.no_grad()
def get_top1_prob(x_pred, model, device):
    B = x_pred.shape[0]
    z_in = torch.cat([x_pred.detach(), torch.zeros_like(x_pred)], dim=-1)
    t_dec = torch.ones(B, dtype=x_pred.dtype, device=device)
    sc    = torch.ones(B, dtype=x_pred.dtype, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _, logits, _ = model(z_in, t_dec, deterministic=True,
                             self_cond_cfg_scale=sc, decoder_step_active=True)
    return F.softmax(logits.float(), dim=-1).max(dim=-1).values


@torch.no_grad()
def generate_staged(model, z0, t_steps, cond_seq, cond_mask_bl, sccfg, device):
    """ODE-32 pass 2 with SAME z0 (no fresh noise), committed positions pinned."""
    cfg = _Cfg()
    z      = restore_cond(z0.clone(), cond_seq, cond_mask_bl)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask_bl)
    use_bf16 = cfg.use_bf16 and device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(t_steps.shape[0] - 2):
            z, x_pred = _ode_step(z=z, t=t_steps[i].item(), t_next=t_steps[i+1].item(),
                                  x_pred_prev=x_pred, model=model, config=cfg,
                                  cfg_scale=1.0, self_cond_cfg_scale=sccfg,
                                  cond_seq=cond_seq, cond_seq_mask=cond_mask_bl)
        z, _ = _ode_step(z=z, t=t_steps[-2].item(), t_next=t_steps[-1].item(),
                         x_pred_prev=x_pred, model=model, config=cfg,
                         cfg_scale=1.0, self_cond_cfg_scale=sccfg,
                         cond_seq=cond_seq, cond_seq_mask=cond_mask_bl)
    return z


@torch.no_grad()
def decode_z(z, model, device):
    B = z.shape[0]
    z_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
    t1 = torch.ones(B, dtype=z.dtype, device=device)
    sc = torch.ones(B, dtype=z.dtype, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _, logits, _ = model(z_in, t1, deterministic=True, self_cond_cfg_scale=sc,
                             decoder_step_active=True)
    return logits.argmax(dim=-1)


def ids_to_text(ids, tokenizer):
    eos = tokenizer.eos_token_id
    texts = []
    for row in ids.tolist():
        try: row = row[:row.index(eos)]
        except ValueError: pass
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return texts


def compute_ppl(texts, ppl_model, ppl_tok, device, max_length=256):
    if not texts: return float("nan")
    enc = ppl_tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length, return_attention_mask=True)
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    total_nll = total_tok = 0
    with torch.no_grad():
        for i in range(0, ids.shape[0], 8):
            ids_b, msk_b = ids[i:i+8], mask[i:i+8]
            out = ppl_model(input_ids=ids_b, attention_mask=msk_b)
            logits = out.logits[:, :-1, :].float()
            tgts, msk_s = ids_b[:, 1:], msk_b[:, 1:].bool()
            nll = -(F.log_softmax(logits, dim=-1).gather(-1, tgts.unsqueeze(-1)).squeeze(-1)
                    * msk_s.float()).sum().item()
            total_nll += nll
            total_tok += msk_s.sum().item()
    return math.exp(total_nll / total_tok) if total_tok else float("nan")


def run_ckpt(ckpt_name, device, ppl_model, ppl_tok, elf_tok):
    print(f"\n{'='*60}\nCheckpoint: {ckpt_name}\n{'='*60}")
    ckpt = torch.load(REPO_ROOT / CHECKPOINTS[ckpt_name], map_location="cpu", weights_only=False)
    model = ELF_B(**MODEL_CFG)
    model.load_state_dict(ckpt["params"], strict=False)
    model.eval().to(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    all_z0 = torch.randn(N_SEQ, MAX_LENGTH, 512, generator=gen, device=device)
    t_steps = get_sampling_steps(N_STEPS, time_schedule="uniform", device=device)

    # Pass 1: generate all sequences
    print("  Running pass 1 (ODE-32, all seqs)...")
    all_xp_p1 = []
    texts_standard = []
    for b in range(0, N_SEQ, BATCH_SIZE):
        z_p1_b, xp_p1_b = generate_ode(model, all_z0[b:b+BATCH_SIZE], t_steps, SCCFG, device)
        all_xp_p1.append(xp_p1_b.detach())
        ids = decode_z(z_p1_b, model, device)
        texts_standard.extend(ids_to_text(ids.cpu(), elf_tok))

    ppl_standard = compute_ppl(texts_standard, ppl_model, ppl_tok, device)
    print(f"  {'standard':<30} PPL = {ppl_standard:.2f}")
    results = {"standard": {"ppl": ppl_standard, "I": 0.0}}

    arm_specs = [("staged_left50", None)] + [(k, v) for k, v in CONF_THRESHOLDS.items()]

    for arm_name, conf_thresh in arm_specs:
        texts = []
        commit_fracs = []
        for b in range(0, N_SEQ, BATCH_SIZE):
            b_end = min(b + BATCH_SIZE, N_SEQ)
            xp_p1_b = all_xp_p1[b // BATCH_SIZE]
            z0_b = all_z0[b:b_end]   # SAME z0 as pass 1

            B_b, L, D = xp_p1_b.shape
            if arm_name == "staged_left50":
                cond_mask = torch.zeros(B_b, L, dtype=xp_p1_b.dtype, device=device)
                cond_mask[:, :L // 2] = 1.0
            else:
                conf = get_top1_prob(xp_p1_b, model, device)
                cond_mask = (conf > conf_thresh).float()

            commit_fracs.append(cond_mask.mean().item())
            z_out = generate_staged(model, z0_b, t_steps, xp_p1_b, cond_mask, SCCFG, device)
            ids = decode_z(z_out, model, device)
            texts.extend(ids_to_text(ids.cpu(), elf_tok))

        ppl = compute_ppl(texts, ppl_model, ppl_tok, device)
        avg_commit = sum(commit_fracs) / len(commit_fracs)
        I = ppl - ppl_standard
        print(f"  {arm_name:<30} PPL = {ppl:.2f}  I={I:+.2f}  commit={avg_commit*100:.1f}%")
        results[arm_name] = {"ppl": ppl, "I": I, "commit_frac": avg_commit}

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading GPT-2 Large PPL model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer
    ppl_tok = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tok.pad_token is None:
        ppl_tok.pad_token = ppl_tok.eos_token
        ppl_tok.pad_token_id = ppl_tok.eos_token_id
    ppl_model = (AutoModelForCausalLM.from_pretrained("gpt2-large", dtype=torch.bfloat16)
                 .to(device).eval())
    elf_tok = T5Tokenizer.from_pretrained("t5-small")

    ckpts_to_run = [args.ckpt] if args.ckpt else list(CHECKPOINTS.keys())
    all_results = {}
    for ck in ckpts_to_run:
        all_results[ck] = run_ckpt(ck, device, ppl_model, ppl_tok, elf_tok)

    out = OUT_DIR / "results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out}")

    print("\n=== SUMMARY ===")
    print("(EXP-55 fresh-noise reference for comparison)")
    print("  kd_cr left50=+109, conf all hurt; kd2 left50=-121, conf all hurt")
    for ck, r in all_results.items():
        print(f"\n{ck} (same noise):")
        for arm, v in r.items():
            cfrac = f"  commit={v.get('commit_frac',0)*100:.0f}%" if arm != "standard" else ""
            print(f"  {arm:<30} PPL={v['ppl']:.1f}  I={v['I']:+.1f}{cfrac}")


if __name__ == "__main__":
    main()
