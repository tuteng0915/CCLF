#!/usr/bin/env python3
"""
EXP-56: Within-ODE progressive commitment (Strategy C)

Hypothesis: partway through a single ODE-32 run, high-confidence positions
can be committed (frozen at their current x_pred estimate) for the remaining
steps, providing stable anchors for the uncommitted positions to refine against.

Key mechanism: At the first ODE step where t_next >= 0.5 (the critical boundary
from EXP-42 CKA / EXP-54c gate), compute decode-branch confidence. Commit
positions above threshold: set cond_seq=x_pred, cond_mask=1, z[committed]=x_pred.
For all remaining steps, v=0 keeps committed z frozen; x_pred at committed
positions is pinned to cond_seq. Same FLOPs as standard ODE-32 + 1 extra forward.

t=0.5 boundary motivated by:
- EXP-42 CKA: B08-B11 diverge sharply at t=0.5 (CKA 0.896→0.427)
- EXP-54c: SC_T_MIN=0.5 essential; tmin=0.0→+1084 PPL catastrophic

Arms:
  standard       : ODE-32, no commitment (reference)
  prog_t05_c90   : commit at t_next>=0.5, threshold=0.90
  prog_t05_c80   : commit at t_next>=0.5, threshold=0.80
  prog_t05_c70   : commit at t_next>=0.5, threshold=0.70

Models: kd_cr, kd2

Usage:
    cd models/ELF-torch
    conda run -n elf python3 experiments/probe_elf/progressive_commit_exp56.py --device cuda:1
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
SCCFG      = 1.0    # consistent with h10 SC baseline (EXP-54b)
COMMIT_T   = 0.5    # commit at first t_next >= this
OUT_DIR    = Path("results/exp56_progressive")

MODEL_CFG = dict(
    text_encoder_dim=512, max_length=MAX_LENGTH,
    num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
    vocab_size=32100, bottleneck_dim=128,
)

CONF_THRESHOLDS = {
    "prog_t05_c90": 0.90,
    "prog_t05_c80": 0.80,
    "prog_t05_c70": 0.70,
}


class _Cfg:
    t_eps = 0.05
    self_cond_prob = 1.0
    num_self_cond_cfg_tokens = 4
    denoiser_noise_scale = 1.0
    use_bf16 = True


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_top1_prob(x_pred, model, device):
    """Decode-branch top-1 softmax probability per position. Returns (B, L)."""
    B = x_pred.shape[0]
    z_in = torch.cat([x_pred.detach(), torch.zeros_like(x_pred)], dim=-1)
    t_dec = torch.ones(B, dtype=x_pred.dtype, device=device)
    sc    = torch.ones(B, dtype=x_pred.dtype, device=device)
    use_bf16 = device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        _, logits, _ = model(z_in, t_dec, deterministic=True,
                             self_cond_cfg_scale=sc, decoder_step_active=True)
    return F.softmax(logits.float(), dim=-1).max(dim=-1).values  # (B, L)


@torch.no_grad()
def generate_progressive(model, z0, t_steps, conf_thresh, device, verbose=False):
    """ODE-32 with mid-ODE commitment at the first t_next >= COMMIT_T.

    conf_thresh=None → standard ODE (no commitment)
    conf_thresh=float → commit positions with decode-branch top-1 prob > thresh

    Returns (z_final, commit_info) where commit_info is None for standard arm.
    """
    cfg = _Cfg()
    B, L, D = z0.shape
    cond_seq  = torch.zeros_like(z0)
    cond_mask = torch.zeros(B, L, dtype=z0.dtype, device=device)

    z      = restore_cond(z0.clone(), cond_seq, cond_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)

    committed = False
    commit_info = None
    use_bf16 = cfg.use_bf16 and device.type == "cuda"

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(t_steps.shape[0] - 2):
            t      = t_steps[i].item()
            t_next = t_steps[i + 1].item()

            step_kw = dict(
                model=model, config=cfg, cfg_scale=1.0, self_cond_cfg_scale=SCCFG,
                cond_seq=cond_seq, cond_seq_mask=cond_mask,
            )
            z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kw)

            # First crossing of COMMIT_T: compute confidence and freeze positions
            if conf_thresh is not None and not committed and t_next >= COMMIT_T:
                conf = get_top1_prob(x_pred, model, device)   # (B, L)
                new_committed = (conf > conf_thresh)           # (B, L) bool
                commit_frac = new_committed.float().mean().item()

                # Update cond_seq and cond_mask for remaining steps
                cond_seq  = x_pred.detach().clone()                    # (B, L, D)
                cond_mask = new_committed.float()                      # (B, L)
                # Also pin z at committed positions to x_pred (jump to clean estimate)
                z = z.clone()
                z[new_committed] = x_pred.detach()[new_committed]

                committed = True
                commit_info = {"t_commit": t_next, "commit_frac": commit_frac,
                               "conf_thresh": conf_thresh}
                if verbose:
                    print(f"    Committed {commit_frac*100:.1f}% positions "
                          f"at t_next={t_next:.3f} (thresh={conf_thresh})")

        # Last step
        step_kw = dict(
            model=model, config=cfg, cfg_scale=1.0, self_cond_cfg_scale=SCCFG,
            cond_seq=cond_seq, cond_seq_mask=cond_mask,
        )
        t      = t_steps[-2].item()
        t_next = t_steps[-1].item()
        z, _ = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kw)

    return z, commit_info


@torch.no_grad()
def decode_z(z, model, device):
    B = z.shape[0]
    z_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
    t1 = torch.ones(B, dtype=z.dtype, device=device)
    sc = torch.ones(B, dtype=z.dtype, device=device)
    use_bf16 = device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        _, logits, _ = model(z_in, t1, deterministic=True, self_cond_cfg_scale=sc,
                             decoder_step_active=True)
    return logits.argmax(dim=-1)


def ids_to_text(ids, tokenizer):
    eos = tokenizer.eos_token_id
    texts = []
    for row in ids.tolist():
        try:
            row = row[:row.index(eos)]
        except ValueError:
            pass
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return texts


def compute_ppl(texts, ppl_model, ppl_tok, device, max_length=256):
    if not texts:
        return float("nan")
    enc = ppl_tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length, return_attention_mask=True)
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    total_nll = total_tok = 0
    with torch.no_grad():
        for i in range(0, ids.shape[0], 8):
            ids_b, msk_b = ids[i:i+8], mask[i:i+8]
            out = ppl_model(input_ids=ids_b, attention_mask=msk_b)
            logits = out.logits[:, :-1, :].float()
            tgts   = ids_b[:, 1:]
            msk_s  = msk_b[:, 1:].bool()
            nll = -(F.log_softmax(logits, dim=-1)
                    .gather(-1, tgts.unsqueeze(-1)).squeeze(-1)
                    * msk_s.float()).sum().item()
            total_nll += nll
            total_tok += msk_s.sum().item()
    return math.exp(total_nll / total_tok) if total_tok else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_ckpt(ckpt_name, device, ppl_model, ppl_tok, elf_tok):
    print(f"\n{'='*60}\nCheckpoint: {ckpt_name}\n{'='*60}")

    ckpt_path = REPO_ROOT / CHECKPOINTS[ckpt_name]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ELF_B(**MODEL_CFG)
    model.load_state_dict(ckpt["params"], strict=False)
    model.eval().to(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    all_z0 = torch.randn(N_SEQ, MAX_LENGTH, 512, generator=gen, device=device)
    t_steps = get_sampling_steps(N_STEPS, time_schedule="uniform", device=device)

    results = {}
    arm_specs = [("standard", None)] + [(k, v) for k, v in CONF_THRESHOLDS.items()]

    for arm_name, conf_thresh in arm_specs:
        texts = []
        commit_fracs = []
        for b in range(0, N_SEQ, BATCH_SIZE):
            z0_b = all_z0[b:b+BATCH_SIZE]
            z_out, cinfo = generate_progressive(
                model, z0_b, t_steps, conf_thresh, device,
                verbose=(b == 0 and conf_thresh is not None),
            )
            ids = decode_z(z_out, model, device)
            texts.extend(ids_to_text(ids.cpu(), elf_tok))
            if cinfo is not None:
                commit_fracs.append(cinfo["commit_frac"])

        ppl = compute_ppl(texts, ppl_model, ppl_tok, device)
        avg_commit = sum(commit_fracs) / len(commit_fracs) if commit_fracs else 0.0
        I = ppl - results.get("standard", {}).get("ppl", ppl)
        if arm_name == "standard":
            I = 0.0

        cfrac_str = f"  commit={avg_commit*100:.1f}%" if conf_thresh is not None else ""
        print(f"  {arm_name:<30} PPL = {ppl:.2f}  I={I:+.2f}{cfrac_str}")
        results[arm_name] = {"ppl": ppl, "I": I, "commit_frac": avg_commit}

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--ckpt", default=None, help="single checkpoint name to run")
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
    for ck, r in all_results.items():
        print(f"\n{ck}:")
        for arm, v in r.items():
            cfrac = f"  commit={v.get('commit_frac',0)*100:.0f}%" if arm != "standard" else ""
            print(f"  {arm:<30} PPL={v['ppl']:.1f}  I={v['I']:+.1f}{cfrac}")


if __name__ == "__main__":
    main()
