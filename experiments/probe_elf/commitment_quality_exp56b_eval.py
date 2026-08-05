#!/usr/bin/env python3
"""
EXP-56b Quality Evaluation: Is I=−186 PPL hacking?

Checks whether prog_t30_c70's PPL improvement is real or an artifact of
generating low-information but grammatically safe text.

Metrics:
  1. MAUVE   — KL divergence between generated and real OWT distributions
  2. Distinct-1/2 — fraction of unique unigrams/bigrams (lexical diversity)
  3. Committed token POS — what fraction of committed positions are function words?
     Tests: does prog_t30 primarily commit high-frequency function words (trivial),
     or does it commit a diverse set of content+function words (non-trivial)?

Arms:
  standard          — ODE-32, no commitment
  prog_t30_c70      — commit at t=0.30, thresh=0.70 (kd2 best, I=−186)
  prog_t40_c70      — commit at t=0.40, thresh=0.70 (kd_cr best, I=−175)

Checkpoints: kd_cr, kd2

Usage:
    cd models/ELF-torch
    conda run -n elf python3 experiments/probe_elf/commitment_quality_exp56b_eval.py --device cuda:0
"""

import argparse
import json
import math
import sys
from collections import Counter
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
CONF_THRESH = 0.70
OUT_DIR     = Path("results/exp56b_quality_eval")

MODEL_CFG = dict(
    text_encoder_dim=512, max_length=MAX_LENGTH,
    num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
    vocab_size=32100, bottleneck_dim=128,
)

# Penn Treebank POS tags for function words
FUNCTION_POS = {
    "DT", "IN", "CC", "TO", "MD", "WDT", "WP", "WP$", "WRB",
    "EX", "PRP", "PRP$", "RP", "UH", "PDT",
}

try:
    import nltk
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk import pos_tag, word_tokenize
    HAS_NLTK = True
except Exception:
    HAS_NLTK = False
    print("[WARNING] NLTK not available; POS analysis will be skipped")

try:
    from mauve import compute_mauve
    HAS_MAUVE = True
except ImportError:
    HAS_MAUVE = False
    print("[WARNING] mauve-text not installed; MAUVE will be skipped")
    print("          Install with: pip install mauve-text")


class _Cfg:
    t_eps = 0.05
    self_cond_prob = 1.0
    num_self_cond_cfg_tokens = 4
    denoiser_noise_scale = 1.0
    use_bf16 = True


@torch.no_grad()
def get_top1_prob(x_pred, model, device):
    B = x_pred.shape[0]
    z_in = torch.cat([x_pred.detach(), torch.zeros_like(x_pred)], dim=-1)
    t_dec = torch.ones(B, dtype=x_pred.dtype, device=device)
    sc    = torch.ones(B, dtype=x_pred.dtype, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _, logits, _ = model(z_in, t_dec, deterministic=True,
                             self_cond_cfg_scale=sc, decoder_step_active=True)
    return F.softmax(logits.float(), dim=-1).max(dim=-1).values  # (B, L)


@torch.no_grad()
def generate_with_commit(model, z0, t_steps, commit_t, device):
    """Returns (z_final, committed_token_ids_list, commit_frac)."""
    cfg = _Cfg()
    B, L, D = z0.shape
    cond_seq  = torch.zeros_like(z0)
    cond_mask = torch.zeros(B, L, dtype=z0.dtype, device=device)

    z         = z0.clone()
    x_pred    = torch.zeros_like(z)
    committed = False
    commit_frac = 0.0
    committed_ids = None  # token ids for committed positions

    use_bf16 = cfg.use_bf16 and device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(t_steps.shape[0] - 1):
            t      = t_steps[i].item()
            t_next = t_steps[i + 1].item()
            z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                                  model=model, config=cfg, cfg_scale=1.0,
                                  self_cond_cfg_scale=SCCFG,
                                  cond_seq=cond_seq, cond_seq_mask=cond_mask)

            if commit_t is not None and not committed and t_next >= commit_t:
                conf = get_top1_prob(x_pred, model, device)
                new_committed = (conf > CONF_THRESH)
                commit_frac = new_committed.float().mean().item()

                # Save committed token ids for POS analysis
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=use_bf16):
                    z_in = torch.cat([x_pred.detach(),
                                      torch.zeros_like(x_pred)], dim=-1)
                    t1 = torch.ones(B, dtype=x_pred.dtype, device=device)
                    sc = torch.ones_like(t1)
                    _, logits, _ = model(z_in, t1, deterministic=True,
                                        self_cond_cfg_scale=sc,
                                        decoder_step_active=True)
                committed_ids_raw = logits.argmax(dim=-1)  # (B, L)
                # Only keep ids at committed positions
                committed_ids = committed_ids_raw * new_committed.long()

                cond_seq  = x_pred.detach().clone()
                cond_mask = new_committed.float()
                z = z.clone()
                z[new_committed] = x_pred.detach().to(z.dtype)[new_committed]
                committed = True

    return z, committed_ids, commit_frac


@torch.no_grad()
def decode_z(z, model, device):
    B = z.shape[0]
    z_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
    t1 = torch.ones(B, dtype=z.dtype, device=device)
    sc = torch.ones(B, dtype=z.dtype, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _, logits, _ = model(z_in, t1, deterministic=True,
                             self_cond_cfg_scale=sc, decoder_step_active=True)
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
            out   = ppl_model(input_ids=ids_b, attention_mask=msk_b)
            logits = out.logits[:, :-1, :].float()
            tgts   = ids_b[:, 1:]
            msk_s  = msk_b[:, 1:].bool()
            nll = -(F.log_softmax(logits, dim=-1)
                    .gather(-1, tgts.unsqueeze(-1)).squeeze(-1)
                    * msk_s.float()).sum().item()
            total_nll += nll
            total_tok += msk_s.sum().item()
    return math.exp(total_nll / total_tok) if total_tok else float("nan")


def compute_distinct(texts, n):
    """Distinct-n: fraction of unique n-grams over all generated tokens."""
    all_ngrams, unique_ngrams = [], set()
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        all_ngrams.extend(ngrams)
        unique_ngrams.update(ngrams)
    if not all_ngrams:
        return 0.0
    return len(unique_ngrams) / len(all_ngrams)


def pos_analysis(committed_token_ids_list, elf_tok):
    """
    Analyze POS distribution of committed tokens.
    committed_token_ids_list: list of (B, L) tensors, 0 = not committed
    Returns: {'func_frac': float, 'content_frac': float, 'total_committed': int}
    """
    if not HAS_NLTK:
        return {}

    all_tokens = []
    for ids_tensor in committed_token_ids_list:
        for row in ids_tensor.tolist():
            # Decode non-zero (committed) positions
            committed_ids = [tid for tid in row if tid != 0]
            if committed_ids:
                text = elf_tok.decode(committed_ids, skip_special_tokens=True)
                words = word_tokenize(text)
                all_tokens.extend(words)

    if not all_tokens:
        return {"func_frac": 0.0, "content_frac": 0.0, "total_committed": 0}

    tagged = pos_tag(all_tokens)
    func_count    = sum(1 for _, tag in tagged if tag in FUNCTION_POS)
    content_count = len(tagged) - func_count

    return {
        "func_frac":    func_count / len(tagged),
        "content_frac": content_count / len(tagged),
        "total_committed": len(tagged),
        "top_committed_words": dict(Counter(w for w, _ in tagged).most_common(20)),
    }


def run_ckpt(ckpt_name, device, ppl_model, ppl_tok, elf_tok, ref_texts):
    print(f"\n{'='*60}\nCheckpoint: {ckpt_name}\n{'='*60}")
    ckpt = torch.load(REPO_ROOT / CHECKPOINTS[ckpt_name], map_location="cpu",
                      weights_only=False)
    model = ELF_B(**MODEL_CFG)
    model.load_state_dict(ckpt["params"], strict=False)
    model.eval().to(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    all_z0 = torch.randn(N_SEQ, MAX_LENGTH, 512, generator=gen, device=device)
    t_steps = get_sampling_steps(N_STEPS, time_schedule="uniform", device=device)

    # Each model: run commit at its optimal time
    best_commit_t = {"kd2": 0.30, "kd_cr": 0.40}
    commit_t = best_commit_t.get(ckpt_name, 0.30)

    arms = [
        ("standard",    None),
        (f"prog_t{int(commit_t*100):02d}_c70", commit_t),
    ]

    results = {}
    ppl_std = None

    for arm_name, ct in arms:
        texts = []
        all_committed_ids = []
        commit_fracs = []

        for b in range(0, N_SEQ, BATCH_SIZE):
            z0_b = all_z0[b:b+BATCH_SIZE]
            z_out, committed_ids, cfrac = generate_with_commit(
                model, z0_b, t_steps, ct, device
            )
            ids = decode_z(z_out, model, device)
            texts.extend(ids_to_text(ids.cpu(), elf_tok))
            if committed_ids is not None:
                all_committed_ids.append(committed_ids.cpu())
                commit_fracs.append(cfrac)

        ppl = compute_ppl(texts, ppl_model, ppl_tok, device)
        if arm_name == "standard":
            ppl_std = ppl

        d1 = compute_distinct(texts, 1)
        d2 = compute_distinct(texts, 2)

        mauve_score = None
        if HAS_MAUVE and ref_texts:
            try:
                out_m = compute_mauve(
                    p_text=ref_texts[:len(texts)],
                    q_text=texts,
                    device_id=device.index if device.type == "cuda" else -1,
                    max_text_length=256,
                    verbose=False,
                )
                mauve_score = out_m.mauve
            except Exception as e:
                print(f"  [MAUVE error: {e}]")

        pos_stats = pos_analysis(all_committed_ids, elf_tok) if all_committed_ids else {}
        avg_commit = sum(commit_fracs) / len(commit_fracs) if commit_fracs else 0.0

        I_val = ppl - ppl_std if ppl_std is not None else 0.0
        results[arm_name] = {
            "ppl":        ppl,
            "I":          I_val,
            "distinct_1": d1,
            "distinct_2": d2,
            "mauve":      mauve_score,
            "commit_frac": avg_commit,
            "pos":        pos_stats,
            "sample_texts": texts[:4],
        }

        commit_str = f"  commit={avg_commit:.0%}" if ct else ""
        mauve_str  = f"  MAUVE={mauve_score:.3f}" if mauve_score else ""
        print(f"  {arm_name:<22} PPL={ppl:.1f}  I={I_val:+.1f}"
              f"  D1={d1:.3f}  D2={d2:.3f}{mauve_str}{commit_str}")
        if pos_stats:
            print(f"    committed POS: func={pos_stats.get('func_frac',0):.1%}"
                  f"  content={pos_stats.get('content_frac',0):.1%}"
                  f"  n={pos_stats.get('total_committed',0)}")

    return results


def load_ref_texts(elf_tok, n=256, seed=0):
    """Load real OWT samples as reference for MAUVE."""
    if not HAS_MAUVE:
        return []
    try:
        from datasets import load_dataset
        ds = load_dataset("openwebtext", split="train", streaming=True,
                          trust_remote_code=True)
        texts = []
        for ex in ds:
            text = ex["text"][:512].strip()
            if len(text) > 50:
                texts.append(text)
            if len(texts) >= n:
                break
        return texts
    except Exception as e:
        print(f"[WARNING] Could not load OWT reference: {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--ckpt",   default=None)
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

    print("Loading OWT reference texts for MAUVE...")
    ref_texts = load_ref_texts(elf_tok)
    if ref_texts:
        print(f"  Loaded {len(ref_texts)} reference texts")
    else:
        print("  No reference texts (MAUVE will be skipped)")

    ckpts = [args.ckpt] if args.ckpt else list(CHECKPOINTS.keys())
    all_results = {}
    for ck in ckpts:
        all_results[ck] = run_ckpt(ck, device, ppl_model, ppl_tok, elf_tok,
                                    ref_texts)

    # Save results (excluding large text lists)
    save = {}
    for ck, cr in all_results.items():
        save[ck] = {}
        for arm, rv in cr.items():
            save[ck][arm] = {k: v for k, v in rv.items() if k != "sample_texts"}
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved → {OUT_DIR / 'results.json'}")

    print("\n=== SUMMARY ===")
    for ck, cr in all_results.items():
        print(f"\n{ck}:")
        for arm, rv in cr.items():
            mauve_s = f"  MAUVE={rv['mauve']:.3f}" if rv.get("mauve") else ""
            print(f"  {arm:<22} PPL={rv['ppl']:.1f}  I={rv['I']:+.1f}"
                  f"  D1={rv['distinct_1']:.3f}  D2={rv['distinct_2']:.3f}{mauve_s}")
            if rv.get("pos"):
                print(f"    func={rv['pos'].get('func_frac',0):.1%}  "
                      f"content={rv['pos'].get('content_frac',0):.1%}")
            print("    Samples:")
            for t in rv.get("sample_texts", [])[:2]:
                print(f"      > {t[:120]}")

    print("\nInterpretation guide:")
    print("  Distinct-1/2:  prog ≈ standard → no diversity collapse")
    print("  MAUVE:         prog ≈ standard → distributions similar (no mode collapse)")
    print("  func_frac:     >> standard func baseline → trivial function-word anchoring")
    print("  content_frac:  >> 0 → non-trivial content word commitment")


if __name__ == "__main__":
    main()
