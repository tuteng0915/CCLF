#!/usr/bin/env python3
"""Unified native-recipe evaluation for the method-comparison slide.

This script removes the protocol mismatch across EXP-55/56/58/59/61:

* all methods use the same checkpoint payload, z0, seed, native noise scale,
  self-conditioning CFG, ODE grid, and GPT-2-large evaluator;
* every retained arm reports unconditional PPL/diversity/repetition/
  degeneration metrics;
* every arm is also evaluated on the same prefix-conditioned continuation
  panel, including suffix ROUGE-L against the held-out continuation.

The historical scripts used noise scale 1 and SC-CFG 1 for two-pass and hard
commit, while ELF's native evaluation recipe uses noise scale 2 and SC-CFG 3.
The point of this script is revalidation, not reproducing those legacy values.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.model import ELF_B
from modules.t5_encoder import get_encoder
from utils.encoder_utils import encode_text
from utils.sampling_utils import _ode_step, get_sampling_steps, restore_cond


CHECKPOINTS = {
    "baseline": "converted/elf_b-owt-baseline_torch.pt",
    "kd2": "converted/elf_b-owt-kd2_torch.pt",
    "kd_cr": "converted/elf_b-owt-kd-cr_torch.pt",
}
DEFAULT_ARMS = {
    "baseline": ["standard", "local_clock_ltr", "hard_commit"],
    "kd2": ["standard", "two_pass_prefix", "hard_commit", "pipeline"],
    "kd_cr": ["standard", "hard_commit", "pipeline"],
}
COMMIT_TIME = {"baseline": 0.50, "kd2": 0.30, "kd_cr": 0.40}
MODEL_CALLS = {
    "standard": 32,
    "local_clock_ltr": 32,
    "hard_commit": 33,  # 32 solver calls + one confidence readout
    "two_pass_prefix": 64,
    "pipeline": 31,
}


class SamplingConfig:
    t_eps = 0.05
    self_cond_prob = 1.0
    num_self_cond_cfg_tokens = 4
    denoiser_noise_scale = 2.0
    use_bf16 = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, required=True)
    parser.add_argument("--arms", nargs="+", choices=MODEL_CALLS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_uncond", type=int, default=256)
    parser.add_argument("--n_cond", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--prefix_length", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--pipeline_groups", type=int, default=16)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--async_delta", type=float, default=0.20)
    parser.add_argument("--label", default="native_panel")
    return parser.parse_args()


def model_config(max_length):
    return dict(
        text_encoder_dim=512,
        max_length=max_length,
        num_time_tokens=4,
        num_self_cond_cfg_tokens=4,
        num_model_mode_tokens=4,
        vocab_size=32100,
        bottleneck_dim=128,
    )


def load_weights(checkpoint):
    return checkpoint.get("ema_params1", checkpoint["params"])


def distinct_n(texts, n):
    unique, total = set(), 0
    for text in texts:
        tokens = text.split()
        grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        unique.update(grams)
        total += len(grams)
    return len(unique) / total if total else 0.0


def repetition_rate(texts, n=4):
    values = []
    for text in texts:
        tokens = text.split()
        grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        values.append(1.0 - len(set(grams)) / len(grams) if grams else 0.0)
    return sum(values) / len(values) if values else 0.0


def unigram_collapse_stats(texts, threshold=0.20):
    """Per-sample repetition controls that catch high-frequency word loops.

    Repeated unigrams such as ``on on on`` can evade a repeated-4-gram metric,
    especially when punctuation is interleaved. We lowercase and keep lexical
    word pieces only, then report both a continuous concentration and the
    established 20% collapse flag used by the earlier ELF evaluations.
    """
    max_fractions, unique_ratios = [], []
    for text in texts:
        words = re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)
        if not words:
            max_fractions.append(1.0)
            unique_ratios.append(0.0)
            continue
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        max_fractions.append(max(counts.values()) / len(words))
        unique_ratios.append(len(counts) / len(words))
    return {
        "mean_max_word_fraction": (
            sum(max_fractions) / len(max_fractions) if max_fractions else 1.0
        ),
        "mean_unique_word_ratio": (
            sum(unique_ratios) / len(unique_ratios) if unique_ratios else 0.0
        ),
        "unigram_collapse_rate": (
            sum(value > threshold for value in max_fractions) / len(max_fractions)
            if max_fractions else 1.0
        ),
    }


def degeneration_rate(texts):
    flags = []
    for text in texts:
        tokens = text.split()
        max_word_fraction = (
            max((tokens.count(tok) for tok in set(tokens)), default=0) / len(tokens)
            if tokens else 1.0
        )
        non_ascii_fraction = (
            sum(ord(char) > 127 for char in text) / len(text) if text else 1.0
        )
        flags.append(
            (not text.strip())
            or max_word_fraction > 0.20
            or non_ascii_fraction > 0.02
        )
    return sum(flags) / len(flags) if flags else 0.0


def rouge_l_f1(hypothesis, reference):
    hyp, ref = hypothesis.split(), reference.split()
    if not hyp or not ref:
        return 0.0
    prev = [0] * (len(ref) + 1)
    for hyp_token in hyp:
        cur = [0]
        for j, ref_token in enumerate(ref, 1):
            cur.append(prev[j - 1] + 1 if hyp_token == ref_token else max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    precision, recall = lcs / len(hyp), lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


@torch.no_grad()
def compute_ppl(texts, evaluator, tokenizer, device, max_length=200):
    if not texts:
        return float("nan")
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)
    total_nll, total_tokens = 0.0, 0
    for start in range(0, ids.shape[0], 8):
        ids_b, mask_b = ids[start:start + 8], mask[start:start + 8]
        logits = evaluator(input_ids=ids_b, attention_mask=mask_b).logits[:, :-1].float()
        targets, valid = ids_b[:, 1:], mask_b[:, 1:].bool()
        nll = -(
            F.log_softmax(logits, dim=-1)
            .gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            * valid.float()
        ).sum().item()
        total_nll += nll
        total_tokens += valid.sum().item()
    return math.exp(total_nll / total_tokens) if total_tokens else float("nan")


def text_metrics(texts, evaluator, ppl_tokenizer, device):
    lengths = [len(text.split()) for text in texts]
    metrics = {
        "ppl": compute_ppl(texts, evaluator, ppl_tokenizer, device),
        "d1": distinct_n(texts, 1),
        "d2": distinct_n(texts, 2),
        "rep4": repetition_rate(texts),
        "degeneration_rate": degeneration_rate(texts),
        "mean_words": sum(lengths) / len(lengths) if lengths else 0.0,
    }
    metrics.update(unigram_collapse_stats(texts))
    return metrics


@torch.no_grad()
def decode(z, model, device):
    batch = z.shape[0]
    z_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
    ones = torch.ones(batch, dtype=z.dtype, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _, logits, _ = model(
            z_in,
            ones,
            deterministic=True,
            self_cond_cfg_scale=ones,
            decoder_step_active=True,
        )
    return logits.argmax(-1)


def decode_texts(ids, tokenizer, start=0):
    texts = []
    eos = tokenizer.eos_token_id
    for row in ids[:, start:].tolist():
        try:
            row = row[:row.index(eos)]
        except ValueError:
            pass
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return texts


def empty_condition(z0):
    batch, length, _ = z0.shape
    return torch.zeros_like(z0), torch.zeros(batch, length, dtype=z0.dtype, device=z0.device)


@torch.no_grad()
def standard_ode(z0, model, t_steps, sccfg, cond_seq=None, cond_mask=None):
    cfg = SamplingConfig()
    if cond_seq is None:
        cond_seq, cond_mask = empty_condition(z0)
    z = restore_cond(z0.clone(), cond_seq, cond_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"):
        for i in range(t_steps.shape[0] - 1):
            z, x_pred = _ode_step(
                z=z,
                t=t_steps[i].item(),
                t_next=t_steps[i + 1].item(),
                x_pred_prev=x_pred,
                model=model,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=sccfg,
                cond_seq=cond_seq,
                cond_seq_mask=cond_mask,
            )
    return z, x_pred


@torch.no_grad()
def confidence_mask(x_pred, model, threshold):
    batch = x_pred.shape[0]
    z_in = torch.cat([x_pred.detach(), torch.zeros_like(x_pred)], dim=-1)
    ones = torch.ones(batch, dtype=x_pred.dtype, device=x_pred.device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=x_pred.device.type == "cuda"):
        _, logits, _ = model(
            z_in,
            ones,
            deterministic=True,
            self_cond_cfg_scale=ones,
            decoder_step_active=True,
        )
    return F.softmax(logits.float(), dim=-1).amax(-1) >= threshold


@torch.no_grad()
def hard_commit_ode(
    z0,
    model,
    t_steps,
    sccfg,
    commit_time,
    threshold,
    cond_seq=None,
    cond_mask=None,
):
    cfg = SamplingConfig()
    if cond_seq is None:
        cond_seq, cond_mask = empty_condition(z0)
    active_cond_seq = cond_seq.clone()
    active_cond_mask = cond_mask.clone()
    z = restore_cond(z0.clone(), active_cond_seq, active_cond_mask)
    x_pred = restore_cond(torch.zeros_like(z), active_cond_seq, active_cond_mask)
    committed = False
    commit_fraction = 0.0
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"):
        for i in range(t_steps.shape[0] - 1):
            z, x_pred = _ode_step(
                z=z,
                t=t_steps[i].item(),
                t_next=t_steps[i + 1].item(),
                x_pred_prev=x_pred,
                model=model,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=sccfg,
                cond_seq=active_cond_seq,
                cond_seq_mask=active_cond_mask,
            )
            if not committed and t_steps[i + 1].item() >= commit_time:
                selected = confidence_mask(x_pred, model, threshold)
                selected &= active_cond_mask < 0.5
                eligible = (cond_mask < 0.5).sum().item()
                commit_fraction = selected.sum().item() / max(eligible, 1)
                active_cond_seq = torch.where(
                    selected.unsqueeze(-1), x_pred.detach(), active_cond_seq
                )
                active_cond_mask = torch.maximum(active_cond_mask, selected.to(active_cond_mask.dtype))
                z = restore_cond(z, active_cond_seq, active_cond_mask)
                x_pred = restore_cond(x_pred, active_cond_seq, active_cond_mask)
                committed = True
    return z, x_pred, commit_fraction


@torch.no_grad()
def two_pass_prefix(
    z0,
    model,
    t_steps,
    sccfg,
    cond_seq=None,
    cond_mask=None,
):
    if cond_seq is None:
        cond_seq, cond_mask = empty_condition(z0)
    draft_z, draft_x = standard_ode(z0, model, t_steps, sccfg, cond_seq, cond_mask)
    batch, length, _ = z0.shape
    second_mask = cond_mask.clone()
    second_seq = cond_seq.clone()
    for row in range(batch):
        available = torch.where(cond_mask[row] < 0.5)[0]
        selected = available[: max(len(available) // 2, 1)]
        second_mask[row, selected] = 1
        second_seq[row, selected] = draft_x[row, selected]
    z, x_pred = standard_ode(z0, model, t_steps, sccfg, second_seq, second_mask)
    return z, x_pred, 0.5


@torch.no_grad()
def pipeline_ode(
    z0,
    model,
    groups,
    sccfg,
    cond_seq=None,
    cond_mask=None,
):
    cfg = SamplingConfig()
    if cond_seq is None:
        cond_seq, cond_mask = empty_condition(z0)
    batch, length, dim = z0.shape
    total = 2 * groups - 1
    chunk = max(1, length // groups)
    positions = torch.arange(length, device=z0.device)
    group_of = torch.clamp(positions // chunk, 0, groups - 1)
    z = restore_cond(z0.clone(), cond_seq, cond_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"):
        for stage in range(total):
            first = max(0, stage - groups + 1)
            last = min(groups - 1, stage)
            time = ((stage - last) + (stage - first)) / (2.0 * groups)
            next_time = min(time + 1.0 / groups, 1.0)
            z_full, pred_full = _ode_step(
                z=z,
                t=time,
                t_next=next_time,
                x_pred_prev=x_pred,
                model=model,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=sccfg,
                cond_seq=cond_seq,
                cond_seq_mask=cond_mask,
            )
            active = ((group_of >= first) & (group_of <= last)).view(1, length, 1)
            active = active.expand(batch, length, dim)
            z = torch.where(active, z_full, z)
            x_pred = torch.where(active, pred_full, x_pred)
            z = restore_cond(z, cond_seq, cond_mask)
            x_pred = restore_cond(x_pred, cond_seq, cond_mask)
    return z, x_pred


@torch.no_grad()
def local_clock_ltr(
    z0,
    model,
    sccfg,
    delta,
    cond_seq=None,
    cond_mask=None,
    n_async_steps=28,
    n_sync_refine=4,
):
    """GS19-style heterogeneous-state LTR intervention for ELF."""
    cfg = SamplingConfig()
    if cond_seq is None:
        cond_seq, cond_mask = empty_condition(z0)
    batch, length, _ = z0.shape
    z = restore_cond(z0.clone(), cond_seq, cond_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
    async_grid = torch.linspace(cfg.t_eps, 0.99, n_async_steps + 1, device=z0.device)

    rank = torch.zeros(batch, length, dtype=torch.float32, device=z0.device)
    for row in range(batch):
        available = torch.where(cond_mask[row] < 0.5)[0]
        if len(available) > 1:
            rank[row, available] = torch.arange(len(available), device=z0.device) / (len(available) - 1)
    rank_term = 1.0 - 2.0 * rank

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"):
        for i in range(n_async_steps):
            time, next_time = async_grid[i].item(), async_grid[i + 1].item()
            _, x_hat = _ode_step(
                z=z,
                t=time,
                t_next=next_time,
                x_pred_prev=x_pred,
                model=model,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=sccfg,
                cond_seq=cond_seq,
                cond_seq_mask=cond_mask,
            )
            eps_hat = (z - time * x_hat) / max(1.0 - time, cfg.t_eps)
            tau = (next_time + delta * rank_term).clamp(0.0, 1.0).unsqueeze(-1)
            z = tau * x_hat + (1.0 - tau).clamp(min=cfg.t_eps) * eps_hat
            x_pred = x_hat
            z = restore_cond(z, cond_seq, cond_mask)
            x_pred = restore_cond(x_pred, cond_seq, cond_mask)

        refine_grid = torch.linspace(0.99, 0.999, n_sync_refine + 1, device=z0.device)
        for i in range(n_sync_refine):
            z, x_pred = _ode_step(
                z=z,
                t=refine_grid[i].item(),
                t_next=refine_grid[i + 1].item(),
                x_pred_prev=x_pred,
                model=model,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=sccfg,
                cond_seq=cond_seq,
                cond_seq_mask=cond_mask,
            )
    return z, x_pred


def get_gutenberg_pairs(tokenizer, n_pairs, prefix_length, suffix_length):
    import nltk
    nltk.download("gutenberg", quiet=True)
    from nltk.corpus import gutenberg

    total_length = prefix_length + suffix_length
    pairs = []
    for filename in gutenberg.fileids():
        words = re.split(r"\s+", gutenberg.raw(filename).strip())
        for start in range(0, len(words) - total_length * 2, total_length):
            ids = tokenizer.encode(
                " ".join(words[start:start + total_length * 2]),
                add_special_tokens=False,
            )
            if len(ids) >= total_length:
                pairs.append((ids[:prefix_length], ids[prefix_length:total_length]))
            if len(pairs) >= n_pairs:
                return pairs
    return pairs


@torch.no_grad()
def build_condition_data(pairs, tokenizer, encoder, device, max_length, prefix_length):
    batch_size = 16
    all_cond_seq, references = [], []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        ids = torch.zeros(len(batch), prefix_length, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for row, (prefix, suffix) in enumerate(batch):
            width = min(prefix_length, len(prefix))
            ids[row, :width] = torch.tensor(prefix[:width], device=device)
            mask[row, :width] = 1
            references.append(tokenizer.decode(suffix, skip_special_tokens=True))
        latents = encode_text(ids, mask, encoder, latent_mean=0.0, latent_std=0.2)
        cond_seq = torch.zeros(len(batch), max_length, 512, dtype=latents.dtype, device=device)
        cond_seq[:, :prefix_length] = latents
        all_cond_seq.append(cond_seq)
    cond_seq = torch.cat(all_cond_seq)
    cond_mask = torch.zeros(len(pairs), max_length, dtype=cond_seq.dtype, device=device)
    cond_mask[:, :prefix_length] = 1
    return cond_seq, cond_mask, references


def run_method(
    arm,
    checkpoint_name,
    z0,
    model,
    t_steps,
    args,
    cond_seq=None,
    cond_mask=None,
):
    if arm == "standard":
        z, _ = standard_ode(z0, model, t_steps, args.sccfg, cond_seq, cond_mask)
        return z, None
    if arm == "hard_commit":
        z, _, fraction = hard_commit_ode(
            z0,
            model,
            t_steps,
            args.sccfg,
            COMMIT_TIME[checkpoint_name],
            args.confidence,
            cond_seq,
            cond_mask,
        )
        return z, fraction
    if arm == "two_pass_prefix":
        z, _, fraction = two_pass_prefix(z0, model, t_steps, args.sccfg, cond_seq, cond_mask)
        return z, fraction
    if arm == "pipeline":
        z, _ = pipeline_ode(z0, model, args.pipeline_groups, args.sccfg, cond_seq, cond_mask)
        return z, None
    if arm == "local_clock_ltr":
        z, _ = local_clock_ltr(z0, model, args.sccfg, args.async_delta, cond_seq, cond_mask)
        return z, None
    raise ValueError(arm)


def evaluate_arm(
    arm,
    checkpoint_name,
    z0,
    model,
    t_steps,
    args,
    elf_tokenizer,
    ppl_model,
    ppl_tokenizer,
    cond_seq=None,
    cond_mask=None,
    suffix_start=0,
    references=None,
):
    texts, fractions = [], []
    for start in range(0, len(z0), args.batch_size):
        end = start + args.batch_size
        z, fraction = run_method(
            arm,
            checkpoint_name,
            z0[start:end],
            model,
            t_steps,
            args,
            cond_seq[start:end] if cond_seq is not None else None,
            cond_mask[start:end] if cond_mask is not None else None,
        )
        ids = decode(z, model, z.device)
        texts.extend(decode_texts(ids.cpu(), elf_tokenizer, suffix_start))
        if fraction is not None:
            fractions.append(fraction)
    metrics = text_metrics(texts, ppl_model, ppl_tokenizer, z0.device)
    if references is not None:
        metrics["rouge_l"] = sum(
            rouge_l_f1(hyp, ref) for hyp, ref in zip(texts, references)
        ) / max(len(texts), 1)
    metrics["commit_fraction"] = (
        sum(fractions) / len(fractions) if fractions else None
    )
    metrics["model_calls"] = MODEL_CALLS[arm]
    metrics["samples"] = texts[:4]
    metrics["texts"] = texts
    return metrics


def main():
    args = parse_args()
    if args.max_length <= args.prefix_length:
        raise ValueError("prefix_length must be smaller than max_length")
    if args.n_steps != 32:
        raise ValueError("this comparison fixes ODE steps at 32")
    arms = args.arms or DEFAULT_ARMS[args.checkpoint]
    if "standard" not in arms:
        arms = ["standard", *arms]
    device = torch.device(args.device)
    SamplingConfig.denoiser_noise_scale = args.noise_scale

    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

    print(
        f"checkpoint={args.checkpoint} arms={arms} seed={args.seed} "
        f"noise_scale={args.noise_scale:g} sccfg={args.sccfg:g} "
        f"n_uncond={args.n_uncond} n_cond={args.n_cond}"
    )
    ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    if ppl_tokenizer.pad_token is None:
        ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
        ppl_tokenizer.pad_token_id = ppl_tokenizer.eos_token_id
    ppl_model = AutoModelForCausalLM.from_pretrained(
        "gpt2-large", torch_dtype=torch.bfloat16
    ).to(device).eval()
    elf_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    checkpoint_path = REPO_ROOT / CHECKPOINTS[args.checkpoint]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ELF_B(**model_config(args.max_length))
    missing, unexpected = model.load_state_dict(load_weights(checkpoint), strict=False)
    if missing or unexpected:
        print(f"load_state_dict missing={missing} unexpected={unexpected}")
    model.to(device).eval()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    total = max(args.n_uncond, args.n_cond)
    all_z0 = args.noise_scale * torch.randn(
        total,
        args.max_length,
        512,
        generator=generator,
        device=device,
    )
    t_steps = get_sampling_steps(args.n_steps, time_schedule="uniform", device=device)

    print("loading T5 encoder and fixed Gutenberg continuation panel")
    _, encoder = get_encoder("t5-small", dtype=torch.float32)
    encoder = encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    pairs = get_gutenberg_pairs(
        elf_tokenizer,
        args.n_cond,
        args.prefix_length,
        args.max_length - args.prefix_length,
    )
    if len(pairs) != args.n_cond:
        raise RuntimeError(f"requested {args.n_cond} conditioned pairs, got {len(pairs)}")
    cond_seq, cond_mask, references = build_condition_data(
        pairs,
        elf_tokenizer,
        encoder,
        device,
        args.max_length,
        args.prefix_length,
    )
    cond_z0 = all_z0[:args.n_cond].clone()
    cond_z0[:, :args.prefix_length] = cond_seq[:, :args.prefix_length].to(cond_z0.dtype)

    results = {}
    for arm in arms:
        print(f"\n[{args.checkpoint}] {arm}: unconditional")
        unconditional = evaluate_arm(
            arm,
            args.checkpoint,
            all_z0[:args.n_uncond],
            model,
            t_steps,
            args,
            elf_tokenizer,
            ppl_model,
            ppl_tokenizer,
        )
        print(
            f"  PPL={unconditional['ppl']:.1f} D1={unconditional['d1']:.3f} "
            f"D2={unconditional['d2']:.3f} rep4={unconditional['rep4']:.3f} "
            f"deg={unconditional['degeneration_rate']:.3f}"
        )

        print(f"[{args.checkpoint}] {arm}: conditioned suffix")
        conditioned = evaluate_arm(
            arm,
            args.checkpoint,
            cond_z0,
            model,
            t_steps,
            args,
            elf_tokenizer,
            ppl_model,
            ppl_tokenizer,
            cond_seq,
            cond_mask,
            args.prefix_length,
            references,
        )
        print(
            f"  PPL={conditioned['ppl']:.1f} D1={conditioned['d1']:.3f} "
            f"D2={conditioned['d2']:.3f} rep4={conditioned['rep4']:.3f} "
            f"deg={conditioned['degeneration_rate']:.3f} "
            f"ROUGE-L={conditioned['rouge_l']:.3f}"
        )
        results[arm] = {"unconditional": unconditional, "conditioned": conditioned}

    output_dir = Path("results/exp64_unified_method_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.label}_{args.checkpoint}_seed{args.seed}.json"
    payload = {
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "arms": arms,
        "seed": args.seed,
        "n_uncond": args.n_uncond,
        "n_cond": args.n_cond,
        "max_length": args.max_length,
        "prefix_length": args.prefix_length,
        "noise_scale": args.noise_scale,
        "sccfg": args.sccfg,
        "n_steps": args.n_steps,
        "pipeline_groups": args.pipeline_groups,
        "confidence": args.confidence,
        "commit_time": COMMIT_TIME[args.checkpoint],
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    main()
