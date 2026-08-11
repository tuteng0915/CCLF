#!/usr/bin/env python3
"""EXP-79: align block clocks before a short bidirectional joint refinement."""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
for path in (str(SRC), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import robust_revisable_commit_exp78 as exp78
import unified_method_eval_exp64 as common
from modules.model import ELF_B
from modules.t5_encoder import get_encoder
from utils.encoder_utils import encode_text
from utils.sampling_utils import _ode_step, get_sampling_steps, restore_cond


CHECKPOINTS = {
    name: common.CHECKPOINTS[name]
    for name in ("baseline", "ct_control", "kd_early")
}
REPRESENTATIONS = ("continuous", "reencoded", "hybrid")
OUT_DIR = Path("results/exp79_late_coupled_blocks")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--block_length", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--maturities", nargs="+", type=int, default=[24, 28])
    parser.add_argument("--parallel_steps", nargs="+", type=int, default=[60])
    parser.add_argument("--freeze_a_maturities", nargs="*", type=int, default=[28])
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=["reencoded"],
    )
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--hybrid_confidence", type=float, default=0.90)
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--skip_reference_gate", action="store_true")
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--prefix_length", type=int, default=64)
    parser.add_argument("--label", default="screen")
    return parser.parse_args()


@torch.no_grad()
def ode_range(z, x_pred, model, grid, start, end, args, cond_seq=None, cond_mask=None):
    if cond_seq is None:
        cond_seq, cond_mask = common.empty_condition(z)
    z = restore_cond(z.clone(), cond_seq, cond_mask)
    x_pred = restore_cond(x_pred.clone(), cond_seq, cond_mask)
    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"
    ):
        for index in range(start, end):
            z, x_pred = _ode_step(
                model=model,
                z=z,
                t=grid[index].item(),
                t_next=grid[index + 1].item(),
                x_pred_prev=x_pred,
                config=common.SamplingConfig(),
                cfg_scale=1.0,
                self_cond_cfg_scale=args.sccfg,
                cond_seq=cond_seq,
                cond_seq_mask=cond_mask,
            )
    return z, x_pred


@torch.no_grad()
def prefix_snapshots(
    z0, model, grid, maturities, args, cond_seq=None, cond_mask=None
):
    wanted = set(maturities) | {args.n_steps}
    snapshots = {}
    z = z0.clone()
    x_pred = torch.zeros_like(z)
    if cond_seq is None:
        cond_seq, cond_mask = common.empty_condition(z)
    for index in range(args.n_steps):
        z, x_pred = ode_range(
            z, x_pred, model, grid, index, index + 1, args, cond_seq, cond_mask
        )
        if index + 1 in wanted:
            snapshots[index + 1] = (z.detach().clone(), x_pred.detach().clone())
    return snapshots


@torch.no_grad()
def t5_reencode(token_ids, encoder, dtype):
    attention = torch.ones_like(token_ids)
    return encode_text(
        token_ids,
        attention,
        encoder,
        latent_mean=0.0,
        latent_std=0.2,
    ).to(dtype=dtype)


@torch.no_grad()
def make_condition(x_pred, model, encoder, representation, threshold):
    token_ids, confidence = exp78.lexical_readout(x_pred, model)
    selected = confidence >= threshold
    reencoded = None
    if representation in ("reencoded", "hybrid"):
        reencoded = t5_reencode(token_ids, encoder, x_pred.dtype)
    if representation == "continuous":
        condition = x_pred.detach()
    elif representation == "reencoded":
        condition = reencoded
    else:
        condition = torch.where(selected.unsqueeze(-1), reencoded, x_pred.detach())
    cosine = F.cosine_similarity(condition.float(), x_pred.float(), dim=-1).mean()
    return condition, token_ids, confidence, selected, float(cosine.item())


def init_records(args):
    names = [
        f"parallel{steps}"
        for steps in sorted({args.n_steps, *args.parallel_steps})
    ]
    names.append("semi_ar64")
    names.extend(
        f"late_{representation}_m{m}"
        for representation in args.representations
        for m in args.maturities
    )
    names.extend(
        f"late_reencoded_m{m}_freeze_a"
        for m in args.freeze_a_maturities
        if m in args.maturities and "reencoded" in args.representations
    )
    return {
        name: {
            "texts": [],
            "prompt_texts": [],
            "prefix_texts": [],
            "suffix_texts": [],
            "boundary_prefix_texts": [],
            "references": [],
            "decoded_prefix_agreement": [],
            "prefix_revision": [],
            "suffix_revision": [],
            "highconf_revision": [],
            "lowconf_revision": [],
            "hybrid_fraction": [],
            "condition_cosine": [],
            "clamp_error": [],
            "samples": [],
        }
        for name in names
    }


def append_ids(
    record,
    ids,
    tokenizer,
    block_length,
    external_prefix_length=0,
    prompt_ids=None,
    references=None,
    maturity_ids=None,
    selected=None,
    suffix_before_ids=None,
):
    prefix_texts = common.decode_texts(
        ids[:, external_prefix_length:block_length].cpu(), tokenizer
    )
    suffix_texts = common.decode_texts(ids[:, block_length:].cpu(), tokenizer)
    # Each block has its own EOS convention. Decoding the concatenated token
    # tensor directly would stop at A's EOS and silently discard all of B.
    texts = [
        f"{prefix.strip()} {suffix.strip()}".strip()
        for prefix, suffix in zip(prefix_texts, suffix_texts)
    ]
    record["texts"].extend(texts)
    record["prefix_texts"].extend(prefix_texts)
    record["suffix_texts"].extend(suffix_texts)
    if prompt_ids is not None:
        prompt_texts = common.decode_texts(prompt_ids.cpu(), tokenizer)
        record["prompt_texts"].extend(prompt_texts)
        record["boundary_prefix_texts"].extend(
            [
                f"{prompt.strip()} {prefix.strip()}".strip()
                for prompt, prefix in zip(prompt_texts, prefix_texts)
            ]
        )
        record["decoded_prefix_agreement"].extend(
            (ids[:, :external_prefix_length].cpu() == prompt_ids.cpu())
            .all(dim=1)
            .float()
            .tolist()
        )
    else:
        record["boundary_prefix_texts"].extend(prefix_texts)
    if references is not None:
        record["references"].extend(references)
    if len(record["samples"]) < 4:
        record["samples"].extend(texts[: 4 - len(record["samples"])])
    if maturity_ids is None:
        return
    changed = (
        ids[:, external_prefix_length : maturity_ids.shape[1]].cpu()
        != maturity_ids[:, external_prefix_length:].cpu()
    )
    record["prefix_revision"].extend(changed.float().mean(dim=1).tolist())
    if suffix_before_ids is not None:
        suffix_changed = ids[:, block_length:].cpu() != suffix_before_ids.cpu()
        record["suffix_revision"].extend(
            suffix_changed.float().mean(dim=1).tolist()
        )
    if selected is not None:
        selected = selected[:, external_prefix_length:].cpu()
        for row in range(changed.shape[0]):
            high = changed[row][selected[row]]
            low = changed[row][~selected[row]]
            record["highconf_revision"].append(
                float(high.float().mean().item()) if high.numel() else float("nan")
            )
            record["lowconf_revision"].append(
                float(low.float().mean().item()) if low.numel() else float("nan")
            )


@torch.no_grad()
def run_batch(
    z0_a,
    z0_b,
    block_model,
    joint_model,
    encoder,
    grid,
    parallel_grids,
    args,
    records,
    tokenizer,
    base_cond_seq_a=None,
    base_cond_mask_a=None,
    prompt_ids=None,
    references=None,
):
    batch = z0_a.shape[0]
    block = args.block_length
    external_prefix_length = args.prefix_length if args.conditional else 0
    snapshots = prefix_snapshots(
        z0_a,
        block_model,
        grid,
        args.maturities,
        args,
        base_cond_seq_a,
        base_cond_mask_a,
    )
    if base_cond_seq_a is None:
        base_cond_seq_a, base_cond_mask_a = common.empty_condition(z0_a)
    joint_base_cond = torch.cat(
        [base_cond_seq_a, torch.zeros_like(z0_b)], dim=1
    )
    joint_base_mask = torch.cat(
        [
            base_cond_mask_a,
            torch.zeros(batch, block, dtype=z0_a.dtype, device=z0_a.device),
        ],
        dim=1,
    )

    # Fully parallel references, including an exact denoiser-call match for
    # the primary m=28 late-coupled arm.
    for steps, parallel_grid in parallel_grids.items():
        z_parallel, _ = ode_range(
            torch.cat([z0_a, z0_b], dim=1),
            torch.zeros(batch, 2 * block, z0_a.shape[-1], device=z0_a.device),
            joint_model,
            parallel_grid,
            0,
            steps,
            args,
            joint_base_cond,
            joint_base_mask,
        )
        append_ids(
            records[f"parallel{steps}"],
            common.decode(z_parallel, joint_model, z0_a.device),
            tokenizer,
            block,
            external_prefix_length,
            prompt_ids,
            references,
        )
        records[f"parallel{steps}"]["clamp_error"].append(
            float(
                (
                    z_parallel[:, :block][base_cond_mask_a.bool()]
                    - base_cond_seq_a[base_cond_mask_a.bool()]
                )
                .abs()
                .max()
                .item()
            )
            if base_cond_mask_a.any()
            else 0.0
        )

    # Native reencoded block Semi-AR baseline.
    z_a_final, _ = snapshots[args.n_steps]
    ids_a_final = common.decode(z_a_final, block_model, z0_a.device)
    cond_a_final = t5_reencode(ids_a_final, encoder, z0_a.dtype)
    cond_a_final = torch.where(
        base_cond_mask_a.unsqueeze(-1).bool(), base_cond_seq_a, cond_a_final
    )
    cond_seq = torch.cat([cond_a_final, torch.zeros_like(z0_b)], dim=1)
    cond_mask = torch.cat(
        [
            torch.ones(batch, block, dtype=z0_a.dtype, device=z0_a.device),
            torch.zeros(batch, block, dtype=z0_a.dtype, device=z0_a.device),
        ],
        dim=1,
    )
    z_b_full, _ = ode_range(
        torch.cat([cond_a_final, z0_b], dim=1),
        torch.zeros(batch, 2 * block, z0_a.shape[-1], device=z0_a.device),
        joint_model,
        grid,
        0,
        args.n_steps,
        args,
        cond_seq,
        cond_mask,
    )
    ids_b_full = common.decode(z_b_full, joint_model, z0_a.device)[:, block:]
    append_ids(
        records["semi_ar64"],
        torch.cat([ids_a_final, ids_b_full], dim=1),
        tokenizer,
        block,
        external_prefix_length,
        prompt_ids,
        references,
    )
    records["semi_ar64"]["clamp_error"].append(
        float((z_b_full[:, :block] - cond_a_final).abs().max().item())
    )

    for maturity in args.maturities:
        z_a_m, x_a_m = snapshots[maturity]
        for representation in args.representations:
            name = f"late_{representation}_m{maturity}"
            condition, maturity_ids, _, selected, cosine = make_condition(
                x_a_m,
                block_model,
                encoder,
                representation,
                args.hybrid_confidence,
            )
            condition = torch.where(
                base_cond_mask_a.unsqueeze(-1).bool(),
                base_cond_seq_a,
                condition,
            )
            if prompt_ids is not None:
                maturity_ids = maturity_ids.clone()
                maturity_ids[:, :external_prefix_length] = prompt_ids
            selected &= base_cond_mask_a < 0.5
            cond_seq = torch.cat([condition, torch.zeros_like(z0_b)], dim=1)
            cond_mask = torch.cat(
                [
                    torch.ones(batch, block, dtype=z0_a.dtype, device=z0_a.device),
                    torch.zeros(batch, block, dtype=z0_a.dtype, device=z0_a.device),
                ],
                dim=1,
            )
            z_b_m_full, x_b_m_full = ode_range(
                torch.cat([condition, z0_b], dim=1),
                torch.zeros(batch, 2 * block, z0_a.shape[-1], device=z0_a.device),
                joint_model,
                grid,
                0,
                maturity,
                args,
                cond_seq,
                cond_mask,
            )
            suffix_before_ids = common.decode(
                z_b_m_full, joint_model, z0_a.device
            )[:, block:]
            z_joint = torch.cat([z_a_m, z_b_m_full[:, block:]], dim=1)
            x_joint = torch.cat([x_a_m, x_b_m_full[:, block:]], dim=1)
            z_joint, _ = ode_range(
                z_joint,
                x_joint,
                joint_model,
                grid,
                maturity,
                args.n_steps,
                args,
                joint_base_cond,
                joint_base_mask,
            )
            final_ids = common.decode(z_joint, joint_model, z0_a.device)
            append_ids(
                records[name],
                final_ids,
                tokenizer,
                block,
                external_prefix_length,
                prompt_ids,
                references,
                maturity_ids=maturity_ids,
                selected=selected,
                suffix_before_ids=suffix_before_ids,
            )
            records[name]["hybrid_fraction"].append(
                float(selected.float().mean().item())
            )
            records[name]["condition_cosine"].append(cosine)
            records[name]["clamp_error"].append(
                float((z_b_m_full[:, :block] - condition).abs().max().item())
            )
            if base_cond_mask_a.any():
                records[name]["clamp_error"].append(
                    float(
                        (
                            z_joint[:, :block][base_cond_mask_a.bool()]
                            - base_cond_seq_a[base_cond_mask_a.bool()]
                        )
                        .abs()
                        .max()
                        .item()
                    )
                )

            if (
                representation == "reencoded"
                and maturity in args.freeze_a_maturities
            ):
                freeze_name = f"late_reencoded_m{maturity}_freeze_a"
                z_freeze = torch.cat([condition, z_b_m_full[:, block:]], dim=1)
                x_freeze = torch.cat([condition, x_b_m_full[:, block:]], dim=1)
                z_freeze, _ = ode_range(
                    z_freeze,
                    x_freeze,
                    joint_model,
                    grid,
                    maturity,
                    args.n_steps,
                    args,
                    cond_seq,
                    cond_mask,
                )
                freeze_readout = common.decode(
                    z_freeze, joint_model, z0_a.device
                )
                freeze_ids = torch.cat(
                    [maturity_ids, freeze_readout[:, block:]], dim=1
                )
                append_ids(
                    records[freeze_name],
                    freeze_ids,
                    tokenizer,
                    block,
                    external_prefix_length,
                    prompt_ids,
                    references,
                    maturity_ids=maturity_ids,
                    selected=selected,
                    suffix_before_ids=suffix_before_ids,
                )
                records[freeze_name]["hybrid_fraction"].append(
                    float(selected.float().mean().item())
                )
                records[freeze_name]["condition_cosine"].append(cosine)
                records[freeze_name]["clamp_error"].append(
                    float((z_freeze[:, :block] - condition).abs().max().item())
                )


def finite_mean(values):
    clean = [value for value in values if value == value]
    return sum(clean) / len(clean) if clean else float("nan")


def text_metrics_without_ppl(texts):
    lengths = [len(text.split()) for text in texts]
    metrics = {
        "ppl": float("nan"),
        "d1": common.distinct_n(texts, 1),
        "d2": common.distinct_n(texts, 2),
        "rep4": common.repetition_rate(texts),
        "degeneration_rate": common.degeneration_rate(texts),
        "mean_words": sum(lengths) / len(lengths) if lengths else 0.0,
    }
    metrics.update(common.unigram_collapse_stats(texts))
    return metrics


@torch.no_grad()
def conditional_boundary_ppl(
    prefix_texts,
    suffix_texts,
    evaluator,
    tokenizer,
    device,
    suffix_tokens=32,
    max_length=1024,
):
    """GPT-2 PPL on the first suffix tokens while conditioning on prefix text."""
    sequences = []
    prefix_lengths = []
    for prefix, suffix in zip(prefix_texts, suffix_texts):
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        separator = " " if prefix.strip() and suffix.strip() else ""
        suffix_ids = tokenizer.encode(separator + suffix, add_special_tokens=False)
        if suffix_tokens is not None:
            suffix_ids = suffix_ids[:suffix_tokens]
        prefix_budget = max(max_length - len(suffix_ids), 0)
        prefix_ids = prefix_ids[-prefix_budget:] if prefix_budget else []
        sequences.append(prefix_ids + suffix_ids)
        prefix_lengths.append(len(prefix_ids))

    pad_id = tokenizer.pad_token_id
    width = max((len(sequence) for sequence in sequences), default=1)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    suffix_mask = torch.zeros_like(ids, dtype=torch.bool)
    for row, (sequence, prefix_length) in enumerate(zip(sequences, prefix_lengths)):
        if not sequence:
            continue
        ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention[row, : len(sequence)] = 1
        suffix_mask[row, prefix_length : len(sequence)] = True

    total_nll, total_tokens = 0.0, 0
    for start in range(0, len(sequences), 8):
        ids_b = ids[start : start + 8].to(device)
        attention_b = attention[start : start + 8].to(device)
        valid = suffix_mask[start : start + 8, 1:].to(device)
        logits = evaluator(
            input_ids=ids_b, attention_mask=attention_b
        ).logits[:, :-1].float()
        targets = ids_b[:, 1:]
        nll = -F.log_softmax(logits, dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        total_nll += float(nll[valid].sum().item())
        total_tokens += int(valid.sum().item())
    return (
        math.exp(total_nll / total_tokens)
        if total_tokens
        else float("nan")
    )


def main():
    args = parse_args()
    if any(m <= 0 or m >= args.n_steps for m in args.maturities):
        raise ValueError("maturities must lie strictly between 0 and n_steps")
    if args.block_length * 2 > 1024:
        raise ValueError("this bounded evaluator supports total length at most 1024")
    if args.conditional and not 0 < args.prefix_length < args.block_length:
        raise ValueError(
            "conditional prefix_length must lie between 0 and block_length"
        )
    if any(steps <= 0 for steps in args.parallel_steps):
        raise ValueError("parallel_steps must be positive")
    if any(m not in args.maturities for m in args.freeze_a_maturities):
        raise ValueError("freeze_a_maturities must be a subset of maturities")

    from transformers import T5Tokenizer

    device = torch.device(args.device)
    total_length = 2 * args.block_length
    common.SamplingConfig.denoiser_noise_scale = args.noise_scale
    checkpoint_path = REPO_ROOT / CHECKPOINTS[args.checkpoint]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = common.load_weights(checkpoint)
    block_model = ELF_B(**common.model_config(args.block_length))
    block_model.load_state_dict(weights, strict=False)
    block_model.to(device).eval()
    joint_model = ELF_B(**common.model_config(total_length))
    joint_model.load_state_dict(weights, strict=False)
    joint_model.to(device).eval()

    elf_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    _, encoder = get_encoder("t5-small", dtype=torch.float32)
    encoder.to(device).eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    z0_a = args.noise_scale * torch.randn(
        args.n_samples,
        args.block_length,
        512,
        generator=generator,
        device=device,
    )
    z0_b = args.noise_scale * torch.randn(
        args.n_samples,
        args.block_length,
        512,
        generator=generator,
        device=device,
    )
    base_cond_seq_a = base_cond_mask_a = prompt_ids = None
    references = None
    if args.conditional:
        pairs = common.get_gutenberg_pairs(
            elf_tokenizer,
            args.n_samples,
            args.prefix_length,
            total_length - args.prefix_length,
        )
        if len(pairs) != args.n_samples:
            raise RuntimeError(
                f"requested {args.n_samples} conditioned pairs, got {len(pairs)}"
            )
        full_cond_seq, full_cond_mask, references = common.build_condition_data(
            pairs,
            elf_tokenizer,
            encoder,
            device,
            total_length,
            args.prefix_length,
        )
        base_cond_seq_a = full_cond_seq[:, : args.block_length]
        base_cond_mask_a = full_cond_mask[:, : args.block_length]
        z0_a[:, : args.prefix_length] = base_cond_seq_a[:, : args.prefix_length]
        prompt_ids = torch.zeros(
            args.n_samples,
            args.prefix_length,
            dtype=torch.long,
            device=device,
        )
        for row, (prefix, _) in enumerate(pairs):
            prompt_ids[row, : min(len(prefix), args.prefix_length)] = torch.tensor(
                prefix[: args.prefix_length], dtype=torch.long, device=device
            )
    grid = get_sampling_steps(args.n_steps, "uniform", device=device)
    parallel_grids = {
        steps: get_sampling_steps(steps, "uniform", device=device)
        for steps in sorted({args.n_steps, *args.parallel_steps})
    }
    records = init_records(args)

    native_reference_agreement = float("nan")
    if not args.skip_reference_gate:
        gate_size = min(args.batch_size, args.n_samples)
        gate_z0 = torch.cat([z0_a[:gate_size], z0_b[:gate_size]], dim=1)
        gate_cond = (
            torch.cat(
                [
                    base_cond_seq_a[:gate_size],
                    torch.zeros_like(z0_b[:gate_size]),
                ],
                dim=1,
            )
            if args.conditional
            else None
        )
        gate_mask = (
            torch.cat(
                [
                    base_cond_mask_a[:gate_size],
                    torch.zeros(
                        gate_size,
                        args.block_length,
                        dtype=z0_a.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )
            if args.conditional
            else None
        )
        native_z, _ = common.standard_ode(
            gate_z0, joint_model, grid, args.sccfg, gate_cond, gate_mask
        )
        local_z, _ = ode_range(
            gate_z0,
            torch.zeros_like(gate_z0),
            joint_model,
            grid,
            0,
            args.n_steps,
            args,
            gate_cond,
            gate_mask,
        )
        native_ids = common.decode(native_z, joint_model, device)
        local_ids = common.decode(local_z, joint_model, device)
        native_reference_agreement = float(
            (native_ids == local_ids).float().mean().item()
        )
        if native_reference_agreement != 1.0:
            raise RuntimeError(
                "native reference gate failed: "
                f"agreement={native_reference_agreement:.6f}"
            )

    for start in range(0, args.n_samples, args.batch_size):
        end = min(start + args.batch_size, args.n_samples)
        print(f"batch {start}:{end}", flush=True)
        run_batch(
            z0_a[start:end],
            z0_b[start:end],
            block_model,
            joint_model,
            encoder,
            grid,
            parallel_grids,
            args,
            records,
            elf_tokenizer,
            (
                base_cond_seq_a[start:end]
                if base_cond_seq_a is not None
                else None
            ),
            (
                base_cond_mask_a[start:end]
                if base_cond_mask_a is not None
                else None
            ),
            prompt_ids[start:end] if prompt_ids is not None else None,
            references[start:end] if references is not None else None,
        )

    for name, record in records.items():
        if record["clamp_error"] and max(record["clamp_error"]) > 1e-6:
            raise RuntimeError(
                f"condition restore gate failed for {name}: "
                f"max_error={max(record['clamp_error']):.6g}"
            )
        if (
            name.endswith("_freeze_a")
            and finite_mean(record["prefix_revision"]) != 0.0
        ):
            raise RuntimeError(
                f"freeze-A gate failed for {name}: "
                f"revision={finite_mean(record['prefix_revision']):.6g}"
            )

    # The two ELF instances are both needed during generation but not during
    # GPT-2 evaluation. Release them before loading the evaluator so the
    # experiment remains safe on a GPU shared with other bounded pilots.
    block_model.cpu()
    joint_model.cpu()
    encoder.cpu()
    del block_model, joint_model, encoder, checkpoint, weights
    torch.cuda.empty_cache()

    ppl_model = None
    ppl_tokenizer = None
    if not args.skip_ppl:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        if ppl_tokenizer.pad_token is None:
            ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
            ppl_tokenizer.pad_token_id = ppl_tokenizer.eos_token_id
        ppl_model = AutoModelForCausalLM.from_pretrained(
            "gpt2-large", torch_dtype=torch.bfloat16
        ).to(device).eval()

    results = {}
    for name, record in records.items():
        if ppl_model is None:
            metrics = text_metrics_without_ppl(record["texts"])
            prefix_ppl = suffix_ppl = boundary_ppl = float("nan")
            prompt_conditioned_ppl = float("nan")
        else:
            metrics = common.text_metrics(
                record["texts"],
                ppl_model,
                ppl_tokenizer,
                device,
                max_length=total_length,
            )
            prefix_ppl = common.compute_ppl(
                record["prefix_texts"], ppl_model, ppl_tokenizer, device,
                max_length=args.block_length,
            )
            suffix_ppl = common.compute_ppl(
                record["suffix_texts"], ppl_model, ppl_tokenizer, device,
                max_length=args.block_length,
            )
            boundary_ppl = conditional_boundary_ppl(
                record["boundary_prefix_texts"],
                record["suffix_texts"],
                ppl_model,
                ppl_tokenizer,
                device,
                suffix_tokens=32,
                max_length=1024,
            )
            prompt_conditioned_ppl = (
                conditional_boundary_ppl(
                    record["prompt_texts"],
                    record["texts"],
                    ppl_model,
                    ppl_tokenizer,
                    device,
                    suffix_tokens=None,
                    max_length=1024,
                )
                if record["prompt_texts"]
                else float("nan")
            )
        if record["references"]:
            metrics["rouge_l"] = sum(
                common.rouge_l_f1(hypothesis, reference)
                for hypothesis, reference in zip(
                    record["texts"], record["references"]
                )
            ) / len(record["references"])
        else:
            metrics["rouge_l"] = float("nan")
        if name.startswith("parallel"):
            denoiser_calls = int(name[len("parallel") :])
            processed_token_calls = denoiser_calls * total_length
            readout_calls, t5_calls = 0, 0
        elif name == "semi_ar64":
            denoiser_calls = 2 * args.n_steps
            processed_token_calls = (
                args.n_steps * args.block_length
                + args.n_steps * total_length
            )
            readout_calls, t5_calls = 1, 1
        else:
            maturity = int(name.rsplit("_m", 1)[1].split("_", 1)[0])
            representation = name[len("late_") : name.rfind("_m")]
            denoiser_calls = args.n_steps + maturity
            processed_token_calls = (
                maturity * args.block_length
                + args.n_steps * total_length
            )
            readout_calls = 1
            t5_calls = int(representation in ("reencoded", "hybrid"))
        metrics.update(
            {
                "denoiser_calls": denoiser_calls,
                "processed_token_calls": processed_token_calls,
                "readout_calls": readout_calls,
                "t5_calls": t5_calls,
                "prefix_ppl": prefix_ppl,
                "suffix_ppl": suffix_ppl,
                "boundary32_conditional_ppl": boundary_ppl,
                "prompt_conditioned_ppl": prompt_conditioned_ppl,
                "decoded_prefix_agreement": finite_mean(
                    record["decoded_prefix_agreement"]
                ),
                "prefix_revision": finite_mean(record["prefix_revision"]),
                "suffix_revision": finite_mean(record["suffix_revision"]),
                "highconf_prefix_revision": finite_mean(record["highconf_revision"]),
                "lowconf_prefix_revision": finite_mean(record["lowconf_revision"]),
                "hybrid_fraction": finite_mean(record["hybrid_fraction"]),
                "condition_cosine": finite_mean(record["condition_cosine"]),
                "max_clamp_restore_error": (
                    max(record["clamp_error"]) if record["clamp_error"] else 0.0
                ),
                "texts": record["texts"],
                "samples": record["samples"],
            }
        )
        results[name] = metrics

    output = {
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "seed": args.seed,
        "n_samples": args.n_samples,
        "block_length": args.block_length,
        "total_length": total_length,
        "n_steps": args.n_steps,
        "maturities": args.maturities,
        "parallel_steps": args.parallel_steps,
        "freeze_a_maturities": args.freeze_a_maturities,
        "representations": args.representations,
        "noise_scale": args.noise_scale,
        "sccfg": args.sccfg,
        "hybrid_confidence": args.hybrid_confidence,
        "conditional": args.conditional,
        "prefix_length": args.prefix_length if args.conditional else 0,
        "skip_ppl": args.skip_ppl,
        "native_reference_agreement": native_reference_agreement,
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / f"{args.label}_{args.checkpoint}_seed{args.seed}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    compact = {
        name: {
            key: value
            for key, value in metrics.items()
            if key
            in (
                "ppl",
                "d1",
                "d2",
                "rep4",
                "degeneration_rate",
                "prefix_ppl",
                "suffix_ppl",
                "boundary32_conditional_ppl",
                "prompt_conditioned_ppl",
                "rouge_l",
                "decoded_prefix_agreement",
                "prefix_revision",
                "suffix_revision",
                "max_clamp_restore_error",
                "hybrid_fraction",
                "denoiser_calls",
                "processed_token_calls",
                "readout_calls",
                "t5_calls",
            )
        }
        for name, metrics in results.items()
    }
    print(json.dumps(compact, indent=2))
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
