#!/usr/bin/env python3
"""EXP-79: align block clocks before a short bidirectional joint refinement."""

import argparse
import json
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
    parser.add_argument("--maturities", nargs="+", type=int, default=[20, 24, 28, 30])
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=list(REPRESENTATIONS),
    )
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--hybrid_confidence", type=float, default=0.90)
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
def prefix_snapshots(z0, model, grid, maturities, args):
    wanted = set(maturities) | {args.n_steps}
    snapshots = {}
    z = z0.clone()
    x_pred = torch.zeros_like(z)
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
    names = ["parallel32", "semi_ar64"]
    names.extend(
        f"late_{representation}_m{m}"
        for representation in args.representations
        for m in args.maturities
    )
    return {
        name: {
            "texts": [],
            "prefix_revision": [],
            "highconf_revision": [],
            "lowconf_revision": [],
            "hybrid_fraction": [],
            "condition_cosine": [],
            "samples": [],
        }
        for name in names
    }


def append_ids(record, ids, tokenizer, maturity_ids=None, selected=None):
    texts = common.decode_texts(ids.cpu(), tokenizer)
    record["texts"].extend(texts)
    if len(record["samples"]) < 4:
        record["samples"].extend(texts[: 4 - len(record["samples"])])
    if maturity_ids is None:
        return
    changed = ids[:, : maturity_ids.shape[1]].cpu() != maturity_ids.cpu()
    record["prefix_revision"].extend(changed.float().mean(dim=1).tolist())
    if selected is not None:
        selected = selected.cpu()
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
    args,
    records,
    tokenizer,
):
    batch = z0_a.shape[0]
    block = args.block_length
    snapshots = prefix_snapshots(z0_a, block_model, grid, args.maturities, args)

    # Fully parallel ODE-32.
    z_parallel, _ = ode_range(
        torch.cat([z0_a, z0_b], dim=1),
        torch.zeros(batch, 2 * block, z0_a.shape[-1], device=z0_a.device),
        joint_model,
        grid,
        0,
        args.n_steps,
        args,
    )
    append_ids(
        records["parallel32"],
        common.decode(z_parallel, joint_model, z0_a.device),
        tokenizer,
    )

    # Native reencoded block Semi-AR baseline.
    z_a_final, _ = snapshots[args.n_steps]
    ids_a_final = common.decode(z_a_final, block_model, z0_a.device)
    cond_a_final = t5_reencode(ids_a_final, encoder, z0_a.dtype)
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
            )
            final_ids = common.decode(z_joint, joint_model, z0_a.device)
            append_ids(
                records[name],
                final_ids,
                tokenizer,
                maturity_ids=maturity_ids,
                selected=selected,
            )
            records[name]["hybrid_fraction"].append(float(selected.float().mean().item()))
            records[name]["condition_cosine"].append(cosine)


def finite_mean(values):
    clean = [value for value in values if value == value]
    return sum(clean) / len(clean) if clean else float("nan")


def main():
    args = parse_args()
    if any(m <= 0 or m >= args.n_steps for m in args.maturities):
        raise ValueError("maturities must lie strictly between 0 and n_steps")
    if args.block_length * 2 > 1024:
        raise ValueError("this bounded evaluator supports total length at most 1024")

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
    grid = get_sampling_steps(args.n_steps, "uniform", device=device)
    records = init_records(args)

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
            args,
            records,
            elf_tokenizer,
        )

    # The two ELF instances are both needed during generation but not during
    # GPT-2 evaluation. Release them before loading the evaluator so the
    # experiment remains safe on a GPU shared with other bounded pilots.
    block_model.cpu()
    joint_model.cpu()
    encoder.cpu()
    del block_model, joint_model, encoder, checkpoint, weights
    torch.cuda.empty_cache()

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
        metrics = common.text_metrics(
            record["texts"],
            ppl_model,
            ppl_tokenizer,
            device,
            max_length=total_length,
        )
        if name == "parallel32":
            denoiser_calls, readout_calls, t5_calls = 32, 0, 0
        elif name == "semi_ar64":
            denoiser_calls, readout_calls, t5_calls = 64, 1, 1
        else:
            maturity = int(name.rsplit("m", 1)[1])
            representation = name[len("late_") : name.rfind("_m")]
            denoiser_calls = 32 + maturity
            readout_calls = 1
            t5_calls = int(representation in ("reencoded", "hybrid"))
        metrics.update(
            {
                "denoiser_calls": denoiser_calls,
                "readout_calls": readout_calls,
                "t5_calls": t5_calls,
                "prefix_revision": finite_mean(record["prefix_revision"]),
                "highconf_prefix_revision": finite_mean(record["highconf_revision"]),
                "lowconf_prefix_revision": finite_mean(record["lowconf_revision"]),
                "hybrid_fraction": finite_mean(record["hybrid_fraction"]),
                "condition_cosine": finite_mean(record["condition_cosine"]),
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
        "representations": args.representations,
        "noise_scale": args.noise_scale,
        "sccfg": args.sccfg,
        "hybrid_confidence": args.hybrid_confidence,
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
                "prefix_revision",
                "hybrid_fraction",
                "denoiser_calls",
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
