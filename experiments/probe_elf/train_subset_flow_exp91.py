#!/usr/bin/env python3
"""EXP-91: real-OWT paired control/subset-conditioned flow pilot."""

import argparse
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
for path in (str(SRC), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import paired_conditional_revalidation_exp80 as exp80
import unified_method_eval_exp64 as common
from modules.model import ELF_B
from modules.t5_encoder import get_encoder
from utils.encoder_utils import encode_text
from utils.sampling_utils import add_noise, net_out_to_v_x, sample_timesteps


OUT_DIR = Path("outputs/exp91_subset_flow")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("control", "anchor"), required=True)
    parser.add_argument(
        "--checkpoint", default="converted/elf_b-owt-baseline_torch.pt"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--train_size", type=int, default=512)
    parser.add_argument("--val_size", type=int, default=128)
    parser.add_argument("--owt_offset", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--anchor_probability", type=float, default=0.5)
    parser.add_argument("--anchor_fraction", type=float, default=0.5)
    parser.add_argument("--anchor_t_min", type=float, default=0.20)
    parser.add_argument("--anchor_t_max", type=float, default=0.40)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


def load_model(path, max_length, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ELF_B(**common.model_config(max_length))
    missing, unexpected = model.load_state_dict(common.load_weights(payload), strict=False)
    if missing or unexpected:
        print(f"load_state_dict missing={missing} unexpected={unexpected}")
    return model.to(device)


def freeze_student(student):
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for block in student.blocks[-4:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    for module in (student.final_layer, student.self_cond_proj):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in student.parameters() if parameter.requires_grad]


def load_token_bank(args):
    pairs = exp80.load_owt_pairs(
        args.train_size + args.val_size,
        0,
        args.max_length,
        args.owt_offset,
    )
    ids = torch.tensor([suffix for _, suffix in pairs], dtype=torch.long)
    if ids.shape != (args.train_size + args.val_size, args.max_length):
        raise RuntimeError(f"unexpected OWT bank shape {tuple(ids.shape)}")
    return ids[: args.train_size], ids[args.train_size :]


def random_anchor_mask(batch, length, fraction, active, generator, device):
    scores = torch.rand(batch, length, generator=generator, device=device)
    count = max(1, min(length - 1, int(round(fraction * length))))
    selected = torch.zeros(batch, length, dtype=torch.bool, device=device)
    order = scores.topk(count, dim=1).indices
    selected.scatter_(1, order, True)
    return selected & active[:, None]


@torch.no_grad()
def teacher_prediction(teacher, z, t, sccfg):
    scale = torch.full((z.shape[0],), sccfg, dtype=z.dtype, device=z.device)
    zeros = torch.zeros_like(z)
    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"
    ):
        output = teacher(
            torch.cat([z, zeros], dim=-1),
            t,
            deterministic=True,
            self_cond_cfg_scale=scale,
            decoder_step_active=None,
        )[0]
    _, predicted_clean = net_out_to_v_x(output, z, t, t_eps=0.05)
    return predicted_clean.detach()


def make_batch(
    ids,
    indices,
    encoder,
    teacher,
    args,
    generator,
    device,
    force_mode=None,
):
    token_ids = ids[indices].to(device)
    attention = torch.ones_like(token_ids, dtype=torch.float32)
    with torch.no_grad():
        x0 = encode_text(
            input_ids=token_ids,
            attention_mask=attention,
            encoder=encoder,
            latent_mean=0.0,
            latent_std=0.2,
            use_bf16=device.type == "cuda",
        ).float()
    batch = token_ids.shape[0]
    native_t = sample_timesteps(
        batch,
        P_mean=-0.8,
        P_std=0.8,
        time_schedule="logit_normal",
        device=device,
        dtype=x0.dtype,
    )
    transition_t = args.anchor_t_min + (
        args.anchor_t_max - args.anchor_t_min
    ) * torch.rand(batch, generator=generator, device=device)
    if force_mode == "anchor":
        use_anchor = torch.ones(batch, dtype=torch.bool, device=device)
    elif force_mode == "control":
        use_anchor = torch.zeros(batch, dtype=torch.bool, device=device)
    elif args.mode == "anchor":
        use_anchor = torch.rand(
            batch, generator=generator, device=device
        ) < args.anchor_probability
    else:
        use_anchor = torch.zeros(batch, dtype=torch.bool, device=device)
    t = torch.where(use_anchor, transition_t, native_t)
    noise = torch.randn(
        x0.shape, generator=generator, device=device, dtype=x0.dtype
    )
    config = common.SamplingConfig()
    config.denoiser_noise_scale = args.noise_scale
    z = add_noise(x0, noise, t, config)
    x_teacher = teacher_prediction(teacher, z, t, args.sccfg)
    anchor_mask = random_anchor_mask(
        batch,
        args.max_length,
        args.anchor_fraction,
        use_anchor,
        generator,
        device,
    )
    z_mixed = torch.where(anchor_mask.unsqueeze(-1), x_teacher, z)
    t_expanded = t[:, None, None]
    velocity_target = (x0 - z) / torch.clamp(1.0 - t_expanded, min=0.05)
    loss_mask = ~anchor_mask
    return z_mixed, x_teacher, t, velocity_target.detach(), loss_mask, use_anchor


def objective(student, batch, args):
    z, x_teacher, t, target, loss_mask, use_anchor = batch
    scale = torch.full((z.shape[0],), args.sccfg, dtype=z.dtype, device=z.device)
    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"
    ):
        output = student(
            torch.cat([z, x_teacher], dim=-1),
            t,
            deterministic=True,
            self_cond_cfg_scale=scale,
            decoder_step_active=None,
        )[0]
    velocity, _ = net_out_to_v_x(output, z, t, t_eps=0.05)
    per_token = (velocity.float() - target.float()).pow(2).mean(-1)
    loss = per_token[loss_mask].mean()
    sync_mask = loss_mask & (~use_anchor[:, None])
    anchor_unresolved = loss_mask & use_anchor[:, None]
    sync_loss = per_token[sync_mask].mean() if sync_mask.any() else torch.zeros((), device=z.device)
    anchor_loss = per_token[anchor_unresolved].mean() if anchor_unresolved.any() else torch.zeros((), device=z.device)
    return loss, sync_loss, anchor_loss


@torch.no_grad()
def validation(student, teacher, encoder, ids, args, device):
    student.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed + 900001)
    records = {"sync": [], "anchor": []}
    for mode in records:
        for start in range(0, len(ids), args.batch_size):
            end = min(start + args.batch_size, len(ids))
            indices = torch.arange(start, end)
            batch = make_batch(
                ids, indices, encoder, teacher, args, generator, device,
                force_mode=mode,
            )
            loss, sync_loss, anchor_loss = objective(student, batch, args)
            records[mode].append(float(anchor_loss if mode == "anchor" else sync_loss))
    student.train()
    return {
        f"val_{mode}_loss": sum(values) / len(values)
        for mode, values in records.items()
    }


def main():
    args = parse_args()
    if not 0 < args.anchor_fraction < 1 or not 0 <= args.anchor_probability <= 1:
        raise ValueError("invalid anchor fraction/probability")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    train_ids, val_ids = load_token_bank(args)
    teacher = load_model(args.checkpoint, args.max_length, device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = load_model(args.checkpoint, args.max_length, device).train()
    parameters = freeze_student(student)
    _, encoder = get_encoder("t5-small", dtype=torch.float32)
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    ema = {name: value.detach().clone() for name, value in student.state_dict().items()}
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    logs = []
    order = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(args.seed))

    for step in range(1, args.steps + 1):
        offset = ((step - 1) * args.batch_size) % len(train_ids)
        if offset + args.batch_size > len(train_ids):
            order = torch.randperm(
                len(train_ids), generator=torch.Generator().manual_seed(args.seed + step)
            )
            offset = 0
        indices = order[offset : offset + args.batch_size]
        batch = make_batch(
            train_ids, indices, encoder, teacher, args, generator, device
        )
        loss, sync_loss, anchor_loss = objective(student, batch, args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        with torch.no_grad():
            for name, value in student.state_dict().items():
                ema[name].mul_(args.ema_decay).add_(
                    value.detach(), alpha=1.0 - args.ema_decay
                )
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "sync_loss": float(sync_loss.detach()),
            "anchor_loss": float(anchor_loss.detach()),
            "grad_norm": float(grad_norm),
            "anchor_example_fraction": float(batch[-1].float().mean()),
        }
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record.update(validation(student, teacher, encoder, val_ids, args, device))
            print(json.dumps(record), flush=True)
        logs.append(record)

    output_dir = Path(args.output_dir) / f"{args.mode}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint_{args.steps}.pt"
    torch.save(
        {
            "params": {name: value.detach().cpu() for name, value in student.state_dict().items()},
            "ema_params1": {name: value.detach().cpu() for name, value in ema.items()},
            "step": args.steps,
            "config": vars(args),
        },
        checkpoint_path,
    )
    (output_dir / "train_metrics.json").write_text(json.dumps(logs, indent=2))
    print(f"Saved -> {checkpoint_path}")


if __name__ == "__main__":
    main()
