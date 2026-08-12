#!/usr/bin/env python3
"""EXP-92: paired conditional/on-policy Triggered Subset Flow screen."""

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
from utils.sampling_utils import (
    _ode_step,
    add_noise,
    get_sampling_steps,
    net_out_to_v_x,
    restore_cond,
)


OUT_DIR = Path("outputs/exp92_onpolicy_subset_flow")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("control", "conditional_oracle", "conditional_onpolicy"),
        required=True,
    )
    parser.add_argument(
        "--checkpoint", default="converted/elf_b-owt-baseline_torch.pt"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--prefix_length", type=int, default=64)
    parser.add_argument("--train_size", type=int, default=512)
    parser.add_argument("--val_size", type=int, default=64)
    parser.add_argument("--owt_offset", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--conditional_probability", type=float, default=0.5)
    parser.add_argument("--anchor_densities", type=float, nargs="+", default=(0.25, 0.50, 0.75))
    parser.add_argument("--hold_horizons", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--trigger_time", type=float, default=0.30)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--lambda_mix", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=100)
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
    count = args.train_size + args.val_size
    suffix_length = args.max_length - args.prefix_length
    pairs = exp80.load_owt_pairs(
        count, args.prefix_length, suffix_length, args.owt_offset
    )
    ids = torch.tensor([prefix + suffix for prefix, suffix in pairs], dtype=torch.long)
    if ids.shape != (count, args.max_length):
        raise RuntimeError(f"unexpected OWT bank shape {tuple(ids.shape)}")
    return ids[: args.train_size], ids[args.train_size :]


@torch.no_grad()
def encode_batch(token_ids, encoder, args, device, generator):
    token_ids = token_ids.to(device)
    attention = torch.ones_like(token_ids, dtype=torch.float32)
    x0 = encode_text(
        input_ids=token_ids,
        attention_mask=attention,
        encoder=encoder,
        latent_mean=0.0,
        latent_std=0.2,
        use_bf16=device.type == "cuda",
    ).float()

    prefix_ids = token_ids[:, : args.prefix_length]
    prefix_attention = torch.ones_like(prefix_ids, dtype=torch.float32)
    prefix_latents = encode_text(
        input_ids=prefix_ids,
        attention_mask=prefix_attention,
        encoder=encoder,
        latent_mean=0.0,
        latent_std=0.2,
        use_bf16=device.type == "cuda",
    ).float()
    conditional = torch.rand(
        token_ids.shape[0], generator=generator, device=device
    ) < args.conditional_probability
    cond_seq = torch.zeros_like(x0)
    cond_seq[:, : args.prefix_length] = prefix_latents
    cond_mask = torch.zeros(
        token_ids.shape[0], args.max_length, dtype=x0.dtype, device=device
    )
    cond_mask[conditional, : args.prefix_length] = 1.0
    cond_seq = cond_seq * cond_mask.unsqueeze(-1)
    return x0, cond_seq, cond_mask, conditional


def random_anchor_mask(eligible, fraction, generator):
    selected = torch.zeros_like(eligible, dtype=torch.bool)
    scores = torch.rand(
        eligible.shape, generator=generator, device=eligible.device
    )
    for row in range(eligible.shape[0]):
        candidates = torch.nonzero(eligible[row], as_tuple=False).flatten()
        if candidates.numel() < 2:
            continue
        count = max(1, min(candidates.numel() - 1, int(round(fraction * candidates.numel()))))
        order = torch.topk(scores[row, candidates], count).indices
        selected[row, candidates[order]] = True
    return selected


@torch.no_grad()
def teacher_prediction(teacher, z, t, args, cond_seq, cond_mask):
    scale = torch.full((z.shape[0],), args.sccfg, dtype=z.dtype, device=z.device)
    zeros = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
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
    return restore_cond(predicted_clean.detach(), cond_seq, cond_mask)


def make_sync_batch(x0, cond_seq, cond_mask, teacher, args, generator):
    batch = x0.shape[0]
    normal = torch.randn(
        batch, generator=generator, device=x0.device, dtype=x0.dtype
    )
    t = torch.sigmoid(-0.8 + 0.8 * normal)
    noise = torch.randn(
        x0.shape, generator=generator, device=x0.device, dtype=x0.dtype
    )
    cfg = common.SamplingConfig()
    cfg.denoiser_noise_scale = args.noise_scale
    z = restore_cond(add_noise(x0, noise, t, cfg), cond_seq, cond_mask)
    x_teacher = teacher_prediction(teacher, z, t, args, cond_seq, cond_mask)
    target = (x0 - z) / torch.clamp(1.0 - t[:, None, None], min=0.05)
    loss_mask = cond_mask < 0.5
    return z, x_teacher, t, target.detach(), loss_mask


def sample_choice(values, generator):
    index = int(torch.randint(
        len(values), (), generator=generator, device=generator.device
    ).item())
    return values[index]


def make_oracle_subset_batch(x0, cond_seq, cond_mask, teacher, args, generator):
    batch = x0.shape[0]
    t = torch.full(
        (batch,), args.trigger_time, dtype=x0.dtype, device=x0.device
    )
    noise = torch.randn(
        x0.shape, generator=generator, device=x0.device, dtype=x0.dtype
    )
    cfg = common.SamplingConfig()
    cfg.denoiser_noise_scale = args.noise_scale
    z = restore_cond(add_noise(x0, noise, t, cfg), cond_seq, cond_mask)
    x_teacher = teacher_prediction(teacher, z, t, args, cond_seq, cond_mask)
    density = float(sample_choice(args.anchor_densities, generator))
    anchors = random_anchor_mask(cond_mask < 0.5, density, generator)
    z_mixed = restore_cond(
        torch.where(anchors.unsqueeze(-1), x_teacher, z), cond_seq, cond_mask
    )
    target = (x0 - z_mixed) / torch.clamp(1.0 - t[:, None, None], min=0.05)
    loss_mask = (cond_mask < 0.5) & (~anchors)
    metadata = {"density": density, "horizon": 0, "state_time": args.trigger_time}
    return (z_mixed, x_teacher, t, target.detach(), loss_mask), metadata


@torch.no_grad()
def make_onpolicy_subset_batch(x0, cond_seq, cond_mask, teacher, args, generator):
    cfg = common.SamplingConfig()
    cfg.denoiser_noise_scale = args.noise_scale
    grid = get_sampling_steps(args.n_steps, "uniform", device=x0.device)
    trigger_index = next(
        index
        for index in range(grid.shape[0] - 1)
        if float(grid[index + 1]) >= args.trigger_time
    )
    horizon = int(sample_choice(args.hold_horizons, generator))
    density = float(sample_choice(args.anchor_densities, generator))

    z = torch.randn(
        x0.shape, generator=generator, device=x0.device, dtype=x0.dtype
    ) * args.noise_scale
    z = restore_cond(z, cond_seq, cond_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
    active_seq, active_mask = cond_seq.clone(), cond_mask.clone()
    anchors = None
    end_index = min(trigger_index + horizon, grid.shape[0] - 2)

    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=x0.device.type == "cuda"
    ):
        for index in range(end_index + 1):
            z, x_pred = _ode_step(
                model=teacher,
                z=z,
                t=grid[index].item(),
                t_next=grid[index + 1].item(),
                x_pred_prev=x_pred,
                config=cfg,
                cfg_scale=1.0,
                self_cond_cfg_scale=args.sccfg,
                cond_seq=active_seq,
                cond_seq_mask=active_mask,
            )
            if index == trigger_index:
                anchors = random_anchor_mask(cond_mask < 0.5, density, generator)
                content = x_pred.detach()
                active_seq = torch.where(anchors.unsqueeze(-1), content, cond_seq)
                active_mask = torch.maximum(active_mask, anchors.to(active_mask.dtype))
            z = restore_cond(z, active_seq, active_mask)
            x_pred = restore_cond(x_pred, active_seq, active_mask)

    if anchors is None:
        raise RuntimeError("on-policy trigger was not reached")
    state_time = float(grid[end_index + 1])
    t = torch.full((x0.shape[0],), state_time, dtype=x0.dtype, device=x0.device)
    # This is exactly the previous-step self-conditioning state that the native
    # sampler would feed at `state_time`; recomputing with zero self-condition
    # would reintroduce an off-policy mismatch.
    x_teacher = x_pred.detach()
    target = (x0 - z) / torch.clamp(1.0 - t[:, None, None], min=0.05)
    loss_mask = (cond_mask < 0.5) & (~anchors)
    metadata = {"density": density, "horizon": horizon, "state_time": state_time}
    return (z, x_teacher, t, target.detach(), loss_mask), metadata


def objective(student, batch, args):
    z, x_teacher, t, target, loss_mask = batch
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
    if not loss_mask.any():
        raise RuntimeError("empty objective mask")
    return per_token[loss_mask].mean()


def make_second_batch(mode, x0, cond_seq, cond_mask, teacher, args, generator):
    if mode == "control":
        return make_sync_batch(x0, cond_seq, cond_mask, teacher, args, generator), {
            "density": 0.0,
            "horizon": 0,
            "state_time": float("nan"),
        }
    if mode == "conditional_oracle":
        return make_oracle_subset_batch(
            x0, cond_seq, cond_mask, teacher, args, generator
        )
    return make_onpolicy_subset_batch(
        x0, cond_seq, cond_mask, teacher, args, generator
    )


@torch.no_grad()
def validation(student, teacher, encoder, ids, args, device):
    student.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed + 900001)
    sync_values, second_values = [], []
    limit = min(len(ids), max(args.batch_size, 16))
    for start in range(0, limit, args.batch_size):
        token_ids = ids[start : start + args.batch_size]
        x0, cond_seq, cond_mask, _ = encode_batch(
            token_ids, encoder, args, device, generator
        )
        sync_values.append(
            float(objective(student, make_sync_batch(
                x0, cond_seq, cond_mask, teacher, args, generator
            ), args))
        )
        second, _ = make_second_batch(
            args.mode, x0, cond_seq, cond_mask, teacher, args, generator
        )
        second_values.append(float(objective(student, second, args)))
    student.train()
    return {
        "val_sync_loss": sum(sync_values) / len(sync_values),
        "val_second_loss": sum(second_values) / len(second_values),
    }


def save_checkpoint(student, ema, args, step, logs):
    output_dir = Path(args.output_dir) / f"{args.mode}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint_{step}.pt"
    torch.save(
        {
            "params": {name: value.detach().cpu() for name, value in student.state_dict().items()},
            "ema_params1": {name: value.detach().cpu() for name, value in ema.items()},
            "step": step,
            "config": vars(args),
        },
        checkpoint_path,
    )
    (output_dir / "train_metrics.json").write_text(json.dumps(logs, indent=2))
    print(f"Saved -> {checkpoint_path}", flush=True)


def main():
    args = parse_args()
    if not 0 <= args.conditional_probability <= 1:
        raise ValueError("conditional_probability must be in [0, 1]")
    if not all(0 < value < 1 for value in args.anchor_densities):
        raise ValueError("anchor densities must be in (0, 1)")
    if not all(value >= 1 for value in args.hold_horizons):
        raise ValueError("hold horizons must be positive")
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
    cpu_generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(train_ids), generator=cpu_generator)
    logs = []

    for step in range(1, args.steps + 1):
        offset = ((step - 1) * args.batch_size) % len(train_ids)
        if offset == 0 and step > 1:
            order = torch.randperm(len(train_ids), generator=cpu_generator)
        indices = order[offset : offset + args.batch_size]
        if len(indices) < args.batch_size:
            indices = order[: args.batch_size]
        data_generator = torch.Generator(device=device).manual_seed(
            args.seed * 1000003 + step * 10 + 1
        )
        sync_generator = torch.Generator(device=device).manual_seed(
            args.seed * 1000003 + step * 10 + 2
        )
        second_generator = torch.Generator(device=device).manual_seed(
            args.seed * 1000003 + step * 10 + 3
        )
        x0, cond_seq, cond_mask, conditional = encode_batch(
            train_ids[indices], encoder, args, device, data_generator
        )
        sync_batch = make_sync_batch(
            x0, cond_seq, cond_mask, teacher, args, sync_generator
        )
        second_batch, metadata = make_second_batch(
            args.mode, x0, cond_seq, cond_mask, teacher, args, second_generator
        )
        sync_loss = objective(student, sync_batch, args)
        second_loss = objective(student, second_batch, args)
        loss = sync_loss + args.lambda_mix * second_loss

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
            "second_loss": float(second_loss.detach()),
            "grad_norm": float(grad_norm),
            "conditional_fraction": float(conditional.float().mean()),
            **metadata,
        }
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record.update(validation(student, teacher, encoder, val_ids, args, device))
            print(json.dumps(record), flush=True)
        logs.append(record)
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(student, ema, args, step, logs)


if __name__ == "__main__":
    main()
