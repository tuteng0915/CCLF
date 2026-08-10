#!/usr/bin/env python3
"""EXP-77: distill block-local transitions inside a staggered sequence."""

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

import train_wave_distill_exp73 as wave
import unified_method_eval_exp64 as common
from modules.model import ELF_B
from utils.sampling_utils import get_sampling_steps, net_out_to_v_x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_checkpoint", required=True)
    parser.add_argument(
        "--teacher_checkpoint", default="converted/elf_b-owt-baseline_torch.pt"
    )
    parser.add_argument("--mode", choices=("sync_control", "off_policy", "on_policy"), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--teacher_steps", type=int, default=16)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--unroll_steps", type=int, default=4)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--canonical_context", action="store_true")
    parser.add_argument("--direction", choices=("ltr", "rtl"), default="ltr")
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lambda_x", type=float, default=1.0)
    parser.add_argument("--lambda_sync", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def config(max_length):
    cfg = common.model_config(max_length)
    cfg.update(per_token_time_conditioning=True, per_layer_time_conditioning=True)
    return cfg


def load_model(path, model_config, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ELF_B(**model_config)
    weights = checkpoint.get("ema_params1", checkpoint["params"])
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        print(f"load {path}: missing={missing} unexpected={unexpected}")
    return model.to(device)


def group_map(length, groups, direction, device):
    group = torch.div(torch.arange(length, device=device) * groups, length, rounding_mode="floor")
    return group if direction == "ltr" else groups - 1 - group


def staggered_tau(global_t, batch, length, delta, direction, device):
    rank = torch.linspace(0.0, 1.0, length, device=device)
    offset = 1.0 - 2.0 * rank
    if direction == "rtl":
        offset = -offset
    envelope = math.sin(math.pi * float(global_t))
    return (float(global_t) + delta * envelope * offset).clamp(0.0, 1.0)[None].expand(batch, -1)


def predict(student, z, x_pred, tau, selected, sccfg, canonical):
    z_input = z
    if canonical:
        z_input = torch.where(selected.unsqueeze(-1), z, x_pred)
    scale = torch.full((z.shape[0],), sccfg, dtype=z.dtype, device=z.device)
    net_out = student(
        torch.cat([z_input, x_pred], dim=-1),
        tau,
        deterministic=True,
        self_cond_cfg_scale=scale,
        decoder_step_active=None,
    )[0]
    velocity, clean = net_out_to_v_x(net_out, z_input, tau, t_eps=0.05)
    return velocity, clean


def masked_mse(left, right, selected):
    values = (left.float() - right.float()).pow(2).mean(-1)
    return values[selected].mean()


def update_ema(ema, model, decay):
    with torch.no_grad():
        for name, value in model.state_dict().items():
            ema[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


def main():
    args = parse_args()
    if args.groups > args.max_length or args.unroll_steps <= 0:
        raise ValueError("invalid groups or unroll_steps")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    common.SamplingConfig.denoiser_noise_scale = args.noise_scale
    teacher = load_model(args.teacher_checkpoint, common.model_config(args.max_length), device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = load_model(args.student_checkpoint, config(args.max_length), device).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    ema = {name: value.detach().clone() for name, value in student.state_dict().items()}
    grid = get_sampling_steps(args.teacher_steps, "uniform", device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    groups = group_map(args.max_length, args.groups, args.direction, device)
    dim = common.model_config(args.max_length)["text_encoder_dim"]
    logs = []

    for step in range(1, args.steps + 1):
        z0 = args.noise_scale * torch.randn(
            args.batch_size, args.max_length, dim, generator=generator, device=device
        )
        teacher_states, teacher_predictions = wave.teacher_trajectory(
            z0, teacher, grid, args.sccfg
        )
        start_index = int(
            torch.randint(1, args.teacher_steps - 1, (1,), generator=generator, device=device)
        )
        global_t = grid[start_index].item()
        if args.mode == "sync_control":
            tau = torch.full(
                (args.batch_size, args.max_length), global_t, device=device
            )
            selected = torch.ones(
                args.batch_size, args.max_length, dtype=torch.bool, device=device
            )
            transitions = 1
        else:
            tau = staggered_tau(
                global_t, args.batch_size, args.max_length, args.delta,
                args.direction, device,
            )
            target_group = int(
                torch.randint(0, args.groups, (1,), generator=generator, device=device)
            )
            selected = (groups == target_group)[None].expand(args.batch_size, -1)
            transitions = args.unroll_steps if args.mode == "on_policy" else 1
        z_current = wave.assemble(teacher_states, tau)
        x_current = wave.assemble(teacher_predictions, tau)
        total_state = torch.zeros((), device=device)
        total_x = torch.zeros((), device=device)
        on_policy_probability = 0.5 * step / args.steps if args.mode == "on_policy" else 0.0

        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            for offset in range(transitions):
                if args.mode == "on_policy" and offset:
                    target_group = (target_group + 1) % args.groups
                    selected = (groups == target_group)[None].expand(args.batch_size, -1)
                tau_next = tau.clone()
                local_dt = 1.0 / args.teacher_steps
                tau_next = torch.where(selected, (tau_next + local_dt).clamp(max=1.0), tau_next)
                target_z = wave.assemble(teacher_states, tau_next)
                target_x = wave.assemble(teacher_predictions, tau)
                velocity, clean = predict(
                    student, z_current, x_current, tau, selected,
                    args.sccfg, args.canonical_context,
                )
                predicted_z = z_current + (tau_next - tau).unsqueeze(-1) * velocity
                total_state = total_state + masked_mse(predicted_z, target_z, selected)
                total_x = total_x + masked_mse(clean, target_x, selected)
                if offset + 1 < transitions:
                    choose_student = (
                        torch.rand(
                            args.batch_size, 1, 1, generator=generator, device=device
                        ) < on_policy_probability
                    )
                    teacher_current = target_z.detach()
                    z_current = torch.where(
                        selected.unsqueeze(-1),
                        torch.where(choose_student, predicted_z.detach(), teacher_current),
                        teacher_current,
                    )
                    teacher_clean = wave.assemble(teacher_predictions, tau_next).detach()
                    x_current = torch.where(
                        selected.unsqueeze(-1),
                        torch.where(choose_student, clean.detach(), teacher_clean),
                        teacher_clean,
                    )
                    tau = tau_next

            scalar_tau = torch.full_like(tau, global_t)
            scalar_next = torch.full_like(tau, min(global_t + 1.0 / args.teacher_steps, 1.0))
            sync_z = wave.assemble(teacher_states, scalar_tau)
            sync_x = wave.assemble(teacher_predictions, scalar_tau)
            all_selected = torch.ones_like(selected)
            sync_velocity, _ = predict(
                student, sync_z, sync_x, scalar_tau, all_selected,
                args.sccfg, False,
            )
            predicted_sync = sync_z + (scalar_next - scalar_tau).unsqueeze(-1) * sync_velocity
            target_sync = wave.assemble(teacher_states, scalar_next)
            sync_loss = F.mse_loss(predicted_sync.float(), target_sync.float())
            loss = (
                total_state / transitions
                + args.lambda_x * total_x / transitions
                + args.lambda_sync * sync_loss
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        update_ema(ema, student, args.ema_decay)
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "state_loss": float((total_state / transitions).detach()),
            "x_loss": float((total_x / transitions).detach()),
            "sync_loss": float(sync_loss.detach()),
            "grad_norm": float(grad_norm),
            "on_policy_probability": on_policy_probability,
            "local_scale_mean": float(student.local_time_scales.detach().abs().mean()),
        }
        logs.append(record)
        if step == 1 or step % args.log_every == 0:
            print(json.dumps(record))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "params": {name: value.detach().cpu() for name, value in student.state_dict().items()},
            "ema_params1": {name: value.detach().cpu() for name, value in ema.items()},
            "step": args.steps,
            "config": vars(args),
        },
        output_dir / f"checkpoint_{args.steps}",
    )
    with open(output_dir / "train_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(logs, handle, indent=2)
    print(f"Saved -> {output_dir}")


if __name__ == "__main__":
    main()
