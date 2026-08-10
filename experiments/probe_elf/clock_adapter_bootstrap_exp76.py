#!/usr/bin/env python3
"""EXP-76 stage-0: freeze ELF and force layerwise local-time adapters to learn."""

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
    parser.add_argument("--checkpoint", default="converted/elf_b-owt-baseline_torch.pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--bank_size", type=int, default=32)
    parser.add_argument("--heldout_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--teacher_steps", type=int, default=16)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--sccfg", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def student_config(max_length):
    cfg = common.model_config(max_length)
    cfg.update(per_token_time_conditioning=True, per_layer_time_conditioning=True)
    return cfg


def load_model(path, config, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ELF_B(**config)
    weights = checkpoint.get("ema_params1", checkpoint["params"])
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        print(f"load {path}: missing={missing} unexpected={unexpected}")
    return model.to(device)


@torch.no_grad()
def build_bank(teacher, count, batch_size, grid, args, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    state_parts, prediction_parts = [], []
    dim = common.model_config(args.max_length)["text_encoder_dim"]
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        z0 = args.noise_scale * torch.randn(
            size, args.max_length, dim, generator=generator, device=device
        )
        states, predictions = wave.teacher_trajectory(z0, teacher, grid, args.sccfg)
        state_parts.append(states.permute(1, 0, 2, 3).cpu().to(torch.bfloat16))
        prediction_parts.append(
            predictions.permute(1, 0, 2, 3).cpu().to(torch.bfloat16)
        )
    return torch.cat(state_parts), torch.cat(prediction_parts)


def batch_trajectory(bank, indices, device):
    return bank[indices.cpu()].to(device).permute(1, 0, 2, 3).float()


def velocity_trajectory(states, grid):
    dt = (grid[1:] - grid[:-1]).view(-1, 1, 1, 1)
    velocity = (states[1:] - states[:-1]) / dt
    return torch.cat([velocity, velocity[-1:]], dim=0)


def make_tau(global_t, batch, length, delta, direction, device):
    rank = torch.linspace(0.0, 1.0, length, device=device)
    offset = 1.0 - 2.0 * rank
    if direction == "rtl":
        offset = -offset
    envelope = math.sin(math.pi * float(global_t))
    tau = (float(global_t) + delta * envelope * offset).clamp(0.0, 1.0)
    return tau[None].expand(batch, -1)


def predict_velocity(model, z, x_pred, tau, sccfg):
    scale = torch.full((z.shape[0],), sccfg, dtype=z.dtype, device=z.device)
    net_out = model(
        torch.cat([z, x_pred], dim=-1),
        tau,
        deterministic=True,
        self_cond_cfg_scale=scale,
        decoder_step_active=None,
    )[0]
    velocity, _ = net_out_to_v_x(net_out, z, tau, t_eps=0.05)
    return velocity


@torch.no_grad()
def evaluate(model, state_bank, pred_bank, grid, args, device):
    model.eval()
    count = min(args.heldout_size, state_bank.shape[0])
    indices = torch.arange(state_bank.shape[0] - count, state_bank.shape[0])
    states = batch_trajectory(state_bank, indices, device)
    predictions = batch_trajectory(pred_bank, indices, device)
    velocities = velocity_trajectory(states, grid)
    global_t = 0.5
    tau_ltr = make_tau(global_t, count, args.max_length, args.delta, "ltr", device)
    tau_rtl = make_tau(global_t, count, args.max_length, args.delta, "rtl", device)
    z_ltr = wave.assemble(states, tau_ltr)
    x_ltr = wave.assemble(predictions, tau_ltr)
    target_ltr = wave.assemble(velocities, tau_ltr)
    z_rtl = wave.assemble(states, tau_rtl)
    x_rtl = wave.assemble(predictions, tau_rtl)
    target_rtl = wave.assemble(velocities, tau_rtl)
    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        v_ltr = predict_velocity(model, z_ltr, x_ltr, tau_ltr, args.sccfg)
        v_rtl = predict_velocity(model, z_rtl, x_rtl, tau_rtl, args.sccfg)
        # Change only the clock while holding the heterogeneous state fixed.
        v_clock_flip = predict_velocity(model, z_ltr, x_ltr, tau_rtl, args.sccfg)
    return {
        "heldout_mse_ltr": float(F.mse_loss(v_ltr.float(), target_ltr.float())),
        "heldout_mse_rtl": float(F.mse_loss(v_rtl.float(), target_rtl.float())),
        "fixed_state_ltr_rtl_cosine": float(
            F.cosine_similarity(
                v_ltr.float().flatten(1), v_clock_flip.float().flatten(1), dim=-1
            ).mean()
        ),
        "natural_ltr_rtl_cosine": float(
            F.cosine_similarity(
                v_ltr.float().flatten(1), v_rtl.float().flatten(1), dim=-1
            ).mean()
        ),
        "fixed_state_clock_delta": float(
            (v_ltr.float() - v_clock_flip.float()).pow(2).mean().sqrt()
        ),
    }


def main():
    args = parse_args()
    if args.bank_size <= args.heldout_size:
        raise ValueError("bank_size must exceed heldout_size")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    common.SamplingConfig.denoiser_noise_scale = args.noise_scale
    checkpoint_path = Path(args.checkpoint)
    teacher = load_model(checkpoint_path, common.model_config(args.max_length), device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = load_model(checkpoint_path, student_config(args.max_length), device)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name, parameter in student.named_parameters():
        if "local_time_projections" in name or "local_time_scales" in name:
            parameter.requires_grad_(True)
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("no local-time adapter parameters found")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    grid = get_sampling_steps(args.teacher_steps, "uniform", device=device)
    print("Building fixed teacher trajectory bank...")
    state_bank, pred_bank = build_bank(
        teacher, args.bank_size, args.batch_size, grid, args, device, args.seed + 100
    )
    initial_metrics = evaluate(student, state_bank, pred_bank, grid, args, device)
    print("initial=" + json.dumps(initial_metrics))
    train_count = args.bank_size - args.heldout_size
    generator = torch.Generator(device=device).manual_seed(args.seed + 200)
    logs = []
    student.train()
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            0, train_count, (args.batch_size,), generator=generator, device=device
        )
        states = batch_trajectory(state_bank, indices, device)
        predictions = batch_trajectory(pred_bank, indices, device)
        velocities = velocity_trajectory(states, grid)
        grid_index = int(
            torch.randint(1, args.teacher_steps, (1,), generator=generator, device=device)
        )
        global_t = grid[grid_index].item()
        direction = "ltr" if step % 2 else "rtl"
        tau = make_tau(
            global_t, args.batch_size, args.max_length, args.delta,
            direction, device,
        )
        z_wave = wave.assemble(states, tau)
        x_wave = wave.assemble(predictions, tau)
        target_velocity = wave.assemble(velocities, tau)
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            predicted_velocity = predict_velocity(
                student, z_wave, x_wave, tau, args.sccfg
            )
            loss = F.mse_loss(predicted_velocity.float(), target_velocity.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            "direction": direction,
            "global_t": global_t,
            "local_scale_mean": float(student.local_time_scales.detach().abs().mean()),
        }
        logs.append(record)
        if step == 1 or step % args.log_every == 0:
            print(json.dumps(record))
    final_metrics = evaluate(student, state_bank, pred_bank, grid, args, device)
    print("final=" + json.dumps(final_metrics))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in student.state_dict().items()}
    torch.save(
        {"params": state, "ema_params1": state, "step": args.steps, "config": vars(args)},
        output_dir / f"checkpoint_{args.steps}",
    )
    with open(output_dir / "train_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"initial": initial_metrics, "final": final_metrics, "logs": logs},
            handle,
            indent=2,
        )
    print(f"Saved -> {output_dir}")


if __name__ == "__main__":
    main()
