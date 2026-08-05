import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config, SamplingConfig
from utils.sampling_utils import (
    restore_cond, _ode_step, _sde_step, _wff_ode_step,
    make_wff_time_vector,
)


# ============================================
# Generation utilities
# ============================================

def mask_after_eos(predicted_ids: torch.Tensor, eos_token_id: int, pad_token_id: int) -> torch.Tensor:
    """Mask everything at/after first EOS token per sequence."""
    eos_mask = (predicted_ids == eos_token_id)
    keep_mask = (eos_mask.to(torch.int32).cumsum(dim=1) == 0)
    return torch.where(keep_mask, predicted_ids, torch.full_like(predicted_ids, pad_token_id))


def shift_left(x: torch.Tensor, shift_per_sample: torch.Tensor, pad_value=0, axis: int = 1) -> torch.Tensor:
    """Shift each sample left along the sequence axis; pad emptied positions."""
    if x.dim() < 2:
        raise ValueError("x must have at least batch and sequence dimensions")
    if axis < 0:
        axis = x.dim() + axis
    if axis == 0:
        raise ValueError("axis=0 is the batch axis and cannot be shifted")
    shift_per_sample = shift_per_sample.to(torch.long)
    if axis != 1:
        x = x.movedim(axis, 1)
    seq_len = x.shape[1]
    base_idx = torch.arange(seq_len, device=x.device)[None, :]
    gather_idx = shift_per_sample[:, None].to(x.device) + base_idx
    valid = gather_idx < seq_len
    gather_idx = gather_idx.clamp(0, seq_len - 1)
    if x.dim() == 2:
        shifted = torch.gather(x, 1, gather_idx)
        shifted = torch.where(valid, shifted, torch.full_like(shifted, pad_value))
    else:
        expand_shape = [-1, -1] + list(x.shape[2:])
        idx = gather_idx.view(*gather_idx.shape, *([1] * (x.dim() - 2))).expand(*expand_shape)
        valid_b = valid.view(*valid.shape, *([1] * (x.dim() - 2))).expand(*expand_shape)
        shifted = torch.gather(x, 1, idx)
        shifted = torch.where(valid_b, shifted, torch.full_like(shifted, pad_value))
    if axis != 1:
        shifted = shifted.movedim(1, axis)
    return shifted



# ============================================
# EXP-13: Decode self-conditioning mode helpers
# ============================================

@torch.no_grad()
def _apply_dec_sc_mode(
    x_pred: torch.Tensor,
    z: torch.Tensor,
    t_val: float,
    model: nn.Module,
    config,
    cond_seq: torch.Tensor,
    cond_seq_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Transform x_pred after each ODE/SDE step before using it as self-cond for the next step.

    All modes add exactly one extra model forward pass (same FLOPs as decode branch).
    'none' is a no-op (standard ELF behavior).
    """
    if not mode or mode == "none":
        return x_pred

    B, L, d = x_pred.shape
    use_bf16 = bool(getattr(config, "use_bf16", True)) and x_pred.is_cuda
    zeros_sc = torch.zeros_like(x_pred)
    # Always pass self_cond_cfg_scale=1 to match the normal ODE forward pass, which adds
    # SC-CFG tokens (+4 positions) so the total sequence length matches the RoPE precomputation.
    sc_scale = torch.ones(B, dtype=x_pred.dtype, device=x_pred.device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        if mode == "decode":
            # Standard dec_sc: decode branch at t=1 with decoder_step_active=True.
            z_in = torch.cat([x_pred.detach(), zeros_sc], dim=-1)
            t_dec = torch.ones(B, dtype=x_pred.dtype, device=x_pred.device)
            x_refined, _, _ = model(z_in, t_dec, decoder_step_active=True,
                                    self_cond_cfg_scale=sc_scale, skip_decoder_logits=True)

        elif mode == "extra_denoise":
            # Compute control: extra denoising pass at current z and t (no decode branch).
            z_in = torch.cat([z.detach(), x_pred.detach()], dim=-1)
            t_batch = torch.full((B,), t_val, dtype=x_pred.dtype, device=x_pred.device)
            x_refined, _, _ = model(z_in, t_batch, decoder_step_active=None,
                                    self_cond_cfg_scale=sc_scale)

        elif mode == "decode_shuffled":
            # Dec branch with sequence positions shuffled (destroys per-token specificity).
            idx = torch.randperm(L, device=x_pred.device)
            z_in = torch.cat([x_pred[:, idx, :].detach(), zeros_sc], dim=-1)
            t_dec = torch.ones(B, dtype=x_pred.dtype, device=x_pred.device)
            x_shuf, _, _ = model(z_in, t_dec, decoder_step_active=True,
                                  self_cond_cfg_scale=sc_scale, skip_decoder_logits=True)
            x_refined = x_shuf[:, torch.argsort(idx), :]

        elif mode == "decode_wrong_t":
            # Dec branch at wrong t (current t - 0.2, min 0.05), tests timing sensitivity.
            wrong_t = max(0.05, t_val - 0.2)
            z_in = torch.cat([x_pred.detach(), zeros_sc], dim=-1)
            t_wrong = torch.full((B,), wrong_t, dtype=x_pred.dtype, device=x_pred.device)
            x_refined, _, _ = model(z_in, t_wrong, decoder_step_active=True,
                                    self_cond_cfg_scale=sc_scale, skip_decoder_logits=True)

        elif mode == "random_residual":
            # Matched-norm random vector (tests whether any signal in x_pred_prev matters).
            norm = x_pred.detach().norm(dim=-1, keepdim=True)
            rand = torch.randn_like(x_pred)
            x_refined = rand * norm / (rand.norm(dim=-1, keepdim=True) + 1e-8)

        else:
            raise ValueError(f"Unknown dec_sc_mode: {mode}")

    return restore_cond(x_refined, cond_seq, cond_seq_mask)


# ============================================
# spec-11: Diffusion Forcing per-position guidance
# ============================================

@torch.no_grad()
def _get_df_entropy(
    x_pred: torch.Tensor,
    model: nn.Module,
    config,
    cond_seq: torch.Tensor,
    cond_seq_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-position token entropy via a decode branch pass.

    Runs the backbone in decode mode (t=1) using x_pred as self-conditioning to obtain
    vocabulary logits, then computes Shannon entropy H (nats). Costs +1 backbone forward.
    Returns H: (B, L).
    """
    B, L, _ = x_pred.shape
    use_bf16 = bool(getattr(config, "use_bf16", True)) and x_pred.is_cuda
    zeros_sc = torch.zeros_like(x_pred)
    sc_scale = torch.ones(B, dtype=x_pred.dtype, device=x_pred.device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        z_in = torch.cat([x_pred.detach(), zeros_sc], dim=-1)
        t_dec = torch.ones(B, dtype=x_pred.dtype, device=x_pred.device)
        _, logits, _ = model(z_in, t_dec, decoder_step_active=True,
                             self_cond_cfg_scale=sc_scale, skip_decoder_logits=False)

    if logits is None:
        raise RuntimeError("decode branch did not return logits; ensure skip_decoder_logits=False")
    log_p = F.log_softmax(logits.float(), dim=-1)
    H = -(torch.exp(log_p) * log_p).sum(dim=-1)  # (B, L) in nats
    return H


@torch.no_grad()
def _apply_df_step(
    z: torch.Tensor,
    x_pred: torch.Tensor,
    t_next: float,
    model: nn.Module,
    config,
    cond_seq: torch.Tensor,
    cond_seq_mask: torch.Tensor,
    sampling_config,
) -> torch.Tensor:
    """Apply Diffusion Forcing per-position commitment guidance to z.

    variant="freeze": positions with H < df_commit_thresh are frozen (z_i <- x_pred_i).
    variant="soft":   all positions interpolated toward x_pred proportional to confidence.

    Returns updated z (x_pred unchanged).
    """
    variant = getattr(sampling_config, "df_variant", "none")
    if not variant or variant == "none":
        return z

    df_t_min = float(getattr(sampling_config, "df_t_min", 0.0))
    if t_next < df_t_min:
        return z

    H = _get_df_entropy(x_pred, model, config, cond_seq, cond_seq_mask)  # (B, L)

    if variant == "freeze":
        thresh = float(getattr(sampling_config, "df_commit_thresh", 0.5))
        committed = (H < thresh).unsqueeze(-1)  # (B, L, 1)
        z_new = torch.where(committed, x_pred.detach(), z)

    elif variant == "soft":
        alpha = float(getattr(sampling_config, "df_soft_alpha", 0.5))
        log_V = math.log(32100)
        confidence = (1.0 - H / log_V).clamp(0.0, 1.0).unsqueeze(-1)  # (B, L, 1)
        z_new = (1.0 - alpha * confidence) * z + alpha * confidence * x_pred.detach()

    else:
        raise ValueError(f"Unknown df_variant: {variant}")

    return restore_cond(z_new, cond_seq, cond_seq_mask)


# ============================================
# Single-batch sampling (PyTorch)
# ============================================

@torch.no_grad()
def _generate_samples_single_batch(
    model: nn.Module,
    generator: torch.Generator,
    z: torch.Tensor,
    t_steps: torch.Tensor,
    cond_seq: Optional[torch.Tensor],
    cond_seq_mask: Optional[torch.Tensor],
    config: Config,
    sampling_config: SamplingConfig,
    cfg_scale: float,
    self_cond_cfg_scale: float,
) -> torch.Tensor:
    """Generate samples for a single batch (PyTorch Euler / SDE rollout)."""
    method = sampling_config.sampling_method
    batch_size, max_length, d_model = z.shape
    if cond_seq is None:
        cond_seq = torch.zeros((batch_size, max_length, d_model), dtype=z.dtype, device=z.device)
        cond_seq_mask = torch.zeros((batch_size, max_length), dtype=z.dtype, device=z.device)

    step_kwargs = dict(
        model=model, config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )

    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)

    n = t_steps.shape[0]
    sde_gamma = getattr(sampling_config, "sde_gamma", 0.0)
    sar_alpha = float(getattr(sampling_config, "sar_alpha", 0.0))
    wff_delta = float(getattr(sampling_config, "wff_delta", 0.0))
    wff_order = str(getattr(sampling_config, "wff_order", "ltr"))
    if method == "wff_ode" and sar_alpha > 0.0:
        raise ValueError("wff_ode and sar_alpha cannot be enabled together")

    # SAR: fix the initial noise and precompute per-position time offsets.
    # t_i = clamp(t + alpha * (0.5 - i/L), 0, 1) shifts left positions ahead and right behind.
    eps_fixed = None
    pos_offset = None
    if sar_alpha > 0.0:
        eps_fixed = z.clone()  # z at t=0 is pure scaled noise
        pos = torch.arange(max_length, dtype=z.dtype, device=z.device)
        pos_offset = sar_alpha * (0.5 - pos / max_length)  # shape (L,)

    # EXP-01: trajectory saving setup.
    save_traj = bool(getattr(sampling_config, "save_trajectory", False))
    traj_steps = [] if save_traj else None

    use_bf16 = bool(getattr(config, "use_bf16", True)) and z.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(n - 2):
            t = t_steps[i].item()
            t_next = t_steps[i + 1].item()
            z_before = z.detach().cpu().clone() if save_traj else None
            if method == "sde":
                z, x_pred = _sde_step(
                    z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                    gamma=sde_gamma, generator=generator, **step_kwargs,
                )
            elif method == "ode":
                z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs)
            elif method == "wff_ode":
                tau = make_wff_time_vector(
                    t, max_length, wff_delta, wff_order,
                    device=z.device, dtype=z.dtype,
                )
                tau_next = make_wff_time_vector(
                    t_next, max_length, wff_delta, wff_order,
                    device=z.device, dtype=z.dtype,
                )
                z, x_pred = _wff_ode_step(
                    z=z, t=tau, t_next=tau_next, x_pred_prev=x_pred, **step_kwargs,
                )
            else:
                raise ValueError(f"Invalid sampling method: {method}")

            if save_traj:
                traj_steps.append({
                    "t": t,
                    "t_next": t_next,
                    "z_t": z_before,
                    "x_pred": x_pred.detach().cpu().clone(),
                })

            # SAR: discard Euler z, reconstruct from x_pred with per-position t_i.
            if sar_alpha > 0.0:
                t_i = (t_next + pos_offset).clamp(0.0, 1.0)  # (L,)
                t_i_bld = t_i[None, :, None]                  # (1, L, 1)
                z = t_i_bld * x_pred + (1.0 - t_i_bld) * eps_fixed
                z = restore_cond(z, cond_seq, cond_seq_mask)

            # EXP-13: refine x_pred before it is used as self-cond in the next step.
            dec_sc_mode = getattr(sampling_config, "dec_sc_mode", "none")
            dec_sc_t_min = float(getattr(sampling_config, "dec_sc_apply_t_min", 0.0))
            if dec_sc_mode and dec_sc_mode != "none" and t_next >= dec_sc_t_min:
                x_pred = _apply_dec_sc_mode(
                    x_pred=x_pred, z=z, t_val=t_next,
                    model=model, config=config,
                    cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
                    mode=dec_sc_mode,
                )

            # spec-11: Diffusion Forcing per-position commitment guidance.
            df_variant = getattr(sampling_config, "df_variant", "none")
            if df_variant and df_variant != "none":
                z = _apply_df_step(
                    z=z, x_pred=x_pred, t_next=t_next,
                    model=model, config=config,
                    cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
                    sampling_config=sampling_config,
                )

        # Last step always with ODE (no SAR reconstruction — z at t=1 is the decoded output).
        t = t_steps[-2].item()
        t_next = t_steps[-1].item()
        z_before = z.detach().cpu().clone() if save_traj else None
        if method == "wff_ode":
            tau = make_wff_time_vector(
                t, max_length, wff_delta, wff_order,
                device=z.device, dtype=z.dtype,
            )
            tau_next = make_wff_time_vector(
                t_next, max_length, wff_delta, wff_order,
                device=z.device, dtype=z.dtype,
            )
            z, x_pred = _wff_ode_step(
                z=z, t=tau, t_next=tau_next, x_pred_prev=x_pred, **step_kwargs,
            )
        else:
            z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs)
        if save_traj:
            traj_steps.append({
                "t": t,
                "t_next": t_next,
                "z_t": z_before,
                "x_pred": x_pred.detach().cpu().clone(),
            })

    if save_traj:
        import os
        traj_dir = getattr(sampling_config, "trajectory_save_dir", "results/exp01/trajectories")
        os.makedirs(traj_dir, exist_ok=True)
        # Use a unique suffix per call to avoid collision across batches.
        import time
        suffix = f"{int(time.time() * 1000) % 10000000:07d}"
        torch.save(traj_steps, os.path.join(traj_dir, f"traj_{suffix}.pt"))

    return z


@torch.no_grad()
def _dlm_decode_batch(z: torch.Tensor, model: nn.Module, t_final_val,
                      config, self_cond_cfg_scale: float,
                      _chunk: int = 8) -> torch.Tensor:
    """Decode z -> tokens with the DLM decoder head.

    Chunks over the batch dimension to avoid OOM when vocab × seq_len × batch is large.
    """
    batch_size = z.shape[0]
    if batch_size > _chunk:
        return torch.cat([
            _dlm_decode_batch(z[i:i + _chunk], model, t_final_val, config, self_cond_cfg_scale, _chunk)
            for i in range(0, batch_size, _chunk)
        ], dim=0)

    if isinstance(t_final_val, torch.Tensor) and t_final_val.dim() == 0:
        t_final = torch.full((batch_size,), t_final_val.item(), dtype=z.dtype, device=z.device)
    else:
        t_final = torch.full((batch_size,), float(t_final_val), dtype=z.dtype, device=z.device)
    sc_batch = (
        torch.full((batch_size,), float(self_cond_cfg_scale), dtype=z.dtype, device=z.device)
        if config.num_self_cond_cfg_tokens > 0 else None
    )
    z_input = torch.cat([z, torch.zeros_like(z)], dim=-1) if config.self_cond_prob > 0 else z
    use_bf16 = bool(getattr(config, "use_bf16", True)) and z.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        _, decoder_logits, _ = model(
            z_input, t_final, deterministic=True,
            self_cond_cfg_scale=sc_batch,
            decoder_step_active=True,
        )
    return decoder_logits.argmax(dim=-1)


def _build_run_name(sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
                    time_schedule, sde_gamma, suffix, sar_alpha=0.0, dec_sc_mode="none",
                    dec_sc_apply_t_min=0.0, df_variant="none", df_commit_thresh=0.5,
                    df_soft_alpha=0.5, df_t_min=0.0,
                    wff_delta=0.0, wff_order="ltr"):
    ts_str = f"-ts_{time_schedule}"
    sccfg_str = f"-sccfg{self_cond_cfg_scale}" if self_cond_cfg_scale != 1.0 else ""
    sde_str = f"-gamma{sde_gamma}" if sampling_method == "sde" else ""
    sar_str = f"-sar{sar_alpha}" if sar_alpha > 0.0 else ""
    wff_str = (
        f"-wff{wff_delta}-{wff_order}" if sampling_method == "wff_ode" else ""
    )
    dec_str = f"-decsc_{dec_sc_mode}" if dec_sc_mode and dec_sc_mode != "none" else ""
    tmin_str = f"-tmin{dec_sc_apply_t_min}" if dec_sc_apply_t_min > 0.0 and dec_str else ""
    if df_variant and df_variant != "none":
        if df_variant == "freeze":
            df_str = f"-df_freeze{df_commit_thresh}"
        elif df_variant == "soft":
            df_str = f"-df_soft{df_soft_alpha}"
        else:
            df_str = f"-df_{df_variant}"
        df_tmin_str = f"-dftmin{df_t_min}" if df_t_min > 0.0 else ""
    else:
        df_str = ""
        df_tmin_str = ""
    return f"{sampling_method}-steps{num_sampling_steps}-cfg{cfg_scale}{sccfg_str}{ts_str}{sde_str}{sar_str}{wff_str}{dec_str}{tmin_str}{df_str}{df_tmin_str}-{suffix}"
