"""One mini-batch forward/backward for the ELF diffusion language model.

Each example in the batch independently picks the decoder (CE) or denoiser
(L2) branch via a Bernoulli draw at `decoder_prob`. A single forward consumes
a mixed input (decoder_z for decoder rows, denoiser_z for denoiser rows) and
both heads run; the CE / L2 losses are then masked to their respective rows
and combined with a single denominator. Self-conditioning + CFG guidance is
applied on the denoiser branch only.
"""

import contextlib
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.train_utils import TrainState, ema_update
from utils.encoder_utils import encode_text
from utils.sampling_utils import (
    sample_cfg_scale, add_noise, sample_timesteps, sample_wff_timesteps,
    _expand_time_like,
    net_out_to_v_x, restore_cond,
)


def _trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def _kd_omega_gate(t, k, t_low=0.25, t_high=0.95):
    """Smooth plateau gate used by the original JAX KD objective."""
    return torch.sigmoid(k * (t - t_low)) * (1.0 - torch.sigmoid(k * (t - t_high)))


def _kd_kl_per_token(teacher_logits, student_logits, temperature):
    """Temperature-scaled KL(teacher || student), without reduction."""
    teacher_log_soft = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    teacher_soft = teacher_log_soft.exp()
    student_log_soft = F.log_softmax(student_logits.float() / temperature, dim=-1)
    return (
        teacher_soft * (teacher_log_soft - student_log_soft)
    ).sum(dim=-1) * (temperature ** 2)


def train_step(
    state: TrainState,
    encoder: nn.Module,
    batch: Dict[str, torch.Tensor],
    config,
) -> Tuple[TrainState, Dict[str, float]]:
    """Perform a single training step."""
    device = next(state.model.parameters()).device
    dtype = next(state.model.parameters()).dtype
    use_bf16 = bool(getattr(config, "use_bf16", True)) and device.type == "cuda"
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob
    latent_mean, latent_std = config.latent_mean, config.latent_std
    decoder_prob = config.decoder_prob
    decoder_noise_scale = config.decoder_noise_scale

    gen = state.dropout_generator

    # encoder_attention_mask: cond sees cond, x sees all
    input_ids = batch["input_ids"].to(device, non_blocking=True).long()
    encoder_attention_mask = batch["encoder_attention_mask"].to(device, dtype=torch.float32, non_blocking=True)
    cond_seq_mask = batch["cond_seq_mask"].to(device, dtype=torch.float32, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, dtype=torch.float32, non_blocking=True)
    label_drop_mask = batch.get("label_drop_mask",
                                torch.zeros((input_ids.shape[0],), dtype=torch.bool)).to(device, non_blocking=True)

    # Label drop before encoding: prevent target tokens from attending to
    # condition tokens so x0 is truly unconditional for dropped samples.
    if config.label_drop_prob > 0:
        drop = label_drop_mask.to(dtype=torch.float32).reshape(-1, 1, 1)  # (B, 1, 1)
        cond_mask = cond_seq_mask  # (B, S)
        # block_mask is 1 only at (non-cond row, cond col) — leaves cond↔cond unchanged
        block_mask = (1 - cond_mask).unsqueeze(-1) * cond_mask.unsqueeze(1)
        encoder_attention_mask = encoder_attention_mask * (1 - drop * block_mask)

    x0 = encode_text(
        input_ids=input_ids,
        attention_mask=encoder_attention_mask,
        encoder=encoder,
        latent_mean=latent_mean,
        latent_std=latent_std,
        use_bf16=use_bf16,
    ).to(dtype)

    batch_size, seq_length = x0.shape[0], x0.shape[1]

    t = sample_timesteps(
        batch_size,
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule,
        device=device, dtype=dtype,
        cliff_mix_prob=float(getattr(config, 'cliff_mix_prob', 0.6)),
        cliff_lo=float(getattr(config, 'cliff_lo', 0.10)),
        cliff_hi=float(getattr(config, 'cliff_hi', 0.45)),
    )

    # Native WFF training uses a local time per token.  Setting
    # per_token_time_conditioning=true with wff_train_prob=0 creates the
    # architecture-matched synchronous fine-tuning control.
    if bool(getattr(config, "per_token_time_conditioning", False)):
        curriculum_steps = int(getattr(config, "wff_curriculum_steps", 0))
        if curriculum_steps > 0:
            curriculum = min(max(float(state.step) / curriculum_steps, 0.0), 1.0)
            probability_start = float(getattr(config, "wff_train_prob_start", 0.0))
            probability_end = float(getattr(config, "wff_train_prob", 0.0))
            probability = probability_start + curriculum * (
                probability_end - probability_start
            )
            delta_start = float(getattr(config, "wff_delta_start", 0.0))
            delta_min_end = float(getattr(config, "wff_delta_min", 0.05))
            delta_max_end = float(getattr(config, "wff_delta_max", 0.20))
            delta_min = delta_start + curriculum * (delta_min_end - delta_start)
            delta_max = delta_start + curriculum * (delta_max_end - delta_start)
        else:
            probability = float(getattr(config, "wff_train_prob", 0.0))
            delta_min = float(getattr(config, "wff_delta_min", 0.05))
            delta_max = float(getattr(config, "wff_delta_max", 0.20))
        denoiser_t, wff_use_wave, wff_delta, _wff_order = sample_wff_timesteps(
            t,
            seq_length,
            probability=probability,
            delta_min=delta_min,
            delta_max=delta_max,
            ltr_probability=float(getattr(config, "wff_ltr_prob", 0.5)),
            rtl_probability=float(getattr(config, "wff_rtl_prob", 0.25)),
            refine_start=float(getattr(config, "wff_refine_start", 1.0)),
        )
    else:
        denoiser_t = t
        wff_use_wave = torch.zeros(batch_size, dtype=dtype, device=device)
        wff_delta = torch.zeros(batch_size, dtype=dtype, device=device)

    noise = torch.randn(x0.shape, dtype=dtype, device=device)

    if config.pad_token == "pad":
        loss_mask = attention_mask
    else:
        loss_mask = torch.ones_like(attention_mask)
    loss_mask = loss_mask * (1 - cond_seq_mask)

    cond_seq_mask = cond_seq_mask.unsqueeze(-1)  # (B, S, 1)

    denoiser_z = add_noise(x0, noise, denoiser_t, config, cond_seq_mask=cond_seq_mask)

    drop = label_drop_mask.unsqueeze(1)  # (B, 1)
    if config.label_drop_prob > 0:
        denoiser_z = torch.where(drop.unsqueeze(-1) & (cond_seq_mask > 0), torch.zeros_like(denoiser_z), denoiser_z)
        x0 = torch.where(drop.unsqueeze(-1) & (cond_seq_mask > 0), torch.zeros_like(x0), x0)

    decoder_targets = input_ids  # (B, S)

    # Per-example branching: each example independently picks decoder (CE) vs.
    # denoiser (L2) instead of one scalar bernoulli per step. Smooths training
    decoder_step_active = torch.bernoulli(
        torch.full((batch_size,), decoder_prob, dtype=torch.float32),
        generator=gen,
    ).to(device=device, dtype=dtype)  # (B,) — 1.0 = decoder mode, 0.0 = denoiser
    decoder_mask_B11 = decoder_step_active.view(-1, 1, 1)
    decoder_mask_B1 = decoder_step_active.view(-1, 1)

    # Decoder-branch input: logit-normal-noised latent (decoder_z) at t=1
    decoder_z_vals = (
        torch.randn((batch_size * seq_length,), dtype=dtype, device=device)
        * config.decoder_p_std + config.decoder_p_mean
    )
    decoder_lambda_t = torch.sigmoid(decoder_z_vals).reshape(batch_size, seq_length, 1)
    decoder_noise = torch.randn(x0.shape, dtype=dtype, device=device) * decoder_noise_scale
    decoder_z = decoder_lambda_t * x0 + (1 - decoder_lambda_t) * decoder_noise

    t_expanded = _expand_time_like(denoiser_t, x0)
    v_target = (x0 - denoiser_z) / torch.clamp(1 - t_expanded, min=t_eps)

    if self_cond_prob > 0:
        use_self_cond_mask = (
            (torch.rand((batch_size,), dtype=dtype, device=device) < self_cond_prob)
            .reshape(-1, 1, 1).to(dtype)
        )
    else:
        use_self_cond_mask = None

    if config.num_self_cond_cfg_tokens > 0:
        self_cond_cfg_scale = sample_cfg_scale(
            batch_size,
            cfg_min=config.self_cond_cfg_min, cfg_max=config.self_cond_cfg_max,
            dtype=dtype, device=device,
        )
    else:
        self_cond_cfg_scale = None

    model = state.model

    def compute_shared_uncond(z, t_input, x_tokens):
        """Unconditional forward shared by self-cond-init and sc-cfg-uncond."""
        z_uncond = restore_cond(torch.zeros_like(z), x_tokens, cond_seq_mask)
        z_input_uncond = torch.cat([z, z_uncond], dim=-1)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            net_out_uncond = model(
                z_input_uncond, t_input,
                deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale,
            )
        return net_out_uncond

    def get_sc_cond_and_uncond(z, t_input, cond_mask, x_tokens, shared_net_out_uncond):
        if config.self_cond_prob == 0:
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                net_out_uncond = model(
                    z, t_input,
                    deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale,
                )
            v_uncond, _ = net_out_to_v_x(net_out_uncond, z, t_input, t_eps)
            return v_uncond, v_uncond

        v_uncond, x_uncond = net_out_to_v_x(shared_net_out_uncond, z, t_input, t_eps)
        x_uncond = restore_cond(x_uncond, x_tokens, cond_mask)

        z_input_cond = torch.cat([z, x_uncond], dim=-1)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            net_out_cond = model(
                z_input_cond, t_input,
                deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale,
            )
        v_cond, _ = net_out_to_v_x(net_out_cond, z, t_input, t_eps)
        return v_cond, v_uncond

    def get_sc_guided_v(z, t_input, base_v_target, x_tokens, shared_net_out_uncond):
        """v target with self-conditioning guidance."""
        v_cond, v_uncond = get_sc_cond_and_uncond(
            z, t_input, cond_mask=cond_seq_mask, x_tokens=x_tokens,
            shared_net_out_uncond=shared_net_out_uncond,
        )
        sc_w = self_cond_cfg_scale.reshape(batch_size, 1, 1)
        sc_guidance = (1 - 1 / sc_w) * (v_cond - v_uncond)
        sc_guidance = torch.where(use_self_cond_mask.bool(), sc_guidance, torch.zeros_like(sc_guidance))
        return (base_v_target + sc_guidance).detach()

    def get_v_target(z, t_input, base_v_target, x_tokens, shared_net_out_uncond):
        """Compute final v target with self-conditioning guidance."""
        if config.num_self_cond_cfg_tokens > 0 and config.self_cond_prob > 0:
            return get_sc_guided_v(
                z, t_input, base_v_target=base_v_target, x_tokens=x_tokens,
                shared_net_out_uncond=shared_net_out_uncond,
            )
        return base_v_target

    model.train()

    # Original JAX KD teacher: clean x0 passed through the decoder at t=1.
    # It is a separate, deterministic, stop-gradient forward.  Using the
    # decoder logits from the noisy mixed forward would instead perform head
    # self-distillation at z_t and is not the historical KD objective.
    lambda_kd = float(getattr(config, "lambda_kd", 0.0))
    teacher_logits_detached = None
    if lambda_kd > 0:
        tau = float(getattr(config, "kd_temperature", 4.0))
        teacher_input = (
            torch.cat([x0, torch.zeros_like(x0)], dim=-1)
            if self_cond_prob > 0 else x0
        )
        teacher_t = torch.ones(batch_size, dtype=dtype, device=device)
        teacher_active = torch.ones(batch_size, dtype=dtype, device=device)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            _, teacher_logits, _ = model(
                teacher_input,
                teacher_t,
                deterministic=True,
                self_cond_cfg_scale=self_cond_cfg_scale,
                decoder_step_active=teacher_active,
            )
        teacher_logits_detached = teacher_logits.float().detach()

    # Per-example branching: build a mixed input (decoder_z for decoder-mode
    # rows, denoiser_z for denoiser-mode rows). One forward computes both
    # heads; we mask CE / L2 losses to their respective rows. 
    decoder_t = torch.ones_like(denoiser_t)
    if denoiser_t.dim() == 2:
        decoder_time_mask = decoder_mask_B1
    else:
        decoder_time_mask = decoder_step_active
    t_mixed = decoder_time_mask * decoder_t + (1.0 - decoder_time_mask) * denoiser_t
    z_mixed = decoder_mask_B11 * decoder_z + (1.0 - decoder_mask_B11) * denoiser_z

    # Self-cond shared forward (run on denoiser_z / t — only relevant for
    # denoiser-mode rows; decoder-mode rows zero out the self-cond half below).
    if self_cond_prob > 0 or config.num_self_cond_cfg_tokens > 0:
        shared_net_out_uncond = compute_shared_uncond(denoiser_z, denoiser_t, x0)
    else:
        shared_net_out_uncond = None

    if config.self_cond_prob > 0:
        _, x_pred_init = net_out_to_v_x(shared_net_out_uncond, denoiser_z, denoiser_t, t_eps)
        x_pred_init = restore_cond(x_pred_init, x0, cond_seq_mask)
        x_pred_cond = x_pred_init * use_self_cond_mask.to(dtype)
        x_pred_cond = restore_cond(x_pred_cond, x0, cond_seq_mask)
        # Zero the self-cond half for decoder-mode rows (matches the old
        # `cat([decoder_z, zeros], -1)` decoder-branch input).
        sc_half = x_pred_cond * (1.0 - decoder_mask_B11)
        model_input = torch.cat([z_mixed, sc_half], dim=-1)
    else:
        model_input = z_mixed

    # D1: register hook BEFORE the main forward so h_N is captured during it.
    _h_captured = {}
    _hook_handles = []
    lambda_intermediate_rec = getattr(config, "lambda_intermediate_rec", 0.0)
    if lambda_intermediate_rec > 0:
        _irec_layer = int(getattr(config, "intermediate_rec_layer", 10))
        _raw = getattr(model, "module", model)
        _prefix_sz = (getattr(_raw, "num_model_mode_tokens", 0)
                      + getattr(_raw, "num_time_tokens", 4)
                      + getattr(_raw, "num_self_cond_cfg_tokens", 4))

        def _cap_h(module, inp, out, _idx=_irec_layer):
            _h_captured[_idx] = out

        _hook_handles.append(_raw.blocks[_irec_layer].register_forward_hook(_cap_h))

    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        net_out, decoder_logits, linear_logits = model(
            model_input, t_mixed,
            deterministic=False,
            self_cond_cfg_scale=self_cond_cfg_scale,
            decoder_step_active=decoder_step_active,  # (B,) tensor
        )

    for _h in _hook_handles:
        _h.remove()

    # CE per-token (used on decoder-mode rows).
    log_probs = F.log_softmax(decoder_logits.to(torch.float32), dim=-1)
    ce_per_token = -log_probs.gather(-1, decoder_targets.unsqueeze(-1)).squeeze(-1)

    # L2 per-token (used on denoiser-mode rows). v_pred is extracted with
    # (denoiser_z, t) — meaningful only for denoiser rows; decoder rows are
    # masked out below.
    v_pred, _ = net_out_to_v_x(net_out, denoiser_z, denoiser_t, t_eps)
    v_final_target = get_v_target(
        denoiser_z, denoiser_t, base_v_target=v_target, x_tokens=x0,
        shared_net_out_uncond=shared_net_out_uncond,
    )
    l2_per_token = ((v_pred - v_final_target) ** 2).mean(dim=-1)

    # Masks: each position is "alive" for exactly one branch.
    loss_mask_f = loss_mask.to(ce_per_token.dtype)
    ce_mask = loss_mask_f * decoder_mask_B1
    l2_mask = loss_mask_f * (1.0 - decoder_mask_B1)

    # Combined loss with a single denominator. In expectation this is
    # decoder_prob * mean_CE + (1 - decoder_prob) * mean_L2.
    total_sum = (ce_per_token * ce_mask).sum() + (l2_per_token * l2_mask).sum()
    loss = total_sum / torch.clamp(loss_mask_f.sum(), min=1.0)

    # Per-branch metrics: mean per-token within each branch.
    ce_loss_val = ((ce_per_token * ce_mask).sum()
                   / torch.clamp(ce_mask.sum(), min=1.0)).detach()
    l2_loss_val = ((l2_per_token * l2_mask).sum()
                   / torch.clamp(l2_mask.sum(), min=1.0)).detach()

    # KD loss: distill the clean-state decoder teacher into the noisy-state
    # linear branch. Applied only to denoiser-mode examples with a smooth
    # temporal gate, matching the original JAX implementation.
    # Idea B: set kd_entropy_mask=true to up-weight wrong-committed positions.
    kd_loss_val = torch.zeros(1, device=device)
    if lambda_kd > 0 and linear_logits is not None:
        tau = float(getattr(config, "kd_temperature", 4.0))
        gate_low = float(getattr(config, "kd_gate_low", 0.0))
        gate_high = float(getattr(config, "kd_gate_high", 1.0))
        gate_k = float(getattr(config, "kd_gate_k", 10.0))
        denoiser_mask_B = 1.0 - decoder_step_active          # (B,) 1=denoiser, 0=decoder
        t_gate = _kd_omega_gate(denoiser_t.float(), gate_k, gate_low, gate_high)
        window_low = float(getattr(config, "kd_window_low", -1.0))
        window_high = float(getattr(config, "kd_window_high", -1.0))
        if window_low >= 0.0 and window_high > window_low:
            window_k = float(getattr(config, "kd_window_k", 40.0))
            # Localize a subset of the historical objective without replacing
            # or amplifying its original smooth temporal weighting.
            t_gate = t_gate * _kd_omega_gate(
                denoiser_t.float(), window_k, window_low, window_high
            )
        if t_gate.dim() == 1:
            t_gate = t_gate.unsqueeze(-1)
        kd_mask_BS = loss_mask_f * denoiser_mask_B.unsqueeze(-1) * t_gate

        kd_per_token = _kd_kl_per_token(
            teacher_logits_detached, linear_logits, tau
        )  # (B, S), KL(teacher || student)

        # Idea B: per-position entropy mask — up-weight wrong-committed positions.
        # Additive form: w_i = 1 + kd_entropy_beta * (1−H/logV) * 1[student≠teacher]
        # Degrades to uniform KD when no wrong-committed positions exist (no NaN risk).
        # kd_entropy_beta (default 3.0): wrong-committed positions get up to (1+beta)x weight.
        if getattr(config, "kd_entropy_mask", False):
            beta = float(getattr(config, "kd_entropy_beta", 3.0))
            with torch.no_grad():
                p_student = F.softmax(linear_logits.float(), dim=-1)         # (B, S, V)
                H_student = -(p_student * (p_student + 1e-10).log()).sum(-1) # (B, S)
                log_V = math.log(linear_logits.shape[-1])
                confidence = (1.0 - H_student / log_V).clamp(0.0, 1.0)      # (B, S)
                teacher_top1 = teacher_logits_detached.argmax(dim=-1)        # (B, S)
                student_top1 = linear_logits.argmax(dim=-1)                  # (B, S)
                wrong_mask = (student_top1 != teacher_top1).float()          # (B, S)
                # Additive: base weight 1 everywhere, +beta*confidence on wrong positions
                entropy_weight = 1.0 + beta * confidence * wrong_mask        # (B, S)
            kd_mask_BS = kd_mask_BS * entropy_weight

        if bool(getattr(config, "kd_normalize_active", False)):
            kd_denom = kd_mask_BS.sum()
        else:
            # JAX reduce_token_loss divides by the ordinary token mask, not by
            # omega(t); the gate therefore controls both support and strength.
            kd_denom = loss_mask_f.sum()
        kd_loss = (kd_per_token * kd_mask_BS).sum() / torch.clamp(kd_denom, min=1.0)
        kd_loss_val = kd_loss.detach()
        loss = loss + lambda_kd * kd_loss

    # L_sc: push the main denoising prediction (net_out, has grad) toward x̂_t^dec.
    # Teacher: decode branch applied to x̂_t^(1) at t=1 (no grad, 1 extra forward).
    # Student: net_out for denoiser-mode rows (gradient flows via main backward).
    # Closes the embedding-space gap so inference-time dec_sc becomes unnecessary:
    # after training, x̂_t^(1) ≈ x̂_t^dec even without the extra decode pass.
    lambda_sc = getattr(config, "lambda_sc", 0.0)
    sc_loss_val = torch.zeros(1, device=device)
    if lambda_sc > 0 and x_pred_init is not None:
        sc_gate_low = float(getattr(config, "sc_gate_low", 0.0))
        sc_gate_high = float(getattr(config, "sc_gate_high", 1.0))
        # Decode-branch teacher: run backbone on x̂_t^(1) at t=1 (no SC input)
        sc_t_ones = torch.ones(batch_size, dtype=dtype, device=device)
        sc_dec_input = torch.cat(
            [x_pred_init.detach(), torch.zeros_like(x_pred_init)], dim=-1
        )
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            sc_dec_raw = model(
                sc_dec_input, sc_t_ones,
                deterministic=True,
                self_cond_cfg_scale=self_cond_cfg_scale,
                decoder_step_active=None,   # skip lin_branch
            )
        x_dec = (sc_dec_raw[0] if isinstance(sc_dec_raw, tuple) else sc_dec_raw)
        x_dec = restore_cond(x_dec.float().detach(), x0.float(), cond_seq_mask)

        denoiser_mask_B = 1.0 - decoder_step_active                          # (B,)
        t_sc_gate = ((denoiser_t >= sc_gate_low) & (denoiser_t <= sc_gate_high)).float()
        if t_sc_gate.dim() == 1:
            t_sc_gate = t_sc_gate.unsqueeze(-1)
        sc_mask_BS = loss_mask_f * denoiser_mask_B.unsqueeze(-1) * t_sc_gate
        sc_diff = net_out.float() - x_dec                                     # (B, S, d)
        sc_per_token = (sc_diff ** 2).mean(dim=-1)                            # (B, S)
        sc_loss = (sc_per_token * sc_mask_BS).sum() / torch.clamp(sc_mask_BS.sum(), min=1.0)
        sc_loss_val = sc_loss.detach()
        loss = loss + lambda_sc * sc_loss

    # Idea 2: cliff CE — cross-entropy from linear branch against ground truth,
    # weighted by two-stage α_nm(t) = σ(k(t−cliff_rise)) − σ(k(t−cliff_fall)).
    # Active only on denoiser-mode samples; pushes lin_branch to commit correctly
    # during the commitment cliff without affecting decoder-mode CE rows.
    lambda_ce_cliff = getattr(config, "lambda_ce_cliff", 0.0)
    ce_cliff_loss_val = torch.zeros(1, device=device)
    if lambda_ce_cliff > 0 and linear_logits is not None:
        k_cliff = float(getattr(config, "ce_cliff_k", 10.0))
        cliff_rise = float(getattr(config, "ce_cliff_rise", 0.30))
        cliff_fall = float(getattr(config, "ce_cliff_fall", 0.60))
        # α_nm(t) peaks between cliff_rise and cliff_fall, ≈0 outside
        alpha_t = (
            torch.sigmoid(k_cliff * (denoiser_t - cliff_rise))
            - torch.sigmoid(k_cliff * (denoiser_t - cliff_fall))
        )
        if alpha_t.dim() == 1:
            alpha_t = alpha_t.unsqueeze(-1)
        denoiser_mask_B = 1.0 - decoder_step_active  # (B,)
        cliff_mask_BS = loss_mask_f * denoiser_mask_B.unsqueeze(-1) * alpha_t
        ce_cliff_per_token = F.cross_entropy(
            linear_logits.float().reshape(-1, linear_logits.shape[-1]),
            decoder_targets.reshape(-1),
            reduction='none',
        ).reshape(batch_size, seq_length)  # (B, S)
        ce_cliff_loss = (ce_cliff_per_token * cliff_mask_BS).sum() / torch.clamp(
            cliff_mask_BS.sum(), min=1.0
        )
        ce_cliff_loss_val = ce_cliff_loss.detach()
        loss = loss + lambda_ce_cliff * ce_cliff_loss

    # D1: Intermediate reconstruction loss — supervise x̂_t at an earlier backbone layer.
    # Hook was registered before the main forward; _h_captured[_irec_layer] holds h_N.
    # Strips the prefix, runs final_layer(h_N), adds MSE(x̂_N, x0) for denoiser rows.
    intermediate_rec_loss_val = torch.zeros(1, device=device)
    if lambda_intermediate_rec > 0 and _irec_layer in _h_captured:
        h_N_full = _h_captured[_irec_layer]                           # [B, prefix+S, 768]
        h_N = h_N_full[:, _prefix_sz:, :].float()                    # [B, S, 768]
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            x_hat_N = _raw.final_layer(h_N)                          # [B, S, 512]
        denoiser_mask_B = 1.0 - decoder_step_active                   # (B,)
        rec_mask_BS = loss_mask_f * denoiser_mask_B.unsqueeze(-1)     # (B, S)
        rec_per_tok = ((x_hat_N.float() - x0.float()) ** 2).mean(-1) # (B, S)
        int_rec_loss = (rec_per_tok * rec_mask_BS).sum() / torch.clamp(rec_mask_BS.sum(), min=1.0)
        intermediate_rec_loss_val = int_rec_loss.detach()
        loss = loss + lambda_intermediate_rec * int_rec_loss

    # D3: Gram-matrix alignment loss — encourage final_layer.linear and self_cond_proj
    # (x̂_t portion) to share compatible column-space geometry, preventing the
    # producer/consumer mismatch identified in EXP-44 Phase 2.
    # L_align = ||G_F - G_P||_F^2 where G_F = (F@F.T)/norm, G_P = (P@P.T)/norm.
    lambda_align = getattr(config, "lambda_align", 0.0)
    align_loss_val = torch.zeros(1, device=device)
    if lambda_align > 0:
        raw = getattr(model, "module", model)
        F_w = raw.final_layer.linear.weight.float()     # [512, 768]
        P_w = raw.self_cond_proj.weight[:, 512:].float()  # [512, 512]
        gram_F = (F_w @ F_w.T) / F_w.shape[1]          # [512, 512]  /768
        gram_P = (P_w @ P_w.T) / P_w.shape[1]          # [512, 512]  /512
        # Normalize by trace so scale doesn't dominate
        tr_F = gram_F.diagonal().mean().clamp(min=1e-6)
        tr_P = gram_P.diagonal().mean().clamp(min=1e-6)
        align_loss = ((gram_F / tr_F - gram_P / tr_P) ** 2).mean()
        align_loss_val = align_loss.detach()
        loss = loss + lambda_align * align_loss

    accum_steps = max(config.grad_accum_steps, 1)
    state.step += 1
    is_optimizer_step = (state.step % accum_steps) == 0

    sync_ctx = model.no_sync() if (not is_optimizer_step and hasattr(model, 'no_sync')) else contextlib.nullcontext()
    with sync_ctx:
        (loss / accum_steps).backward()

    raw_model = getattr(model, "module", model)
    if hasattr(raw_model, "local_time_scales"):
        local_time_scale_abs_val = raw_model.local_time_scales.detach().abs().mean()
        local_grads = [
            parameter.grad.detach().float().norm()
            for parameter in raw_model.local_time_projections.parameters()
            if parameter.grad is not None
        ]
        if raw_model.local_time_scales.grad is not None:
            local_grads.append(raw_model.local_time_scales.grad.detach().float().norm())
        local_time_grad_norm_val = (
            torch.stack(local_grads).norm()
            if local_grads else torch.zeros((), device=device)
        )
    else:
        local_time_scale_abs_val = torch.zeros((), device=device)
        local_time_grad_norm_val = torch.zeros((), device=device)

    if is_optimizer_step:
        torch.nn.utils.clip_grad_norm_(_trainable_params(model), max_norm=1.0)
        state.optimizer.step()
        if state.lr_scheduler is not None:
            state.lr_scheduler.step()
        ema_update(state.ema_params1, state.model, config.ema_decay1)
        state.optimizer.zero_grad(set_to_none=True)

    metrics = {
        "loss": loss.detach(),
        "l2_loss": l2_loss_val,
        "ce_loss": ce_loss_val,
        "kd_loss": kd_loss_val,
        "sc_loss": sc_loss_val,
        "ce_cliff_loss": ce_cliff_loss_val,
        "intermediate_rec_loss": intermediate_rec_loss_val,
        "align_loss": align_loss_val,
        "wff_fraction": wff_use_wave.mean().detach(),
        "wff_mean_delta": wff_delta.mean().detach(),
        "local_time_scale_abs": local_time_scale_abs_val,
        "local_time_grad_norm": local_time_grad_norm_val,
    }
    return state, metrics
