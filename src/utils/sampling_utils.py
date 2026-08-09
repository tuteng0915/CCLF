import math
from typing import Optional

import torch
import torch.nn.functional as F


# ============================================
# Noise Schedulers (how to compute z from x0 and noise)
# ============================================

def _expand_time_like(t: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast scalar-per-example or per-token time over the channel axis."""
    if t.dim() == 1:
        if t.shape[0] != reference.shape[0]:
            raise ValueError(f"Expected {reference.shape[0]} timesteps, got {t.shape[0]}")
        return t[:, None, None]
    if t.dim() == 2:
        if t.shape != reference.shape[:2]:
            raise ValueError(
                f"Expected per-token time shape {tuple(reference.shape[:2])}, got {tuple(t.shape)}"
            )
        return t.unsqueeze(-1)
    raise ValueError(f"Time must have shape (B,) or (B,S), got {tuple(t.shape)}")


def add_noise(x0, noise, t, config, cond_seq_mask=None):
    """Flow-matching interpolation z = t*x0 + (1-t)*noise*scale, preserving cond tokens."""
    t_expanded = _expand_time_like(t, x0)
    z = t_expanded * x0 + (1 - t_expanded) * noise * config.denoiser_noise_scale
    if cond_seq_mask is not None:
        z = cond_seq_mask * x0 + (1 - cond_seq_mask) * z
    return z


# ============================================
# Time Schedulers (how to sample t)
# ============================================

def sample_timesteps(
    batch_size: int,
    P_mean: float = -0.8,
    P_std: float = 0.8,
    time_schedule: str = 'logit_normal',
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    cliff_mix_prob: float = 0.6,
    cliff_lo: float = 0.10,
    cliff_hi: float = 0.45,
):
    """Sample timesteps using various time schedules.

    Args:
        batch_size: Number of samples
        P_mean: Mean for logit-normal distribution
        P_std: Std for logit-normal distribution
        time_schedule: 'logit_normal', 'uniform', or 'cliff'
        cliff_mix_prob: For 'cliff' — fraction of batch drawn from [cliff_lo, cliff_hi]
        cliff_lo / cliff_hi: Commitment cliff boundaries (probe: dG/dt peaks at t≈0.25–0.35)

    Returns:
        Sampled timesteps in [0, 1]
    """
    if time_schedule == 'logit_normal':
        z = torch.randn((batch_size,), dtype=dtype, device=device) * P_std + P_mean
        return torch.sigmoid(z)
    if time_schedule == 'uniform':
        return torch.rand((batch_size,), dtype=dtype, device=device)
    if time_schedule == 'cliff':
        # cliff_mix_prob fraction from U[cliff_lo, cliff_hi] (commitment cliff),
        # rest from logit_normal. Concentrates training on dG/dt-high region.
        logit_t = torch.sigmoid(
            torch.randn((batch_size,), dtype=dtype, device=device) * P_std + P_mean
        )
        cliff_t = cliff_lo + (cliff_hi - cliff_lo) * torch.rand(
            (batch_size,), dtype=dtype, device=device
        )
        use_cliff = torch.rand((batch_size,), dtype=dtype, device=device) < cliff_mix_prob
        return torch.where(use_cliff, cliff_t, logit_t)
    if time_schedule == 'probe_dGdt':
        # Data-driven schedule: sample t proportional to dG/dt from ELF-B OWT probe.
        # Weights computed from probe_geo_v1/probe_geo.json:
        #   p(bin_i) ∝ max(dG/dt_i, 0) + eps,  20 bins of width 0.05 on [0, 1].
        # 69% of mass lands on cliff t=[0.15, 0.40] where dG/dt is highest.
        # Hardcoded so training is reproducible without a file-system dependency.
        _PROBE_WEIGHTS = torch.tensor([
            0.0205, 0.0297, 0.0261, 0.0809, 0.1986, 0.2045, 0.1264, 0.0807,
            0.0464, 0.0288, 0.0179, 0.0120, 0.0097, 0.0097, 0.0097, 0.0097,
            0.0097, 0.0097, 0.0592, 0.0097,
        ], dtype=torch.float32, device=device)
        bins = torch.multinomial(_PROBE_WEIGHTS, batch_size, replacement=True)  # (B,)
        t_lo = bins.to(dtype) * 0.05
        t = t_lo + 0.05 * torch.rand((batch_size,), dtype=dtype, device=device)
        return t.clamp(0.0, 1.0)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def sample_wff_timesteps(
    base_t: torch.Tensor,
    seq_length: int,
    probability: float,
    delta_min: float,
    delta_max: float,
    ltr_probability: float = 0.5,
    rtl_probability: float = 0.25,
    refine_start: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample native heterogeneous local times for WFF training.

    Returns `(tau, use_wave, delta, order_id)`, where `tau` is `(B,S)` and
    order ids are 0=LTR, 1=RTL, 2=random.  Non-wave examples retain the
    original scalar time at every position, providing an exact synchronous
    training control inside the same model.
    """
    if base_t.dim() != 1:
        raise ValueError(f"base_t must have shape (B,), got {tuple(base_t.shape)}")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("WFF probability must lie in [0,1]")
    if not 0.0 <= delta_min <= delta_max:
        raise ValueError("Require 0 <= delta_min <= delta_max")
    if not 0.0 < refine_start <= 1.0:
        raise ValueError("refine_start must lie in (0,1]")
    if delta_max > refine_start / math.pi:
        raise ValueError(
            "WFF delta must be <= refine_start/pi so local clocks remain monotone"
        )
    if ltr_probability < 0.0 or rtl_probability < 0.0 or ltr_probability + rtl_probability > 1.0:
        raise ValueError("Invalid WFF ordering probabilities")

    batch_size = base_t.shape[0]
    device, dtype = base_t.device, base_t.dtype
    synchronous = base_t[:, None].expand(batch_size, seq_length)
    use_wave = torch.rand(batch_size, device=device) < probability
    delta = delta_min + (delta_max - delta_min) * torch.rand(
        batch_size, device=device, dtype=dtype
    )

    order_draw = torch.rand(batch_size, device=device)
    order_id = torch.full((batch_size,), 2, dtype=torch.long, device=device)
    order_id = torch.where(order_draw < ltr_probability, torch.zeros_like(order_id), order_id)
    order_id = torch.where(
        (order_draw >= ltr_probability)
        & (order_draw < ltr_probability + rtl_probability),
        torch.ones_like(order_id),
        order_id,
    )

    if seq_length <= 1:
        ltr_offset = torch.zeros(seq_length, dtype=dtype, device=device)
    else:
        rank = torch.linspace(0.0, 1.0, seq_length, dtype=dtype, device=device)
        ltr_offset = 1.0 - 2.0 * rank
    rtl_offset = -ltr_offset

    random_scores = torch.rand(batch_size, seq_length, device=device)
    random_rank = random_scores.argsort(dim=1).argsort(dim=1).to(dtype)
    if seq_length > 1:
        random_rank = random_rank / float(seq_length - 1)
    random_offset = 1.0 - 2.0 * random_rank

    offsets = random_offset
    offsets = torch.where((order_id == 0)[:, None], ltr_offset[None, :], offsets)
    offsets = torch.where((order_id == 1)[:, None], rtl_offset[None, :], offsets)
    # Match the inference clock exactly: the heterogeneous offset vanishes at
    # both endpoints, so every token begins at pure noise and finishes at the
    # same clean endpoint. Delta <= 1/pi keeps d tau_i / d t non-negative.
    normalized = (base_t / refine_start).clamp(0.0, 1.0)
    envelope = torch.sin(math.pi * normalized)[:, None]
    wave_tau = (
        base_t[:, None] + delta[:, None] * envelope * offsets
    ).clamp(0.0, 1.0)
    tau = torch.where(use_wave[:, None], wave_tau, synchronous)
    effective_delta = torch.where(use_wave, delta, torch.zeros_like(delta))
    return tau, use_wave.to(dtype), effective_delta, order_id


def make_wff_time_vector(
    global_t: float,
    seq_length: int,
    delta: float,
    order: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    refine_start: float = 1.0,
) -> torch.Tensor:
    """Construct a monotone wavefront clock with synchronous endpoints.

    `sin(pi*s)` turns the offset on only in the interior, so every position
    starts at pure noise and reaches clean time exactly.  Delta must be at
    most 1/pi to keep all local clocks monotone in global solver time.
    """
    if not 0.0 < refine_start <= 1.0:
        raise ValueError(f"refine_start must be in (0,1], got {refine_start}")
    if delta < 0.0 or delta > refine_start / math.pi:
        raise ValueError(
            f"wff_delta must be in [0, refine_start/pi], got {delta}"
        )
    if order not in {"ltr", "rtl"}:
        raise ValueError(f"Unsupported WFF order: {order}")
    if seq_length <= 1:
        offset = torch.zeros(seq_length, dtype=dtype, device=device)
    else:
        rank = torch.linspace(0.0, 1.0, seq_length, dtype=dtype, device=device)
        offset = 1.0 - 2.0 * rank
    if order == "rtl":
        offset = -offset
    normalized = min(max(float(global_t) / refine_start, 0.0), 1.0)
    envelope = math.sin(math.pi * normalized)
    return (float(global_t) + float(delta) * envelope * offset).clamp(0.0, 1.0)


def get_sampling_steps(
    n_steps: int, time_schedule: str = "logit_normal",
    P_mean: float = -0.8, P_std: float = 0.8,
    device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a length-(n_steps+1) tensor of t values in [0, 1] for a sampling run.

    - "uniform": evenly-spaced linspace from 0 to 1 (deterministic).
    - "logit_normal": sorted logit-normal samples with 0 / 1 endpoints (random).
    """
    if time_schedule == "uniform":
        return torch.linspace(0.0, 1.0, n_steps + 1, dtype=dtype, device=device)
    if time_schedule == "logit_normal":
        steps = sample_timesteps(
            batch_size=n_steps - 1,
            P_mean=P_mean, P_std=P_std, time_schedule=time_schedule,
            device=device, dtype=dtype,
        )
        steps = torch.sort(steps).values
        endpoints_lo = torch.zeros((1,), dtype=dtype, device=steps.device)
        endpoints_hi = torch.ones((1,), dtype=dtype, device=steps.device)
        return torch.cat([endpoints_lo, steps, endpoints_hi], dim=0)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


# ============================================
# CFG Scale Sampling (how to sample cfg scale)
# ============================================

def sample_cfg_scale(batch_size, cfg_min=0.0, cfg_max=3.0,
                     dtype=torch.float32, device=None):
    """Sample CFG scale from log-uniform distribution in [cfg_min, cfg_max]."""
    u = torch.rand((batch_size,), dtype=dtype, device=device)
    a = float(1.0 + cfg_min)
    b = float(1.0 + cfg_max)
    log_ratio = torch.tensor(b / a, dtype=dtype, device=u.device).log()
    return a * torch.exp(u * log_ratio) - 1.0


# ============================================
# Conditioning helpers (preserve clean tokens during sampling)
# ============================================

def restore_cond(z_updated, cond_seq, cond_seq_mask):
    """Restore clean conditioning tokens in z after a denoising step."""
    mask = cond_seq_mask
    target_ndim = max(z_updated.dim(), cond_seq.dim())
    while mask.dim() < target_ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, cond_seq, z_updated)


def restore_vx(v, x, cond_seq, cond_seq_mask):
    """Restore cond positions: x -> clean cond_seq, v -> 0 (cond tokens don't move)."""
    if cond_seq is not None:
        x = restore_cond(x, cond_seq, cond_seq_mask)
        v = restore_cond(v, torch.zeros_like(cond_seq), cond_seq_mask)
    return v, x


# ============================================
# Flow-matching forward passes (with optional self-cond / CFG)
# ============================================

def net_out_to_v_x(net_out, z, t, t_eps=5e-2):
    """Convert x_pred network output to v and x.

    When the model returns a tuple (denoised_output, decoder_logits),
    decoder logits are discarded here (used separately in training).
    """
    if isinstance(net_out, tuple):
        net_out = net_out[0]
    t_reshaped = _expand_time_like(t, z)
    x = net_out
    denom = torch.clamp(1.0 - t_reshaped, min=t_eps)
    v = (x - z) / denom
    return v, x


def _forward_sample_self_cond(
    model, z, t_batch, x_pred_prev, config,
    self_cond_cfg_scale, cond_seq, cond_seq_mask,
):
    """Forward pass with self-conditioning."""
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob

    def _restore(v, x):
        return restore_vx(v, x, cond_seq=cond_seq, cond_seq_mask=cond_seq_mask)

    if config.num_self_cond_cfg_tokens > 0:
        if x_pred_prev is None:
            x_pred_prev = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
        self_cond_scale_batch = torch.full((z.shape[0],), float(self_cond_cfg_scale),
                                           dtype=z.dtype, device=z.device)
        net_out_cond = model(z_input_cond, t_batch, deterministic=True,
                             self_cond_cfg_scale=self_cond_scale_batch)
        v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
        return _restore(v_cond, x_cond)

    # No self-conditioning
    if self_cond_prob == 0:
        net_out = model(z, t_batch, deterministic=True)
        v, x = net_out_to_v_x(net_out, z, t_batch, t_eps)
        return _restore(v, x)

    # Combined unconditional and conditional forward pass
    v_uncond = x_uncond = None
    if self_cond_cfg_scale != 1 or x_pred_prev is None:
        z_uncond = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_uncond = torch.cat([z, z_uncond], dim=-1)
        net_out_uncond = model(z_input_uncond, t_batch, deterministic=True)
        v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, z, t_batch, t_eps)
        v_uncond, x_uncond = _restore(v_uncond, x_uncond)
        if self_cond_cfg_scale == 0.0 or x_pred_prev is None:
            return v_uncond, x_uncond

    z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
    net_out_cond = model(z_input_cond, t_batch, deterministic=True)
    v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
    v_cond, x_cond = _restore(v_cond, x_cond)
    if self_cond_cfg_scale == 1:
        return v_cond, x_cond

    v_out = v_uncond + self_cond_cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + self_cond_cfg_scale * (x_cond - x_uncond)
    return _restore(v_out, x_out)


def _forward_sample(
    model, z, t_batch, x_pred_prev, config,
    cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask,
):
    """Forward pass with optional self-conditioning and CFG."""
    v_cond, x_cond = _forward_sample_self_cond(
        model, z, t_batch, x_pred_prev, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    if cfg_scale == 1.0:
        return v_cond, x_cond

    # Unconditional forward: zero out cond prefix, no self-cond state, no restore
    z_uncond = restore_cond(z, torch.zeros_like(z), cond_seq_mask)
    x_pred_prev_uncond = (
        None if x_pred_prev is None
        else restore_cond(x_pred_prev, torch.zeros_like(x_pred_prev), cond_seq_mask)
    )
    v_uncond, x_uncond = _forward_sample_self_cond(
        model, z_uncond, t_batch, x_pred_prev_uncond, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=torch.zeros_like(cond_seq), cond_seq_mask=cond_seq_mask,
    )

    v_out = v_uncond + cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + cfg_scale * (x_cond - x_uncond)
    return restore_vx(v_out, x_out, cond_seq, cond_seq_mask)


def _ode_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask,
):
    """Single ODE (Euler) step for sampling."""
    t_batch = torch.full((z.shape[0],), float(t), dtype=z.dtype, device=z.device)
    v_pred, x_pred = _forward_sample(
        model=model, z=z, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z + (t_next - t) * v_pred, x_pred


def _wff_ode_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask,
):
    """Native WFF Euler step using a local clock for every token position."""
    batch_size, seq_length = z.shape[:2]

    def _batch_time(value):
        if value.dim() == 1:
            if value.shape[0] != seq_length:
                raise ValueError(f"Expected {seq_length} local times, got {value.shape[0]}")
            return value[None, :].expand(batch_size, -1)
        if value.dim() == 2 and value.shape == (batch_size, seq_length):
            return value
        raise ValueError(
            f"WFF time must have shape (S,) or (B,S), got {tuple(value.shape)}"
        )

    t_batch = _batch_time(t).to(dtype=z.dtype, device=z.device)
    t_next_batch = _batch_time(t_next).to(dtype=z.dtype, device=z.device)
    v_pred, x_pred = _forward_sample(
        model=model, z=z, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    dt = (t_next_batch - t_batch).unsqueeze(-1)
    z_next = z + dt * v_pred
    if cond_seq is not None:
        z_next = restore_cond(z_next, cond_seq, cond_seq_mask)
    return z_next, x_pred


def _sde_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask, gamma, generator,
):
    """Per-step SDE-style sampler with hybrid (t-and-step) noise scaling.

    t_back = t * (1 - gamma * h), where h = t_next - t. alpha = 1 - gamma*h is the
    signal-preservation fraction, constant in t. gamma=0 degenerates to a plain ODE step.
    Uniform-N-step equivalence with old multiplicative gamma_old: gamma_hybrid = gamma_old * N.
    """
    h = float(t_next - t)
    alpha = max(0.0, min(1.0, 1.0 - gamma * h))
    t_back = alpha * float(t)
    if z.is_cuda:
        eps = torch.randn(z.shape, dtype=z.dtype, device=z.device) * config.denoiser_noise_scale
    else:
        eps = torch.randn(z.shape, generator=generator, dtype=z.dtype) * config.denoiser_noise_scale
    z_back = restore_cond(alpha * z + (1.0 - alpha) * eps, cond_seq, cond_seq_mask)
    t_batch = torch.full((z.shape[0],), t_back, dtype=z.dtype, device=z.device)
    v_pred, x_pred = _forward_sample(
        model=model, z=z_back, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z_back + (t_next - t_back) * v_pred, x_pred
