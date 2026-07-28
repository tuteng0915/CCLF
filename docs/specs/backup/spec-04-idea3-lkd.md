# Spec 04 — Idea 3: L_KD (Decode-Teacher Knowledge Distillation)

**Type**: code + training experiment  
**Priority**: high (core training component, strongest probe support)  
**Session**: 2 (after Idea A inference experiment)  
**Output**: fine-tuned ELF-B checkpoint + training curves in `outputs/elf_b-owt-kd/`

---

## Background

ELF's decode branch outperforms its linear branch by +15–27 pp in top-1 accuracy
across the entire denoising plateau (t=0.25–0.95). The linear branch is used at every
denoising step, but cannot yet extract the token identity that is already geometrically
encoded in the backbone's hidden state.

**Core gap (from probe_decode_branch.json):**

| t    | lin top-1 | dec top-1 | G_dec (gap) |
|------|-----------|-----------|-------------|
| 0.30 | 45.5%     | 70.0%     | +24.5 pp    |
| 0.35 | 59.2%     | 86.8%     | +27.6 pp    |
| 0.50 | 76.8%     | 96.3%     | +19.6 pp    |
| 0.95 | 81.0%     | 96.9%     | +15.9 pp    |

**Goal**: Internalize the decode branch's distributional advantage into the linear branch
via knowledge distillation. After training, the linear branch should approximate p_t^dec
without needing an extra forward pass.

---

## Formula

$$\mathcal{L}_\text{KD}(t) = \tau^2 \cdot \lambda_\text{kd} \cdot \frac{1}{L}
\sum_{i=1}^{L} \text{KL}\!\left(\text{sg}(p_{1,i}^{\text{dec}}) \,\middle\|\, p_{t,i}\right) \cdot \omega(t)$$

Where:
- $p_{1,i}^{\text{dec}}$ = teacher: decode branch distribution on **clean x0** at t=1, stop-gradient
- $p_{t,i}$ = student: linear branch distribution at denoising time t (differentiable)
- $\omega(t) = \sigma(k_\omega(t-0.25)) \cdot (1 - \sigma(k_\omega(t-0.95)))$ — gate active on [0.25, 0.95]
- $\tau$ = temperature (default 4.0), $\lambda_\text{kd}$ = loss weight (default 1.0)

Default hyperparameters: `lambda_kd=1.0`, `kd_temperature=4.0`, `kd_gate_k=10.0`

---

## Implementation

### Files to modify

#### `src/configs/config.py`

Add under `# Decoder objective`:

```python
# L_KD (Idea 3): decode-teacher knowledge distillation
lambda_kd: float = 0.0       # loss weight; 0.0 = disabled (no compute overhead)
kd_temperature: float = 4.0  # τ_KD — scales KL loss by τ²
kd_gate_k: float = 10.0      # sigmoid sharpness k_ω for ω(t) gate
```

---

#### `src/train_step.py`

**Step 1** — Add gate helper at module level (before `train_step`):

```python
def _kd_omega_gate(t, k):
    """ω(t) = σ(k(t−0.25))·(1−σ(k(t−0.95))): active on [0.25, 0.95]."""
    return jax.nn.sigmoid(k * (t - 0.25)) * (1.0 - jax.nn.sigmoid(k * (t - 0.95)))
```

**Step 2** — Compute teacher OUTSIDE `loss_fn`, just before `grad_fn = jax.value_and_grad(loss_fn)`:

```python
if config.lambda_kd > 0:
    teacher_t = jnp.ones((batch_size,), dtype=x0.dtype)
    teacher_input = (
        jnp.concatenate([x0, jnp.zeros_like(x0)], axis=-1)
        if config.self_cond_prob > 0 else x0
    )
    _, teacher_logits_raw = state.apply_fn(
        {"params": state.params}, teacher_input, teacher_t,
        deterministic=True,
        self_cond_cfg_scale=self_cond_cfg_scale,
        decoder_step_active=jnp.array(True),
    )
    # stop_gradient: teacher is a constant from the optimizer's perspective
    teacher_log_probs = jax.lax.stop_gradient(
        jax.nn.log_softmax(teacher_logits_raw.astype(jnp.float32), axis=-1)
    )
else:
    teacher_log_probs = None
```

Teacher is computed with `state.params` (not the `params` argument to `loss_fn`), so
it is outside `jax.value_and_grad` — no gradient flows through it by construction.

**Step 3** — Extend `_denoiser_branch` inside `loss_fn`:

```python
def _denoiser_branch(_):
    # Enable decoder head in denoiser branch when KD is active
    _dec_active = jnp.array(config.lambda_kd > 0)  # static at trace time
    net_out, student_logits = state.apply_fn(
        {"params": params}, denoiser_input, denoiser_t,
        deterministic=False,
        rngs={"dropout": model_dropout_rng},
        self_cond_cfg_scale=self_cond_cfg_scale,
        decoder_step_active=_dec_active,
    )
    v_pred, _ = net_out_to_v_x(net_out, denoiser_z, denoiser_t, t_eps)
    v_final_target = get_v_target(
        params, denoiser_z, denoiser_t, base_v_target=v_target, x_tokens=x0,
    )
    per_dim_loss = (v_pred - v_final_target) ** 2
    l2_loss = reduce_token_loss(jnp.mean(per_dim_loss, axis=-1), loss_mask)

    # L_KD
    kd_loss = jnp.zeros(())
    if config.lambda_kd > 0:
        student_log_probs = jax.nn.log_softmax(
            student_logits.astype(jnp.float32), axis=-1
        )
        teacher_probs = jnp.exp(teacher_log_probs)  # from closure, already stop-grad'd
        # KL(P||Q) = Σ P·(log P − log Q), gradient only through Q (student)
        kl_per_pos = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(-1)  # [B, L]
        omega = _kd_omega_gate(denoiser_t, config.kd_gate_k)  # [B]
        kl_gated = kl_per_pos * omega[:, None]
        kd_loss = (config.kd_temperature ** 2 * config.lambda_kd
                   * reduce_token_loss(kl_gated, loss_mask))

    return l2_loss + kd_loss, jnp.zeros(()), l2_loss, kd_loss
```

**Step 4** — Extend `_decoder_branch` return to match 4-tuple shape:

```python
def _decoder_branch(_):
    ...
    return ce_loss, ce_loss, jnp.zeros(()), jnp.zeros(())
    #      ^^^^^^^^  ^^^^^^  ^^^^^^^^^^^^  ^^^^^^^^^^^^
    #      total     ce      l2            kd
```

**Step 5** — Unpack 4-tuple from `jax.lax.cond`:

```python
loss, ce_loss_val, l2_loss_val, kd_loss_val = jax.lax.cond(
    decoder_step_active, _decoder_branch, _denoiser_branch, None,
)
```

**Step 6** — Propagate `kd_loss` through pmean and rescaling (alongside existing `l2_loss`):

```python
kd_loss_val = jax.lax.pmean(kd_loss_val, axis_name="batch")
...
active_kd_loss_val = jnp.where(
    denoiser_prob_arr > 0.0, kd_loss_val / denoiser_prob_arr,
    jnp.zeros_like(kd_loss_val),
)
metrics = {
    "loss": loss,
    "l2_loss": active_l2_loss_val,
    "ce_loss": active_ce_loss_val,
    "kd_loss": active_kd_loss_val,   # NEW
}
```

---

#### `src/configs/training_configs/train_owt_ELF-B-kd.yml` (new file)

Copy `train_owt_ELF-B.yml`, update output dir and run name, add:

```yaml
output_dir: "outputs/elf_b-owt-kd"
wandb_run_name: elf_b-owt-kd

# L_KD (Idea 3)
lambda_kd: 1.0
kd_temperature: 4.0
kd_gate_k: 10.0
```

---

## Compute Overhead

| Branch | Before | After (lambda_kd > 0) |
|--------|--------|-----------------------|
| Decoder (20%) | 1 backbone pass | unchanged |
| Denoiser (80%) | 1 backbone pass, dec head OFF | 1 backbone pass (dec head ON) + 1 teacher pass on x0 |

- Decoder head in denoiser: tiny overhead (2 linear projections, d_hidden→d_enc→V)
- Teacher pass: 1 full backbone forward on x0 at t=1 per denoiser step → ~+100% on denoiser steps → ~**+80% total training time**
- When `lambda_kd=0`: zero overhead (Python-level `if` → no extra code path in JIT)

If overhead is prohibitive, teacher can be cached across K steps (amortization).

---

## Ablations to run

1. **Baseline**: `lambda_kd=0` (standard ELF-B fine-tune, same data/steps)
2. **KD-only**: `lambda_kd=1.0`, no L_ce, no L_anc
3. **KD temperature sweep**: τ ∈ {1, 2, 4, 8}
4. **Gate ablation**: ω(t)=1 (no gate) vs. current gated version

---

## Idea C (Cliff Importance Sampling) — natural companion to this spec

RESEARCH_NOTES groups Idea 3+C as the first training run. Adding Idea C:

- **Source data**: `dec_top1_gt_mean` from `results/elf/probe_decode_v1/probe_decode_branch.json`
- **Weights**: p(t) ∝ dG/dt + ε, where dG/dt is computed by finite difference over the 21-point t_grid
- **Implementation**: new `time_schedule: 'cliff'` option in `sampling_utils.sample_timesteps`
  using `jax.random.choice` over the 21-point grid with the dG/dt+ε weights
- **New config fields**: `cliff_sampling: bool = False`, `cliff_sampling_eps: float = 0.1`
- **⚠️ Double-concentration risk**: ω(t) gate + cliff sampling both focus on t∈[0.25,0.35].
  Monitor the effective t distribution and KD loss during training.

Can be implemented in the same PR as L_KD — small change to `sampling_utils.py` only.

---

## Verification

1. **Gate sanity** (run locally):
   ```python
   t = jnp.linspace(0, 1, 200)
   w = jax.nn.sigmoid(10*(t-0.25)) * (1 - jax.nn.sigmoid(10*(t-0.95)))
   assert w[t < 0.20].max() < 0.05   # near-zero before cliff
   assert w[(t > 0.40) & (t < 0.80)].min() > 0.95  # near-one in plateau
   assert w[t > 0.98].max() < 0.05   # near-zero after plateau
   ```

2. **KL direction**: log that `kd_loss` is positive and decreasing; `teacher CE < student CE` at same t

3. **No regression**: run with `lambda_kd=0` → `kd_loss=0`, all other metrics unchanged vs. original `train_step.py`

4. **Teacher sanity**: print `teacher top-1 accuracy` on a held-out batch at start of training; expect ≈90%+ (consistent with probe result: dec_top1 ≈ 0.97 at t=1)
