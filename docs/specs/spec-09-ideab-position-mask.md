# Spec 09 — Idea B: Position-Adaptive KD Mask

**Type**: code modification + training experiment  
**Priority**: medium  
**Session**: 4 (after spec-07 KD run confirms L_KD works)  
**Depends on**: spec-07 (L_KD baseline)  
**Output**: `models/ELF/outputs/elf_b-owt-kd-mask/`

---

## Background

In the stable-but-imperfect plateau (t=0.35–0.95), ~13–17% of positions are
"wrong-committed": low entropy (H < thresh) but wrong top-1 token. These are the
highest-value targets for KD correction, but currently receive equal weight as
low-entropy correct positions and high-entropy uncertain positions.

The mask focuses KD weight on positions that need it most.

**Formula**:
```
w_{t,i} = (1 − H(p_{t,i}) / log V) · 1[argmax p_{t,i} ≠ argmax p_{1,i}^dec]
```
- `1 − H/logV`: high weight for low-entropy (committed) positions
- `1[wrong top-1 vs teacher]`: high weight only when student disagrees with teacher
- Combined: high weight = committed AND wrong

---

## Implementation

**File**: `src/train_step.py`  
**Location**: replace `reduce_token_loss(kl_gated, loss_mask)` in the L_KD block

### New config fields (`config.py`)

```python
# Idea B: position-adaptive KD mask
kd_position_mask: bool = False   # enable per-position weighting
kd_mask_min_weight: float = 0.1  # floor weight (avoid zeroing out uncertain positions)
```

### Modified L_KD block in `_denoiser_branch`

```python
if config.lambda_kd > 0:
    student_log_probs = jax.nn.log_softmax(student_logits.astype(jnp.float32), axis=-1)
    teacher_probs = jnp.exp(teacher_log_probs)
    kl_per_pos = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(-1)  # [B, L]

    if config.kd_position_mask:
        # Per-position weight: (1 - H/logV) * 1[wrong top-1]
        V = student_log_probs.shape[-1]
        H_student = -(jnp.exp(student_log_probs) * student_log_probs).sum(-1)  # [B, L]
        entropy_weight = 1.0 - H_student / jnp.log(V)                          # [B, L]
        wrong_mask = (jnp.argmax(student_log_probs, axis=-1)
                      != jnp.argmax(teacher_log_probs, axis=-1)).astype(jnp.float32)  # [B, L]
        pos_weight = entropy_weight * wrong_mask
        # Add floor to avoid zeroing all positions when none are wrong-committed
        pos_weight = pos_weight + config.kd_mask_min_weight
        pos_weight = pos_weight / pos_weight.mean()  # normalize to keep loss scale stable
        kl_per_pos = kl_per_pos * pos_weight

    omega = _kd_omega_gate(denoiser_t, config.kd_gate_k)
    kl_gated = kl_per_pos * omega[:, None]
    kd_loss = (config.kd_temperature ** 2 * config.lambda_kd
               * reduce_token_loss(kl_gated, loss_mask))
```

---

## Training configs

### train_owt_ELF-B-kd-mask.yml

```yaml
# (copy train_owt_ELF-B-kd.yml and add:)
lambda_kd: 1.0
kd_temperature: 4.0
kd_gate_k: 10.0
kd_position_mask: true
kd_mask_min_weight: 0.1
output_dir: "outputs/elf_b-owt-kd-mask"
wandb_run_name: elf_b-owt-kd-mask
```

---

## Ablation matrix

| Run | kd_position_mask | output_dir | Purpose |
|-----|-----------------|------------|---------|
| uniform KD | false | elf_b-owt-kd (spec-07) | baseline |
| masked KD | true | elf_b-owt-kd-mask | this spec |

For the combined best system (after individual ablations):

| Run | lambda_kd | kd_mask | lambda_ce | output_dir |
|-----|-----------|---------|-----------|------------|
| KD+mask+CE | 1.0 | true | 0.5 | elf_b-owt-full |

---

## Metrics to track

Same as spec-07 plus:
- **pos_weight distribution**: log mean and std of `pos_weight` (unnormalized) per training step
  → confirms mask is non-trivial (std > 0, mean > 0 beyond floor)
- **wrong-committed fraction**: run probe on KD-mask checkpoint, compare `comm_wrong` at t=0.50
  vs uniform-KD checkpoint → should decrease

---

## Success criteria

- Masked KD Gen.PPL < Uniform KD Gen.PPL
- `comm_wrong` fraction at t=0.50 decreases more than uniform KD
- Mean `pos_weight` > floor for ≥50% of denoiser steps (mask is active)

---

## Notes

- Normalization: `pos_weight / pos_weight.mean()` keeps the average KD weight = 1.0,
  so `lambda_kd` retains its interpretation as the loss coefficient.
- `kd_mask_min_weight=0.1` prevents the mask from zeroing the loss when t is very early
  (before any positions are committed). The floor ensures gradients flow even during
  warm-up.
- student_log_probs used for both KD and mask computation — no extra forward pass.
