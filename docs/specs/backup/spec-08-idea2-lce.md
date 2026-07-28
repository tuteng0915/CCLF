# Spec 08 — Idea 2: L_ce with Commit–Release Schedule

**Type**: code modification + training experiment  
**Priority**: medium  
**Session**: 3 (after spec-07 KD run)  
**Depends on**: spec-07 (best KD checkpoint as starting point)  
**Output**: `models/ELF/outputs/elf_b-owt-kd-ce/`

---

## Background

w→c correction rate peaks at 20.8%/step at t=0.30 (commitment cliff), then drops to
1–1.5%/step on the plateau. CE supervision is 10–20× more effective at the cliff.

L_ce adds a commit-release schedule that applies CE only during the cliff window, then
releases it to avoid over-committing wrong-locked positions in the plateau.

**Formula** (two-stage α_nm):
```
α_nm(t) = α1·σ(k1·(t − 0.30)) − α2·σ(k2·(t − 0.60))
```
- Stage 1 rises at t≈0.30: supervise crystallization during the cliff
- Stage 2 falls at t≈0.60: release CE as plateau begins and wrong-committed positions freeze

Default: `α1=1.0, α2=1.0, k1=20.0, k2=20.0` (sharp transitions)

---

## Implementation

**File**: `src/train_step.py`  
**Location**: inside `_denoiser_branch`, after the L_KD block

### New config fields (`config.py`)

```python
# L_ce (Idea 2): commit-release CE schedule
lambda_ce: float = 0.0          # CE loss weight; 0.0 = disabled
ce_alpha1: float = 1.0           # amplitude of rise at cliff
ce_alpha2: float = 1.0           # amplitude of fall at plateau onset
ce_k1: float = 20.0              # sigmoid sharpness for rise (t≈0.30)
ce_k2: float = 20.0              # sigmoid sharpness for fall (t≈0.60)
```

### New gate helper (`train_step.py`, module level)

```python
def _ce_alpha_nm(t, alpha1, alpha2, k1, k2):
    """Two-stage commit-release schedule: rises at t≈0.30, falls at t≈0.60."""
    return (alpha1 * jax.nn.sigmoid(k1 * (t - 0.30))
            - alpha2 * jax.nn.sigmoid(k2 * (t - 0.60)))
```

### Addition to `_denoiser_branch`

After the L_KD block, before `return`:

```python
# L_ce: cross-entropy supervision at the commitment cliff
ce_loss_den = jnp.zeros(())
if config.lambda_ce > 0:
    # student_logits already computed when lambda_kd > 0; else recompute
    if config.lambda_kd == 0:
        _, student_logits = state.apply_fn(
            {"params": params}, denoiser_input, denoiser_t,
            deterministic=False, rngs={"dropout": model_dropout_rng},
            self_cond_cfg_scale=self_cond_cfg_scale,
            decoder_step_active=jnp.array(True),
        )
    log_probs = jax.nn.log_softmax(student_logits.astype(jnp.float32), axis=-1)
    ce_per_pos = -jnp.take_along_axis(log_probs, decoder_targets[..., None], axis=-1).squeeze(-1)
    alpha_nm = _ce_alpha_nm(denoiser_t, config.ce_alpha1, config.ce_alpha2,
                             config.ce_k1, config.ce_k2)  # [B]
    alpha_nm = jnp.clip(alpha_nm, 0.0, None)  # no negative supervision
    ce_gated = ce_per_pos * alpha_nm[:, None]
    ce_loss_den = config.lambda_ce * reduce_token_loss(ce_gated, loss_mask)

return l2_loss + kd_loss + ce_loss_den, jnp.zeros(()), l2_loss, kd_loss, ce_loss_den
```

Note: the return tuple grows to 5 elements. Update `_decoder_branch` and all unpacking accordingly.

---

## Training configs

### train_owt_ELF-B-kd-ce.yml

```yaml
# (copy train_owt_ELF-B-kd.yml and add:)
lambda_kd: 1.0
kd_temperature: 4.0
kd_gate_k: 10.0
lambda_ce: 0.5
ce_alpha1: 1.0
ce_alpha2: 1.0
ce_k1: 20.0
ce_k2: 20.0
output_dir: "outputs/elf_b-owt-kd-ce"
wandb_run_name: elf_b-owt-kd-ce
```

---

## Ablation matrix

| Run | lambda_kd | lambda_ce | output_dir | Purpose |
|-----|-----------|-----------|------------|---------|
| KD-only | 1.0 | 0 | elf_b-owt-kd (spec-07) | baseline for this ablation |
| KD+CE | 1.0 | 0.5 | elf_b-owt-kd-ce | main run |
| CE-only | 0 | 0.5 | elf_b-owt-ce-only | isolate CE effect |
| two-stage | 1.0 | 0.5 | elf_b-owt-kd-ce | (k1=k2=20, α1=α2=1) |
| three-stage | 1.0 | 0.5 | elf_b-owt-kd-ce3 | add +α3·σ(k3(t−0.90)) |

### Three-stage config override

```bash
--config_override lambda_ce=0.5 ce_alpha1=1.0 ce_alpha2=1.0 \
                  ce_k1=20.0 ce_k2=20.0 \
                  ce_alpha3=0.3 ce_k3=20.0 ce_t3=0.90
```

(Requires adding `ce_alpha3`, `ce_k3`, `ce_t3` to config and updating `_ce_alpha_nm`.)

---

## Metrics to track

Same as spec-07 plus:
- `ce_loss` (denoiser branch CE, now tracked separately from decoder branch CE)
- α_nm(t) curve at initialization (verify it rises at t=0.30 and falls at t=0.60)

---

## Success criteria

- KD+CE Gen.PPL < KD-only Gen.PPL
- CE-only better than baseline but worse than KD+CE (CE alone insufficient)
- two-stage ≥ three-stage (t3 recommit adds noise, not signal)

---

## Notes

- ⚠️ If lambda_kd=0 and lambda_ce>0, student_logits must be recomputed (decoder head
  not activated in default denoiser path). The implementation above handles this case.
- decoder_targets = batch["input_ids"] (the original token ids), same as decoder branch.
  This is correct: we're supervising the denoiser's token prediction against ground truth.
- α_nm can become slightly negative between the two sigmoid stages — `jnp.clip(0)` ensures
  we don't accidentally add negative CE (which would push the model away from correct tokens).
