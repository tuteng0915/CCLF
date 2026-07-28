# Spec 10 — Idea 1: Cosine L_anc (Ablation)

**Type**: code modification + training experiment (ablation only)  
**Priority**: low (weakest component; run last)  
**Session**: 5 (after spec-08 or spec-09)  
**Depends on**: any prior KD checkpoint (used as base)  
**Output**: `models/ELF/outputs/elf_b-owt-kd-anc/`

---

## Background

ρ(t) = ‖r_t‖/‖x̂_t‖ ≥ 0.82 throughout the denoising trajectory. The contextual
residual r_t = x̂_t − a_t has an irreducible magnitude floor; an L2 anchor loss that
fights this floor is inefficient. A cosine anchor loss targets only angular alignment.

**Formula**:
```
L_anc(t) = (1/L) Σ_i (1 − cos(x̂_{t,i}, sg(a_{t,i}))) · β(t)
β(t) = σ(k_β(t − 0.20)) · (1 − σ(k_β(t − 0.55)))  ← active during commitment window
```
where a_{t,i} = nearest contextual centroid in embedding space.

**Expected impact**: low — this is primarily a regularizer. λ_anc << λ_KD.

---

## Implementation

**File**: `src/train_step.py`  
**New config fields** (`config.py`):

```python
# L_anc (Idea 1): cosine anchor regularizer
lambda_anc: float = 0.0     # loss weight; 0.0 = disabled; keep << lambda_kd
anc_gate_k: float = 20.0    # sigmoid sharpness k_β
centroid_path: str = None   # path to token_centroids.npz (for a_{t,i})
```

### Gate helper (`train_step.py`, module level)

```python
def _anc_beta_gate(t, k):
    """β(t) = σ(k(t−0.20))·(1−σ(k(t−0.55))): active during commitment window."""
    return jax.nn.sigmoid(k * (t - 0.20)) * (1.0 - jax.nn.sigmoid(k * (t - 0.55)))
```

### Centroid loading

At the top of `train_step` (or in `train.py` initialization):

```python
if config.lambda_anc > 0 and config.centroid_path:
    import numpy as np
    _centroids = np.load(config.centroid_path)["centroids"].astype(np.float32)
    centroids = jnp.array(_centroids) / config.latent_std  # normalize to ELF latent space
    # shape: [V, d_enc]
```

### Addition to `_denoiser_branch`

```python
# L_anc: cosine anchor regularizer during commitment window
anc_loss = jnp.zeros(())
if config.lambda_anc > 0:
    # x_pred from denoiser: shape [B, L, d_enc]
    # Find nearest centroid per position
    # x_pred: [B, L, d]  centroids: [V, d]
    # cos_sim: [B, L, V] → max over V → nearest centroid
    x_norm = x_pred / (jnp.linalg.norm(x_pred, axis=-1, keepdims=True) + 1e-8)
    c_norm = centroids / (jnp.linalg.norm(centroids, axis=-1, keepdims=True) + 1e-8)
    # Nearest centroid (stop-gradient: anchor is fixed)
    cos_sim = jnp.einsum('bld,vd->blv', x_norm, jax.lax.stop_gradient(c_norm))
    nn_idx = jnp.argmax(cos_sim, axis=-1)  # [B, L]
    a_t = jax.lax.stop_gradient(centroids[nn_idx])  # [B, L, d]

    # Cosine loss
    a_norm = a_t / (jnp.linalg.norm(a_t, axis=-1, keepdims=True) + 1e-8)
    cosine_dist = 1.0 - (x_norm * jax.lax.stop_gradient(a_norm)).sum(-1)  # [B, L]
    beta = _anc_beta_gate(denoiser_t, config.anc_gate_k)
    anc_loss = config.lambda_anc * reduce_token_loss(
        cosine_dist * beta[:, None], loss_mask
    )
```

⚠️ The `jnp.einsum('bld,vd->blv', ...)` over V=32100 is expensive: `[B, L, V]` is
`[32, 1024, 32100]` ≈ 4 GB in float32. **Use chunked computation or token-level NN
lookup instead** if memory is a concern:

```python
# Alternative: use pre-indexed centroids (lookup by current top-1 token)
top1_student = jnp.argmax(student_log_probs, axis=-1)  # [B, L] (reuse from KD block)
a_t = jax.lax.stop_gradient(centroids[top1_student])   # [B, L, d]
```

This approximation (anchor = centroid of current top-1 predicted token) is faster and
avoids the V-dimensional matmul. Use this unless full NN is required.

---

## Training configs

```yaml
# train_owt_ELF-B-kd-anc.yml (copy from kd.yml, add:)
lambda_anc: 0.05          # small; << lambda_kd
anc_gate_k: 20.0
centroid_path: "results/data/token_centroids.npz"
output_dir: "outputs/elf_b-owt-kd-anc"
wandb_run_name: elf_b-owt-kd-anc
```

---

## Ablation matrix

| Run | lambda_anc | anchor type | output_dir |
|-----|-----------|-------------|------------|
| KD-only (no L_anc) | 0 | — | elf_b-owt-kd (spec-07) |
| KD + cosine L_anc | 0.05 | top-1 centroid | elf_b-owt-kd-anc |
| KD + L2 L_anc | 0.05 | top-1 centroid (L2 dist) | elf_b-owt-kd-anc-l2 |

L2 variant: replace `cosine_dist` with `jnp.sum((x_pred - a_t)**2, axis=-1)`.

---

## Metrics to track

Same as spec-07 plus:
- `anc_loss` (should be small; <10% of total loss)
- ρ(t) = ‖x̂_t − a_t‖/‖x̂_t‖ on held-out probe: should decrease vs KD-only

---

## Success criteria

- KD+cosine-anc Gen.PPL ≈ KD-only Gen.PPL (neutral: confirms it's a regularizer, not a booster)
- KD+cosine-anc Gen.PPL ≥ KD+L2-anc Gen.PPL (cosine at least as good as L2)
- ρ(t) decreases on probe checkpoint (geometric anchoring confirmed)

**If cosine L_anc hurts**: reduce λ_anc to 0.01 or disable. Do NOT waste compute on
extensive tuning — this is a low-priority ablation.

---

## Notes

- `token_centroids.npz` is at `results/data/token_centroids.npz` (computed in spec-01).
  22250/32100 tokens have valid centroids; the rest are zero vectors. The top-1 centroid
  approximation naturally handles this: if the model predicts a token with zero centroid,
  the loss is zero for that position (anchor = zero, cosine = undefined → handled by ε).
- β(t) gate [0.20, 0.55]: active during commitment formation, off during plateau.
  This intentionally limits L_anc to the window where geometric anchoring is still forming.
