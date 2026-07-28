# Spec 07 — Idea 3+C: L_KD Training Run (+ Cliff Importance Sampling)

**Type**: training experiment  
**Priority**: high (first training run, core result)  
**Session**: 2 (after Idea A inference experiment)  
**Depends on**: spec-05 (baseline), spec-04 implementation (L_KD code, already done)  
**Output**: `models/ELF/outputs/elf_b-owt-kd/`

---

## Background

L_KD implementation is complete (spec-04). This spec covers:
1. The actual training run
2. Optionally adding Idea C (cliff importance sampling) in the same run
3. Ablation runs to isolate each component

See `spec-04-idea3-lkd.md` for the full formula and implementation details.

---

## Idea C: Cliff Importance Sampling (optional addition to this run)

**Motivation**: ELF's logit-normal schedule (P_mean=-1.5) concentrates t around 0.18,
below the commitment cliff. L_KD with ω(t) gate fires only ~31% of steps. Idea C shifts
the distribution toward the cliff.

**Formula**: p(t) ∝ dG/dt + ε, computed from dec_top1 in probe_decode_branch.json.

### Implementation (`src/utils/sampling_utils.py`)

Add new option to `sample_timesteps`:

```python
def sample_timesteps(rng, batch_size, P_mean=-0.8, P_std=0.8,
                     time_schedule='logit_normal', cliff_weights=None):
    if time_schedule == 'cliff':
        assert cliff_weights is not None
        # cliff_weights: [21] normalized probabilities at t_grid=[0,0.05,...,1.0]
        t_grid = jnp.linspace(0.0, 1.0, len(cliff_weights))
        indices = jax.random.choice(rng, len(cliff_weights),
                                    shape=(batch_size,), p=jnp.array(cliff_weights))
        return t_grid[indices]
    ...
```

**Config** (`config.py`): add `cliff_sampling: bool = False`, `cliff_sampling_eps: float = 0.1`

**Precompute weights** (run once offline):

```python
import json, numpy as np

with open("results/elf/probe_decode_v1/probe_decode_branch.json") as f:
    d = json.load(f)

t = np.array(d["t"])           # [21]
G = np.array(d["dec_top1_gt_mean"])  # [21]
dG = np.diff(G) / np.diff(t)  # [20] forward differences
dG = np.clip(dG, 0, None)     # keep only positive changes

# Extend to 21 points (append last value), add floor ε, normalize
dG21 = np.append(dG, dG[-1])
eps = 0.1
weights = dG21 + eps
weights /= weights.sum()
print(list(np.round(weights, 4)))
```

Store the resulting 21-element list in the YAML config as `cliff_weights`.

⚠️ **Double-concentration risk**: ω(t) gate + cliff sampling both focus on t∈[0.25,0.35].
Monitor the effective t distribution in early training steps.

---

## Experiment runs

### Run 1 — L_KD only (main experiment)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python src/train.py \
    --config src/configs/training_configs/train_owt_ELF-B-kd.yml \
    --checkpoint embedded-language-flows/ELF-B-owt
```

Config: `train_owt_ELF-B-kd.yml` (already created, `lambda_kd=1.0`)

### Run 2 — L_KD + Cliff sampling (Idea 3+C)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python src/train.py \
    --config src/configs/training_configs/train_owt_ELF-B-kd.yml \
    --checkpoint embedded-language-flows/ELF-B-owt \
    --config_override output_dir=outputs/elf_b-owt-kd-cliff \
                      wandb_run_name=elf_b-owt-kd-cliff \
                      time_schedule=cliff \
                      cliff_sampling=true
```

---

## Metrics to track (W&B)

| Metric | Expected | Notes |
|--------|----------|-------|
| `kd_loss` | ↓ over training | primary KD signal |
| `l2_loss` | should not degrade | ensure L2 not crowded out |
| `ce_loss` | should not degrade | ensure decoder not harmed |
| `loss` | ↓ overall | weighted sum |
| Gen.PPL | ↓ vs baseline | epoch-end eval |

**Early diagnostic**: at step 0 (before any gradient update), log:
- mean `kd_loss` over first 100 steps → confirms KD fires
- histogram of t values that triggered ω(t) > 0.5 → confirm gate is active

---

## Ablation matrix (run after main experiment)

| Run | lambda_kd | time_schedule | output_dir |
|-----|-----------|---------------|------------|
| baseline | 0 | logit_normal | elf_b-owt-baseline (spec-05) |
| KD-only | 1.0 | logit_normal | elf_b-owt-kd |
| KD+cliff | 1.0 | cliff | elf_b-owt-kd-cliff |
| cliff-only | 0 | cliff | elf_b-owt-cliff-only |

The `cliff-only` run (no KD, just reweighted t sampling) isolates the importance sampling effect.

---

## Success criteria

- `kd_loss` decreases and stabilizes (doesn't diverge or oscillate)
- Gen.PPL (KD run, epoch 5) < Gen.PPL (baseline, epoch 5)
- `l2_loss` and `ce_loss` within ±5% of baseline at same training steps
- Cliff-sampling run: t histogram visibly shifted toward [0.25, 0.40]

---

## Open questions to answer from this run

1. Does p_t^lin (linear branch) improve on the held-out probe after KD training? → run probe_decode_branch.py on the KD checkpoint and compare G_dec gap
2. Does KD-cliff double-concentrate or complement? → check t distribution + kd_loss per-t
