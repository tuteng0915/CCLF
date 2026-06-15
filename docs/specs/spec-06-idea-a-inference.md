# Spec 06 — Idea A: Decode-Branch Self-Conditioning at Inference

**Type**: code modification + experiment  
**Priority**: high (validates gap before any training)  
**Session**: 1 (no training required; run before or alongside spec-05)  
**Output**: `results/elf/idea_a_inference/`

---

## Background

ELF currently uses x̂_t^(1) (linear branch output from the first forward pass) as its
self-conditioning signal. The decode branch at the same t achieves +15–27 pp higher
top-1 accuracy. Idea A asks: can we just use the decode-branch embedding as the SC signal
at inference time, without any retraining?

**Three-pass scheme (per sampling step):**
```
Pass 1:  [z_t, x_pred_prev] → backbone(t) → x_pred_raw    (standard denoiser)
Pass 1b: [x_pred_raw, zeros] → backbone(t=1) → x_pred_dec  (decode branch)
Pass 2:  [z_t, x_pred_dec]  → backbone(t) → x_pred_final  (main pass with better SC)
```

**Cost**: ~3× per sampling step vs current 1× (no retraining).

---

## Implementation

**File**: `src/utils/sampling_utils.py`  
**Function**: `_forward_sample_self_cond`

### Current flow (2 paths: uncond or cond)

```python
# default path (self_cond_cfg_scale=1, x_pred_prev not None):
z_input_cond = jnp.concatenate([z, x_pred_prev], axis=-1)
net_out_cond = model_apply_fn({"params": model_params}, z_input_cond, t_batch, ...)
v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
return v_cond, x_cond
```

### New flow (add decode-branch refinement step)

Add a `use_decode_sc: bool` flag to the function signature and to `_forward_sample` and
`_ode_step` / `_sde_step`. When True:

```python
# After getting x_cond from the standard forward:
if use_decode_sc:
    # Pass 1b: run x_cond through backbone at t=1 to get decode-branch embedding
    t_one = jnp.ones_like(t_batch)
    x_dec_input = jnp.concatenate([x_cond, jnp.zeros_like(x_cond)], axis=-1)
    net_out_dec, _ = model_apply_fn(
        {"params": model_params}, x_dec_input, t_one,
        deterministic=True, decoder_step_active=jnp.array(False),
    )
    _, x_pred_dec = net_out_to_v_x(net_out_dec, x_cond, t_one, t_eps)
    x_pred_dec = restore_cond(x_pred_dec, cond_seq, cond_seq_mask)

    # Pass 2: re-run with decode-branch x_pred_dec as SC
    z_input_dec = jnp.concatenate([z, x_pred_dec], axis=-1)
    net_out_final = model_apply_fn({"params": model_params}, z_input_dec, t_batch, ...)
    v_cond, x_cond = net_out_to_v_x(net_out_final, z, t_batch, t_eps)
    v_cond, x_cond = _restore_vx(v_cond, x_cond)
```

**Config change**: add to `SamplingConfig`:
```python
use_decode_sc: bool = False
```

**YAML usage** (in `uncond_sampling_configs.yml`):
```yaml
- sampling_method: ode
  num_sampling_steps: [1, 2, 4, 8, 16, 50]
  use_decode_sc: true
```

---

## Experiment

Run generation with and without `use_decode_sc` at {1, 2, 4, 8, 16, 50} steps,
starting from the original pretrained checkpoint (no fine-tuning needed).

```bash
# on new-ncl, from ~/tt_workspace/model/CCLF/CCLF/models/ELF/
CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
python src/eval.py \
    --config src/configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint embedded-language-flows/ELF-B-owt \
    --sampling_configs_path src/configs/sampling_configs/idea_a_eval.yml \
    --output_dir results/elf/idea_a_inference \
    --num_samples 1000
```

Create `src/configs/sampling_configs/idea_a_eval.yml`:
```yaml
# Baseline (no decode SC)
- sampling_method: ode
  num_sampling_steps: [1, 2, 4, 8, 16, 50]
  use_decode_sc: false
  time_schedule: uniform

# Idea A (with decode SC)
- sampling_method: ode
  num_sampling_steps: [1, 2, 4, 8, 16, 50]
  use_decode_sc: true
  time_schedule: uniform
```

---

## Metrics to collect

For each (steps, use_decode_sc) pair:
- **Gen.PPL** (GPT-2-large) — primary metric
- Wall-clock time per sample (confirm ~3× overhead)

Additionally, run a **probe-style intermediate logging** during generation to get:
- G(t) = top-1 accuracy of x_pred_final at each step vs ground-truth tokens
- Rec@1(t) = same for x_pred_raw (without decode SC)

This intermediate logging is optional — implement only if not too invasive to the scan loop.

---

## Output

`results/elf/idea_a_inference/`
- Per-config `metrics.jsonl` with Gen.PPL
- `summary.md` with PPL table comparing baseline vs Idea A at each step count

---

## Success criteria

| Result | Interpretation |
|--------|---------------|
| Gen.PPL (Idea A) < Gen.PPL (baseline) at same step count | Gap is exploitable at inference |
| Improvement concentrated at low step counts (1–4 steps) | Decode SC helps most when budget is tight |
| Gen.PPL (Idea A, 4 steps) ≈ Gen.PPL (baseline, 8 steps) | Effective step reduction |
| No improvement | Gap is training problem, not forward-pass problem → strengthens Idea 3 necessity |

---

## Notes

- `decoder_step_active=jnp.array(False)` in Pass 1b: we only need the backbone output
  (x_pred_dec as embedding), not the logits.
- `t=1` in Pass 1b: backbone at t=1 = decode branch mode; no additional noise.
- `jax.lax.scan` compatibility: `use_decode_sc` is a static bool (config-level), so
  the scan body can use Python-level `if`. No dynamic branching needed.
- ⚠️ The JIT-compiled scan will be 3× longer if `use_decode_sc=True` (3 model calls per
  step). Expect recompilation when switching between True/False configs.
