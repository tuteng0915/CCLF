# EXP-62 Spec — Controlled Fine-Tuning Checkpoint Panel

**Status**: SUPERSEDED FOR KD CAUSAL CLAIMS — valid negative result for a mismatched noisy-head self-distillation objective
**Priority**: closed; replaced by EXP-63
**Base model**: official ELF-B OpenWebText baseline  
**Pilot config**: `src/configs/training_configs/finetune_owt_ELF-B-panel.yml`

The launcher uses `conda run --no-capture-output` so loss and failure logs are
visible in tmux during the 2,000-step run rather than buffered until process
exit.

## 1. Motivation

Most intervention results currently depend on two historically trained
checkpoints, `kd_cr` and `kd2`. They differ not only in the intended KD
schedule, but also in training duration (roughly 700k versus 400k steps),
trajectory through optimization, and possibly checkpoint-selection details.
They also show opposite and unstable responses to self-conditioning,
Diffusion Forcing, Pipeline ODE, and sequence length.

Consequently, a difference between `kd_cr` and `kd2` cannot by itself identify
a KD-time-window mechanism. We need a small controlled population of
fine-tuned checkpoints, not more analyses of two historical endpoints.

## 2. Scientific questions

1. How much of the observed behavior is ordinary continued-training drift?
2. Does the KD effect persist across matched training budgets?
3. Does the KD time window produce a systematic early-to-late trend under a
   fixed initialization and data order?
4. Which mechanism measurements predict downstream sampler stability across
   checkpoints?

## 3. Stage A — objective and drift controls (required first)

All runs start from the same official baseline checkpoint and share data,
batch order policy, optimizer, learning rate, total optimizer steps, sequence
length, and architecture.

| family | KD | randomization | purpose |
|---|---|---|---|
| `ct_control` | none (`lambda_kd=0`) | fixed seed 42 | ordinary continued-training drift |
| `kd_full` | full trajectory, `lambda_kd=1` | same initialization and data order | controlled KD effect |

The primary causal contrast is

```text
kd_full(step) - ct_control(step),
```

not `kd_cr - baseline` or `kd2 - baseline`.

The 2,000-step pilot saves the matched final checkpoints and serves only as a
go/no-go comparison. If promoted, the formal runs must save matched
intermediate checkpoints so the contrast can be evaluated at the same
training budgets. Random-seed replication is not part of the panel.

## 4. Stage B — temporal KD panel

Run the three equal-width windows with the same fixed initialization and data
order under the native logit-normal denoiser time distribution:

| family | KD gate | intended role |
|---|---|---|
| `kd_early` | `[0.05, 0.30]` | before/through first local evidence |
| `kd_transition` | `[0.30, 0.55]` | endpoint-affinity and collapse window |
| `kd_late` | `[0.55, 0.80]` | post-selection refinement |

The base denoising and decoder losses see the same timestep distribution in
every family; only the KD mask changes. The KD loss is normalized over active
positions, so `lambda_kd` has the same nominal scale, though the number and
difficulty of active examples must still be logged.

Compare each window against the matched continued-training and full-KD runs at
the same optimizer budget. This avoids an uninformative random-seed grid.

## 5. Pilot budget and formal budget

### Pilot

- sequence length: 128;
- 2,000 optimizer steps;
- global batch size: 32;
- AdamW, learning rate `1e-5`, 20-step warmup;
- save final EMA and raw parameters;
- native-noise evaluation at length 128.

This pilot creates inexpensive experimental objects for mechanism screening.
It is not directly comparable to the historical 400k/700k, length-1024 KD
runs in training compute.

### Stage A pilot result (`2026-08-06`)

Both matched runs completed 2,000 steps and saved raw and EMA parameters. The
first gate uses EMA weights, the identical seed-42 initial-noise bank,
`n=256`, length 128, native `z0 = 2 epsilon`, ODE-32, and SC-CFG 3:

| training | PPL | D1 | D2 | rep-4 | degeneration |
|---|---:|---:|---:|---:|---:|
| continued-training control | 261.8 | 0.394 | 0.860 | 0.008 | 0.008 |
| full-trajectory KD | 220.0 | 0.441 | 0.873 | 0.007 | 0.004 |
| full-KD minus control | **-41.8** | **+0.047** | **+0.013** | **-0.001** | **-0.004** |

The initial metric-only reading was positive, but unselected samples from both
KD runs were code-like, fragmented pseudo-text. The original degeneration
heuristic missed most of these failures. Lower GPT-2 PPL therefore cannot be
read as better generation.

### Stage B result and objective audit (`2026-08-08`)

All three windows completed 2,000 steps. Under the same ODE-32 gate:

| training | PPL | D1 | D2 | rep-4 | degeneration |
|---|---:|---:|---:|---:|---:|
| continued-training control | 261.8 | 0.394 | 0.860 | 0.008 | 0.008 |
| noisy-head KD, full | 220.0 | 0.441 | 0.873 | 0.007 | 0.004 |
| noisy-head KD, early | **159.5** | 0.451 | 0.866 | 0.014 | 0.016 |
| noisy-head KD, transition | 311.3 | 0.379 | 0.855 | 0.004 | 0.000 |
| noisy-head KD, late | 273.0 | 0.396 | 0.861 | 0.007 | 0.004 |

The PPL ordering survives ODE-16/32/64, but degeneration grows with solver
steps and is strongest for early KD: at ODE-64, early KD reaches PPL 53.5
while the automatic degeneration flag rises to 9.8%, and inspected samples
are plainly repetitive fragments. This is metric gaming, not a method win.

The implementation audit then found that this panel did not train the
historical JAX objective. PyTorch used the decoder logits from the same noisy
mixed forward as the teacher, a hard temporal mask, and active-only
normalization. JAX uses a separate stop-gradient clean-`x0`, `t=1` decoder
teacher, a smooth sigmoid plateau, and ordinary-token normalization. Thus the
window ordering above applies only to this mismatched noisy-head
self-distillation objective. EXP-63 replaces it with a corrected replication.

### Formal promotion

Promote at most two families after the pilot:

- sequence length 1024;
- matched effective token budget and optimizer steps;
- checkpoints saved at multiple training budgets to separate recipe from
  training duration.

Do not formally train all five families unless the pilot shows an ordered
effect.

## 6. Compact checkpoint fingerprint

Every trained checkpoint gets the same evaluation card:

1. native ODE generation: Gen.PPL, D1/D2, repetition, degeneration;
2. true-rollout `tau_first`, `tau_stable`, and revision count;
3. perturbation amplification versus self-correction around the transition;
4. self-conditioning susceptibility;
5. standard versus Pipeline ODE response under EXP-61's corrected protocol;
6. for retained families only, GS16 endpoint specificity/collapse timing.

This is intentionally smaller than rerunning the entire experiment archive.
It tests the axes that currently show checkpoint instability.
Run it through the staged settings in [`EVAL-PROTOCOL.md`](EVAL-PROTOCOL.md):
length-128 smoke first, then the native length-1024 and solver-budget checks
only for checkpoints promoted beyond screening.

## 7. Analysis

Treat this as a controlled intervention panel, not an estimate of a population
over training seeds. Report:

- pilot recipe effect relative to the matched `ct_control` at 2,000 steps;
- for promoted formal runs, recipe effect at each matched saved budget;
- temporal-window trend across controlled interventions;
- correlation between mechanism fingerprints and sampler response, labelled
  exploratory at this panel size.

The design supports conclusions about these matched checkpoints. It does not
support a claim about training-seed variance or average recipe effects over
random initialization.

## 8. Decision rules

- **Pilot controlled KD effect**: `kd_full` materially differs from the matched
  continued-training control at 2,000 steps on the compact fingerprint.
- **Formal controlled KD effect**: after promotion, the difference persists
  across matched saved training budgets rather than appearing at one endpoint.
- **Temporal mechanism**: early/transition/late windows show an ordered or
  sign-changing effect under the shared initialization and data order.
- **Training drift**: `ct_control` changes the mechanism or sampler metrics as
  much as KD. Existing KD claims must be expressed relative to continued
  training, not the original baseline.

## 9. Optional extensions — not part of the first panel

- KD-strength sweep only after a stable temporal effect is found;
- SC-consistency or WFF objectives only as method variants against this panel;
- instruction SFT or a new dataset only after the OpenWebText control panel is
  understood, because changing data and objective together would destroy the
  causal interpretation.
