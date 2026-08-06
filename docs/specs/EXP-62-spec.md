# EXP-62 Spec — Controlled Fine-Tuning Checkpoint Panel

**Status**: RUNNING (Stage A matched 2k-step control/full-KD pilot)
**Priority**: P0 — establish a controlled KD effect before more checkpoint-specific interventions
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

Save matched intermediate checkpoints so this contrast can be evaluated at
the same training budgets. Random-seed replication is not part of the panel.

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

- recipe effect relative to the matched `ct_control` at each saved budget;
- temporal-window trend across controlled interventions;
- correlation between mechanism fingerprints and sampler response, labelled
  exploratory at this panel size.

The design supports conclusions about these matched checkpoints. It does not
support a claim about training-seed variance or average recipe effects over
random initialization.

## 8. Decision rules

- **Controlled KD effect**: `kd_full` moves consistently away from the matched
  continued-training control across saved training budgets.
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
