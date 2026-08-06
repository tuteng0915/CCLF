# EXP-60 Spec — Native Wavefront Flow Forcing Training Pilot

**Status**: DONE / NEGATIVE — paired pilot does not improve the sampler interaction
**Priority**: closed method pilot
**Model**: ELF-B `kd_cr`, OpenWebText, length 128  
**Code**: `models/ELF-torch/src/` and
`models/ELF-torch/experiments/probe_elf/eval_wff_pilot.py`

## Why run this after the negative GS19 result?

GS19 showed that imposing heterogeneous states on a checkpoint trained only
with a scalar global time severely hurts ELF and Plaid. That is evidence
against an inference-only asynchronous pipeline, but it does not distinguish
an intrinsically bad schedule from a train--test mismatch. EXP-60 performs
this one narrow follow-up: give the model native per-token time conditioning
during training and ask whether the failure survives.

This is not evidence for Wavefront Flow Forcing unless the WFF-specific
training-by-sampler interaction is positive.

## Model and clock

For global progress `s`, position rank `q_i in [0,1]`, direction sign
`r_i = 1 - 2 q_i`, and wave width `Delta`:

```text
tau_i(s) = clip(s + Delta sin(pi s) r_i, 0, 1),  Delta <= 1/pi.
```

The sine envelope gives exact synchronous endpoints,
`tau_i(0)=0` and `tau_i(1)=1`, while allowing the prefix to lead in the
interior. Reversing `r_i` gives the RTL control; random permutations are
included during training to separate directionality from symmetry breaking.

The original scalar-time prefix remains. For local clocks, token `i` receives

```text
h_i <- h_i + tanh(g) [e(tau_i) - e(mean_j tau_j)],
```

where `e` is ELF's existing time embedder and the new scalar gate `g` is
initialized to zero. Therefore a newly converted checkpoint exactly matches
the old scalar-time checkpoint before fine-tuning.

## Paired pilot

Both arms start from `converted/elf_b-owt-kd-cr_torch.pt`, use the same seed,
data, batch size, learning rate, KD objective, and 500 optimizer steps.

| arm | per-token architecture | heterogeneous examples |
|---|---:|---:|
| synchronous control | yes | 0% |
| native WFF | yes | 50% |

Within heterogeneous examples, order is 50% LTR, 25% RTL, and 25% fixed
random, with `Delta ~ Uniform(0.05, 0.20)`. The remaining 50% retain ordinary
synchronous denoising to protect the base task.

## Evaluation

Use identical initial noise for each checkpoint and sampler arm:

1. ordinary ODE-32;
2. native WFF ODE-32, LTR `Delta=0.10`;
3. native WFF ODE-32, LTR `Delta=0.20`;
4. native WFF ODE-32, RTL `Delta=0.20`.

Use the native ODE initialization (`z0 = 2 epsilon`) and SC-CFG 3. The first
Pipeline audit showed that a method conclusion can reverse between noise scale
1 and 2, so SC-CFG and noise scale must be written into every WFF result.

Primary quality metrics are GPT-2-large Gen.PPL, Distinct-1/2, 4-gram
repetition, and qualitative degeneration. If quality is retained, a second
stage measures `tau_first`, `tau_stable`, revisions, branch entropy, and the
GS16/GS17 endpoint-alignment diagnostics.

## Decision rule

The causal comparison is the interaction

```text
[WFF-trained: WFF sampler - standard sampler]
  - [control-trained: WFF sampler - standard sampler].
```

- **Proceed** only if native WFF training materially removes the WFF sampler's
  quality penalty relative to the matched control, while its standard sampler
  remains healthy. Then expand to three seeds and transition metrics.
- **Stop** if both checkpoints fail similarly under WFF sampling, or if WFF
  training degrades ordinary ODE generation. This would strengthen GS19's
  conclusion that asynchronous clocks are mismatched to the learned dynamics,
  not merely unseen at training time.
- A learned nonzero local-time gate is necessary but not sufficient evidence;
  it only proves that the model used the new input.

## Result (`2026-08-06`)

Both matched arms completed 500 steps from the same converted `kd_cr`
checkpoint. Evaluation used EMA weights, fixed seed 42, `n=256`, noise scale 2,
SC-CFG 3, length 128, and ODE-32.

| training | sampler | PPL | delta vs own standard | D1 | D2 | rep-4 |
|---|---|---:|---:|---:|---:|---:|
| sync control | standard | 1086.7 | 0.0 | 0.408 | 0.944 | 0.000 |
| sync control | LTR, delta 0.10 | 1053.8 | -32.9 | 0.404 | 0.937 | 0.001 |
| sync control | LTR, delta 0.20 | 979.8 | -106.9 | 0.383 | 0.915 | 0.005 |
| sync control | RTL, delta 0.20 | 1002.3 | -84.4 | 0.394 | 0.926 | 0.001 |
| WFF-trained | standard | 1114.0 | 0.0 | 0.425 | 0.949 | 0.000 |
| WFF-trained | LTR, delta 0.10 | 1126.6 | +12.6 | 0.422 | 0.941 | 0.001 |
| WFF-trained | LTR, delta 0.20 | 1031.7 | -82.3 | 0.398 | 0.921 | 0.005 |
| WFF-trained | RTL, delta 0.20 | 1019.2 | -94.8 | 0.404 | 0.931 | 0.001 |

The pre-registered interaction, defined as WFF-trained sampler delta minus
control-trained sampler delta, is:

| sampler | interaction delta PPL | interpretation |
|---|---:|---|
| LTR, delta 0.10 | +45.5 | worse after WFF training |
| LTR, delta 0.20 | +24.6 | worse after WFF training |
| RTL, delta 0.20 | -10.4 | small improvement only in the reverse-direction arm |

The WFF-trained standard sampler is also 27.3 PPL worse than the matched
control. Stronger waves lower PPL in both training arms but reduce D1/D2 and
increase repetition, so they are not clean quality gains.

The local-time gate barely moved:

| training | raw gate | EMA gate |
|---|---:|---:|
| sync control | +1.03e-4 | +3.41e-6 |
| WFF-trained | -9.49e-4 | -1.41e-5 |

This rejects the proposed explanation that the inference-only failure is
removed simply by exposing the model to heterogeneous clocks during a short
matched fine-tune. Do not expand this implementation to more seeds or formal
training. Any future revisit must first remove the near-zero gate bottleneck
and pass a cheap training-by-sampler interaction gate.

## Preflight checks before launching

1. Run `python tests/test_wff.py`. The fresh ELF flow head is intentionally
   zero-initialized, so the gate-gradient unit test explicitly initializes a
   nonzero synthetic output head; otherwise every backbone gradient is
   mechanically zero and the test gives a false alarm.
2. Verify `delta=0` WFF sampling is numerically identical to ordinary ODE.
3. Verify both arms load the same `kd_cr` weights and that the only config
   difference is `wff_train_prob` (`0.0` versus `0.5`).
4. After the first logged optimizer step, require finite loss and a nonzero
   `local_time_gate` gradient in the WFF arm. Stop rather than interpreting a
   dead gate as a negative scientific result.

The first server launches exposed a Transformers compatibility error: current
T5 mask utilities require boolean 2D padding masks, whereas ELF stored an
equivalent float32 `[B,L,L]` pairwise mask even for unconditional data. The
wrapper now converts mask dtype to bool, and the unconditional collator returns
the exactly equivalent 2D key-padding mask. This is an infrastructure failure,
not an EXP-60 result; both matched arms must be relaunched from step zero.
