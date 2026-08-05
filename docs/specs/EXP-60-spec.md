# EXP-60 Spec — Native Wavefront Flow Forcing Training Pilot

**Status**: IMPLEMENTED; runtime validation and paired training pending  
**Priority**: P2 method pilot  
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

