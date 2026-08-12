# EXP-92 Spec — On-Policy Conditional Triggered Subset Flow

**Status:** STAGE 0 DONE / ON-POLICY NEGATIVE; LOSS-BALANCED FOLLOW-UP ACTIVE
**Purpose:** retain the portable conditional benefit of temporary random
anchors while correcting the state-distribution and objective mismatches in
the negative EXP-91 training pilot.

## Why EXP-91 is not the final test

EXP-91 trained on oracle-corrupted unconditional states. At inference, however,
the useful intervention is applied to a model-generated trajectory, after a
real prompt has been clamped, and the selected predicted-clean states persist
for several solver steps before release. It also alternated synchronous and
subset examples, so the asynchronous objective could move ordinary generation
without an explicit preservation constraint.

The negative result therefore rejects that particular surrogate objective,
not Triggered Subset Flow Forcing as a family.

## Central hypothesis

At the model-native transition, a broad, position-correct subset can serve as
temporary lexical context for the unresolved suffix. Training on the same
heterogeneous states should let the student exploit this context without
requiring anchors at inference.

For an observed prefix `P`, selected suffix subset `A`, and unresolved suffix
`U`, construct a teacher trajectory up to trigger `t_*`, hold the teacher's
predicted-clean states on `A` for `H` solver intervals, and train the student at
a state sampled from that held trajectory:

```text
z_mix[i] = prompt[i]              if i in P
           x_hat_teacher[i]       if i in A
           z_rollout[i]           if i in U.
```

The paired objective is

```text
L = L_sync(z_oracle) + lambda_mix L_U(z_mix)
    + lambda_anchor L_A^preserve(z_mix),
```

where every update contains the ordinary synchronous loss. `L_A^preserve` is
initially diagnostic or low-weight: anchors should remain useful but revisable,
not become irreversible discrete commitments.

## Stage 0: isolate the missing ingredient

Train matched 200-step arms with identical OWT documents, optimizer, trainable
parameters, and number of student forward/backward calls:

1. **paired control:** two synchronous objectives per update;
2. **conditional oracle subset:** real prefixes, but the mixed state is built
   from the oracle-corrupted trajectory as in EXP-91;
3. **conditional on-policy subset:** real prefixes and a frozen-teacher rollout
   to the trigger, followed by a random subset held for `H` steps.

This factorial asks whether the EXP-91 failure came from missing conditional
context or from off-policy state construction. Do not add an architectural
anchor flag at this stage; that would confound state-source correction with a
new input channel.

Default screen:

- ELF base, length 128, prefix 64;
- 50% conditional and 50% unconditional documents in every arm;
- native 32-step ODE, trigger near `t=.30`;
- random suffix density sampled from `{.25, .50, .75}`;
- hold horizon sampled from `{1, 2, 4}`;
- `lambda_mix=1`; paired synchronous preservation on every update;
- checkpoints at 100 and 200 steps; expand to 500 only after a generation
  gate passes.

## Selector audit: why random can beat confidence

Do not infer that uncertainty itself is useful from the current ranking. At the
same trigger and density, compare random and top-confidence subsets on:

- selected-token reliability: confidence and final identity agreement;
- spatial coverage: gap distribution and sequence span;
- lexical coverage: content/function-token mix and token-frequency quantile;
- redundancy: mean pairwise similarity among selected states;
- causal utility: change in unresolved-token NLL/margin after anchoring;
- downstream influence: how many unresolved positions change their prediction.

Test whether per-token selection utility is better described by

```text
utility(i) ~= reliability(i) * influence(i) * nonredundancy(i)
```

than by reliability alone. This is a mechanistic audit, not a selector-tuning
loop; thresholds and regressions must be frozen before final quality panels.

## Evaluation and promotion gate

Evaluate checkpoints with three paired inference seeds in both U/C scopes,
using Standard-32 and the frozen random-anchor policy. Report PPL, shuffled-
prompt PPL, prompt gain, ROUGE-L, D1/D2, Rep-4, unigram collapse/degeneration,
revision, and model-call counts.

The primary statistic is a difference in differences relative to the matched
control:

```text
DeltaDelta_C = (C-PPL_student - C-PPL_base)_subset
               - (C-PPL_student - C-PPL_base)_control.
```

Promote only if C-PPL interaction is favorable in at least `2/3` seeds and its
mean is favorable, while Standard generation, prompt gain, Rep-4, and
degeneration do not regress materially. A lower training loss is insufficient.

If conditional oracle succeeds but on-policy does not, debug rollout-state
construction. If on-policy succeeds and oracle does not, treat exposure to the
actual trajectory as essential. If neither succeeds, stop before adding longer
training or an explicit anchor indicator.

## Stage-0 result (2026-08-12)

All three 200-step arms completed. Prompt clamp error was exactly zero and the
requested random density was exact in every logged batch. Three paired
generation seeds (`42/123/456`, `n_U=n_C=32`) give:

| Training arm | Standard U-PPL | Random U-PPL | Standard C-PPL | Random C-PPL | Standard gain | Random gain | Random C-RL | Random C-D1 | Random C-Rep4 | Random C-Deg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| paired control | 287.36 | **201.02** | **765.59** | **524.99** | .1598 | **.2133** | .0740 | **.6377** | **.0158** | .0208 |
| conditional oracle | 287.63 | 206.45 | 787.24 | 532.94 | .1651 | .2004 | .0753 | .6340 | .0164 | .0208 |
| conditional on-policy | **286.99** | 210.11 | 779.13 | 550.11 | **.1668** | .2015 | **.0750** | .6398 | .0165 | .0208 |

Relative to the paired control, random-minus-Standard interactions are:

| Training arm | `DeltaDelta` U-PPL | `DeltaDelta` C-PPL | `DeltaDelta` gain | `DeltaDelta` C-RL | `DeltaDelta` C-D1 | `DeltaDelta` C-Rep4 | `DeltaDelta` C-Deg. |
|---|---:|---:|---:|---:|---:|---:|---:|
| conditional oracle | +5.16 | -13.70 | -.0182 | +.0004 | -.0065 | +.0003 | +.0104 |
| conditional on-policy | +9.46 | +11.58 | -.0189 | +.0020 | +.0061 | -.0001 | +.0104 |

Negative PPL is favorable. Conditional oracle has a favorable C-PPL
interaction in `2/3` seeds (`-68.6/-26.1/+53.6`) but worsens absolute random
C-PPL, prompt gain, D1, and degeneration. On-policy is unfavorable on mean C-
PPL and succeeds in only `1/3` seeds. Both fail the preregistered gate.

The on-policy transition loss was about three times the synchronous loss at
initialization (`2.38` versus `.77`), so `lambda_mix=1` did not actually balance
the preservation and transition gradients. A single preregistered follow-up
uses `lambda_mix=.25` with all other data, states, and seeds frozen. If that
still fails, stop the current straight-to-endpoint target: GS15 already warns
that real residual motion is not a linear path to the endpoint.
