# EXP-92 Spec — On-Policy Conditional Triggered Subset Flow

**Status:** ACTIVE / STAGE-0 FACTORIZATION
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
