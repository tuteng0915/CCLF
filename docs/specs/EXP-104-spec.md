# EXP-104 Spec — Distilled Online Trigger Controller

**Status:** IMPLEMENTED / CALIBRATION BANK PENDING
**Purpose:** distill EXP-102's four-step native entropy-response teacher into a
controller that decides from the current, unmodified trajectory without
enumerating counterfactual trigger branches.

## Training target

Use the frozen EXP-102 target:

```text
four-step unresolved entropy reduction,
anchor branch relative to paired native control.
```

Training banks are the already opened seed/offset pairs `(42,0)`,
`(123,1000)`, and `(2026,6000)`, for 192 trajectories and 1536
trajectory/trigger cells. Inputs contain only the EXP-101 current-state event
statistics plus normalized native trigger time:

```text
mean and q10 confidence,
mean entropy,
mean lexical margin,
one-step lexical revision,
one-step predicted-clean instability,
confidence and entropy change,
trigger step / 32.
```

A shared two-layer MLP predicts the standardized teacher response for each
candidate time. The loss combines cellwise MSE with within-trajectory pairwise
ranking. Architecture, optimizer, epoch count, and training seed are fixed
before the calibration bank is opened.

## Causal online policy

The controller may inspect candidate steps `8,10,12,14` in order. It triggers
at the first step whose predicted response exceeds a scalar threshold and is
forced to trigger at step 14 otherwise. It never observes a future trigger
state before making an earlier decision. This distinguishes it from EXP-102's
enumerative teacher.

Seed 2027 / offset 7000 is calibration. It may select only the scalar response
threshold. Seed 2028 / offset 8000 remains unopened until the controller and
threshold are frozen.

## Gates

Open the final bank only if calibration satisfies all of:

- predicted teacher-response pairwise accuracy above `.55`;
- paired final NLL CI upper bound below zero relative to fixed step 14;
- `Delta D1 >= -.005`, `Delta Rep-4 <= .005`,
  `Delta degeneration <= .015`, and `Delta prompt gain >= -.01`;
- a nonzero but nonsaturated early-trigger fraction.

The final bank is evaluated once with no retuning. Report C-PPL, NLL bootstrap,
quality panel, trigger histogram, and mean number of controller checks. A
failed final gate closes current-state distillation; a positive result opens a
training-time controller integration experiment.

## Implementation

- trainer and calibration:
  `experiments/interventions/train_distilled_trigger_controller_exp104.py`;
- frozen final evaluator:
  `experiments/interventions/eval_distilled_trigger_controller_exp104.py`.

