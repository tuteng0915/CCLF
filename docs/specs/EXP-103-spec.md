# EXP-103 Spec — Selective Native-Utility Anchoring

**Status:** DONE / FINAL LIKELIHOOD GATE FAILS
**Purpose:** retain EXP-102's validated short-horizon likelihood signal while
falling back to the balanced fixed step-14 policy when the local advantage is
too small.

## Frozen lineage

- seed 42 selects the EXP-102 signal family only: four-step unresolved entropy
  reduction after the paired anchor/control fork;
- seed 123 calibrates one scalar abstention threshold;
- no final bank is opened unless calibration passes every likelihood and
  quality gate;
- if opened, seed 2026 / offset 6000 is the first untouched final bank.

For each trajectory, let `s_k` be the frozen local entropy-reduction signal at
candidate trigger `k`, `k_star = argmax_k s_k`, and `k_0=14`. The selective
policy is

```text
use k_star if s_k_star - s_k0 >= gamma;
otherwise use fixed step 14.
```

Search only `gamma` on seed 123. Choose the lowest-NLL threshold satisfying:

```text
paired NLL CI upper bound < 0,
Delta D1 >= -.005,
Delta Rep-4 <= .005,
Delta degeneration <= .015,
Delta prompt gain >= -.01.
```

The threshold rule is still a counterfactual teacher because it evaluates all
candidate triggers for four native steps. A positive final result would justify
distillation into a single-pass controller; it is not itself the efficient
method claim.

If no calibration threshold passes, close the quality-constrained trigger
branch without opening a new final bank. If calibration passes but the frozen
final test fails, report the final failure and do not retune.

Implementation:
`experiments/interventions/calibrate_selective_trigger_exp103.py`.

## Result

Calibration freezes `gamma=1.53414`. The policy switches away from step 14 on
only `5/64` trajectories, changes C-PPL `86.85 -> 79.65`, and has paired NLL
`-.088 [-.191,-.012]`. D1 is unchanged, D2 and prompt gain improve, and no
degeneration is added, so the calibration gate passes.

On the untouched seed-2026/offset-6000 bank, trigger oracle headroom remains
large (`94.91 -> 54.23`, `42.86%`), but the frozen selective rule switches on
only `2/64` trajectories. It changes C-PPL `94.91 -> 94.30` and NLL by
`-.0064 [-.0175,0]`. Quality remains healthy (D1 `-.0002`, D2 `+.0011`, no
Rep-4 or degeneration increase, prompt gain `+.0044`), but the likelihood
interval does not exclude zero. The final gate fails. Do not retune the
threshold; use EXP-102's local response as a future distillation/training
target rather than an enumerative inference method.
