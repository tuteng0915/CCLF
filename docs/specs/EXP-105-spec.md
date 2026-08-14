# EXP-105 Spec — Causal Online Response Trigger

**Status:** CALIBRATION PASSED / FINAL BANK UNOPENED
**Purpose:** turn EXP-102's transferable local entropy response into a causal
trigger rule without enumerating all future trigger times.

## Intervention and decision rule

At candidate native steps `8,10,12`, fork only the *current* Plaid state into:

1. the one-step 75% top-confidence anchor used by EXP-95/101; and
2. a paired unmodified control with common ancestral noise.

Advance both forks for four native updates and measure unresolved-token entropy
reduction of anchor relative to control. Rewind to the control state. Trigger
at the first candidate whose measured response exceeds one scalar threshold;
if none fires, use step 14:

```text
r_k = H_unresolved(control after 4 steps)
      - H_unresolved(anchor after 4 steps)

k_selected = min {k in {8,10,12}: r_k >= gamma},
             or 14 when the set is empty.
```

The rule uses a bounded counterfactual lookahead, but it is causal: at step
`k` it never inspects states or candidate decisions from a later trigger time.
Its cost is up to three paired four-step probes. This experiment tests signal
validity, not yet an efficient sampler claim.

## Frozen calibration and final protocol

- seed 42 selected the response family in EXP-102;
- seed 2027 / offset 7000 calibrates only `gamma`;
- choose the passing response threshold with the lowest paired final NLL;
- seed 2028 / offset 8000 is generated only after this protocol and threshold
  are committed.

A calibration/final result passes only if:

```text
paired NLL CI upper < 0,
Delta D1 >= -.005,
Delta Rep-4 <= .005,
Delta degeneration <= .015,
Delta prompt gain >= -.01,
0 < early-trigger fraction < 1.
```

Calibration freezes `gamma=0.9086204767`. It selects steps `8/10/12/14` on
`1/1/2/60` trajectories, changes C-PPL `84.63 -> 83.63`, and yields paired NLL
`-.0124 [-.0266,-.0022]`. Quality deltas are D1 `-.0013`, D2 `+.0007`, Rep-4
`-.0011`, degeneration `0`, and prompt gain `-.0054`; all gates pass. No
parameter may change after opening seed 2028.

Implementation:
`experiments/interventions/calibrate_online_response_trigger_exp105.py`.
