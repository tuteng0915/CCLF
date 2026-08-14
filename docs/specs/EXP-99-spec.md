# EXP-99 Spec — Plaid Temporary-Anchor Subset Headroom

**Status:** IMPLEMENTED / HEADROOM GATE PENDING  
**Purpose:** determine whether Plaid's successful early one-step temporary
anchoring still contains a learnable subset-selection gap before training a
non-additive selector.

## Question

EXP-95 established a balanced Plaid operating point at native step 14,
density `.75`, and horizon `1`. EXP-93 found a large best-of-16 random-mask
gap on ELF, but that result cannot be transferred to Plaid. For trigger state
`s` and fixed-density subset `A`, define

```text
U(A; s) = NLL_standard(s) - NLL_anchor(A; s).
```

The diagnostic headroom is

```text
H_M(s) = max_{m <= M} U(A_m; s) - mean_{m <= M} U(A_m; s).
```

The best-of-M subset uses final conditional NLL and is therefore an oracle
upper bound, not an inference method.

## Protocol

- Plaid native ancestral sampler, 32 steps, 64-token prefix and continuation;
- trigger step `14`, horizon `1`, densities `.50` and `.75`;
- `n=64` conditional trajectories and `M=16` masks for the first gate;
- fixed prompt, initial latent noise, and every ancestral noise draw across
  masks within each trajectory;
- a separate `mask_seed` changes only subset identity;
- report Standard, top confidence, mean random, oracle best/worst, mask utility
  IQR, random-beats-confidence probability, revision, PPL, prompt gain,
  ROUGE-L, D1/D2, Rep-4, and degeneration;
- compute corpus-level D1/D2 separately for each `n=64` random-mask panel and
  then average across masks; never flatten `M*n` texts against an `n`-text arm;
- bootstrap trajectories rather than the correlated masks.

The two densities are isolated runs. Density `.75` is primary because it is
the EXP-95 formal operating point; density `.50` tests whether more unresolved
positions create greater selector headroom.

## Gate

Proceed to a learned selector only if, on an independent Plaid bank,

```text
(PPL_mean-random - PPL_oracle-best) / PPL_mean-random >= 5%.
```

The gap must not be explained solely by worse D1, Rep-4, or degeneration. If
the gate fails at both densities, close Plaid subset learning and treat the
high-coverage random/top-confidence policy as practically saturated; the next
adaptive variable becomes trigger timing.

If the gate passes, EXP-100 will build trajectory-disjoint train/validation/
test utility banks and train a non-additive set scorer. No selector feature,
architecture, or hyperparameter may be chosen on the final test bank.

## Implementation

- runner: `experiments/interventions/eval_plaid_subset_headroom_exp99.py`;
- shared anchor sampler: `eval_temporary_anchor_portability_exp90.py` now
  accepts an independent `mask_seed` without changing native solver noise;
- output: `results/exp99_plaid_subset_headroom/` on the experiment server.
