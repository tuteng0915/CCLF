# EXP-GS21 Spec — Common-Factor-Controlled Collective Dynamics (P1)

**Status**: PLANNED  
**Priority**: P1 — determine whether GS5 reflects coordination or a shared factor  
**Models**: ELF baseline first; reuse GS18 true rollouts  
**Proposed script**: `experiments/global_state/analyze_connected_coupling.py`  
**Output**: `results/global_state/<model>/<checkpoint>/connected_coupling_<label>.{json,npz}`

## 1. Confound

GS5's position-shuffle null preserves neither token composition nor all
sequence-level factors. Correlated margin increments may be caused by document
difficulty, global confidence, logit scale, or a shared semantic factor rather
than position-to-position coordination.

## 2. Data lines

Run the same analysis on:

1. oracle states, for direct comparison with GS5;
2. true rollout states from GS18, which are primary for mechanism claims.

Use terminal-token margins for true rollouts and ground-truth margins for
oracle states. Never merge these two interpretations.

## 3. Residualization

For every adjacent checkpoint pair, construct margin increment `dm_{n,i}` and
residualize in stages:

```text
M0: raw dm
M1: dm - mean_i(dm) within each sequence
M2: M1 residualized by token frequency, POS, absolute position, and current margin
M3: M2 residualized by sequence-level logit norm and mean entropy
```

Primary connected correlation:

```text
C_conn(d,t) = Corr(M3_{n,i,t}, M3_{n,i+d,t})
xi_conn(t)  = sum_d max(C_conn(d,t), 0)
```

## 4. Nulls

1. position shuffle within sequence, stratified by POS and frequency bin;
2. sequence shuffle at fixed position and token-frequency bin;
3. circular position shift, preserving local autocorrelation but breaking
   semantic neighborhood alignment;
4. sign-flip null within sequence;
5. marginal-variance-matched Gaussian surrogate.

The same random permutations must be applied across nearby checkpoints when
testing increment dynamics.

## 5. Additional tests

- correlation as a function of token distance `d`, with bootstrap bands;
- left/right asymmetry;
- conditioned analysis around already stable versus unstable neighbors;
- peak width and peak time on normalized log-SNR;
- optional attention-edge stratification: high-attention pairs versus
  distance-matched low-attention pairs.

Attention stratification is correlational unless attention edges are ablated.

## 6. Scale and statistics

- `n_sequences>=128`;
- at least 33 checkpoints;
- 1000 null permutations;
- bootstrap sequences, not position pairs;
- report family-wise bands across time and distance.

## 7. Decision rule

Use "collective coordination" only if:

1. `xi_conn` survives M3 residualization;
2. it exceeds every matched null;
3. it forms a reproducible transition-localized peak on true rollouts;
4. the peak aligns with GS18 stable commitment or GS16 affinity collapse.

If only M0 is positive, the correct interpretation is a shared sequence-level
fluctuation. If oracle survives but rollout does not, the effect concerns
controlled recoverability, not sampler coordination.

