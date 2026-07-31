# EXP-GS18 Spec — Unified True-Rollout Transition Timeline (P0)

**Status**: PLANNED  
**Priority**: P0 — answer when the lexical transition happens  
**Models**: ELF baseline first; ELF KD checkpoints as secondary; LangFlow next  
**Proposed script**: `experiments/global_state/analyze_transition_timeline.py`  
**Output**: `results/global_state/<model>/<checkpoint>/transition_timeline_<label>.{json,npz}`

## 1. Motivation

The paper currently aligns oracle margin crossing, collective peaks, rollout
recovery, branching, and GS15 by comparing broad visual windows from different
experiments. This does not establish that they are the same event.

GS18 computes all feasible transition times on the **same true rollouts** and
tests their ordering at trajectory and token level.

## 2. Dense rollout observables

For each true rollout, save at every solver step:

- raw `Z_s` and self-conditioning `SC_s`;
- predicted-clean representation and native logits;
- top-1 token and true-terminal-token margin;
- residual geometry and velocity outputs from GS17;
- endpoint affinities from GS16.

The reference token is the rollout's own terminal decode. This measures
internal stabilization, not external text accuracy.

## 3. Event definitions

Per token position:

```text
tau_first   = first time top1 equals terminal token
tau_stable  = first time top1 equals terminal token at every later checkpoint
tau_margin  = first persistent crossing of terminal-token margin above 0
tau_conf    = first persistent crossing of a preregistered entropy threshold
```

Per trajectory:

```text
tau_50_stable = time when 50% of positions are stably committed
tau_90_stable = time when 90% are stably committed
tau_affinity  = maximum negative derivative of endpoint-affinity entropy
tau_velocity  = maximum rise of self-endpoint velocity advantage
tau_collective = peak of common-factor-controlled coupling from GS21
tau_branch    = maximum contraction rate of calibrated branch entropy from GS19
```

Use monotone interpolation only between adjacent saved checkpoints. Do not
claim sub-step temporal precision.

## 4. Core analyses

1. Distribution of `tau_first` and `tau_stable`; report
   `tau_stable - tau_first`.
2. Within-trajectory ordering of the six trajectory-level events.
3. Spearman correlation and paired differences between event times.
4. Event-aligned curves centered on `tau_50_stable`.
5. Token-level survival curves, with sequence as a cluster.
6. Distance-to-anchor analysis: whether a token stabilizes earlier when nearby
   positions have already stabilized, controlling for frequency, position,
   punctuation, and current margin.

The last analysis is descriptive unless paired with an intervention; do not
call it causal coordination.

## 5. Scale and statistics

Pilot:

- ELF baseline, `n_traj=16`, 65 checkpoints.

Formal:

- `n_traj>=64` across 3 seeds;
- 65 or 129 checkpoints;
- repeat on LangFlow at matched log-SNR percentiles.

Use:

- hierarchical bootstrap over trajectories;
- cluster-robust intervals for token-level survival models;
- permutation test for event ordering by shuffling event labels within
  trajectory.

## 6. Decision rule

The paper may call the transition "coordinated" only if:

1. affinity collapse, branch contraction, or controlled collective coupling
   forms a reproducible narrow window;
2. that window precedes or overlaps `tau_50_stable`;
3. alignment is significant within the same trajectories, not only visually
   similar across separate experiments.

If event times are broad and weakly correlated, the correct conclusion is
heterogeneous gradual stabilization, not a collective transition.

## 7. Required figure

One main-paper figure with three aligned panels:

1. token survival curves (`first` versus `stable`);
2. endpoint-affinity entropy and branch entropy;
3. velocity specificity and controlled collective coupling.

Show median event times and bootstrap intervals on the same normalized
log-SNR axis.

