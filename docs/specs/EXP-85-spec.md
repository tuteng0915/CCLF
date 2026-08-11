# EXP-85 Spec — Anchor-Mediated Endpoint Collapse

**Status:** READY / P0
**Purpose:** test whether anchors improve generation by resolving endpoint
uncertainty rather than by generic ODE regularization.

At pre-transition, transition, and post-transition checkpoints, anchor exact
density-matched `.10/.25/.50` subsets using correct high-confidence content,
within-sequence shuffled content, a reachable alternative endpoint, random
tokens, and a mask/readout sham. Release after four solver intervals.

On unanchored positions report immediate and final changes in endpoint entropy,
self rank, first/stable time, alternative-endpoint capture, and the complete
generation-quality panel.

- **Coordination mediation:** correct anchors lower endpoint entropy and
  stabilize unanchored positions; shuffled/random controls do not; alternative
  anchors redirect the joint endpoint, with the largest effect near collapse.
- **Solver regularization:** PPL changes without endpoint entropy/rank changes.
- **Wrong lock-in:** entropy falls only by increasing wrong-endpoint capture or
  degeneration.

Implement by extending the paired endpoint-bank logic in EXP67/74 rather than
creating another independent sampler.
