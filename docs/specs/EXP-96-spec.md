# EXP-96 Spec — Event-Triggered Revisable Coupling

**Status:** CLOSED BY EXP-94 / NO FIXED-SCHEDULE HEADROOM
**Purpose:** replace a globally fixed coupling step with a per-sample maturity
event while preserving the same average compute.

For the leading block `A`, define continuous and lexical instability

```text
r_t = 1 - mean_i cos(xhat_0,t[i], xhat_0,t-delta[i]),
q_t = mean_i 1[y_t[i] != y_t-delta[i]].
```

Couple the suffix only after a statistic remains below a frozen threshold for
two observations. Compare fixed schedules, the continuous trigger, lexical-
revision trigger, confidence trigger, and an offline oracle best schedule used
only as an upper bound. Calibrate thresholds on a disjoint panel, then freeze
them. Match adaptive arms to the best fixed arm in mean token-calls; report
the distribution and tail of per-sample compute, not just its mean.

Promote only if an adaptive trigger improves paired C-PPL and boundary PPL in
`2/3` seeds at matched mean work without worse tail degeneration, D1, Rep-4,
or prompt gain. If the oracle upper bound is negligible, stop the adaptive
branch rather than tuning thresholds.
