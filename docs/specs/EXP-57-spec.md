# EXP-57 Spec: Stacked h10 SC + Progressive Commitment

**Date:** 2026-07-31
**Status:** DONE
**Script:** `models/ELF-torch/experiments/probe_elf/stacked_h10_prog_exp57.py`

---

## Motivation

EXP-54b established h10 SC gives kd2 I=−130 (best inference result at the time).
EXP-56 established progressive commitment gives kd_cr I=−164.
EXP-56b extended this: kd2 + prog_t30_c70 I=−186.

Can we stack h10 SC (which improves x_pred quality by bypassing B11's anti-correlation)
with progressive commitment (which locks high-confidence positions mid-ODE)?

The intuition: h10 SC produces a better x_pred → higher-quality confidence estimates → 
better positions get committed → stronger anchors for remaining ODE steps.

---

## Setup

- COMMIT_T = 0.5, CONF_THRESH = 0.70 (same as EXP-56)
- SC_T_MIN = 0.5 (same as EXP-54c gate)
- N=256, ODE-32, SCCFG=1

Arms:
1. **standard**: no SC, no commitment
2. **h10_only**: h10 SC from t≥0.5 (same as EXP-54)
3. **prog_only**: progressive commitment at t=0.5 (same as EXP-56)
4. **h10_prog**: h10 SC AND progressive commitment — h10 SC active from t≥0.5, commitment triggered at t=0.5 based on h10-improved x_pred

---

## Results

### kd_cr

| Arm | PPL | I | Commit% |
|-----|-----|---|---------|
| standard | 332.1 | 0 | — |
| h10_only | 207.9 | -124.1 | — |
| prog_only | 168.0 | **-164.0** | 70% |
| h10_prog | 242.3 | -89.8 | 34% |

### kd2

| Arm | PPL | I | Commit% |
|-----|-----|---|---------|
| standard | 284.7 | 0 | — |
| h10_only | 155.4 | **-129.3** | — |
| prog_only | 213.0 | -71.7 | 61% |
| h10_prog | 190.0 | -94.7 | 44% |

---

## Key Findings

### 1. h10_prog is WORSE than both individual methods for kd_cr

kd_cr ranking: prog_only (I=−164) >> h10_only (I=−124) >> h10_prog (I=−90)

h10_prog is antisynergistic for kd_cr. Despite h10 SC improving x_pred quality (I=−124 for h10_only),
combining with commitment at t=0.5 HURTS relative to prog_only alone.

### 2. h10_prog is intermediate but suboptimal for kd2

kd2 ranking: h10_only (I=−129) > h10_prog (I=−95) > prog_only (I=−72)

h10_prog slightly outperforms prog_only for kd2 (−95 vs −72), but is much worse than h10_only.

### 3. Why stacking fails: commitment fraction collapse

| Arm | kd_cr commit% | kd2 commit% |
|-----|:-------------:|:-----------:|
| prog_only | 70% | 61% |
| h10_prog | **34%** | **44%** |

With h10_prog, far fewer positions get committed! h10 SC replaces x_pred with final_layer(h10),
which produces a different probability distribution. For many positions where standard x_pred was
confident (>70%), h10's x_pred is less confident (below 70% threshold). Result: fewer anchors,
weaker second-half ODE steering.

### 4. h10 SC destabilizes the confidence signal

The h10 bypass (h10 → final_layer → x̂_t, skipping B11) gives different confidence estimates
than the standard ODE x_pred. For kd_cr:
- Standard x_pred at t=0.5: backbone well-organized (EXP-42 CKA), high confidence (70% commit)
- h10 x_pred at t=0.5: bypasses B11 reorganization → different, possibly less reliable confidence
- Net: fewer positions committed, and at worse quality → h10_prog I=−90 < prog_only I=−164

For kd2:
- h10 SC is specifically designed to fix kd2's B11 anti-correlation (EXP-43/44)
- But combined with commitment, kd2 commit drops from 61% → 44%
- The h10 x_pred is better quality but fewer positions achieve >70% confidence
- Net: h10_prog is slightly better than prog_only (−95 vs −72) because quality > quantity

### 5. Critical context: EXP-56b showed prog_t30 is far better for kd2

EXP-57 uses prog_only at t=0.5 for kd2 (I=−72). EXP-56b showed:
- prog_t30_c70 for kd2: I=−186 (commit=46%) — vastly better!
- The fix is NOT to add h10 SC, but to commit EARLIER

The correct hierarchy for kd2:
- prog_t30_c70 (I=−186) >> h10_only (I=−130) >> h10_prog_t50 (I=−95) >> prog_t50_c70 (I=−72)

---

## Conclusion

Stacking h10 SC + progressive commitment does not work. The two methods are antisynergistic:
- h10 SC changes the confidence landscape (reduces commit fraction)
- Progressive commitment relies on high commit fraction for effective anchoring

**Best inference methods (after EXP-55–57):**

| Model | Best method | I | PPL |
|-------|-------------|---|-----|
| kd_cr | prog_t40_c70 (EXP-56b) | **−175.1** | 157 |
| kd2 | prog_t30_c70 (EXP-56b) | **−186.4** | 98 |

Both models: within-ODE progressive commitment at t=0.3–0.4 is the clear winner.
h10 SC, though effective for kd2 in isolation, is superseded by early progressive commitment.

---

## Follow-up

- **EXP-56c**: Sweep earlier commit times (t ∈ {0.1, 0.15, 0.20, 0.25}) for kd2 to find true optimum
- **EXP-56d**: Threshold sweep (0.5, 0.6) at t=0.3 for kd2 — can we commit more positions without hurting quality?
- **EXP-58**: Cascading commitment — commit at t=0.2, then again at t=0.5 for remaining positions
