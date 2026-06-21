"""
Idea C: cliff importance sampling — analysis & visualization
Tests the proposed t-sampling distribution p(t) ∝ dG/dt + ε WITHOUT any training.

Run:
    python analyze_idea_c.py

Outputs:
    - Prints stats showing where sampling mass concentrates
    - Saves idea_c_distribution.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── 1. Load probe data ──────────────────────────────────────────────────────
PROBE_PATH = Path(__file__).parents[3] / "results/elf/probe_geo_v1/probe_geo.json"
d = json.load(open(PROBE_PATH))

t_probe = np.array(d["t"])          # shape (21,), [0.0, 0.05, ..., 1.0]
G = np.array(d["cos_nn_correct_mean"])  # G(t): geometric commitment

# ── 2. Compute dG/dt via central differences ────────────────────────────────
# Central diff for interior, forward/backward at endpoints
dGdt = np.gradient(G, t_probe)      # numpy central diff at uniform spacing

print("=== G(t) and dG/dt ===")
print(f"{'t':>5}  {'G(t)':>7}  {'dG/dt':>7}")
for ti, gi, di in zip(t_probe, G, dGdt):
    print(f"{ti:5.2f}  {gi:7.4f}  {di:7.4f}")

# ── 3. Build cliff importance sampling distribution ─────────────────────────
# p(t) ∝ max(dG/dt, 0) + ε  (only increasing parts; ε ensures minimum coverage)
# ε = 0.1 * mean(positive dGdt) — sample at least 1/10 of cliff density anywhere
eps_fraction = 0.1
pos_dGdt = np.maximum(dGdt, 0.0)
epsilon = eps_fraction * pos_dGdt.mean()

unnorm = pos_dGdt + epsilon

# Normalize to valid PMF over the 21 discrete t values
# (In practice training uses continuous sampling; we show the density)
dt = t_probe[1] - t_probe[0]  # 0.05
pmf = unnorm / (unnorm.sum() * dt)  # probability density
pmf_prob = (unnorm * dt) / (unnorm * dt).sum()  # discrete PMF

uniform_prob = np.ones(len(t_probe)) / len(t_probe)

# ── 4. Stats ────────────────────────────────────────────────────────────────
print("\n=== Cliff importance sampling PMF ===")
print(f"{'t':>5}  {'p(t)/uniform':>14}  {'G(t)':>7}  {'dG/dt':>7}")
for ti, pi, ui, gi, di in zip(t_probe, pmf_prob, uniform_prob, G, dGdt):
    ratio = pi / ui
    bar = "█" * int(ratio * 10)
    print(f"{ti:5.2f}  {ratio:14.2f}x  {gi:7.4f}  {di:7.4f}  {bar}")

# Cliff region [0.10, 0.35]
cliff_mask = (t_probe >= 0.10) & (t_probe <= 0.35)
cliff_mass_cliff = pmf_prob[cliff_mask].sum()
cliff_mass_uniform = uniform_prob[cliff_mask].sum()
print(f"\nCliff region [0.10, 0.35]:")
print(f"  Cliff sampling mass:   {cliff_mass_cliff:.3f}  ({cliff_mass_cliff/cliff_mass_uniform:.1f}x over uniform)")
print(f"  Uniform sampling mass: {cliff_mass_uniform:.3f}")

# ── 5. Implementation hint: training t-sampling ──────────────────────────────
print("\n=== Implementation for train_step.py ===")
print("Current:  t = jax.random.uniform(rng, (B,))  # uniform [0, 1]")
print("""
Cliff:    # precompute at startup and store as config field
          # t_probe, cliff_pmf = build_cliff_pmf(probe_geo_path, epsilon=0.1)
          # Then sample:
          t_idx = jax.random.choice(rng, len(t_probe), (B,), p=cliff_pmf)
          t = t_probe[t_idx]
          # Or continuous via inverse-CDF / piecewise-linear interpolation
""")

# ── 6. Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: G(t) curve
ax = axes[0]
ax.plot(t_probe, G, "b-o", markersize=4, label="G(t)")
ax.axvspan(0.10, 0.35, alpha=0.15, color="red", label="cliff [0.10, 0.35]")
ax.set_xlabel("t (noise level)")
ax.set_ylabel("G(t)")
ax.set_title("Geometric Commitment G(t)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: dG/dt
ax = axes[1]
ax.plot(t_probe, dGdt, "g-o", markersize=4)
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.axvspan(0.10, 0.35, alpha=0.15, color="red", label="cliff")
ax.set_xlabel("t")
ax.set_ylabel("dG/dt")
ax.set_title("G(t) Gradient (Cliff signal)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Sampling distribution comparison
ax = axes[2]
width = 0.02
ax.bar(t_probe - width/2, uniform_prob, width=width, alpha=0.6, label="Uniform", color="blue")
ax.bar(t_probe + width/2, pmf_prob, width=width, alpha=0.7, label=f"Cliff (ε={eps_fraction})", color="red")
ax.axvspan(0.10, 0.35, alpha=0.10, color="orange", label="cliff region")
ax.set_xlabel("t")
ax.set_ylabel("P(t)")
ax.set_title("Sampling Distribution: Uniform vs Cliff")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = Path(__file__).parent / "idea_c_distribution.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nSaved plot to {out_path}")
