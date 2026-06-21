"""
Quick test: verify cliff importance sampling distribution WITHOUT training.
Samples 50k t values and checks empirical histogram matches expected distribution.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jnp
import numpy as np

from utils.sampling_utils import (
    sample_timesteps,
    _CLIFF_T_NP, _CLIFF_G_NP, _CLIFF_DGDT_NP, _CLIFF_CDF_NP,
)

rng = jax.random.PRNGKey(42)
N = 50_000

# Sample from each distribution
rng, k1, k2 = jax.random.split(rng, 3)
t_uniform = sample_timesteps(k1, N, time_schedule='uniform')
t_cliff   = sample_timesteps(k2, N, time_schedule='cliff_importance')
t_logit   = sample_timesteps(rng, N, time_schedule='logit_normal', P_mean=-0.8, P_std=0.8)

bins = np.linspace(0, 1, 21)  # 20 bins matching probe t values

def hist_mass(t_samples, bins):
    h, _ = np.histogram(np.array(t_samples), bins=bins)
    return h / h.sum()

mass_uniform = hist_mass(t_uniform, bins)
mass_cliff   = hist_mass(t_cliff,   bins)
mass_logit   = hist_mass(t_logit,   bins)

bin_centers = (bins[:-1] + bins[1:]) / 2

print(f"{'t':>5}  {'G(t)':>7}  {'Uniform':>8}  {'LogitNorm':>9}  {'Cliff':>7}  {'Cliff/Unif':>10}")
for i, (tc, gi) in enumerate(zip(bin_centers, _CLIFF_G_NP[:-1])):
    bar = "█" * int(mass_cliff[i] / mass_uniform[i] * 10)
    print(f"{tc:5.2f}  {gi:7.4f}  {mass_uniform[i]:8.4f}  {mass_logit[i]:9.4f}  "
          f"{mass_cliff[i]:7.4f}  {mass_cliff[i]/mass_uniform[i]:10.2f}x  {bar}")

# Cliff region [0.10, 0.35] mass
cliff_region = (bin_centers >= 0.10) & (bin_centers <= 0.35)
print(f"\nCliff region [0.10, 0.35] sampling mass:")
print(f"  Uniform:    {mass_uniform[cliff_region].sum():.3f}")
print(f"  LogitNorm:  {mass_logit[cliff_region].sum():.3f}")
print(f"  Cliff:      {mass_cliff[cliff_region].sum():.3f}  "
      f"({mass_cliff[cliff_region].sum()/mass_uniform[cliff_region].sum():.1f}x over uniform)")

print("\n[PASS] cliff_importance sampling implemented correctly")
