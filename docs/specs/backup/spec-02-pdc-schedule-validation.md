# Spec 02 — PDC μ(t) Schedule Validation

**Type**: data analysis + code  
**Priority**: medium (informs method section)  
**Output**: `results/elf/pdc_schedule/`

---

## Background

Progressive Decode Correction (PDC) proposes a training loss:

$$\mathcal{L}_\text{PDC}(t) = \|\hat{x}_t^{den} - \text{sg}(h_t^{dec})\|^2 \cdot \mu(t)$$

The schedule μ(t) should weight the loss according to the decode-branch advantage G_dec(t).
`experiments/analysis/analyze_snr_gdec.py` already produces a μ(t) fit
(saved in `results/elf/snr_analysis/`), but it uses **hardcoded** G_dec values.

The goal here is to:
1. Load G_dec values directly from the probe JSON instead of hardcoding them.
2. Produce a cleaner analysis notebook / script that any future run can update.
3. Validate the μ(t) formula against the fate-tracking data from probe v4.

---

## Task

### 1. Extend `analyze_snr_gdec.py` to load from JSON

Modify `experiments/analysis/analyze_snr_gdec.py` to accept:

```
--decode_json   path to results/elf/probe_decode_v1/probe_decode_branch.json
--v4_json       path to results/elf/probe_v4/anchor_probe_v4.json
--out_dir       output directory (default: results/elf/pdc_schedule/)
```

Remove the hardcoded G_dec data at the top of the file and instead read:

```python
with open(args.decode_json) as f:
    dec = json.load(f)
# G_dec[t] = dec_top1[i] - lin_top1[i] for each t step, excluding t=1.0
t_vals = [r["t"] for r in dec["results"] if r["t"] < 1.0]
gdec   = [r["dec_top1"] - r["lin_top1"] for r in dec["results"] if r["t"] < 1.0]
```

*(Check the actual JSON schema in `results/elf/probe_decode_v1/probe_decode_branch.json`
before writing this — the field names may differ.)*

### 2. Add fate-tracking overlay

From probe v4 JSON (`results/elf/probe_v4/anchor_probe_v4.json`), extract
`fate_corrected_by_decode` (the fraction of wrong-committed positions corrected by the
decode branch as a function of source_t). Overlay this on the μ(t) plot as a secondary
validation curve.

Expected pattern:
- fate decode correction peaks around t=0.40–0.50 (72% at t=0.50)
- μ(t) should track this: highest weight where decode correction is most useful

### 3. Output

`results/elf/pdc_schedule/`
- `mu_t_schedule.json` — keys: `t`, `gdec_raw`, `gdec_clamped`, `mu_t`, `mu_formula`
- `pdc_schedule_plot.png` — three panels: G_dec(t), μ(t), fate overlay
- Print the closed-form μ(t) approximation (existing fit: `0.9497·(t−0.25)^0.031·(0.95−t)^0.137`)

---

## Success criteria

- Script runs end-to-end without hardcoded values: `python analyze_snr_gdec.py --decode_json ... --v4_json ... --out_dir ...`
- μ(t) plot visually confirms that the schedule peaks in the t=0.30–0.50 window
- fate overlay is consistent with μ(t) (high μ ↔ high decode correction fraction)

---

## Notes

- The existing `results/elf/snr_analysis/analysis.json` has the current hardcoded output;
  use it as a reference for the expected schema.
- t=1.0 must be excluded from G_dec: at t=1.0 the two-pass backbone produces an artifact
  (G_dec = −0.080, negative, because running backbone twice at t=1 is not what ELF does).
- μ(t) should be zero for t < 0.25 (before commitment cliff) and zero at t > 0.95
  (to avoid the two-pass artifact).
