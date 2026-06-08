# TODO

Tasks are broken down into specs in `docs/specs/`. Each spec is self-contained
and can be executed independently.

## Active specs

| # | Spec | Type | Priority | Output |
|---|---|---|---|---|
| 01 | [spec-01-centroid-probe.md](specs/spec-01-centroid-probe.md) | code + experiment | **high** | `results/elf/probe_v3_centroid/` |
| 02 | [spec-02-pdc-schedule-validation.md](specs/spec-02-pdc-schedule-validation.md) | data analysis | medium | `results/elf/pdc_schedule/` |
| 03 | [spec-03-method-section-draft.md](specs/spec-03-method-section-draft.md) | writing | medium | `docs/method_draft.md` |

## Dependency order

```
spec-01 (centroid probe)
    └── unblocks: Conjecture 5 verification in spec-03 method section

spec-02 (μ(t) schedule)
    └── unblocks: training details in spec-03 §3.3

spec-03 (method draft)
    └── depends on: spec-01 results, spec-02 μ(t) formula
```

## Future (not yet specced)

- Implement PDC training loss as ELF fine-tuning (depends on spec-01–03 being done)
- Cross-model comparison section for paper (ELF / MDLM / DUO / LangFlow SNR story)
