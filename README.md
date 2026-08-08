# Progressive Anchoring (ELF extension)

Research project: annealed lexical commitment schedule for continuous diffusion LMs.

## Structure

```
docs/
  result.md                      # canonical ledger of major experiments and complete evaluation metrics
  proposal.md                    # unified research proposal (merged from 4 reports)
  deep-research-report.md        # report 1: 渐进锚定
  deep-research-report (1).md    # report 2: Coupled Semantic-Lexical Flow
  deep-research-report (2).md    # report 3: 统一语言流
  deep-research-report (3).md    # report 4: 退火式词汇承诺 (final synthesis)

experiments/
  probe_anchor/
    probe_anchor.py              # anchor emergence probing script (v2, verification edition)
    results_v1/
      anchor_probe.json          # v1 results: τ=1.0, 64 seqs, seq_len=256
    results_v2/                  # v2 results (tau sweep, JSD, topk_final) — pending

papers/
  2605.10938v1.pdf               # ELF paper (Hu et al. 2026)
  hu2026/report.md               # paper reading notes

```

## Server

Remote: `new-ncl` (ncl-cr3.ddns.comp.nus.edu.sg, port 5008, via gateway.ncl.sg)  
ELF code: `~/ELF/` (official repo clone)  
Conda env: `~/miniforge3/envs/elf/`  
Run probing: `CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`

## Key findings (probing v1)

ELF-B OWT checkpoint shows a 4-phase commitment pattern:
1. **Prior-dominated** (t=0–0.15): entropy peaks at 0.49, top-5≈0
2. **Rapid commitment** (t=0.15–0.40): entropy drops 5×, top-5 shoots to 83%
3. **Stable-but-imperfect** (t=0.40–0.95): H≈0.05, top-5≈90%, revision≈8%
4. **Final refinement** (t=1.0): top-5 jumps to 98%, 19% of positions correct

ELF commits at t≈0.25, **not** at t=1. The stable-but-imperfect plateau is the
target for Progressive Anchoring.
