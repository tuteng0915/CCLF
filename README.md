# Progressive Anchoring (ELF extension)

Research project: annealed lexical commitment schedule for continuous diffusion LMs.
Working name: **CCLF** (Coupled Continuous–Lexical Flow).

## Structure

```
docs/
  proposal.md                    # unified research proposal
  research_log.md                # full experimental record with findings
  probe_findings_section.md      # condensed probe results section for paper
  deep-research-report*.md       # background research reports (4 drafts)
  CCLF.pptx                      # slide deck

experiments/
  probe_elf/
    probe_anchor.py              # ELF anchor emergence probing (v1/v2)
    probe_anchor_v3.py           # v3: commitment state + transition matrix + geometry
    probe_anchor_v4.py           # v4: residual norm, temporal JSD, fate tracking, bimodality
    probe_decode_branch.py       # lin vs dec two-pass gap analysis
  probe_langflow/
    probe_langflow.py            # LangFlow commitment probing
  probe_mdlm/
    probe_mdlm.py                # MDLM (AbsorbingState) probing
  probe_duo/
    probe_duo.py                 # DUO (UniformState) probing
  analysis/
    analyze_snr_gdec.py          # SNR comparison + μ(t) schedule derivation
    compute_token_centroids.py   # contextual centroid computation (portable path)
    compute_token_centroids_pt.py# PyTorch version of centroid computation
    compute_token_centroids_server.py  # original server version (~/ELF/src path)

results/
  elf/
    probe_v1/                    # τ=1.0, 64 seqs, seq_len=256
    probe_v2/                    # tau sweep + JSD verification
    probe_v3/                    # commitment state decomposition + geometry
    probe_v4/                    # residual norm, temporal JSD, fate tracking
    probe_decode_v1/             # decode branch gap analysis (lin vs dec)
    snr_analysis/                # SNR(t) comparison across models + μ(t) fit
  langflow/
    probe_v1/                    # LangFlow probing (gamma direction bug, see research_log)
    probe_v2/                    # corrected direction
    probe_stub/                  # early stub test (ignore)
  mdlm/
    probe_v1/                    # kuleshov-group/mdlm-owt (AbsorbingState)
  duo/
    probe_v1/                    # GPT-2 fallback run (Jun 1, pre-real-model)
    probe_v2/                    # s-sahoo/duo real model (Jun 2)
  scratch/                       # stub/test outputs (ignore)
  data/
    token_centroids.npz          # contextual centroid E[v] for T5-small vocab

models/                          # external repos (git submodules, do not edit)
  ELF/                           # Hu et al. 2026 official repo (JAX/Flax)
  LangFlow/                      # arXiv:2604.11748 (PyTorch)
  mdlm/                          # kuleshov-group/mdlm
  duo/                           # s-sahoo/duo

scripts/
  launch_all.sh                  # launch all probes in sequence
  launch_centroid.sh / _pt.sh    # centroid computation scripts
  launch_langflow.sh / _v2.sh    # LangFlow probe launch
  launch_v4.sh                   # ELF v4 probe launch
  fix_cudnn.sh                   # cuDNN version fix (9.1 → 9.23)
  setup_mdlm_duo.sh              # HuggingFace model setup
  patches/                       # HuggingFace model compatibility patches
    patch_duo_hf.py
    patch_mdlm_hf.py / _hf2.py / _hf3.py

logs/                            # run logs from server
```

## Server

Remote: `new-ncl` (ncl-cr3.ddns.comp.nus.edu.sg, port 5008, via gateway.ncl.sg)  
ELF code: `~/ELF/` (official repo clone)  
Conda env: `~/miniforge3/envs/elf/`  
Run probing: `CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`

## Key findings summary

### ELF 4-phase commitment (probe v1–v4)

ELF-B OWT checkpoint commits at t≈0.25, far before t=1:

| Phase | t range | Signal |
|---|---|---|
| Prior-dominated | 0–0.15 | H peak 0.49, top-5≈0 |
| Commitment cliff | 0.15–0.35 | H drops 5×, top-5 → 85%, stab_jsd_p50 peaks at log(2) |
| Stable plateau | 0.35–0.95 | H≈0.05, top-5≈90%, many positions locked-but-wrong |
| Final refinement | 1.0 | top-5 → 98%, 19% of positions corrected |

### Decode branch gap (probe_decode_v1)

Decode branch has corrective capacity throughout plateau (G_dec ≈ +0.15–+0.25).  
ELF only uses it at t=1 — **Progressive Decode Correction** is the target method.

Training objective candidate:
$$\mathcal{L}_\text{PDC}(t) = \|\hat{x}_t^{den} - \text{sg}(h_t^{dec})\|^2 \cdot \mu(t)$$
where μ(t) activates after t≈0.25 and decays to 0 near t=1.

### Cross-model comparison (ELF vs LangFlow vs MDLM vs DUO)

| Model | commit_time_mean | Cliff location |
|---|---|---|
| ELF (continuous AbsorbingState-like) | ~0.22 | t ≈ 0.25 |
| MDLM (discrete AbsorbingState) | 0.866 | t ≈ 0.85 |
| DUO (discrete UniformState) | 0.904 | t ≈ 0.80–0.95 |
| LangFlow (continuous, always-on supervision) | 0.909 | t ≈ 0.75–0.90 |

ELF's early cliff is due to its linear flow SNR schedule (SNR at t=0.25 is 105× higher than LangFlow).

## Pending work

- [ ] Re-run ELF anchor distance probe with contextual centroid (replacing raw T5 embedding)
- [ ] Write method section: Revisable Lexical Anchoring training objective
