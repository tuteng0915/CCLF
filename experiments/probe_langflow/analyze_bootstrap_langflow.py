"""
EXP-26: LangFlow Contextual Bootstrapping (LangFlow analog of EXP-09)

Tests whether positions near already-committed neighbors commit earlier.
Reuses EXP-25 per-position commitment data (no GPU needed).

ELF kd_cr finding (EXP-09): positions within d=5 of an already-committed
neighbor show +65pp higher commitment rate at t=0.5→0.7.

For LangFlow, the cliff is at t≈0.83-0.93, so we analyze bootstrapping
in that region. Key question: does early function-word commitment at t≈0.83
accelerate nearby content-word commitment at t≈0.88?

Usage (from CCLF root):
  conda run -n elf python experiments/probe_langflow/analyze_bootstrap_langflow.py \
    --exp25_dir results/exp25_langflow \
    --out_dir results/exp26_langflow
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def has_committed_neighbor(committed_mask: np.ndarray, d_near: int) -> np.ndarray:
    """
    Vectorized: for each position, True if any position within d_near is committed.
    Uses numpy sliding-window max (no loop over L).
    committed_mask: [L] bool  →  returns [L] bool
    """
    from scipy.ndimage import maximum_filter1d
    committed_float = committed_mask.astype(np.float32)
    # maximum_filter1d with size=2*d_near+1 centered at each position
    has_near = maximum_filter1d(committed_float, size=2 * d_near + 1, mode="constant", cval=0) > 0
    return has_near


def compute_bootstrapping(commit_tidx: np.ndarray, t_grid: np.ndarray,
                           is_func: np.ndarray, d_near: int = 5):
    """
    For each consecutive t transition, compute bootstrapping statistics.
    Vectorized over positions (no inner position loop).

    commit_tidx: [n_samples, L] int (index into t_grid; len(t_grid) = never committed)
    is_func: [n_samples, L] bool (True for function words)
    """
    n_samples, L = commit_tidx.shape
    n_t = len(t_grid)
    NEVER = n_t

    results = []

    for ti in range(n_t - 1):
        t_cur = float(t_grid[ti])
        t_next = float(t_grid[ti + 1])

        # Accumulators over all samples
        near_comm = 0; near_tot = 0
        far_comm = 0;  far_tot = 0
        near_func_comm = 0; near_func_tot = 0
        far_func_comm = 0;  far_func_tot = 0

        for si in range(n_samples):
            committed  = commit_tidx[si] <= ti          # [L] bool
            uncommitted = ~committed                     # [L] bool
            commits_next = (commit_tidx[si] == ti + 1)  # [L] bool
            func = is_func[si]                           # [L] bool

            if not uncommitted.any():
                continue

            # Vectorized nearest-neighbor check
            has_near = has_committed_neighbor(committed, d_near)  # [L] bool

            u = uncommitted
            near  = u &  has_near
            far   = u & ~has_near

            near_comm += int((near & commits_next).sum())
            near_tot  += int(near.sum())
            far_comm  += int((far & commits_next).sum())
            far_tot   += int(far.sum())

            near_func_comm += int((near & func & commits_next).sum())
            near_func_tot  += int((near & func).sum())
            far_func_comm  += int((far & func & commits_next).sum())
            far_func_tot   += int((far & func).sum())

        near_rate = near_comm / near_tot if near_tot > 0 else None
        far_rate  = far_comm  / far_tot  if far_tot  > 0 else None
        delta = (near_rate - far_rate) if (near_rate is not None and far_rate is not None) else None

        near_func_rate = near_func_comm / near_func_tot if near_func_tot > 0 else None
        far_func_rate  = far_func_comm  / far_func_tot  if far_func_tot  > 0 else None

        results.append({
            "t_cur": t_cur, "t_next": t_next,
            "near_rate": near_rate, "far_rate": far_rate, "delta": delta,
            "near_total": near_tot, "far_total": far_tot,
            "near_func_rate": near_func_rate, "far_func_rate": far_func_rate,
            "near_func_total": near_func_tot, "far_func_total": far_func_tot,
        })

    return results


def main():
    p = argparse.ArgumentParser(description="EXP-26: LangFlow contextual bootstrapping")
    p.add_argument("--exp25_dir", default="results/exp25_langflow")
    p.add_argument("--out_dir", default="results/exp26_langflow")
    p.add_argument("--d_near", type=int, default=5, help="Distance threshold for 'near' neighbors")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] Loading EXP-25 data from {args.exp25_dir}")
    commit_tidx = np.load(f"{args.exp25_dir}/commit_tidx.npy")  # [n, L]
    gt_tokens = np.load(f"{args.exp25_dir}/gt_tokens.npy")       # [n, L]
    t_grid = np.load(f"{args.exp25_dir}/t_grid.npy")             # [n_t]

    with open(f"{args.exp25_dir}/coarse_fine_results.json") as f:
        cf = json.load(f)

    n_samples, L = commit_tidx.shape
    n_t = len(t_grid)
    NEVER = n_t

    print(f"[config] n_samples={n_samples}, L={L}, n_t={n_t}")
    print(f"         t range: [{t_grid[0]:.3f}, {t_grid[-1]:.3f}]")
    print(f"         d_near={args.d_near}")

    # Identify function words using the token IDs
    # Load tokenizer (same as in probe_coarsefine_langflow.py)
    import sys
    _PROBE_DIR = Path(__file__).parent
    _LF_SRC = _PROBE_DIR.parents[1] / "models" / "LangFlow"
    sys.path.insert(0, str(_LF_SRC))
    sys.path.insert(0, str(_PROBE_DIR))
    from probe_coarsefine_langflow import FUNCTION_WORDS, is_function_word

    # We need the tokenizer for is_function_word
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    is_func = np.zeros_like(commit_tidx, dtype=bool)
    for si in range(n_samples):
        for pos in range(L):
            is_func[si, pos] = is_function_word(gt_tokens[si, pos], tokenizer)

    print(f"[classify] function word positions: {is_func.mean():.3f}")

    print(f"\n[EXP-26] Computing bootstrapping (d_near={args.d_near})...")
    print(f"         (focusing on cliff region t=0.70-1.00)")
    results = compute_bootstrapping(commit_tidx, t_grid, is_func, args.d_near)

    # Save full results
    out_path = out_dir / f"bootstrap_d{args.d_near}.json"
    out = {
        "results": results,
        "args": vars(args),
        "n_samples": n_samples,
        "L": L,
        "n_t": n_t,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {out_path}")

    # Print summary for cliff region (t > 0.70)
    print("\n── EXP-26 LangFlow Contextual Bootstrapping ─────────────────────────")
    print(f"  d_near={args.d_near}  (ELF EXP-09 used d=5)")
    print(f"  {'t_cur→t_next':15s}  {'near':>8}  {'far':>8}  {'Δ':>8}  "
          f"{'near_n':>7}  {'far_n':>7}")
    print(f"  {'-'*65}")

    for row in results:
        if row["t_cur"] < 0.70:
            continue
        near = f"{row['near_rate']*100:.1f}%" if row["near_rate"] is not None else "n/a"
        far = f"{row['far_rate']*100:.1f}%" if row["far_rate"] is not None else "n/a"
        delta = f"{row['delta']*100:+.1f}pp" if row["delta"] is not None else "n/a"
        print(f"  {row['t_cur']:.3f}→{row['t_next']:.3f}      {near:>8}  {far:>8}  {delta:>8}  "
              f"{row['near_total']:>7}  {row['far_total']:>7}")

    # Find peak bootstrapping effect
    cliff_rows = [r for r in results if r["t_cur"] >= 0.70 and r["delta"] is not None]
    if cliff_rows:
        peak = max(cliff_rows, key=lambda r: r["delta"])
        print(f"\n  Peak Δ = {peak['delta']*100:+.1f}pp at t={peak['t_cur']:.3f}→{peak['t_next']:.3f}")
        print(f"  ELF kd_cr reference: peak Δ = +65pp at d=5, t=0.5→0.7")

    print(f"\n  Overall: LangFlow bootstrapping peak vs ELF:")
    if cliff_rows:
        peak_delta = peak["delta"] * 100
        if abs(peak_delta) < 5:
            print(f"    NO SIGNIFICANT bootstrapping (peak |Δ|={abs(peak_delta):.1f}pp < 5pp)")
        elif peak_delta > 0:
            print(f"    POSITIVE bootstrapping: committed neighbors ACCELERATE further commits (+{peak_delta:.1f}pp)")
        else:
            print(f"    NEGATIVE bootstrapping: committed neighbors SUPPRESS further commits ({peak_delta:.1f}pp)")


if __name__ == "__main__":
    main()
