"""
EXP-28: LangFlow Asymmetric Bootstrapping — Does Function Word Commitment
        Directionally Help Content Word Commitment?

Extension of EXP-26: tests if the bootstrapping is asymmetric:
  - func→content: does having a committed function word neighbor help uncommitted content words?
  - content→func: does having a committed content word neighbor help uncommitted function words?

For ELF, EXP-09 showed that earlier function-word commitment (EXP-08) and spatial bootstrapping
(EXP-09) may be linked — function words commit early and then help surrounding content words.
This test makes that causal link explicit.

Usage (from CCLF root):
  conda run -n elf python experiments/probe_langflow/analyze_asymmetric_bootstrap_langflow.py \
    --exp25_dir results/exp25_langflow \
    --out_dir results/exp28_langflow
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d


def compute_asymmetric_bootstrap(commit_tidx, t_grid, is_func, d_near=5):
    """
    For each t_i → t_{i+1} transition, compute:
      - fc_near_rate: rate of content-word positions committing when a FUNC-WORD neighbor committed
      - cf_near_rate: rate of func-word positions committing when a CONTENT-WORD neighbor committed
      - baseline_rate: overall uncommitted commit rate (no neighbor conditioning)
    """
    n_samples, L = commit_tidx.shape
    n_t = len(t_grid)
    NEVER = n_t

    results = []

    for ti in range(n_t - 1):
        t_cur  = float(t_grid[ti])
        t_next = float(t_grid[ti + 1])

        # Accumulators
        # fc = function-word-committed-neighbor → content-word-uncommitted transitions
        fc_near_comm = 0;  fc_near_tot = 0
        cf_near_comm = 0;  cf_near_tot = 0
        base_func_comm = 0; base_func_tot = 0
        base_cont_comm = 0; base_cont_tot = 0

        for si in range(n_samples):
            c_tidx = commit_tidx[si]  # [L]
            func   = is_func[si]       # [L] bool

            committed   = c_tidx <= ti          # [L]
            uncommitted = ~committed             # [L]
            commits_next = (c_tidx == ti + 1)   # [L]

            if not uncommitted.any():
                continue

            committed_func    = committed & func
            committed_content = committed & ~func

            def neighbor_mask(mask):
                mf = mask.astype(np.float32)
                return maximum_filter1d(mf, size=2 * d_near + 1, mode="constant", cval=0) > 0

            has_func_neighbor    = neighbor_mask(committed_func)    # [L]
            has_content_neighbor = neighbor_mask(committed_content)  # [L]

            # fc: uncommitted content-word positions with a func-word committed neighbor
            fc_mask = uncommitted & ~func & has_func_neighbor
            fc_near_comm += int((fc_mask & commits_next).sum())
            fc_near_tot  += int(fc_mask.sum())

            # cf: uncommitted func-word positions with a content-word committed neighbor
            cf_mask = uncommitted & func & has_content_neighbor
            cf_near_comm += int((cf_mask & commits_next).sum())
            cf_near_tot  += int(cf_mask.sum())

            # Baseline: all uncommitted func/content positions (no neighbor conditioning)
            u_func = uncommitted & func
            u_cont = uncommitted & ~func
            base_func_comm += int((u_func & commits_next).sum())
            base_func_tot  += int(u_func.sum())
            base_cont_comm += int((u_cont & commits_next).sum())
            base_cont_tot  += int(u_cont.sum())

        def safe_rate(num, den):
            return float(num / den) if den > 0 else None

        fc_rate   = safe_rate(fc_near_comm, fc_near_tot)
        cf_rate   = safe_rate(cf_near_comm, cf_near_tot)
        base_func = safe_rate(base_func_comm, base_func_tot)
        base_cont = safe_rate(base_cont_comm, base_cont_tot)

        fc_delta = (fc_rate - base_cont) if (fc_rate is not None and base_cont is not None) else None
        cf_delta = (cf_rate - base_func) if (cf_rate is not None and base_func is not None) else None

        results.append({
            "t_cur": t_cur, "t_next": t_next,
            "fc_rate": fc_rate, "fc_total": fc_near_tot,
            "cf_rate": cf_rate, "cf_total": cf_near_tot,
            "base_func_rate": base_func, "base_cont_rate": base_cont,
            "fc_delta": fc_delta,  # fc_rate - baseline_content = func→content boost
            "cf_delta": cf_delta,  # cf_rate - baseline_func = content→func boost
        })

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp25_dir", default="results/exp25_langflow")
    p.add_argument("--out_dir",   default="results/exp28_langflow")
    p.add_argument("--d_near", type=int, default=5)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    commit_tidx = np.load(f"{args.exp25_dir}/commit_tidx.npy")
    gt_tokens   = np.load(f"{args.exp25_dir}/gt_tokens.npy")
    t_grid      = np.load(f"{args.exp25_dir}/t_grid.npy")

    n_samples, L = commit_tidx.shape

    # Load tokenizer + function word classifier
    import sys
    _PROBE_DIR = Path(__file__).parent
    sys.path.insert(0, str(_PROBE_DIR))
    from probe_coarsefine_langflow import is_function_word
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    is_func = np.zeros_like(commit_tidx, dtype=bool)
    for si in range(n_samples):
        for pos in range(L):
            is_func[si, pos] = is_function_word(gt_tokens[si, pos], tokenizer)

    print(f"[EXP-28] d_near={args.d_near}, n_samples={n_samples}, L={L}, n_t={len(t_grid)}")
    print(f"         func_frac={is_func.mean():.3f}")

    results = compute_asymmetric_bootstrap(commit_tidx, t_grid, is_func, args.d_near)

    out = {"args": vars(args), "n_samples": n_samples, "L": L, "results": results}
    out_path = out_dir / f"asymmetric_bootstrap_d{args.d_near}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {out_path}")

    print("\n── EXP-28 LangFlow Asymmetric Bootstrapping ─────────────────────────────")
    print(f"  d_near={args.d_near}")
    print(f"  {'t_cur→t_next':14s}  {'fc_rate':>9}  {'base_cont':>9}  {'fc_Δ':>7}  "
          f"{'cf_rate':>9}  {'base_func':>9}  {'cf_Δ':>7}  {'fc_n':>6}  {'cf_n':>6}")
    print(f"  {'-'*85}")

    cliff_rows = [r for r in results if r["t_cur"] >= 0.70]
    for row in cliff_rows:
        fc_r = f"{row['fc_rate']*100:.1f}%" if row["fc_rate"] is not None else "n/a"
        cf_r = f"{row['cf_rate']*100:.1f}%" if row["cf_rate"] is not None else "n/a"
        bc   = f"{row['base_cont_rate']*100:.1f}%" if row["base_cont_rate"] is not None else "n/a"
        bf   = f"{row['base_func_rate']*100:.1f}%" if row["base_func_rate"] is not None else "n/a"
        fd   = f"{row['fc_delta']*100:+.1f}pp" if row["fc_delta"] is not None else "n/a"
        cd   = f"{row['cf_delta']*100:+.1f}pp" if row["cf_delta"] is not None else "n/a"
        print(f"  {row['t_cur']:.3f}→{row['t_next']:.3f}  {fc_r:>9}  {bc:>9}  {fd:>7}  "
              f"{cf_r:>9}  {bf:>9}  {cd:>7}  {row['fc_total']:>6}  {row['cf_total']:>6}")

    # Find peak deltas in cliff region
    valid_fc = [(r["fc_delta"], r) for r in cliff_rows if r["fc_delta"] is not None]
    valid_cf = [(r["cf_delta"], r) for r in cliff_rows if r["cf_delta"] is not None]
    if valid_fc:
        pk_fc = max(valid_fc, key=lambda x: x[0])
        print(f"\n  Peak func→content boost: {pk_fc[0]*100:+.1f}pp at t={pk_fc[1]['t_cur']:.3f}→{pk_fc[1]['t_next']:.3f}")
    if valid_cf:
        pk_cf = max(valid_cf, key=lambda x: x[0])
        print(f"  Peak content→func boost: {pk_cf[0]*100:+.1f}pp at t={pk_cf[1]['t_cur']:.3f}→{pk_cf[1]['t_next']:.3f}")


if __name__ == "__main__":
    main()
