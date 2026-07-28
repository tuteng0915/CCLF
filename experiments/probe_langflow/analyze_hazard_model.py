"""
EXP-26v2: Spatial Commitment Analysis — Hazard Model + Moran's I

PROBLEM WITH EXP-26: Risk-set collapse (far_n=6 at peak), common-cause confound,
multiple comparison across 51 t values. Bootstrap CIs are too wide to be conclusive.

FIX 1: Discrete-time logistic hazard model with has_committed_neighbor as
        time-varying covariate, + controls for log_freq and is_function.
        This gives proper multivariate inference over the full time horizon.

FIX 2: Moran's I spatial autocorrelation test per time step.
        Tests whether commitment is spatially clustered (either direction)
        independently of any causal mechanism.

The two approaches answer different questions:
  Moran's I: is commitment spatially autocorrelated at each t? (tests clustering)
  Hazard model: does having a committed neighbor raise YOUR hazard at time t?
                (tests temporal order; still confounded by shared local context)

Usage (from CCLF root, CPU-only):
  conda run -n elf python experiments/probe_langflow/analyze_hazard_model.py \\
    --exp25_dir results/exp25_langflow \\
    --freq_dir  results/exp27v2_langflow \\
    --out_dir   results/exp26v2_langflow
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression


SPECIAL_IDS = {50256}

FUNCTION_WORDS = frozenset([
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "every", "any", "no", "all", "both", "each", "either", "neither", "some", "another",
    "of", "in", "to", "for", "on", "at", "with", "by", "from", "as",
    "into", "through", "about", "between", "after", "before", "around",
    "up", "out", "down", "under", "over", "above", "below", "across",
    "against", "along", "among", "behind", "beneath", "beside", "besides",
    "beyond", "during", "except", "inside", "near", "off", "since", "than",
    "throughout", "toward", "underneath", "until", "upon", "within", "without",
    "and", "but", "or", "nor", "if", "when", "where", "while", "because",
    "since", "although", "though", "unless", "until", "so", "yet",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "shall", "can",
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "who", "whom", "whose", "what", "which", "whoever", "whatever",
    "not", "very", "just", "also", "even", "only", "well",
    "already", "still", "then", "there", "here",
    ".", ",", "!", "?", ";", ":", "'", '"', "(", ")", "-", "–", "—",
])


def is_function_token(token_str: str) -> bool:
    surf = token_str.lstrip("Ġ").lower()
    return surf in FUNCTION_WORDS or token_str.strip() in FUNCTION_WORDS


def compute_morans_i_per_t(commit_tidx: np.ndarray, t_grid: np.ndarray,
                           n_perm: int = 500, t_stride: int = 5):
    """
    Compute Moran's I for commitment status at each t value within each sequence.

    For each t_i and each sequence s:
      x_s_t[j] = I(commit_tidx[s,j] <= t_i)   — binary committed status at t_i
      W_s = tridiagonal adjacency matrix over non-special positions

    Global I is computed by pooling across sequences (row normalization).

    Permutation p-value: shuffle x within each sequence, recompute I.
    """
    NEVER = len(t_grid)
    n_seqs, L = commit_tidx.shape
    n_t = len(t_grid)

    results = []

    t_indices = list(range(5, n_t, t_stride))  # skip very early (too few committed)

    for ti in t_indices:
        t_val = float(t_grid[ti])

        # committed_matrix[s, j] = 1 if position j committed by t_i
        committed = (commit_tidx <= ti).astype(float)  # [n_seqs, L]

        # Moran's I for 1D sequence: W = tridiagonal
        # I = (N/W) * (sum_s sum_{j≠k} w_jk (x_j - mean)(x_k - mean)) / sum_s sum_j (x_j - mean)^2
        # For tridiagonal W on length-L sequence: W = 2(L-1)

        numerator = 0.0
        denominator = 0.0
        W_total = 0.0
        N_total = 0

        for s in range(n_seqs):
            x = committed[s]  # [L]
            x_mean = x.mean()
            x_dev = x - x_mean

            # Tridiagonal spatial lag (only adjacent pairs)
            # sum_{j} x_dev[j] * x_dev[j+1] (undirected: count each pair twice = x2)
            lag_sum = 2.0 * float(np.dot(x_dev[:-1], x_dev[1:]))
            w_s = 2 * (L - 1)  # number of directed edge contributions

            numerator  += lag_sum
            denominator += float(np.dot(x_dev, x_dev))
            W_total += w_s
            N_total += L

        if denominator < 1e-9 or W_total < 1:
            continue

        I_obs = (N_total / W_total) * (numerator / denominator)

        # Permutation test: shuffle committed status within each sequence
        I_perm = []
        for _ in range(n_perm):
            num_p = 0.0
            den_p = 0.0
            for s in range(n_seqs):
                x = committed[s].copy()
                np.random.shuffle(x)
                x_mean = x.mean()
                x_dev  = x - x_mean
                lag_sum = 2.0 * float(np.dot(x_dev[:-1], x_dev[1:]))
                w_s = 2 * (L - 1)
                num_p += lag_sum
                den_p += float(np.dot(x_dev, x_dev))

            if den_p < 1e-9:
                continue
            I_perm.append((N_total / W_total) * (num_p / den_p))

        if len(I_perm) < 10:
            continue

        I_perm_arr = np.array(I_perm)
        perm_mean = float(I_perm_arr.mean())
        perm_std  = float(I_perm_arr.std() + 1e-15)
        z_score   = (I_obs - perm_mean) / perm_std
        p_val = float(np.mean(I_perm_arr >= I_obs))  # one-sided: p(clustering)

        results.append({
            "t_idx": ti,
            "t_val": t_val,
            "commit_rate": float(committed.mean()),
            "I_obs": float(I_obs),
            "I_perm_mean": perm_mean,
            "I_perm_std":  perm_std,
            "z_score": float(z_score),
            "p_val_cluster": p_val,  # fraction of permutations with I >= I_obs
        })

    return results


def compute_hazard_model(commit_tidx: np.ndarray, t_grid: np.ndarray,
                         log_freq_arr: np.ndarray, is_func_arr: np.ndarray,
                         t_stride: int = 3):
    """
    Discrete-time logistic hazard model.

    For each t_i, pool all positions that are "at risk" (not yet committed by t_{i-1}):
      y = I(commits exactly at t_i)   — event indicator
      X = [has_committed_neighbor_at_t_{i-1}, log_freq_norm, is_function, t_normalized]

    has_committed_neighbor(s, j, t_i) = any position in {j-1, j+1} ∩ [0,L-1] with commit_tidx <= t_{i-1}

    This gives a single pooled logistic regression over all (position, t) pairs where the
    position is at risk. The coefficient on has_committed_neighbor tests whether neighbor
    commitment raises YOUR hazard, controlling for frequency and function-word status.
    """
    NEVER = len(t_grid)
    n_seqs, L = commit_tidx.shape
    n_t = len(t_grid)

    # Reshape to [n_seqs, L] for spatial access
    log_freq_mat = log_freq_arr.reshape(n_seqs, L)
    is_func_mat  = is_func_arr.reshape(n_seqs, L)

    # Normalize covariates
    freq_mean = log_freq_arr.mean()
    freq_std  = log_freq_arr.std() + 1e-9
    log_freq_norm = (log_freq_arr - freq_mean) / freq_std

    # Collect data points: (y, has_neighbor, log_freq_norm, is_func, t_norm)
    rows_y     = []
    rows_X     = []

    for ti in range(1, n_t, t_stride):  # start at 1 (need t_{i-1})
        t_val = float(t_grid[ti])
        t_prev = float(t_grid[ti - 1])

        # committed_before: [n_seqs, L] — committed before t_i (i.e., by t_{i-1})
        committed_before = (commit_tidx <= ti - 1)  # by t_{i-1}

        for s in range(n_seqs):
            # at-risk: positions not yet committed before t_i
            at_risk = ~committed_before[s]   # [L]
            event   = (commit_tidx[s] == ti) # [L] commits exactly at ti

            at_risk_pos = np.where(at_risk)[0]
            if len(at_risk_pos) == 0:
                continue

            for j in at_risk_pos:
                # has_committed_neighbor at t_{i-1}
                neighbors = []
                if j > 0:
                    neighbors.append(committed_before[s, j - 1])
                if j < L - 1:
                    neighbors.append(committed_before[s, j + 1])
                has_neighbor = 1.0 if any(neighbors) else 0.0

                y = 1.0 if event[j] else 0.0
                freq_val = float((log_freq_mat[s, j] - freq_mean) / freq_std)
                func_val = float(is_func_mat[s, j])
                t_norm   = t_val  # already in [0,1]

                rows_y.append(y)
                rows_X.append([has_neighbor, freq_val, func_val, t_norm])

    if len(rows_y) < 100:
        print(f"[hazard] Too few data points: {len(rows_y)}")
        return None

    y_arr = np.array(rows_y)
    X_arr = np.array(rows_X)
    feat_names = ["has_committed_neighbor", "log_freq_norm", "is_function", "t_normalized"]

    print(f"[hazard] {len(y_arr):,} person-period observations")
    print(f"         events: {int(y_arr.sum())} ({100*y_arr.mean():.2f}%)")
    print(f"         neighbor_has_committed: {int(X_arr[:,0].sum())} ({100*X_arr[:,0].mean():.2f}%)")

    if y_arr.sum() < 10:
        print("[hazard] Too few events.")
        return None

    lr = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
    lr.fit(X_arr, y_arr)

    coefs = lr.coef_[0].tolist()
    intercept = float(lr.intercept_[0])

    print(f"\n[HAZARD MODEL RESULTS]")
    for name, b in zip(feat_names, coefs):
        print(f"  β_{name} = {b:+.4f}  OR = {np.exp(b):.3f}")

    # Bootstrap CI
    n = len(y_arr)
    boot_coefs = []
    np.random.seed(42)
    for _ in range(200):
        idx = np.random.choice(n, n, replace=True)
        Xb, yb = X_arr[idx], y_arr[idx]
        if yb.sum() < 5 or (1-yb).sum() < 5:
            continue
        try:
            lr2 = LogisticRegression(max_iter=200, solver="lbfgs", C=1.0)
            lr2.fit(Xb, yb)
            boot_coefs.append(lr2.coef_[0].tolist())
        except Exception:
            continue

    boot_arr = np.array(boot_coefs) if boot_coefs else None
    ci_results = {}
    if boot_arr is not None and len(boot_arr) >= 10:
        ci_lo = np.quantile(boot_arr, 0.025, axis=0)
        ci_hi = np.quantile(boot_arr, 0.975, axis=0)
        for i, name in enumerate(feat_names):
            ci_results[name] = {
                "beta":  coefs[i],
                "OR":    float(np.exp(coefs[i])),
                "ci_lo": float(ci_lo[i]),
                "ci_hi": float(ci_hi[i]),
                "OR_lo": float(np.exp(ci_lo[i])),
                "OR_hi": float(np.exp(ci_hi[i])),
            }
            print(f"  95% CI for β_{name}: [{ci_lo[i]:+.4f}, {ci_hi[i]:+.4f}]  "
                  f"OR: [{np.exp(ci_lo[i]):.3f}, {np.exp(ci_hi[i]):.3f}]")

    return {
        "n_observations": int(len(y_arr)),
        "n_events":       int(y_arr.sum()),
        "features":       feat_names,
        "coefs":          coefs,
        "intercept":      intercept,
        "OR":             [float(np.exp(b)) for b in coefs],
        "bootstrap_ci":   ci_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp25_dir", default="results/exp25_langflow")
    ap.add_argument("--freq_dir",  default="results/exp27v2_langflow")
    ap.add_argument("--out_dir",   default="results/exp26v2_langflow")
    ap.add_argument("--n_perm",    type=int, default=500,
                    help="Permutations for Moran's I p-value")
    ap.add_argument("--t_stride",  type=int, default=5,
                    help="Run Moran's I every t_stride steps")
    ap.add_argument("--hazard_stride", type=int, default=3,
                    help="Step stride for pooled hazard model")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load frequency lookup ─────────────────────────────────────────────
    freq_path = Path(args.freq_dir) / "freq_commitment_v2.json"
    if not freq_path.exists():
        raise FileNotFoundError(f"Run analyze_freq_commitment_v2.py first: {freq_path}")
    with open(freq_path) as f:
        freq_data = json.load(f)
    per_type = freq_data.get("per_type", {})
    log_ppm_lookup = {int(k): d["log_ppm"] for k, d in per_type.items() if d["log_ppm"] is not None}
    is_func_lookup = {int(k): d["is_function"] for k, d in per_type.items()}
    freq_vals = list(log_ppm_lookup.values())
    median_log_ppm = float(np.median(freq_vals)) if freq_vals else 0.0
    print(f"[freq] loaded {len(log_ppm_lookup)} types, median log_ppm={median_log_ppm:.2f}")

    # ── 2. Load exp25 commitment data ─────────────────────────────────────────
    print(f"\n[exp25] loading from {args.exp25_dir}")
    commit_tidx = np.load(f"{args.exp25_dir}/commit_tidx.npy")   # [n, L]
    gt_tokens   = np.load(f"{args.exp25_dir}/gt_tokens.npy")     # [n, L]
    t_grid      = np.load(f"{args.exp25_dir}/t_grid.npy")        # [n_t]
    n_t = len(t_grid)
    NEVER = n_t
    n_seqs, L = commit_tidx.shape

    flat_tokens = gt_tokens.reshape(-1).astype(int)
    log_freq_arr = np.array([
        log_ppm_lookup.get(int(tid), median_log_ppm) for tid in flat_tokens
    ])
    is_func_arr = np.array([
        1.0 if is_func_lookup.get(int(tid), False) else 0.0 for tid in flat_tokens
    ])

    # ── 3. Moran's I per timestep ─────────────────────────────────────────────
    print(f"\n[moran] Computing Moran's I at each t (n_perm={args.n_perm}, stride={args.t_stride}) ...")
    np.random.seed(42)
    moran_results = compute_morans_i_per_t(commit_tidx, t_grid,
                                           n_perm=args.n_perm,
                                           t_stride=args.t_stride)

    print(f"\n[Moran's I summary]")
    print(f"  {'t_val':>6}  {'I_obs':>7}  {'z_score':>8}  {'p_cluster':>10}  {'commit_rate':>12}")
    for r in moran_results[::max(1, len(moran_results)//10)]:
        print(f"  {r['t_val']:6.3f}  {r['I_obs']:7.4f}  {r['z_score']:8.3f}  "
              f"{r['p_val_cluster']:10.4f}  {r['commit_rate']:12.4f}")

    # Peak Moran's I
    if moran_results:
        peak = max(moran_results, key=lambda r: r["I_obs"])
        print(f"\n  Peak Moran's I = {peak['I_obs']:.4f} at t={peak['t_val']:.3f} "
              f"(z={peak['z_score']:.2f}, p={peak['p_val_cluster']:.4f})")
        sig_count = sum(1 for r in moran_results if r["p_val_cluster"] < 0.05)
        print(f"  {sig_count}/{len(moran_results)} t values with p < 0.05 (clustering)")

    # ── 4. Discrete-time hazard model ──────────────────────────────────────────
    print(f"\n[hazard] Building discrete-time hazard model ...")
    hazard_result = compute_hazard_model(
        commit_tidx, t_grid, log_freq_arr, is_func_arr,
        t_stride=args.hazard_stride,
    )

    # ── 5. Interpretation ───────────────────────────────────────────────────────
    print(f"\n[INTERPRETATION]")
    if hazard_result:
        ci = hazard_result["bootstrap_ci"]
        if "has_committed_neighbor" in ci:
            hn = ci["has_committed_neighbor"]
            print(f"  has_committed_neighbor: OR = {hn['OR']:.3f} "
                  f"[{hn['OR_lo']:.3f}, {hn['OR_hi']:.3f}]")
            if hn["ci_lo"] > 0:
                print(f"  → Committed neighbor RAISES hazard (evidence for spatial clustering / propagation)")
                print(f"    (BUT: common-cause confound cannot be ruled out without structural intervention)")
            elif hn["ci_hi"] < 0:
                print(f"  → Committed neighbor LOWERS hazard (competition effect?)")
            else:
                print(f"  → CI spans zero: no conclusive spatial effect after controlling for freq+POS")

        if moran_results:
            sig_frac = sum(1 for r in moran_results if r["p_val_cluster"] < 0.05) / len(moran_results)
            print(f"\n  Moran's I: {100*sig_frac:.0f}% of t values show significant clustering (p<0.05)")
            if sig_frac > 0.5:
                print(f"  → Commitment is spatially clustered; consistent with common local syntactic structure")
                print(f"    OR with causal propagation — Moran's I cannot distinguish between the two")
            else:
                print(f"  → No systematic spatial clustering of commitment")

    # ── 6. Save results ─────────────────────────────────────────────────────────
    out = {
        "metadata": {
            "exp25_dir": args.exp25_dir,
            "freq_dir":  args.freq_dir,
            "n_seqs": int(n_seqs),
            "L": int(L),
            "n_t": int(n_t),
            "n_perm_moran": args.n_perm,
            "hazard_stride": args.hazard_stride,
        },
        "morans_i": moran_results,
        "hazard_model": hazard_result,
    }
    out_path = out_dir / "hazard_morans_v2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
