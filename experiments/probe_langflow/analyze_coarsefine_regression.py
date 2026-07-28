"""
EXP-25v2: LangFlow Coarse-to-Fine — Occurrence-Level Logistic Regression

PROBLEM WITH EXP-25: Type-level comparison of function vs content words doesn't
control for frequency. Function words are high-frequency; the observed Δt* could
be entirely explained by frequency, not POS structure.

FIX: Occurrence-level logistic regression:
   committed_by_t_i ~ is_function_word + log_freq(token)

At each t_i, the coefficient β_func reveals whether function words commit earlier
AFTER controlling for token frequency. If β_func is still large and negative,
POS structure has an independent effect.

REQUIRES: results/exp27v2_langflow/freq_commitment_v2.json
          (run analyze_freq_commitment_v2.py first)

Usage (from CCLF root, CPU-only):
  conda run -n elf python experiments/probe_langflow/analyze_coarsefine_regression.py \\
    --exp25_dir results/exp25_langflow \\
    --freq_dir  results/exp27v2_langflow \\
    --out_dir   results/exp25v2_langflow
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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


def bootstrap_coef_ci(X: np.ndarray, y: np.ndarray, n_boot: int = 200, alpha: float = 0.95) -> dict:
    """Bootstrap percentile CI for logistic regression coefficients."""
    n = len(y)
    coefs = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        Xb, yb = X[idx], y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        try:
            lr = LogisticRegression(max_iter=200, solver="lbfgs", C=1.0)
            lr.fit(Xb, yb)
            coefs.append(lr.coef_[0].tolist())
        except Exception:
            continue
    if len(coefs) < 10:
        return {}
    arr = np.array(coefs)
    lo = (1 - alpha) / 2
    hi = 1 - lo
    return {
        "coef_mean":  arr.mean(axis=0).tolist(),
        "coef_ci_lo": np.quantile(arr, lo, axis=0).tolist(),
        "coef_ci_hi": np.quantile(arr, hi, axis=0).tolist(),
        "n_bootstrap": len(coefs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp25_dir", default="results/exp25_langflow")
    ap.add_argument("--freq_dir",  default="results/exp27v2_langflow")
    ap.add_argument("--out_dir",   default="results/exp25v2_langflow")
    ap.add_argument("--n_boot",    type=int, default=200)
    ap.add_argument("--t_stride",  type=int, default=5,
                    help="Run regression every t_stride t values to reduce computation")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load frequency table ─────────────────────────────────────────────
    freq_path = Path(args.freq_dir) / "freq_commitment_v2.json"
    if not freq_path.exists():
        raise FileNotFoundError(f"Run analyze_freq_commitment_v2.py first: {freq_path}")
    with open(freq_path) as f:
        freq_data = json.load(f)

    per_type = freq_data.get("per_type", {})
    log_ppm_lookup = {}
    is_func_lookup = {}
    for tok_id_str, d in per_type.items():
        log_ppm_lookup[int(tok_id_str)] = d["log_ppm"]
        is_func_lookup[int(tok_id_str)] = d["is_function"]
    print(f"[freq] loaded log_ppm for {len(log_ppm_lookup)} token types")

    # For tokens not in the frequency lookup (rare in OWT), assign median log_ppm
    freq_vals = [v for v in log_ppm_lookup.values() if v is not None]
    median_log_ppm = float(np.median(freq_vals)) if freq_vals else 0.0
    print(f"[freq] median log_ppm = {median_log_ppm:.3f}")

    # ── 2. Load exp25 commitment data ───────────────────────────────────────
    print(f"\n[exp25] loading from {args.exp25_dir}")
    commit_tidx = np.load(f"{args.exp25_dir}/commit_tidx.npy")   # [n, L]
    gt_tokens   = np.load(f"{args.exp25_dir}/gt_tokens.npy")     # [n, L]
    t_grid      = np.load(f"{args.exp25_dir}/t_grid.npy")        # [n_t]
    n_t = len(t_grid)
    NEVER = n_t
    n_seqs, L = commit_tidx.shape
    N = n_seqs * L
    print(f"[exp25] {n_seqs} seqs × {L} pos = {N} occurrences, n_t={n_t}")

    # ── 3. Build occurrence-level feature matrix ────────────────────────────
    # Variables: (token_id, commit_tidx, is_function, log_ppm, seq_id, pos_id)
    flat_tokens = gt_tokens.reshape(-1).astype(int)
    flat_tidx   = commit_tidx.reshape(-1).astype(int)
    seq_ids     = np.repeat(np.arange(n_seqs), L)
    pos_ids     = np.tile(np.arange(L), n_seqs)

    # Build per-occurrence features
    is_func_arr = np.array([
        1.0 if is_func_lookup.get(int(tid), False) else 0.0
        for tid in flat_tokens
    ])
    log_freq_arr = np.array([
        log_ppm_lookup.get(int(tid), median_log_ppm) if int(tid) not in SPECIAL_IDS
        else median_log_ppm
        for tid in flat_tokens
    ])

    # Normalize log_freq for numerical stability
    freq_mean = log_freq_arr.mean()
    freq_std  = log_freq_arr.std() + 1e-9
    log_freq_norm = (log_freq_arr - freq_mean) / freq_std

    n_func    = int(is_func_arr.sum())
    n_content = int((1 - is_func_arr).sum())
    print(f"[occ] function-word occurrences: {n_func} ({100*n_func/N:.1f}%)")
    print(f"      content-word occurrences:  {n_content} ({100*n_content/N:.1f}%)")

    # ── 4. Marginal comparison (no regression) ──────────────────────────────
    # For each t: fraction committed = P(commit_tidx <= t_i)
    print(f"\n[marginal] Computing commit-by-t for function vs content ...")
    func_mask    = is_func_arr.astype(bool)
    content_mask = ~func_mask

    marginal = {}
    for ti in range(n_t):
        committed = flat_tidx <= ti
        f_rate = committed[func_mask].mean()   if func_mask.any()    else float("nan")
        c_rate = committed[content_mask].mean() if content_mask.any() else float("nan")
        marginal[float(t_grid[ti])] = {
            "func_commit_rate":    float(f_rate),
            "content_commit_rate": float(c_rate),
            "delta":               float(f_rate - c_rate),
        }

    # Summary: where does Δ(func-content) peak?
    deltas = [(t, d["delta"]) for t, d in marginal.items()]
    peak_t, peak_delta = max(deltas, key=lambda x: x[1])
    print(f"  Peak Δ(func-content) = {peak_delta:+.4f} at t={peak_t:.3f}")

    # ── 5. Logistic regression at selected t values ─────────────────────────
    # At each t_i:  y = I(committed by t_i), X = [is_function, log_freq_norm]
    # Model: logit P(committed by t_i) = intercept + β_func * is_func + β_freq * log_freq
    #
    # Feature matrix: [is_func, log_freq_norm]
    # Feature names:  ["is_function", "log_freq"]
    X_full = np.column_stack([is_func_arr, log_freq_norm])

    t_indices_to_run = list(range(0, n_t, args.t_stride))
    if (n_t - 1) not in t_indices_to_run:
        t_indices_to_run.append(n_t - 1)

    print(f"\n[regression] Running logistic regression at {len(t_indices_to_run)} t values ...")
    regression_results = []

    for ti in t_indices_to_run:
        t_val = float(t_grid[ti])
        y = (flat_tidx <= ti).astype(int)
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos < 20 or n_neg < 20:
            continue

        try:
            lr = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
            lr.fit(X_full, y)
            beta_func = float(lr.coef_[0][0])
            beta_freq = float(lr.coef_[0][1])
            intercept = float(lr.intercept_[0])
        except Exception as e:
            print(f"  [warn] t={t_val:.3f}: regression failed: {e}")
            continue

        # Bootstrap CI
        ci = bootstrap_coef_ci(X_full, y, n_boot=args.n_boot)

        # Likelihood ratio test vs intercept-only
        from scipy.special import expit
        y_arr = y.astype(float)
        p_hat = expit(lr.predict_log_proba(X_full)[:, 1] + lr.predict_log_proba(X_full)[:, 0] * 0)
        # Use predicted probabilities
        p_full = lr.predict_proba(X_full)[:, 1]
        # Null model (intercept only)
        p_null = y_arr.mean()
        ll_full = np.sum(y_arr * np.log(p_full + 1e-15) + (1 - y_arr) * np.log(1 - p_full + 1e-15))
        ll_null = np.sum(y_arr * np.log(p_null + 1e-15) + (1 - y_arr) * np.log(1 - p_null + 1e-15))
        lr_stat = 2 * (ll_full - ll_null)
        p_lrt   = 1 - stats.chi2(df=2).cdf(lr_stat)

        row = {
            "t_idx": ti,
            "t_val": t_val,
            "n_committed": int(n_pos),
            "n_total": int(len(y)),
            "commit_rate": float(n_pos / len(y)),
            "beta_function": beta_func,
            "beta_log_freq": beta_freq,
            "intercept": intercept,
            "OR_function": float(np.exp(beta_func)),    # odds ratio
            "OR_log_freq": float(np.exp(beta_freq)),
            "LR_stat": float(lr_stat),
            "LR_p": float(p_lrt),
            "bootstrap_ci": ci,
        }
        regression_results.append(row)
        print(f"  t={t_val:.3f}: β_func={beta_func:+.3f} OR={np.exp(beta_func):.2f}  "
              f"β_freq={beta_freq:+.3f} OR={np.exp(beta_freq):.2f}  "
              f"LR_p={p_lrt:.3g}")

    # ── 6. Summary statistics ───────────────────────────────────────────────
    print(f"\n[SUMMARY]")
    # Find t value closest to t=0.85 (typical high-acc region)
    t85_idx = np.argmin(np.abs(t_grid - 0.85))
    res_at_t85 = next((r for r in regression_results if r["t_idx"] == t85_idx or
                       abs(r["t_val"] - t_grid[t85_idx]) < 0.03), None)
    if res_at_t85:
        print(f"  At t≈0.85:")
        print(f"    β_function = {res_at_t85['beta_function']:+.3f} (OR={res_at_t85['OR_function']:.2f})")
        print(f"    β_log_freq = {res_at_t85['beta_log_freq']:+.3f} (OR={res_at_t85['OR_log_freq']:.2f})")
        print(f"    Interpretation:")
        if res_at_t85['beta_function'] < -0.1:
            print(f"      → Function words commit EARLIER even after controlling for frequency")
        elif res_at_t85['beta_function'] > 0.1:
            print(f"      → Function words commit LATER after controlling for frequency (frequency drove earlier finding)")
        else:
            print(f"      → No significant function word effect after controlling for frequency")
        if res_at_t85['beta_log_freq'] < -0.05:
            print(f"      → Higher frequency tokens commit EARLIER (confirming frequency effect)")
        else:
            print(f"      → No significant frequency effect at this t")

    # ── 7. Save results ─────────────────────────────────────────────────────
    out = {
        "metadata": {
            "exp25_dir":  args.exp25_dir,
            "freq_dir":   args.freq_dir,
            "n_seqs":     int(n_seqs),
            "L":          int(L),
            "n_func_occ": int(n_func),
            "n_content_occ": int(n_content),
            "features":   ["is_function", "log_freq_norm"],
            "note": "β_function: positive = function words commit LATER (controlling for freq)",
        },
        "marginal": {str(t): v for t, v in marginal.items()},
        "marginal_summary": {
            "peak_delta_t":     float(peak_t),
            "peak_delta_value": float(peak_delta),
        },
        "regression": regression_results,
    }
    out_path = out_dir / "coarsefine_regression_v2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
