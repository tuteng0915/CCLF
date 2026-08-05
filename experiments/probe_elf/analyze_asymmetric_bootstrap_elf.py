"""
EXP-09v2: ELF Asymmetric Bootstrapping — Full 2x2 Direction Analysis
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter1d

ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(ROOT))

FUNCTION_WORDS = frozenset({
    "the","a","an","in","on","at","to","of","for","with","by","from","as","into",
    "through","during","before","after","above","below","between","out","off","over",
    "under","again","further","then","once","and","but","or","nor","yet","so","if",
    "because","while","although","though","since","unless","until","when","where",
    "which","who","whom","that","this","these","those","it","its","he","she","they",
    "we","i","you","me","him","her","us","them","my","your","his","her","our","their",
    "is","are","was","were","be","been","being","have","has","had","do","does","did",
    "will","would","shall","should","may","might","must","can","could","not","no",
    "all","each","every","both","few","more","most","other","some","such","than","too",
    "very","just","about","up","there","here","how","what","only",
})


def is_function_word(token_id, tokenizer):
    piece = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    if piece is None:
        return False
    surf = piece.lstrip("Ginstalled").strip()
    # Handle GPT2-style Ġ prefix
    if piece.startswith('Ġ'):
        surf = piece[1:].lower()
    else:
        surf = piece.lower()
    return surf in FUNCTION_WORDS


def has_near(mask, d):
    return maximum_filter1d(mask.astype(np.float32), size=2*d+1, mode="constant", cval=0) > 0


def compute_asymmetric(commit_times, y_tokens, t_values, is_func, d_near):
    n_samples, L = commit_times.shape
    N_T = len(t_values)
    results = []
    for step in range(N_T - 1):
        t_cur, t_next = t_values[step], t_values[step + 1]
        acc = {k: 0 for k in ["fc_c","fc_t","cf_c","cf_t","bf_c","bf_t","bc_c","bc_t"]}
        for si in range(n_samples):
            c = commit_times[si]
            func = is_func[si]
            committed = c <= step
            uncommitted = ~committed
            commits_here = c == step + 1
            if not uncommitted.any():
                continue
            comm_func = committed & func
            comm_cont = committed & ~func
            near_func = has_near(comm_func, d_near)
            near_cont = has_near(comm_cont, d_near)
            u_func = uncommitted & func
            u_cont = uncommitted & ~func
            # fc: uncommitted content with committed func neighbor
            m = u_cont & near_func
            acc["fc_c"] += int((m & commits_here).sum())
            acc["fc_t"] += int(m.sum())
            # cf: uncommitted func with committed content neighbor
            m = u_func & near_cont
            acc["cf_c"] += int((m & commits_here).sum())
            acc["cf_t"] += int(m.sum())
            # baseline
            acc["bf_c"] += int((u_func & commits_here).sum())
            acc["bf_t"] += int(u_func.sum())
            acc["bc_c"] += int((u_cont & commits_here).sum())
            acc["bc_t"] += int(u_cont.sum())
        def r(a, b): return float(a/b) if b > 0 else None
        fc_r = r(acc["fc_c"], acc["fc_t"])
        cf_r = r(acc["cf_c"], acc["cf_t"])
        base_f = r(acc["bf_c"], acc["bf_t"])
        base_c = r(acc["bc_c"], acc["bc_t"])
        results.append({
            "step": step, "t_cur": t_cur, "t_next": t_next,
            "fc_rate": fc_r, "fc_n": acc["fc_t"],
            "cf_rate": cf_r, "cf_n": acc["cf_t"],
            "base_func": base_f, "base_cont": base_c,
            "fc_delta": (fc_r - base_c) if (fc_r is not None and base_c is not None) else None,
            "cf_delta": (cf_r - base_f) if (cf_r is not None and base_f is not None) else None,
        })
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="kd_cr")
    p.add_argument("--d_near", type=int, default=5)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = str(ROOT / f"models/ELF-torch/results/exp09v2_{args.checkpoint}")

    data_dir = ROOT / f"models/ELF-torch/results/exp09_{args.checkpoint}"
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    commit_times = np.load(data_dir / "commit_times_matrix.npy")
    y_tokens     = np.load(data_dir / "y_tokens_ref.npy")
    with open(data_dir / "contextual_bootstrap.json") as f:
        cb = json.load(f)
    t_values = cb["t_values"]

    print(f"[EXP-09v2] checkpoint={args.checkpoint}, d_near={args.d_near}")
    print(f"           shape={commit_times.shape}, t_values={t_values}")

    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    n_samples, L = commit_times.shape
    is_func = np.zeros_like(commit_times, dtype=bool)
    for si in range(n_samples):
        for pos in range(L):
            tok = int(y_tokens[si, pos])
            piece = tokenizer.convert_ids_to_tokens([tok])[0]
            if piece is None:
                continue
            if piece.startswith('Ġ'):
                surf = piece[1:].lower()
            else:
                surf = piece.lower()
            if surf in FUNCTION_WORDS:
                is_func[si, pos] = True

    print(f"[classify] func_frac={is_func.mean():.3f}")

    results = compute_asymmetric(commit_times, y_tokens, t_values, is_func, args.d_near)

    out = {"args": vars(args), "t_values": t_values, "results": results}
    out_path = out_dir / f"asymmetric_bootstrap_d{args.d_near}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {out_path}")

    print(f"\n── EXP-09v2 ELF Asymmetric Bootstrapping ({args.checkpoint}) ─────────────────")
    print(f"  {'step':5s}  {'t_cur→t_next':12s}  {'fc_rate':>8}  {'base_cont':>9}  {'fc_Δ':>7}  {'cf_rate':>8}  {'base_func':>9}  {'cf_Δ':>7}")
    for row in results:
        fc = f"{row['fc_rate']*100:.1f}%" if row["fc_rate"] is not None else "n/a"
        cf = f"{row['cf_rate']*100:.1f}%" if row["cf_rate"] is not None else "n/a"
        bc = f"{row['base_cont']*100:.1f}%" if row["base_cont"] is not None else "n/a"
        bf = f"{row['base_func']*100:.1f}%" if row["base_func"] is not None else "n/a"
        fd = f"{row['fc_delta']*100:+.1f}pp" if row["fc_delta"] is not None else "n/a"
        cd = f"{row['cf_delta']*100:+.1f}pp" if row["cf_delta"] is not None else "n/a"
        print(f"  {row['step']:5d}  {row['t_cur']:.1f}→{row['t_next']:.1f}          "
              f"{fc:>8}  {bc:>9}  {fd:>7}  {cf:>8}  {bf:>9}  {cd:>7}")


if __name__ == "__main__":
    main()
