"""
EXP-13v2 Analysis: Compare decode (tmin=0.5) vs extra_denoise (tmin=0.5) vs none.

Reads generated .jsonl files, computes:
  1. Sample diversity check (are outputs identical across checkpoints?)
  2. First 3 samples from each condition for qualitative inspection
  3. PPL from metrics.jsonl if available
  4. Degeneration detection: fraction of outputs with >50% repeated bigrams

Usage (from ELF-torch root):
  python experiments/probe_elf/analyze_exp13v2.py
"""

import json
import os
import glob
import re
from collections import Counter


BASE = "outputs"
CHECKPOINTS = ["baseline", "kd_cr", "kd2"]
MODES = ["none", "decode-tmin0.5", "extra_denoise-tmin0.5", "decode_shuffled-tmin0.5", "random_residual-tmin0.5"]

# Maps mode label to directory suffix pattern
MODE_DIR_SUFFIX = {
    "none": "ode-steps32-cfg1-ts_uniform-uncond",
    "decode-tmin0.5": "ode-steps32-cfg1-ts_uniform-decsc_decode-tmin0.5-uncond",
    "extra_denoise-tmin0.5": "ode-steps32-cfg1-ts_uniform-decsc_extra_denoise-tmin0.5-uncond",
    "decode_shuffled-tmin0.5": "ode-steps32-cfg1-ts_uniform-decsc_decode_shuffled-tmin0.5-uncond",
    "random_residual-tmin0.5": "ode-steps32-cfg1-ts_uniform-decsc_random_residual-tmin0.5-uncond",
}


def load_texts(path, max_n=512):
    texts = []
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "all_generated_*.jsonl")))
        if not files:
            return []
        fpath = files[0]
    else:
        fpath = path
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get("generated", d.get("text", ""))
            if t.strip():
                texts.append(t)
            if len(texts) >= max_n:
                break
    return texts


def load_ppl(path):
    metrics_path = os.path.join(path, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        return None
    ppls = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "ppl" in d:
                ppls.append(d["ppl"])
            elif "gen_ppl" in d:
                ppls.append(d["gen_ppl"])
    if ppls:
        return sum(ppls) / len(ppls)
    return None


def bigram_repeat_fraction(text):
    """Fraction of bigrams that are repeated (high = degenerate)."""
    words = text.split()
    if len(words) < 4:
        return 0.0
    bigrams = list(zip(words[:-1], words[1:]))
    counts = Counter(bigrams)
    repeated = sum(v for v in counts.values() if v > 1)
    return repeated / len(bigrams)


def degeneration_rate(texts, thresh=0.4):
    """Fraction of texts where >thresh of bigrams are repeated."""
    if not texts:
        return float("nan")
    degen = sum(1 for t in texts if bigram_repeat_fraction(t) > thresh)
    return degen / len(texts)


def main():
    results = {}

    for ckpt in CHECKPOINTS:
        outdir = f"{BASE}/exp13v2_{ckpt}"
        results[ckpt] = {}
        for mode, suffix in MODE_DIR_SUFFIX.items():
            path = os.path.join(outdir, suffix)
            if not os.path.isdir(path):
                results[ckpt][mode] = {"status": "MISSING", "path": path}
                continue
            texts = load_texts(path)
            if not texts:
                results[ckpt][mode] = {"status": "EMPTY", "path": path}
                continue
            ppl = load_ppl(path)
            degen = degeneration_rate(texts)
            results[ckpt][mode] = {
                "status": "OK",
                "n": len(texts),
                "ppl": ppl,
                "degen_rate": degen,
                "samples": [t[:120] for t in texts[:3]],
            }

    # Print summary table
    print("\n" + "=" * 80)
    print("EXP-13v2 RESULTS SUMMARY (decode mode with tmin=0.5)")
    print("=" * 80)
    print(f"\n{'Mode':<30} {'n':>5} {'PPL':>10} {'Degen%':>8}  Sample")
    print("-" * 80)

    for ckpt in CHECKPOINTS:
        print(f"\n### {ckpt.upper()} (seed={'42' if ckpt=='baseline' else '123' if ckpt=='kd_cr' else '456'}) ###")
        for mode in MODES:
            r = results[ckpt].get(mode, {"status": "MISSING"})
            if r["status"] != "OK":
                print(f"  {mode:<28} {r['status']}")
                continue
            ppl_str = f"{r['ppl']:.1f}" if r["ppl"] is not None else "N/A"
            degen_str = f"{r['degen_rate']*100:.1f}%"
            sample = r["samples"][0][:80] if r["samples"] else ""
            print(f"  {mode:<28} {r['n']:>5} {ppl_str:>10} {degen_str:>8}  {sample!r}")

    # Cross-checkpoint identical text check
    print("\n\n### IDENTICAL TEXT CHECK (extra_denoise-tmin0.5) ###")
    extra_texts = {}
    for ckpt in CHECKPOINTS:
        r = results[ckpt].get("extra_denoise-tmin0.5", {})
        if r.get("status") == "OK":
            extra_texts[ckpt] = set(r["samples"])
    if len(extra_texts) >= 2:
        ckpts = list(extra_texts.keys())
        for i in range(len(ckpts)):
            for j in range(i + 1, len(ckpts)):
                a, b = ckpts[i], ckpts[j]
                shared = extra_texts[a] & extra_texts[b]
                print(f"  {a} vs {b}: {len(shared)}/{min(len(extra_texts[a]),len(extra_texts[b]))} identical first-3 samples")

    # Key comparison: decode vs extra_denoise
    print("\n### KEY COMPARISON: decode-tmin0.5 vs extra_denoise-tmin0.5 ###")
    for ckpt in CHECKPOINTS:
        dec = results[ckpt].get("decode-tmin0.5", {})
        ext = results[ckpt].get("extra_denoise-tmin0.5", {})
        if dec.get("status") == "OK" and ext.get("status") == "OK":
            dec_ppl = dec["ppl"] or float("nan")
            ext_ppl = ext["ppl"] or float("nan")
            ratio = ext_ppl / dec_ppl if dec_ppl > 0 else float("nan")
            verdict = "H0 (info)" if ratio > 2 else "H1 (compute)" if ratio < 1.5 else "ambiguous"
            print(f"  {ckpt}: decode PPL={dec_ppl:.1f} | extra_denoise PPL={ext_ppl:.1f} "
                  f"| ratio={ratio:.2f}x → {verdict}")
            print(f"    decode degen={dec['degen_rate']*100:.1f}% | extra_denoise degen={ext['degen_rate']*100:.1f}%")

    # Save JSON
    out_path = "results/exp13v2/analysis.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
