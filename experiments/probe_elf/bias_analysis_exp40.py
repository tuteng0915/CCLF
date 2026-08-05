"""
EXP-40: unembed_bias Vocabulary Analysis

Pure weight analysis — no GPU, no data loading.

The logit computed by the decode head is:
  logits_v = GELU(h @ proj_kernel + proj_bias) @ unembed_kernel + unembed_bias

The unembed_bias term shifts every token's logit by a fixed amount regardless of context.
If KD mostly changes unembed_bias (as seen in EXP-15v2 where R=2.59 for unembed_bias),
we can directly read off WHICH tokens are promoted/demoted.

Analysis:
  1. Δbias_kd_cr = unembed_bias_kd_cr - unembed_bias_baseline
  2. Δbias_kd2   = unembed_bias_kd2   - unembed_bias_baseline
  3. Top/bottom 50 tokens by Δbias for each checkpoint
  4. Non-ASCII (multilingual) fraction of top-promoted vs top-demoted tokens
  5. High-frequency vs low-frequency token distribution of Δbias

Requires: tokenizer vocab file or T5 tokenizer to decode token ids.

Output: results/exp40_bias_analysis/bias_analysis.json + bias_analysis_top50.txt

Usage (from ELF-torch root):
  conda run -n elf python experiments/probe_elf/bias_analysis_exp40.py
"""

import json
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

CHECKPOINTS = {
    "baseline": "converted/elf_b-owt-baseline_torch.pt",
    "kd_cr":    "converted/elf_b-owt-kd-cr_torch.pt",
    "kd2":      "converted/elf_b-owt-kd2_torch.pt",
}

TOP_K = 50


def load_bias(ckpt_path: str) -> torch.Tensor:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return raw["params"]["unembed_bias"].float()


def load_tokenizer():
    try:
        from transformers import T5Tokenizer, AutoTokenizer
        # Try T5 tokenizer used by ELF (same vocab as T5-base)
        try:
            tok = AutoTokenizer.from_pretrained("google/t5-v1_1-base")
        except Exception:
            tok = T5Tokenizer.from_pretrained("t5-base")
        return tok
    except ImportError:
        return None


def token_str(tok, idx: int) -> str:
    if tok is None:
        return f"<id={idx}>"
    try:
        s = tok.convert_ids_to_tokens([idx])[0]
        return s if s else f"<id={idx}>"
    except Exception:
        return f"<id={idx}>"


def is_non_ascii(s: str) -> bool:
    """Returns True if the token string contains non-ASCII characters."""
    try:
        s.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def analyze_delta(delta: torch.Tensor, tok, label: str, report_lines: list):
    """Analyze a delta-bias vector and append to report_lines."""
    V = delta.shape[0]
    delta_np = delta.numpy()

    report_lines.append(f"\n{'='*60}")
    report_lines.append(f"Δbias: {label}")
    report_lines.append(f"  mean={delta_np.mean():.4f}  std={delta_np.std():.4f}  "
                        f"max={delta_np.max():.4f}  min={delta_np.min():.4f}")

    top_idx   = np.argsort(delta_np)[::-1][:TOP_K]
    bot_idx   = np.argsort(delta_np)[:TOP_K]

    report_lines.append(f"\n  Top {TOP_K} promoted tokens (Δbias most positive):")
    non_ascii_top = 0
    for rank, idx in enumerate(top_idx):
        ts = token_str(tok, int(idx))
        na = is_non_ascii(ts)
        if na:
            non_ascii_top += 1
        report_lines.append(f"    #{rank+1:3d}  id={idx:6d}  Δ={delta_np[idx]:+.4f}  '{ts}'"
                             + ("  [non-ASCII]" if na else ""))

    report_lines.append(f"\n  Top {TOP_K} demoted tokens (Δbias most negative):")
    non_ascii_bot = 0
    for rank, idx in enumerate(bot_idx):
        ts = token_str(tok, int(idx))
        na = is_non_ascii(ts)
        if na:
            non_ascii_bot += 1
        report_lines.append(f"    #{rank+1:3d}  id={idx:6d}  Δ={delta_np[idx]:+.4f}  '{ts}'"
                             + ("  [non-ASCII]" if na else ""))

    # Non-ASCII fraction in top/bottom
    report_lines.append(f"\n  Non-ASCII fraction:")
    report_lines.append(f"    Top-{TOP_K} promoted: {non_ascii_top}/{TOP_K} = {non_ascii_top/TOP_K:.1%}")
    report_lines.append(f"    Top-{TOP_K} demoted:  {non_ascii_bot}/{TOP_K} = {non_ascii_bot/TOP_K:.1%}")

    # Distribution stats: what fraction of total variance is in top-K?
    total_var = (delta_np ** 2).sum()
    topk_var  = (delta_np[top_idx] ** 2).sum() + (delta_np[bot_idx] ** 2).sum()
    report_lines.append(f"\n  Top+Bot {TOP_K} token pairs capture "
                        f"{topk_var/total_var:.1%} of total squared Δbias")

    # Percentile analysis
    pcts = [50, 75, 90, 95, 99]
    abs_delta = np.abs(delta_np)
    for p in pcts:
        report_lines.append(f"  |Δbias| p{p}: {np.percentile(abs_delta, p):.4f}")

    return {
        "mean": float(delta_np.mean()),
        "std": float(delta_np.std()),
        "max": float(delta_np.max()),
        "min": float(delta_np.min()),
        "non_ascii_top50_frac": non_ascii_top / TOP_K,
        "non_ascii_bot50_frac": non_ascii_bot / TOP_K,
        "top50_ids": top_idx.tolist(),
        "top50_vals": delta_np[top_idx].tolist(),
        "bot50_ids": bot_idx.tolist(),
        "bot50_vals": delta_np[bot_idx].tolist(),
    }


def main():
    os.makedirs("results/exp40_bias_analysis", exist_ok=True)

    print("Loading biases...")
    biases = {name: load_bias(path) for name, path in CHECKPOINTS.items()}
    baseline_bias = biases["baseline"]

    print("Loading tokenizer...")
    tok = load_tokenizer()
    if tok is None:
        print("  (transformers not available, token ids only)")

    report_lines = ["EXP-40: unembed_bias Vocabulary Analysis", "=" * 60]

    # Absolute biases of each checkpoint
    report_lines.append("\n--- Absolute bias stats per checkpoint ---")
    for name, bias in biases.items():
        b = bias.numpy()
        report_lines.append(f"  {name}: mean={b.mean():.4f}  std={b.std():.4f}  "
                             f"L2={np.linalg.norm(b):.4f}")

    results = {}

    # Delta biases
    for name in ["kd_cr", "kd2"]:
        delta = biases[name] - baseline_bias
        label = f"{name} - baseline"
        stats = analyze_delta(delta, tok, label, report_lines)
        results[f"delta_{name}_vs_baseline"] = stats
        print("\n".join(report_lines[-30:]))

    # kd_cr vs kd2 delta
    delta_cr_vs_kd2 = biases["kd_cr"] - biases["kd2"]
    stats = analyze_delta(delta_cr_vs_kd2, tok, "kd_cr - kd2", report_lines)
    results["delta_kd_cr_vs_kd2"] = stats

    # Cosine similarity between the delta vectors
    d_cr  = (biases["kd_cr"] - baseline_bias).numpy()
    d_kd2 = (biases["kd2"]   - baseline_bias).numpy()
    cos_sim = np.dot(d_cr, d_kd2) / (np.linalg.norm(d_cr) * np.linalg.norm(d_kd2) + 1e-8)
    report_lines.append(f"\n--- Cosine similarity between Δbias_kd_cr and Δbias_kd2: {cos_sim:.4f} ---")
    results["cosine_sim_delta_cr_vs_kd2"] = float(cos_sim)
    print(f"\nCosine similarity Δkd_cr vs Δkd2: {cos_sim:.4f}")

    # Save full report
    report_path = "results/exp40_bias_analysis/bias_analysis_top50.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved report → {report_path}")

    # Save JSON summary
    json_path = "results/exp40_bias_analysis/bias_analysis.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON  → {json_path}")


if __name__ == "__main__":
    main()
