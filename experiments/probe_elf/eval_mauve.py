"""
EXP-13 MAUVE Evaluation

Computes MAUVE scores for generated texts vs OpenWebText reference.
Focuses on coherent modes (sccfg3 standard ODE) and extra_denoise mode.

Usage (from ELF-torch root):
  python experiments/probe_elf/eval_mauve.py \
    --output_dir results/mauve_eval

MAUVE: measures distributional similarity between generated and reference texts
using GPT-2 feature space. Score 0–1 (higher = better).
"""

import argparse
import json
import os
import glob
import sys

import numpy as np


def load_jsonl_texts(path, max_samples=512):
    texts = []
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "all_generated_*.jsonl"))
        if not files:
            return []
        path = sorted(files)[0]
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d.get("generated", d.get("text", ""))
            if text.strip():
                texts.append(text)
            if len(texts) >= max_samples:
                break
    return texts


def get_reference_texts(max_samples=512):
    from datasets import load_dataset
    print("Loading OpenWebText reference texts...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True,
                      trust_remote_code=True)
    texts = []
    for item in ds:
        t = item.get("text", "")
        if len(t.strip()) < 100:
            continue
        texts.append(t[:2000])
        if len(texts) >= max_samples:
            break
    print(f"  Loaded {len(texts)} reference texts")
    return texts


def compute_mauve(generated_texts, reference_texts, featurize_model="gpt2-large",
                  device="cuda", max_len=256):
    import mauve
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"  Computing MAUVE (featurize_model={featurize_model}, max_len={max_len})...")
    result = mauve.compute_mauve(
        p_text=reference_texts,
        q_text=generated_texts,
        device_id=int(device.replace("cuda:", "")) if "cuda:" in device else 0,
        max_text_length=max_len,
        verbose=False,
        featurize_model_name=featurize_model,
        batch_size=8,
        num_buckets=500,
    )
    return result.mauve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/mauve_eval")
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Define which directories to evaluate
    base = "outputs"
    eval_configs = [
        # Production eval (standard sccfg3) — coherent text
        {
            "name": "baseline_ode32_sccfg3",
            "path": f"{base}/elf_b-owt-baseline-eval-pt-full/ode-steps32-cfg1-sccfg3-ts_uniform-uncond",
        },
        {
            "name": "baseline_ode16_sccfg3",
            "path": f"{base}/elf_b-owt-baseline-eval-pt-full/ode-steps16-cfg1-sccfg3-ts_uniform-uncond",
        },
        {
            "name": "kd_cr_ode8_sccfg3",
            "path": f"{base}/elf_b-owt-eval-sar-kd-cr/ode-steps8-cfg1-sccfg3-ts_uniform-uncond",
        },
        {
            "name": "kd_cr_ode16_sccfg3",
            "path": f"{base}/elf_b-owt-eval-sar-kd-cr/ode-steps16-cfg1-sccfg3-ts_uniform-uncond",
        },
        {
            "name": "kd2_ode32_sccfg3",
            "path": f"{base}/elf_b-owt-kd2-eval-pt-full/ode-steps32-cfg1-sccfg3-ts_uniform-uncond",
        },
        # EXP-13 variants — extra_denoise produces coherent text
        {
            "name": "kd2_ode32_extra_denoise",
            "path": f"{base}/exp13_kd2/ode-steps32-cfg1-ts_uniform-decsc_extra_denoise-uncond",
        },
        {
            "name": "baseline_ode32_extra_denoise",
            "path": f"{base}/exp13_baseline/ode-steps32-cfg1-ts_uniform-decsc_extra_denoise-uncond",
        },
        {
            "name": "kd_cr_ode32_extra_denoise",
            "path": f"{base}/exp13_kd_cr/ode-steps32-cfg1-ts_uniform-decsc_extra_denoise-uncond",
        },
    ]

    # Load reference texts once
    ref_texts = get_reference_texts(max_samples=args.max_samples)

    results = []
    for cfg in eval_configs:
        texts = load_jsonl_texts(cfg["path"], max_samples=args.max_samples)
        if not texts:
            print(f"  SKIP {cfg['name']}: no texts found at {cfg['path']}")
            continue
        print(f"\n[MAUVE] {cfg['name']}: {len(texts)} generated texts")
        print(f"  Sample: {repr(texts[0][:100])}")
        try:
            score = compute_mauve(texts, ref_texts, device=args.device, max_len=args.max_len)
            print(f"  MAUVE score: {score:.4f}")
            results.append({
                "name": cfg["name"],
                "path": cfg["path"],
                "n_generated": len(texts),
                "mauve": float(score),
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"name": cfg["name"], "path": cfg["path"], "error": str(e)})

    out_path = os.path.join(args.output_dir, "mauve_scores.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n\n=== MAUVE SUMMARY ===")
    for r in results:
        if "mauve" in r:
            print(f"  {r['name']:45s}: {r['mauve']:.4f}")
        else:
            print(f"  {r['name']:45s}: ERROR - {r.get('error', 'unknown')}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
