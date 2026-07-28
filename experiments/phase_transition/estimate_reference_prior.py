"""EXP-PT1 (part 1): estimate three reference priors and the real oracle
posterior at each t, saving compact per-position/per-t metrics for later
prior-subtraction analysis (analyze_prior_subtraction.py).

References (see docs/specs/EXP-PT1-spec.md for exact definitions/simplifications):
  A. gauss   -- per-channel Gaussian matched to the real z_t mean/variance at
                that t (diagonal covariance), Monte-Carlo averaged over softmax.
  B. swap    -- cross-sequence state swap (derangement over the batch axis).
  C. shuffle -- global per-sequence position shuffle (derangement over L).
               NOTE: this is an approximation of "shuffle non-target positions
               only" -- see spec section 1, Reference C, for why.

We do NOT persist full (N, L, V) distributions to disk (V=32k-50k makes that
scale to 100+GB at the full 512-sample protocol). Instead we save, per t,
per-position compact arrays (rank, margin, top-1 id, KL) -- everything
analyze_prior_subtraction.py's decision rules (section 3.4/3.5 of the suite
doc) actually need.

Usage (ELF):
    cd /home/wjzhang/tt_workspace/model/CCLF/CCLF
    CUDA_VISIBLE_DEVICES=6 conda run -n elf python \\
        experiments/phase_transition/estimate_reference_prior.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/phase_transition/elf/baseline \\
        --n_samples 16 --n_t_steps 6 --label pilot

Usage (LangFlow):
    CUDA_VISIBLE_DEVICES=6 conda run -n elf python \\
        experiments/phase_transition/estimate_reference_prior.py \\
        --model langflow --out_dir results/phase_transition/langflow/owt \\
        --n_samples 16 --n_t_steps 6 --label pilot
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "langflow"], required=True)
    p.add_argument("--checkpoint", default="baseline", help="ELF: checkpoint key/path. LangFlow: HF repo id.")
    p.add_argument("--config", default=None, help="ELF only: path to training-config yaml.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_t_steps", type=int, default=6)
    p.add_argument("--t_min", type=float, default=0.05)
    p.add_argument("--t_max", type=float, default=0.95)
    p.add_argument("--n_oracle", type=int, default=3, help="noise seeds for the real oracle posterior")
    p.add_argument("--n_gauss", type=int, default=4, help="noise seeds for Reference A (Gaussian)")
    p.add_argument("--n_swap", type=int, default=4, help="permutation seeds for Reference B (cross-seq swap)")
    p.add_argument("--n_shuffle", type=int, default=4, help="permutation seeds for Reference C (context shuffle)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def derangement(n, generator):
    """Random permutation of range(n) with no fixed points (n>=2)."""
    if n < 2:
        return torch.arange(n)
    while True:
        perm = torch.randperm(n, generator=generator)
        if (perm != torch.arange(n)).all():
            return perm


@torch.no_grad()
def backbone_probs(adapter, z, t, batch_size):
    """Return softmax(logits) (N, L, V) for a full batch, chunked."""
    out = adapter.forward_state(z.cpu(), None, t, batch_size=batch_size)
    return F.softmax(out["logits"], dim=-1)


def average_probs_over_seeds(adapter, make_state_fn, n_seeds, t, batch_size):
    """make_state_fn(seed_idx) -> z (N,L,d) tensor. Averages softmax(logits)
    over seeds (correct Jensen treatment, matches EXP-05v3)."""
    probs_sum = None
    for s in range(n_seeds):
        z = make_state_fn(s)
        probs = backbone_probs(adapter, z, t, batch_size)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / n_seeds


def rank_of_gt(probs, gt_ids):
    """0-indexed rank of the ground-truth token in `probs` (N,L,V) -> (N,L) long.

    Implemented as a strict-greater-than count rather than a full argsort:
    equivalent for float32 probabilities (ties are measure-zero) and avoids
    an O(N*L*V log V) sort + an O(N*L*V) nonzero() scan over the whole
    tensor -- both were the dominant cost of every PT1/PT2/PT5 script before
    this fix (confirmed via wall-clock: this is why the first full-scale
    ELF runs at seq_len=1024 were much slower than their LangFlow
    counterparts at seq_len=128, well beyond what the forward-pass FLOPs
    difference alone would predict)."""
    gt_vals = probs.gather(-1, gt_ids.unsqueeze(-1).to(probs.device))
    rank = (probs > gt_vals).sum(-1)
    return rank.long()


def logit_of_gt(log_probs, gt_ids):
    """log_probs (N,L,V), gt_ids (N,L) -> (N,L) gathered value."""
    return log_probs.gather(-1, gt_ids.unsqueeze(-1)).squeeze(-1)


def kl_per_position(p, q):
    """p, q: (N,L,V) -> (N,L) KL(p||q)."""
    return (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(-1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()
    print(f"[PT1] Loading {args.model} model...")
    if args.model == "elf":
        from adapters.elf_adapter import ELFAdapter
        assert args.config, "--config is required for --model elf"
        adapter = ELFAdapter.load(args.checkpoint, args.config, device)
        # ELF's RoPE table is precomputed for config.max_length and does NOT
        # support arbitrary shorter sequences (no runtime slicing) -- every
        # existing ELF probe script (EXP-01v3/05v3/etc.) therefore always
        # uses seq_len == config.max_length. --seq_len is ignored here; the
        # suite doc itself allows 1024 as the "ELF-only replication" length
        # (section 2), which is what config.max_length already is.
        if args.seq_len != adapter.seq_len:
            print(f"[PT1] NOTE: --seq_len {args.seq_len} ignored for ELF; "
                  f"using config.max_length={adapter.seq_len} (RoPE table is fixed-length).")
        ids, mask = adapter.load_owt_sequences(args.n_samples, seq_len=adapter.seq_len)
        x_clean = adapter.encode_clean(ids, mask).cpu()
        gt_ids = ids
    else:
        from adapters.langflow_adapter import LangFlowAdapter
        adapter = LangFlowAdapter.load(args.checkpoint, device)
        ids, mask, x_clean = adapter.load_owt_sequences(args.n_samples, seq_len=args.seq_len)
        gt_ids = ids

    N, L, d = x_clean.shape
    print(f"[PT1] {N} sequences, L={L}, d={d}. Loaded in {time.time()-t0:.1f}s")

    t_grid = np.linspace(args.t_min, args.t_max, args.n_t_steps)
    ref_names = ["gauss", "swap", "shuffle"]

    def compute_t_record(t, f_i=None):
        """Runs oracle + all three references at time t. If f_i (N,L) int
        array is given, also gathers raw/residual log-values at that fixed
        default-competitor token (needed for m_raw/m_res margins)."""
        t = float(t)

        def make_oracle(s, t=t):
            eps = adapter.sample_epsilon((N, L, d))
            return adapter.make_oracle_state(x_clean.to(device), eps, t)

        p_probs = average_probs_over_seeds(adapter, make_oracle, args.n_oracle, t, args.batch_size)
        log_p = torch.log(p_probs + 1e-12)

        with torch.no_grad():
            eps0 = adapter.sample_epsilon((N, L, d))
            z_real_sample = adapter.make_oracle_state(x_clean.to(device), eps0, t).cpu()
            mu_t = z_real_sample.mean(dim=(0, 1))
            std_t = (z_real_sample.var(dim=(0, 1)) + 1e-8).sqrt()

        def make_gauss(s, mu=mu_t, std=std_t):
            noise = torch.randn(N, L, d, generator=gen)
            return (mu + std * noise).to(device)

        def make_swap(s, t=t):
            eps = adapter.sample_epsilon((N, L, d))
            z_real = adapter.make_oracle_state(x_clean.to(device), eps, t)
            perm = derangement(N, gen)
            return z_real[perm]

        def make_shuffle(s, t=t):
            eps = adapter.sample_epsilon((N, L, d))
            z_real = adapter.make_oracle_state(x_clean.to(device), eps, t)
            perm_l = derangement(L, gen)
            return z_real[:, perm_l, :]

        q_probs = {
            "gauss": average_probs_over_seeds(adapter, make_gauss, args.n_gauss, t, args.batch_size),
            "swap": average_probs_over_seeds(adapter, make_swap, args.n_swap, t, args.batch_size),
            "shuffle": average_probs_over_seeds(adapter, make_shuffle, args.n_shuffle, t, args.batch_size),
        }

        rank_raw = rank_of_gt(p_probs, gt_ids)
        raw_top1 = p_probs.argmax(-1)
        ell_gt = log_p.gather(-1, gt_ids.unsqueeze(-1)).squeeze(-1)          # (N,L)

        record = {
            "t": t,
            "rank_raw": rank_raw.numpy().astype(np.int32),
            "raw_top1": raw_top1.numpy().astype(np.int32),
            "ell_gt": ell_gt.numpy().astype(np.float32),
            "G_oracle": (raw_top1 == gt_ids).float().mean().item(),
        }
        if f_i is not None:
            f_i_t = torch.from_numpy(f_i).long()
            record["ell_f"] = log_p.gather(-1, f_i_t.unsqueeze(-1)).squeeze(-1).numpy().astype(np.float32)

        for name in ref_names:
            q = q_probs[name]
            log_q = torch.log(q + 1e-12)
            e = log_p - log_q                       # residual "logit" (N,L,V)
            residual_top1 = e.argmax(-1)
            rank_res = rank_of_gt(e, gt_ids)
            e_gt = e.gather(-1, gt_ids.unsqueeze(-1)).squeeze(-1)
            kl_pq = kl_per_position(p_probs, q).mean().item()

            record[f"rank_res_{name}"] = rank_res.numpy().astype(np.int32)
            record[f"residual_top1_{name}"] = residual_top1.numpy().astype(np.int32)
            record[f"e_gt_{name}"] = e_gt.numpy().astype(np.float32)
            record[f"G_{name}"] = (q.argmax(-1) == gt_ids).float().mean().item()
            record[f"KL_{name}"] = kl_pq
            if f_i is not None:
                f_i_t = torch.from_numpy(f_i).long()
                record[f"e_f_{name}"] = e.gather(-1, f_i_t.unsqueeze(-1)).squeeze(-1).numpy().astype(np.float32)

        return record

    # Pre-pass: define the default competitor f_i as the native top-1 at the
    # smallest t in the grid (PT2's "Earliest-time native top-1" definition,
    # reused here so PT1/PT2 share one convention -- see spec section 2).
    print("[PT1] Pre-pass at t_min to fix default competitor f_i...")
    f_i = compute_t_record(t_grid[0])["raw_top1"]

    per_t_records = []
    print(f"\n{'t':>6} | {'G_oracle':>9} | {'G_gauss':>8} | {'G_swap':>8} | {'G_shuf':>8} | "
          f"{'rank_or':>8} | {'KL_gauss':>9} | {'KL_swap':>9} | {'KL_shuf':>9}")
    print("-" * 100)

    for t in t_grid:
        record = compute_t_record(t, f_i=f_i)
        per_t_records.append(record)
        print(f"{record['t']:6.3f} | {record['G_oracle']:9.4f} | {record['G_gauss']:8.4f} | "
              f"{record['G_swap']:8.4f} | {record['G_shuffle']:8.4f} | "
              f"{record['rank_raw'].mean():8.2f} | {record['KL_gauss']:9.4f} | "
              f"{record['KL_swap']:9.4f} | {record['KL_shuffle']:9.4f}")

    # Null-mode token: most frequent ground-truth token overall, EXCLUDING
    # padding positions (doc section 2: "Exclude padding. Analyze special
    # tokens separately rather than mixing them into lexical results.").
    # Bug found post-hoc (see EXP-PT1-spec.md): without this, ELF's most
    # frequent token was literally the T5 </s>/pad id (5.3% of positions in
    # a 1024-token, mostly-padded corpus), contaminating
    # frac_null_mode_but_residual_specific with a pad-vs-non-pad signal
    # instead of the intended "high-frequency real word" signal.
    mask_flat = mask.numpy().reshape(-1).astype(bool)
    gt_flat = gt_ids.numpy().reshape(-1)[mask_flat]
    vals, counts = np.unique(gt_flat, return_counts=True)
    null_mode_token = int(vals[np.argmax(counts)])

    npz_path = out_dir / f"prior_subtraction_raw_{args.label}.npz"
    save_dict = {"t_grid": t_grid, "gt_ids": gt_ids.numpy().astype(np.int32),
                 "f_i": f_i, "null_mode_token": null_mode_token,
                 "mask": mask.numpy().astype(np.int32)}
    for i, rec in enumerate(per_t_records):
        for k, v in rec.items():
            save_dict[f"t{i}_{k}"] = v if isinstance(v, np.ndarray) else np.asarray(v)
    np.savez_compressed(npz_path, **save_dict)

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_samples": N, "seq_len": L, "d_model": d, "t_grid": t_grid.tolist(),
        "n_oracle": args.n_oracle, "n_gauss": args.n_gauss,
        "n_swap": args.n_swap, "n_shuffle": args.n_shuffle,
        "null_mode_token": null_mode_token,
        "per_t_scalars": [
            {k: v for k, v in rec.items() if not isinstance(v, np.ndarray)}
            for rec in per_t_records
        ],
        "raw_npz": str(npz_path),
    }
    json_path = out_dir / f"prior_reference_summary_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[PT1] Saved compact per-position arrays to {npz_path}")
    print(f"[PT1] Saved summary to {json_path}")
    print(f"[PT1] Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
