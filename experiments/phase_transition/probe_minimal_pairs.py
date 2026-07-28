"""EXP-PT8 (part 2): Controlled Minimal-Pair Evidence Sources.

IMPORTANT relation to the suite doc (see docs/specs/EXP-PT8-spec.md for full
discussion): doc's design wants a "cue" position (e.g. a plural subject) and
a separate "target" position (e.g. the verb) that differ from the cue, with
minimal pairs that swap ONLY the cue while holding the target's correct
identity fixed. BLiMP's pairs (used here, see build_minimal_pairs.py) differ
at exactly ONE token -- for subject-verb-agreement-style UIDs that one token
IS the target (the verb), while the cue (the subject noun) is identical in
both sentences. This means BLiMP pairs can't directly give a "cue-swapped,
target-held-fixed" triplet; only a "context-held-fixed test" is possible:

For each pair, run TWO fixed-noise oracle trajectories (same epsilon for
both, matching doc's "same fixed-noise oracle path" requirement):
  - "good" trajectory: x_clean = encode(good_ids) -- correct grammar
    throughout, including the target position.
  - "bad" trajectory: x_clean = encode(bad_ids) -- IDENTICAL context (same
    cue), but the target position's OWN noising target is the ungrammatical
    word.
At the critical (target) position, across t, track rank/margin of BOTH the
good word and the bad word in BOTH trajectories. The key comparison is
whether, in the "bad" trajectory, the grammatically-correct word (V_good)
stays anomalously competitive despite not being the nominal denoising
target -- i.e. whether the surrounding correct cue exerts a "grammatical
pull" even when the model is formally being asked to reconstruct the wrong
word.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n elf python \\
        experiments/phase_transition/probe_minimal_pairs.py \\
        --model elf --checkpoint baseline \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --pairs_json results/phase_transition/minimal_pairs_t5.json \\
        --out_dir results/phase_transition/elf/baseline --label full
"""

import argparse
import json
import sys
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
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--pairs_json", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--n_t_steps", type=int, default=9)
    p.add_argument("--t_min", type=float, default=0.05)
    p.add_argument("--t_max", type=float, default=0.95)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def rank_at_position(adapter, z, t, positions, target_ids, batch_size):
    out = adapter.forward_state(z.cpu(), None, t, batch_size=batch_size)
    logits = out["logits"]
    N = logits.shape[0]
    idx = torch.arange(N)
    logits_at_pos = logits[idx, positions, :]  # (N, V)
    probs = F.softmax(logits_at_pos.float(), dim=-1)
    target_probs = probs[idx, target_ids]
    rank = (probs > target_probs.unsqueeze(-1)).sum(-1)
    return rank.float(), target_probs


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    with open(args.pairs_json) as f:
        pairs_data = json.load(f)
    pairs = pairs_data["pairs"]
    seq_len_data = pairs_data["seq_len"]
    print(f"[PT8] Loaded {len(pairs)} minimal pairs (seq_len={seq_len_data})")

    print(f"[PT8] Loading {args.model} model...")
    if args.model == "elf":
        from adapters.elf_adapter import ELFAdapter
        assert args.config, "--config is required for --model elf"
        adapter = ELFAdapter.load(args.checkpoint, args.config, device)
        seq_len = adapter.seq_len
        assert seq_len == seq_len_data, (
            f"pairs were built with seq_len={seq_len_data}, but ELF needs seq_len={seq_len} "
            "(RoPE table is fixed-length) -- rebuild pairs with --seq_len matching config.max_length")
    else:
        from adapters.langflow_adapter import LangFlowAdapter
        adapter = LangFlowAdapter.load(args.checkpoint, device)
        seq_len = seq_len_data

    good_ids = torch.tensor([p["good_ids"] for p in pairs], dtype=torch.long)
    bad_ids = torch.tensor([p["bad_ids"] for p in pairs], dtype=torch.long)
    crit_pos = torch.tensor([p["critical_position"] for p in pairs], dtype=torch.long)
    good_target = good_ids[torch.arange(len(pairs)), crit_pos]
    bad_target = bad_ids[torch.arange(len(pairs)), crit_pos]
    N, L = good_ids.shape

    def encode_chunked(ids, attn_mask=None):
        # adapters' encode_clean doesn't internally chunk over N -- fine for
        # the modest batch sizes every other PT script uses, but at ELF's
        # seq_len=1024 a single T5-encoder call over all N=480 pairs OOMs
        # (hit this directly during pilot testing). Chunk it here instead of
        # touching the shared adapter method (used, validated, by every
        # other already-DONE PT run).
        chunks = []
        for i in range(0, ids.shape[0], args.batch_size):
            if attn_mask is not None:
                chunks.append(adapter.encode_clean(
                    ids[i:i + args.batch_size], attn_mask[i:i + args.batch_size]).cpu())
            else:
                chunks.append(adapter.encode_clean(ids[i:i + args.batch_size]).cpu())
        return torch.cat(chunks, dim=0)

    if args.model == "elf":
        attn_mask = (good_ids != adapter.tokenizer.pad_token_id).long()
        x_good = encode_chunked(good_ids, attn_mask)
        x_bad = encode_chunked(bad_ids, attn_mask)
    else:
        x_good = encode_chunked(good_ids)
        x_bad = encode_chunked(bad_ids)

    d = x_good.shape[-1]
    t_grid = np.linspace(args.t_min, args.t_max, args.n_t_steps)

    results = {"t": [], "rank_good_in_good_traj": [], "rank_good_in_bad_traj": [],
               "rank_bad_in_bad_traj": [], "prob_good_in_bad_traj": [], "prob_bad_in_bad_traj": []}

    for t in t_grid:
        t = float(t)
        eps = adapter.sample_epsilon((N, L, d))  # SAME epsilon reused for both trajectories
        z_good = adapter.make_oracle_state(x_good.to(device), eps, t)
        z_bad = adapter.make_oracle_state(x_bad.to(device), eps, t)

        rank_good_good, _ = rank_at_position(adapter, z_good, t, crit_pos, good_target, args.batch_size)
        rank_good_bad, prob_good_bad = rank_at_position(adapter, z_bad, t, crit_pos, good_target, args.batch_size)
        rank_bad_bad, prob_bad_bad = rank_at_position(adapter, z_bad, t, crit_pos, bad_target, args.batch_size)

        results["t"].append(t)
        results["rank_good_in_good_traj"].append(rank_good_good.mean().item())
        results["rank_good_in_bad_traj"].append(rank_good_bad.mean().item())
        results["rank_bad_in_bad_traj"].append(rank_bad_bad.mean().item())
        results["prob_good_in_bad_traj"].append(prob_good_bad.mean().item())
        results["prob_bad_in_bad_traj"].append(prob_bad_bad.mean().item())

        print(f"  t={t:.3f}  rank(good|good_ctx)={rank_good_good.mean().item():7.1f}  "
              f"rank(good|bad_ctx)={rank_good_bad.mean().item():7.1f}  "
              f"rank(bad|bad_ctx)={rank_bad_bad.mean().item():7.1f}  "
              f"P(good|bad_ctx)={prob_good_bad.mean().item():.4f}  "
              f"P(bad|bad_ctx)={prob_bad_bad.mean().item():.4f}")

    # Per-UID breakdown at the final t (does the "grammatical pull" effect
    # concentrate in some linguistic categories more than others?).
    uids = [p["uid"] for p in pairs]
    eps = adapter.sample_epsilon((N, L, d))
    z_bad_final = adapter.make_oracle_state(x_bad.to(device), eps, args.t_max)
    rank_good_bad_final, prob_good_bad_final = rank_at_position(
        adapter, z_bad_final, args.t_max, crit_pos, good_target, args.batch_size)
    rank_bad_bad_final, prob_bad_bad_final = rank_at_position(
        adapter, z_bad_final, args.t_max, crit_pos, bad_target, args.batch_size)
    by_uid = {}
    for uid in set(uids):
        mask = torch.tensor([u == uid for u in uids])
        by_uid[uid] = {
            "n": int(mask.sum()),
            "rank_good_in_bad_traj": float(rank_good_bad_final[mask].mean()),
            "rank_bad_in_bad_traj": float(rank_bad_bad_final[mask].mean()),
            "prob_good_in_bad_traj": float(prob_good_bad_final[mask].mean()),
            "prob_bad_in_bad_traj": float(prob_bad_bad_final[mask].mean()),
        }

    # Per-pair raw arrays at t_max, for post-hoc bootstrap CI (rigor-audit
    # follow-up, same "free win" pattern as PT1/PT2/PT3 -- resampling unit
    # here is the pair, since each BLiMP pair is an independent sentence,
    # not a shared sequence like the other PT scripts).
    npz_path = out_dir / f"minimal_pairs_raw_{args.label}.npz"
    np.savez_compressed(
        npz_path,
        uids=np.asarray(uids),
        rank_good_in_bad_traj=rank_good_bad_final.numpy(),
        rank_bad_in_bad_traj=rank_bad_bad_final.numpy(),
        prob_good_in_bad_traj=prob_good_bad_final.numpy(),
        prob_bad_in_bad_traj=prob_bad_bad_final.numpy(),
    )
    print(f"[PT8] Saved per-pair raw arrays to {npz_path}")

    summary = {
        "model": args.model, "checkpoint": args.checkpoint, "label": args.label,
        "n_pairs": N, "t_grid": t_grid.tolist(), "results": results, "by_uid": by_uid,
        "notes": [
            "BLiMP pairs differ at exactly one token (the grammatical target itself), "
            "so this tests 'does correct context pull evidence toward the grammatical word "
            "even when the model is asked to reconstruct the ungrammatical one', NOT doc's "
            "literal 'swap only the cue, target position fixed' design -- see script docstring.",
            "Fixed-noise: the SAME epsilon draw is reused for the good and bad trajectory at "
            "each t, matching doc's protocol.",
        ],
    }
    json_path = out_dir / f"minimal_pairs_probe_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[PT8] Per-UID breakdown at t={args.t_max}:")
    for uid, s in by_uid.items():
        print(f"  {uid} (n={s['n']}): rank(good|bad_ctx)={s['rank_good_in_bad_traj']:.1f}  "
              f"rank(bad|bad_ctx)={s['rank_bad_in_bad_traj']:.1f}")
    print(f"[PT8] Saved {json_path}")


if __name__ == "__main__":
    main()
