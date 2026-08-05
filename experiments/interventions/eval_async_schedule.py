"""EXP-GS19: Order-Controlled Asynchronous Denoising Ablation.

If synchronous (single global scalar time) denoising creates a coordination
bottleneck, does giving each position its own LOCAL progress help, and if
so, is it because of linguistic direction (left-to-right), symmetry breaking
(any non-synchronous order), or reliable anchors (confidence-adaptive)? See
docs/specs/EXP-GS19-spec.md.

This is explicitly a HETEROGENEOUS-STATE INTERVENTION with train-test
mismatch, not a native Wavefront-Flow-Forcing sampler: neither backbone ever
saw a single scalar time per forward call during training. At each global
step s -> s_next, we (1) run the ordinary scalar-time model call to get
xhat and estimate the implied noise eps_hat = (z - alpha(s)*xhat)/sigma(s),
then (2) reconstruct EACH position at its own scheduled local progress tau_i
using that SAME (xhat_i, eps_hat_i) pair evaluated at alpha(tau_i)/sigma(tau_i)
(verified analytically and by a unit test that this reduces EXACTLY to the
ordinary solver step when tau_i=s_next for every position, i.e. Delta=0
recovers the synchronous baseline for free). `noise_params()` dispatches the
(alpha,sigma) schedule by `adapter.name` -- ELF's flow-matching schedule is
alpha=t, sigma=1-t; Plaid's VDM schedule uses PlaidAdapter's own gamma/
noise_schedule modules (verified: `noise_params` on Plaid matches
`adapter._alpha_sigma` exactly, since both come from the same underlying
gamma computation).

Implementation notes / deviations from the spec:
  - ELF and Plaid supported (spec: "other models only after a native
    intervention is defined" -- LangFlow not yet added, would need a
    `noise_params` branch using its own native_logsnr-derived alpha/sigma,
    same pattern as Plaid).
  - Orderings implemented: synchronous, left-to-right (LTR), right-to-left
    (RTL), fixed-random, confidence-adaptive. NOT implemented for this pilot:
    reversed-random, block-random, and the separate norm-matched random
    state perturbation control -- deferred, noted below.
  - Single Delta=0.20 (spec pilot asks {0.10, 0.20}).
  - Gen.PPL reuses this repo's own established convention (gpt2-large
    reference LM via models/ELF-torch/src/utils/metrics_utils.py:Metrics,
    the same machinery behind every other "real PPL" number in this repo,
    e.g. EXP-37/48/54).
  - distinct-n and repetition-rate are simple n-gram implementations (no
    existing shared helper in this repo beyond the pilot 4-gram
    repetition_rate() in analyze_global_failures.py, reused here).
  - Endpoint-specificity/velocity from GS16/GS17 are NOT computed per arm in
    this pilot (would require building a fresh calibrated endpoint bank per
    arm x Delta combination -- deferred to a formal-scale follow-up).
  - Pilot n_samples well below the spec's own pilot scale (256 paired x 3
    seeds) -- see --n_samples default.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n elf python \\
        experiments/interventions/eval_async_schedule.py \\
        --config models/ELF-torch/src/configs/training_configs/eval_exp37c_baseline.yml \\
        --out_dir results/global_state/elf/baseline --n_samples 32 --label pilot
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).parent
_GS_DIR = _THIS_DIR.parent / "global_state"
_PT_DIR = _THIS_DIR.parent / "phase_transition"
for p in (_THIS_DIR, _GS_DIR, _PT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import decode_text, load_adapter  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["elf", "plaid"], default="elf")
    p.add_argument("--checkpoint", default="baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default="pilot")
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--deltas", type=float, nargs="+", default=[0.20])
    p.add_argument("--orderings", type=str, nargs="+",
                    default=["synchronous", "ltr", "rtl", "fixed_random", "confidence_adaptive"])
    p.add_argument("--n_async_steps", type=int, default=28)
    p.add_argument("--n_sync_refine", type=int, default=4)
    p.add_argument("--ppl_model", default="gpt2-large")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def repetition_rate(token_ids, n=4):
    ids = list(token_ids)
    if len(ids) < n + 1:
        return 0.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def distinct_n(texts, n=2):
    grams = set()
    total = 0
    for t in texts:
        words = t.split()
        g = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        grams.update(g)
        total += len(g)
    return float(len(grams) / total) if total > 0 else 0.0


def noise_params(adapter, t):
    """(alpha, sigma) at time t, adapter convention (0=noisiest,1=clean), such
    that z_t = alpha*x_clean + sigma*eps -- generalizes the reconstruction
    trick across noising conventions. ELF's flow-matching schedule is
    alpha=t, sigma=1-t (clamped by t_eps, matching adapter.solver_step's own
    internal denom); Plaid (and any other VDM-parameterized adapter, e.g.
    LangFlow) uses PlaidAdapter._alpha_sigma via native_logsnr. Dispatches on
    adapter.name rather than modifying the shared adapter classes."""
    if getattr(adapter, "name", None) == "elf":
        alpha = float(t)
        sigma = max(1.0 - float(t), adapter.t_eps)
        return alpha, sigma
    if getattr(adapter, "name", None) == "plaid":
        alpha, sigma, _ = adapter._alpha_sigma(t)
        return float(alpha), float(sigma)
    raise NotImplementedError(f"noise_params not implemented for adapter '{adapter.name}'")


def rank_for_ordering(ordering, L, rng, confidence=None):
    """Returns q: (L,) int array, rank 0 = 'gets to advance fastest'."""
    if ordering == "synchronous":
        return np.zeros(L, dtype=np.float64)  # unused when delta effectively 0 for this arm
    if ordering == "ltr":
        return np.arange(L, dtype=np.float64)
    if ordering == "rtl":
        return np.arange(L - 1, -1, -1, dtype=np.float64)
    if ordering == "fixed_random":
        return rng.permutation(L).astype(np.float64)
    if ordering == "confidence_adaptive":
        # lower entropy = more confident = smaller rank (advances faster)
        order = np.argsort(confidence)  # ascending entropy
        q = np.empty(L, dtype=np.float64)
        q[order] = np.arange(L)
        return q
    raise ValueError(ordering)


@torch.no_grad()
def run_arm(adapter, eps, ordering, delta, n_async_steps, n_sync_refine, batch_size,
            device, seed):
    N, L, d = eps.shape
    t_start = adapter.t_eps
    t_end_async = 0.99  # global schedule target before the sync refinement tail
    s_grid = np.linspace(t_start, t_end_async, n_async_steps + 1).tolist()
    rng = np.random.RandomState(seed)

    z = eps.to(device)
    sc = torch.zeros_like(z)
    fixed_q = None
    if ordering == "fixed_random":
        fixed_q = np.stack([rank_for_ordering(ordering, L, rng) for _ in range(N)], axis=0)

    top1_history = []  # list of (N,L) per async step, for revision-count / tau_first-stable
    for k in range(n_async_steps):
        s, s_next = float(s_grid[k]), float(s_grid[k + 1])
        # xhat MUST come from the same self-conditioning machinery the real
        # sampler uses (adapter.solver_step -> _ode_step -> _forward_sample),
        # NOT adapter.forward_state -- forward_state's self-conditioning path
        # is a simpler always-deterministic-SC readout used elsewhere in this
        # repo purely to probe an already-computed state, and differs from
        # the sampling path's self_cond_prob-mixed forward pass. Verified
        # numerically: using forward_state's xhat here made the
        # delta=0/synchronous arm match a plain solver_step rollout only
        # ~8% of tokens (should be ~100%); switching to solver_step's own
        # xhat fixes this (see scratchpad/verify_sync_equiv.py in this
        # session's history).
        z_next_ordinary, xhat = adapter.solver_step(z, sc, s, s_next)
        # separate read-out call (logits) for diagnostics only (top1 tracking,
        # confidence-adaptive ranking) -- matches every other GS15-18 script's
        # convention of using forward_state to read out an already-reached state.
        out = adapter.forward_state(z, sc, s, batch_size=batch_size)
        top1_history.append(out["logits"].argmax(-1).cpu())

        if ordering == "synchronous" or delta == 0.0:
            z = z_next_ordinary
            sc = xhat
            continue

        alpha_s, sigma_s = noise_params(adapter, s)
        eps_hat = (z - alpha_s * xhat) / sigma_s

        q = np.zeros((N, L), dtype=np.float64)
        for n in range(N):
            if ordering == "confidence_adaptive":
                probs = torch.softmax(out["logits"][n].float(), dim=-1)
                ent = (-(probs * torch.log(probs + 1e-12)).sum(-1)).numpy()
                q[n] = rank_for_ordering(ordering, L, rng, confidence=ent)
            elif ordering == "fixed_random":
                q[n] = fixed_q[n]
            else:
                q[n] = rank_for_ordering(ordering, L, rng)
        rank_term = 1.0 - 2.0 * q / max(L - 1, 1)  # (N,L) in [-1,1]
        tau_np = np.clip(s_next + delta * rank_term, 0.0, 1.0)
        tau = torch.from_numpy(tau_np).float().to(device).unsqueeze(-1)

        # per-position reconstruction at local progress tau, using the SAME
        # (xhat, eps_hat) pair estimated at the current global step s -- now
        # via the adapter-appropriate (alpha,sigma) schedule instead of the
        # ELF-specific linear (tau, 1-tau) formula. For a given adapter,
        # alpha/sigma are scalars (shared across positions/batch) but tau
        # varies per position, so this still has to be computed per-position
        # via alpha(tau)/sigma(tau); since noise_params is schedule-only
        # (no state dependence) we evaluate it once per unique tau by just
        # calling it elementwise through numpy/torch broadcasting where the
        # schedule is affine in the ELF case and via the VDM closed form for
        # Plaid (both are pointwise functions of tau, safe to vectorize).
        if getattr(adapter, "name", None) == "elf":
            alpha_tau = tau
            sigma_tau = (1.0 - tau).clamp(min=adapter.t_eps)
        elif getattr(adapter, "name", None) == "plaid":
            gamma_0, gamma_1 = adapter.modules["gamma_bounds"]()
            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float64)
            try:
                t_native = 1.0 - tau.double()
                g_tilde = adapter.modules["noise_schedule"](t_native.reshape(-1))
                gamma_tau = (gamma_0 + (gamma_1 - gamma_0) * g_tilde).reshape(tau.shape)
                alpha_tau = torch.sigmoid(-gamma_tau).sqrt().float()
                sigma_tau = torch.sigmoid(gamma_tau).sqrt().float()
            finally:
                torch.set_default_dtype(prev_dtype)
        else:
            raise NotImplementedError(adapter.name)

        z = alpha_tau * xhat + sigma_tau * eps_hat
        sc = xhat

    # final synchronous refinement steps (all arms identical from here)
    s_refine = np.linspace(t_end_async, 0.999, n_sync_refine + 1).tolist()
    for k in range(n_sync_refine):
        z, sc = adapter.solver_step(z, sc, s_refine[k], s_refine[k + 1])
        out = adapter.forward_state(z, sc, s_refine[k + 1], batch_size=batch_size)
        top1_history.append(out["logits"].argmax(-1).cpu())

    final_logits = adapter.forward_state(z, sc, s_refine[-1], batch_size=batch_size)["logits"]
    final_top1 = final_logits.argmax(-1).cpu()
    return final_top1, torch.stack(top1_history, dim=0)  # (S,N,L)


def timeline_stats(top1_hist, final_top1):
    S, N, L = top1_hist.shape
    match = (top1_hist == final_top1.unsqueeze(0)).numpy()
    tau_first_idx = np.full((N, L), -1)
    tau_stable_idx = np.full((N, L), -1)
    for n in range(N):
        for i in range(L):
            if match[:, n, i].any():
                tau_first_idx[n, i] = int(np.argmax(match[:, n, i]))
            for s in range(S):
                if match[s:, n, i].all():
                    tau_stable_idx[n, i] = s
                    break
    revisions = np.zeros((N, L), dtype=np.int64)
    for n in range(N):
        for i in range(L):
            seq = top1_hist[:, n, i].numpy()
            revisions[n, i] = int((seq[1:] != seq[:-1]).sum())
    never_stable_frac = float(np.mean(tau_stable_idx < 0))
    return {
        "mean_tau_first_step": float(np.mean(tau_first_idx[tau_first_idx >= 0])),
        "mean_tau_stable_step": float(np.mean(tau_stable_idx[tau_stable_idx >= 0])),
        "never_stable_frac": never_stable_frac,
        "mean_revisions": float(revisions.mean()),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[GS19] Loading {args.model} model...")
    adapter = load_adapter(args.model, args.checkpoint, args.config, device)
    tokenizer = adapter.tokenizer

    N, L, d = args.n_samples, adapter.seq_len, adapter.d_model
    print(f"[GS19] N={N}, L={L}, orderings={args.orderings}, deltas={args.deltas}")

    print(f"[GS19] loading PPL reference model ({args.ppl_model})...")
    sys.path.insert(0, str(Path("models/ELF-torch/src").resolve()))
    from utils.metrics_utils import Metrics
    metrics = Metrics(gen_ppl_eval_model_name_or_path=args.ppl_model,
                       eval_ppl_batch_size=16, eval_context_size=L)

    eps = adapter.sample_epsilon((N, L, d), generator=torch.Generator(device=device).manual_seed(args.seed))
    mask_full = torch.ones(L, dtype=torch.long)

    records = []
    for ordering in args.orderings:
        deltas_to_run = [0.0] if ordering == "synchronous" else args.deltas
        for delta in deltas_to_run:
            print(f"[GS19] running arm ordering={ordering} delta={delta:g} ...")
            final_top1, top1_hist = run_arm(
                adapter, eps, ordering, delta, args.n_async_steps, args.n_sync_refine,
                args.batch_size, device, args.seed + 7)

            texts = [decode_text(tokenizer, final_top1[n], mask_full) for n in range(N)]
            rep_rates = [repetition_rate(final_top1[n].tolist()) for n in range(N)]
            degenerate_frac = float(np.mean([r > 0.3 for r in rep_rates]))
            ppl_result = metrics.record_generative_perplexity(
                text_samples=texts, max_length=L, retokenize=True)
            tstats = timeline_stats(top1_hist, final_top1)

            rec = {
                "ordering": ordering, "delta": delta,
                "gen_ppl": float(ppl_result["ppl"]),
                "mean_repetition_rate": float(np.mean(rep_rates)),
                "degenerate_frac": degenerate_frac,
                "distinct_2": distinct_n(texts, 2), "distinct_3": distinct_n(texts, 3),
                **tstats,
            }
            records.append(rec)
            print(f"  [GS19] {ordering:20s} delta={delta:g}  gen_ppl={rec['gen_ppl']:.1f}  "
                  f"rep_rate={rec['mean_repetition_rate']:.3f}  degen={degenerate_frac:.2f}  "
                  f"tau_first_step={rec['mean_tau_first_step']:.2f}  "
                  f"tau_stable_step={rec['mean_tau_stable_step']:.2f}  "
                  f"never_stable={rec['never_stable_frac']:.3f}  "
                  f"revisions={rec['mean_revisions']:.2f}")

    summary = {
        "checkpoint": args.checkpoint, "label": args.label, "n_samples": N,
        "orderings": args.orderings, "deltas": args.deltas,
        "n_async_steps": args.n_async_steps, "n_sync_refine": args.n_sync_refine,
        "ppl_model": args.ppl_model, "records": records,
        "notes": [
            "Heterogeneous-state intervention (train-test mismatch), not a native "
            "WFF sampler -- see module docstring.",
            "Orderings implemented: synchronous/ltr/rtl/fixed_random/"
            "confidence_adaptive. NOT implemented: reversed_random, block_random, "
            "norm-matched random perturbation control.",
            "Endpoint-specificity/velocity (GS16/GS17 metrics) not computed per arm "
            "in this pilot.",
            f"Pilot scale (n_samples={N}, single seed, delta={args.deltas}) -- spec's "
            "own pilot asks 256 paired samples x 3 seeds at Delta 0.10 AND 0.20.",
        ],
    }
    json_path = out_dir / f"async_schedule_{args.label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GS19] Saved summary to {json_path}")


if __name__ == "__main__":
    main()
