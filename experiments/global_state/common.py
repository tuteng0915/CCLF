"""Shared utilities for the Global-State-Formation (EXP-GSx) experiment series.

Factored out of probe_sequence_hierarchy.py (EXP-GS1) so GLOBAL-2+ scripts can
reuse the same POS-bucket definition and pooling/similarity helpers without
duplicating them -- see docs/specs/EXP-GS1-spec.md Section 1 for why these
particular targets (POS histogram, mean pooling, cosine agreement) were chosen.
"""

import re

import numpy as np

POS_BUCKETS = ["NOUN", "VERB", "ADJ", "ADV", "DET", "ADP", "PRON", "OTHER"]
_POS_MAP = {}
for tag in ["NN", "NNS", "NNP", "NNPS"]:
    _POS_MAP[tag] = "NOUN"
for tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
    _POS_MAP[tag] = "VERB"
for tag in ["JJ", "JJR", "JJS"]:
    _POS_MAP[tag] = "ADJ"
for tag in ["RB", "RBR", "RBS"]:
    _POS_MAP[tag] = "ADV"
for tag in ["DT", "PDT", "WDT"]:
    _POS_MAP[tag] = "DET"
for tag in ["IN"]:
    _POS_MAP[tag] = "ADP"
for tag in ["PRP", "PRP$", "WP", "WP$"]:
    _POS_MAP[tag] = "PRON"


def masked_mean_pool(z, mask):
    """z: (N,L,d) torch, mask: (N,L) torch -> (N,d) torch."""
    m = mask.unsqueeze(-1).float()
    return (z * m).sum(1) / m.sum(1).clamp(min=1.0)


def pos_histogram(text, buckets=POS_BUCKETS):
    from nltk import pos_tag
    from nltk.tokenize import word_tokenize
    words = [w for w in word_tokenize(text) if re.search(r"[A-Za-z]", w)]
    hist = np.zeros(len(buckets), dtype=np.float64)
    if not words:
        hist[buckets.index("OTHER")] = 1.0
        return hist
    tags = pos_tag(words)
    for _, tag in tags:
        bucket = _POS_MAP.get(tag, "OTHER")
        hist[buckets.index(bucket)] += 1.0
    hist /= hist.sum()
    return hist


def cosine_rows(a, b):
    """a, b: (N,d) numpy -> (N,) cosine similarity per row."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a_n * b_n).sum(1)


def decode_text(tokenizer, ids_1d, mask_1d):
    valid = ids_1d[mask_1d.bool()]
    return tokenizer.decode(valid.tolist(), skip_special_tokens=True)


def nearest_topic(pooled, centroids):
    """pooled: (N,d) numpy, centroids: (K,d) numpy -> (N,) nearest-centroid index.

    Cosine-based, NOT squared-Euclidean. Found (2026-07-26, LangFlow GS6 debug)
    that classifying a NON-clean state (e.g. a rollout endpoint) against
    centroids fit on clean pooled embeddings breaks under Euclidean distance:
    LangFlow rollout endpoints have norm ~1.15 vs ~3.70 for the clean
    embeddings the centroids were fit on, so every query lands nearest to
    whichever centroid happens to have the smallest norm regardless of actual
    content (verified: all 8 test docs' rollout endpoints -> centroid 0,
    cosine sim to their OWN clean embedding was a healthy 0.40-0.67, i.e. the
    real per-document signal was there, just swamped by the norm gap under
    Euclidean distance). Cosine similarity is scale-invariant and fixes this
    without needing to know why the query distribution's norm differs from
    the fitting distribution's."""
    p_n = pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12)
    c_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    sims = p_n @ c_n.T  # (N,K)
    return sims.argmax(axis=1)


def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=0, statistic=np.mean):
    """values: 1D array-like of per-unit metric values (e.g. per-doc/per-pair/
    per-trajectory) -> (point_estimate, lo, hi) percentile bootstrap CI.

    Added 2026-07-27 during a rigor self-audit: unlike the PT series (which
    bootstraps every headline number, 2000 resamples), the GS series had been
    reporting plain point estimates from n=4-24 unit samples with no
    uncertainty quantification. This is a thin generic resampler over
    whatever axis the caller treats as the unit of replication (docs, pairs,
    trajectories) -- it doesn't know or care what the metric means."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    point = float(statistic(values))
    if n < 2:
        return point, point, point
    rng = np.random.RandomState(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot[b] = statistic(values[idx])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return point, float(lo), float(hi)


def load_adapter(model_name, checkpoint, config, device):
    """Uniform ELF/LangFlow adapter loading for the GS scripts.

    Fixes two issues found when actually exercising the (previously
    untested) LangFlow branch: (1) ELFAdapter.load needs a config path,
    LangFlowAdapter.load does not and has its own internal default
    checkpoint -- passing an ELF-style --checkpoint value like "baseline"
    into LangFlowAdapter.load would try to snapshot_download("baseline") and
    fail; (2) LangFlowAdapter has no `t_eps` attribute (ELF's free-running
    rollout start point) -- attaches a 0.05 fallback (matches ELF's typical
    value) so downstream code can always read `adapter.t_eps` uniformly.
    """
    if model_name == "elf":
        from adapters.elf_adapter import ELFAdapter
        assert config, "--config is required for --model elf"
        adapter = ELFAdapter.load(checkpoint, config, device)
    else:
        from adapters.langflow_adapter import LangFlowAdapter
        if checkpoint and checkpoint != "baseline":
            adapter = LangFlowAdapter.load(checkpoint=checkpoint, device=device)
        else:
            adapter = LangFlowAdapter.load(device=device)  # use its own HF default
        if not hasattr(adapter, "t_eps"):
            adapter.t_eps = 0.05
    return adapter


def load_owt_docs(adapter, model_name, n_samples, seq_len=None):
    """Uniform OWT loading: returns (ids, mask, x_clean, tokenizer) for
    either adapter. ELFAdapter.load_owt_sequences returns 2 values (needs a
    separate encode_clean call); LangFlowAdapter.load_owt_sequences returns
    3 (ids, mask, clean_emb already computed) -- this hides that difference
    from callers."""
    if model_name == "elf":
        seq_len = seq_len or adapter.seq_len
        ids, mask = adapter.load_owt_sequences(n_samples, seq_len=seq_len)
        x_clean = adapter.encode_clean(ids, mask).cpu()
    else:
        seq_len = seq_len or adapter.seq_len
        ids, mask, x_clean = adapter.load_owt_sequences(n_samples, seq_len=seq_len)
        x_clean = x_clean.cpu()
    return ids, mask, x_clean, adapter.tokenizer
