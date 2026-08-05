"""
EXP-29: kNN word visualization per position per timestep.

For each checkpoint (baseline/kd_cr/kd2), reads the already-collected oracle
state files from EXP-07 64-step runs. For each sequence × position × t, finds
the top-K cosine-nearest token centroids in x̂_t space.

Usage (from ELF-torch root):
  CUDA_VISIBLE_DEVICES=2 conda run -n elf python experiments/probe_elf/probe_knn_words.py \
      --output_dir results/exp29
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

CKPTS = ["baseline", "kd_cr", "kd2"]
CKPT_DIRS = {
    "baseline": "results/exp07_baseline_64/states",
    "kd_cr":    "results/exp07_kd_cr_64/states",
    "kd2":      "results/exp07_kd2_64/states",
}
CENTROIDS_PATH = "../../results/data/token_centroids.npz"
N_SEQS = 5      # number of example sequences (first N from each state file)
N_POS  = 36     # number of positions to track per sequence
K      = 3      # top-K nearest neighbors


@torch.no_grad()
def compute_knn(x_hat, centroids_norm, k):
    """
    x_hat: (N_SEQS, N_POS, d)
    centroids_norm: (V, d)  - L2-normalized
    returns ids (N_SEQS, N_POS, k), sims (N_SEQS, N_POS, k)
    """
    x_norm = F.normalize(x_hat, dim=-1)
    sims = torch.matmul(x_norm, centroids_norm.T)   # (N_SEQS, N_POS, V)
    top = torch.topk(sims, k=k, dim=-1)
    return top.indices, top.values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/exp29")
    parser.add_argument("--n_seqs", type=int, default=N_SEQS)
    parser.add_argument("--n_pos",  type=int, default=N_POS)
    parser.add_argument("--k",      type=int, default=K)
    args = parser.parse_args()

    n_seqs, n_pos, k = args.n_seqs, args.n_pos, args.k
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── centroids ─────────────────────────────────────────────────────────────
    npz = np.load(CENTROIDS_PATH)
    centroids = torch.tensor(npz["centroids"], dtype=torch.float32, device=device)
    centroids_norm = F.normalize(centroids, dim=-1)
    print(f"Centroids: {centroids.shape}")

    # ── tokenizer ─────────────────────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained("t5-small")

    # ── sequence metadata (from baseline, t=1.00 — cleanest ground truth) ────
    baseline_files = sorted(glob.glob(os.path.join(CKPT_DIRS["baseline"], "states_t*.pt")))
    last_file = baseline_files[-1]   # t=1.00
    meta = torch.load(last_file, map_location="cpu", weights_only=False)
    y_tokens_meta  = meta["y_tokens"][:n_seqs, :n_pos].numpy()  # (n_seqs, n_pos)
    attn_mask_meta = meta["attn_mask"][:n_seqs, :n_pos].numpy()

    seqs_info = []
    for s in range(n_seqs):
        ids   = y_tokens_meta[s].tolist()
        valid = attn_mask_meta[s].astype(bool).tolist()
        words = [tok.convert_ids_to_tokens(i) for i in ids]
        seqs_info.append({"seq_idx": s, "tokens": words, "gt_ids": ids, "valid": valid})

    # ── main loop ─────────────────────────────────────────────────────────────
    knn_per_ckpt = {}

    for ckpt in CKPTS:
        state_files = sorted(glob.glob(os.path.join(CKPT_DIRS[ckpt], "states_t*.pt")))
        print(f"\n── {ckpt}: {len(state_files)} t-steps ──")
        if not state_files:
            print(f"  [WARNING] no state files found for {ckpt}, skipping")
            continue

        # Accumulate across t-values
        # t_knn_ids[t_idx][s][p] = [id0, id1, id2]
        t_vals     = []
        t_knn_ids  = []   # list of (n_seqs, n_pos, k) numpy arrays
        t_knn_sims = []

        for sf in state_files:
            d = torch.load(sf, map_location="cpu", weights_only=False)
            t_val = float(d["t"])
            t_vals.append(round(t_val, 4))

            x_hat = d["x_hat"][:n_seqs, :n_pos, :].to(device).float()
            ids, sims = compute_knn(x_hat, centroids_norm, k)
            t_knn_ids.append(ids.cpu().numpy())
            t_knn_sims.append(sims.cpu().numpy())
            print(f"  t={t_val:.3f}", end="\r", flush=True)

        n_t = len(t_vals)
        # ids_arr: (n_t, n_seqs, n_pos, k)
        ids_arr  = np.array(t_knn_ids,  dtype=np.int32)
        sims_arr = np.array(t_knn_sims, dtype=np.float16)

        # Per-sequence dict: t_grid + per_pos knn
        ckpt_seqs = []
        for s in range(n_seqs):
            pos_list = []
            for p in range(n_pos):
                ids_tp  = ids_arr[:, s, p, :].tolist()    # (n_t, k) ints
                sims_tp = sims_arr[:, s, p, :].tolist()   # (n_t, k) floats
                words_tp = [
                    [tok.convert_ids_to_tokens(int(iid)) for iid in ids_at_t]
                    for ids_at_t in ids_tp
                ]
                pos_list.append({"ids": ids_tp, "sims": sims_tp, "words": words_tp})
            ckpt_seqs.append({"t_grid": t_vals, "pos_knn": pos_list})
        knn_per_ckpt[ckpt] = ckpt_seqs
        print(f"\n  done.")

    # ── save JSON ─────────────────────────────────────────────────────────────
    output = {
        "checkpoints": CKPTS,
        "n_seqs": n_seqs,
        "n_pos":  n_pos,
        "k":      k,
        "seqs_info": seqs_info,
        "knn": knn_per_ckpt,
    }
    json_path = os.path.join(args.output_dir, "knn_words.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    print(f"\nSaved {json_path}")

    # ── generate HTML ─────────────────────────────────────────────────────────
    html_path = os.path.join(args.output_dir, "knn_viz.html")
    generate_html(output, html_path)
    print(f"Saved {html_path}")


# ─────────────────────────────────────────────────────────────────────────────
# HTML generation
# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>EXP-29 — kNN Word Visualization</title>
<style>
:root {
  --bg: #f7f7f5; --surface: #fff; --border: #d0cec8; --text: #1a1a1a;
  --muted: #6b6b6b; --correct: #2d8a4e; --near: #b08020; --wrong: #c03030;
  --correct-bg: #d4f0df; --near-bg: #fdf0c0; --wrong-bg: #fce0e0;
  --btn-active: #2563eb; --btn-bg: #e8e8e8; --btn-text: #333;
  --cell-w: 60px; --cell-h: 22px; --font-cell: 11px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #141414; --surface: #1e1e1e; --border: #333; --text: #e8e8e8;
    --muted: #999; --correct: #52c47a; --near: #e0b040; --wrong: #e06060;
    --correct-bg: #1a3d28; --near-bg: #3a3010; --wrong-bg: #3d1a1a;
    --btn-active: #3b82f6; --btn-bg: #2a2a2a; --btn-text: #ccc;
  }
}
:root[data-theme="dark"] {
  --bg: #141414; --surface: #1e1e1e; --border: #333; --text: #e8e8e8;
  --muted: #999; --correct: #52c47a; --near: #e0b040; --wrong: #e06060;
  --correct-bg: #1a3d28; --near-bg: #3a3010; --wrong-bg: #3d1a1a;
  --btn-active: #3b82f6; --btn-bg: #2a2a2a; --btn-text: #ccc;
}
:root[data-theme="light"] {
  --bg: #f7f7f5; --surface: #fff; --border: #d0cec8; --text: #1a1a1a;
  --muted: #6b6b6b; --correct: #2d8a4e; --near: #b08020; --wrong: #c03030;
  --correct-bg: #d4f0df; --near-bg: #fdf0c0; --wrong-bg: #fce0e0;
  --btn-active: #2563eb; --btn-bg: #e8e8e8; --btn-text: #333;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: 'Courier New', monospace; font-size: 13px; padding: 16px; }
h1 { font-size: 16px; font-weight: 700; margin-bottom: 4px; letter-spacing: 0.02em; }
.subtitle { color: var(--muted); font-size: 11px; margin-bottom: 16px; }
.controls { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }
.ctrl-group { display: flex; flex-direction: column; gap: 4px; }
.ctrl-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.btn-row { display: flex; gap: 4px; flex-wrap: wrap; }
button { padding: 4px 10px; border: 1px solid var(--border); background: var(--btn-bg);
         color: var(--btn-text); border-radius: 3px; cursor: pointer; font-size: 11px; font-family: inherit; }
button.active { background: var(--btn-active); color: #fff; border-color: var(--btn-active); }
.legend { display: flex; gap: 14px; margin-bottom: 10px; font-size: 11px; }
.leg-item { display: flex; align-items: center; gap: 4px; }
.leg-sq { width: 12px; height: 12px; border-radius: 2px; }
.grid-wrap { overflow-x: auto; }
table { border-collapse: separate; border-spacing: 1px; }
th.header-t { font-size: 9px; color: var(--muted); writing-mode: vertical-lr;
              text-orientation: mixed; transform: rotate(180deg); padding: 2px 1px;
              vertical-align: bottom; max-height: 52px; white-space: nowrap; }
td.pos-label { font-size: 10px; white-space: nowrap; padding: 1px 6px 1px 2px;
               color: var(--muted); min-width: 80px; vertical-align: middle; }
td.cell { width: var(--cell-w); height: var(--cell-h); font-size: var(--font-cell);
          text-align: center; vertical-align: middle; border-radius: 2px;
          cursor: pointer; white-space: nowrap; overflow: hidden;
          text-overflow: ellipsis; padding: 0 2px; position: relative; }
td.cell.correct { background: var(--correct-bg); color: var(--correct); }
td.cell.near    { background: var(--near-bg);    color: var(--near); }
td.cell.wrong   { background: var(--wrong-bg);   color: var(--wrong); }
/* tooltip */
#tooltip { position: fixed; background: var(--surface); border: 1px solid var(--border);
           border-radius: 4px; padding: 8px 10px; font-size: 11px; z-index: 1000;
           pointer-events: none; display: none; min-width: 160px; box-shadow: 0 2px 8px rgba(0,0,0,0.18); }
#tooltip .tt-title { font-weight: 700; margin-bottom: 4px; }
#tooltip .tt-row { display: flex; justify-content: space-between; gap: 12px; padding: 1px 0; }
#tooltip .tt-rank { color: var(--muted); }
#tooltip .tt-word { flex: 1; }
#tooltip .tt-sim  { color: var(--muted); font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<h1>EXP-29 — kNN Word Visualization per Position per Timestep</h1>
<p class="subtitle">Oracle protocol (fixed ε, sweep t) · top-1 cell color = correct / near-rank / wrong · hover for top-3</p>

<div class="controls">
  <div class="ctrl-group">
    <span class="ctrl-label">Checkpoint</span>
    <div class="btn-row" id="ckpt-btns"></div>
  </div>
  <div class="ctrl-group">
    <span class="ctrl-label">Sequence</span>
    <div class="btn-row" id="seq-btns"></div>
  </div>
</div>

<div class="legend">
  <div class="leg-item"><div class="leg-sq" style="background:var(--correct-bg);border:1px solid var(--correct)"></div> <span style="color:var(--correct)">top-1 correct</span></div>
  <div class="leg-item"><div class="leg-sq" style="background:var(--near-bg);border:1px solid var(--near)"></div> <span style="color:var(--near)">correct in top-3</span></div>
  <div class="leg-item"><div class="leg-sq" style="background:var(--wrong-bg);border:1px solid var(--wrong)"></div> <span style="color:var(--wrong)">wrong</span></div>
</div>

<div class="grid-wrap">
  <table id="knn-table"></table>
</div>
<div id="tooltip"></div>

<script>
const DATA = __DATA__;

let curCkpt = DATA.checkpoints[0];
let curSeq  = 0;

function buildButtons() {
  const cb = document.getElementById('ckpt-btns');
  DATA.checkpoints.forEach(c => {
    const b = document.createElement('button');
    b.textContent = c; b.dataset.ckpt = c;
    if (c === curCkpt) b.classList.add('active');
    b.onclick = () => { curCkpt = c; updateActiveBtns('ckpt-btns', c, 'ckpt'); render(); };
    cb.appendChild(b);
  });
  const sb = document.getElementById('seq-btns');
  DATA.seqs_info.forEach((s, i) => {
    const b = document.createElement('button');
    const preview = s.tokens.slice(0,6).join(' ');
    b.textContent = `seq${i}: ${preview}…`;
    b.style.maxWidth = '220px'; b.style.overflow = 'hidden'; b.style.textOverflow = 'ellipsis';
    if (i === 0) b.classList.add('active');
    b.onclick = () => { curSeq = i; updateActiveBtns('seq-btns', i, 'idx'); render(); };
    sb.appendChild(b);
  });
}

function updateActiveBtns(id, val, attr) {
  document.getElementById(id).querySelectorAll('button').forEach(b => {
    b.classList.toggle('active', (attr === 'ckpt' ? b.dataset.ckpt : Number(b.dataset.idx)) === val);
  });
  document.getElementById(id).querySelectorAll('button').forEach((b,i) => {
    if (attr === 'idx') b.classList.toggle('active', i === val);
  });
}

function render() {
  const table = document.getElementById('knn-table');
  table.innerHTML = '';
  const seqInfo = DATA.seqs_info[curSeq];
  const ckptData = DATA.knn[curCkpt];
  if (!ckptData) { table.innerHTML = '<tr><td style="color:var(--wrong)">Data not found for this checkpoint</td></tr>'; return; }
  const seqKnn = ckptData[curSeq];
  const t_grid = seqKnn.t_grid;
  const n_pos = seqKnn.pos_knn.length;
  const gt_ids = seqInfo.gt_ids;

  // Header row
  const thead = document.createElement('thead');
  const hrow = document.createElement('tr');
  const emptyTh = document.createElement('th');
  emptyTh.style.minWidth = '80px';
  hrow.appendChild(emptyTh);
  t_grid.forEach(t => {
    const th = document.createElement('th');
    th.className = 'header-t';
    th.textContent = t.toFixed(2);
    hrow.appendChild(th);
  });
  thead.appendChild(hrow);
  table.appendChild(thead);

  // Body rows (one per position)
  const tbody = document.createElement('tbody');
  for (let p = 0; p < n_pos; p++) {
    if (!seqInfo.valid[p]) continue;
    const pos = seqKnn.pos_knn[p];
    const tr = document.createElement('tr');

    // Position label
    const td_lbl = document.createElement('td');
    td_lbl.className = 'pos-label';
    td_lbl.textContent = `${p}: ${seqInfo.tokens[p]}`;
    tr.appendChild(td_lbl);

    // One cell per t-step
    t_grid.forEach((t, ti) => {
      const ids   = pos.ids[ti];
      const sims  = pos.sims[ti];
      const words = pos.words[ti];
      const gtId  = gt_ids[p];
      const top1Word = words[0];
      const isCorrect = ids[0] === gtId;
      const inTop3    = ids.includes(gtId);

      const td = document.createElement('td');
      td.className = 'cell ' + (isCorrect ? 'correct' : inTop3 ? 'near' : 'wrong');
      td.textContent = cleanWord(top1Word);
      td.addEventListener('mouseenter', e => showTooltip(e, p, ti, seqInfo.tokens[p], t, ids, words, sims, gtId));
      td.addEventListener('mouseleave', hideTooltip);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
}

function cleanWord(w) {
  if (!w) return '?';
  return w.startsWith('▁') ? w.slice(1) : w;
}

function showTooltip(e, p, ti, gtWord, t, ids, words, sims, gtId) {
  const tt = document.getElementById('tooltip');
  let html = `<div class="tt-title">pos ${p}: ${gtWord} | t=${t.toFixed(3)}</div>`;
  words.forEach((w, i) => {
    const isGt = ids[i] === gtId;
    const sim = typeof sims[i] === 'number' ? sims[i].toFixed(3) : Number(sims[i]).toFixed(3);
    const style = isGt ? `color:var(--correct);font-weight:700` : '';
    html += `<div class="tt-row"><span class="tt-rank" style="${style}">#${i+1}</span>` +
            `<span class="tt-word" style="${style}">${w || '?'}</span>` +
            `<span class="tt-sim">${sim}</span></div>`;
  });
  tt.innerHTML = html;
  tt.style.display = 'block';
  positionTooltip(e);
}

function positionTooltip(e) {
  const tt = document.getElementById('tooltip');
  const pad = 10;
  let x = e.clientX + pad, y = e.clientY + pad;
  const rect = tt.getBoundingClientRect();
  if (x + rect.width > window.innerWidth)  x = e.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight) y = e.clientY - rect.height - pad;
  tt.style.left = x + 'px'; tt.style.top = y + 'px';
}

function hideTooltip() {
  document.getElementById('tooltip').style.display = 'none';
}

document.addEventListener('mousemove', e => {
  if (document.getElementById('tooltip').style.display === 'block') positionTooltip(e);
});

buildButtons();
render();
</script>
</body>
</html>
"""


def generate_html(data: dict, out_path: str):
    import json as _json
    data_json = _json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    html = HTML_TEMPLATE.replace("__DATA__", data_json)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
