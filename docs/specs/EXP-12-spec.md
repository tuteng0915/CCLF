# EXP-12 Spec — Residual Rank Analysis (Near-Miss Distribution)

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：量化"错误预测的质量"——baseline 是完全错还是差一点？**

EXP-16 告诉我们每个位置何时"第一次正确"预测到真实 token。但这只是二元的（对/错）。EXP-12 追问一个更细粒度的问题：

**在模型预测错误的那些位置（和时刻），真实 token 的排名是多少？**

- 如果 rank ≈ 1000（true token is deep in the ranking list）：模型完全没有这个概念，需要的是更好的表示
- 如果 rank ≈ 3（true token is the 3rd prediction）：模型几乎对了，只是最后几名之差，可能是 decode path 的几何问题

**要验证的核心假说**：
- KD 训练大幅降低"错误位置"的真实 token 排名（即使错也"差一点"）
- baseline 在 t=0.30 的 wrong positions 中，true token 的平均排名很高（如 rank 300+）
- kd-cr 在相同位置，true token 平均排名很低（rank 10-20），说明 KD 提升了"几乎正确"的质量

**重要性**：这一发现有助于理解 dec_sc 的作用机制：如果错误预测的 true token 已在 top-5，那么一个轻微的 calibration（如 decode branch 的调整）就能把它提升到 top-1。

**与其他实验的关系**：
- EXP-12 使用 EXP-07b 的 layer_states（layer_feats[-1]，768-dim L11 隐状态），复用已有数据
- EXP-12 的 rank 分布与 EXP-16 的承诺时序共同解释：为什么 kd-cr 能更早承诺（因为 true token 在更早的 t 时已在 top-3）

---

## Implementation

**Script:** `experiments/probe_elf/analyze_residual_rank.py`

**关键修正**（EXP-12 初始版本有 bug，已修复）：

原始 bug：直接用 x̂_t（512-dim）与 unembed_kernel 做 cosine，得到 frac_correct ≈ 0。  
修复：使用 `layer_feats[-1]`（L11，768-dim）通过完整 decode path：
```python
hidden = F.gelu(x_768 @ proj_kernel + proj_bias, approximate="tanh")  # (B, 512)
logits = hidden @ unembed_kernel + unembed_bias                         # (B, V)
```
其中 `proj_kernel: (768, 512)`，`unembed_kernel: (512, 32100)`。

**Usage:**
```bash
CUDA_VISIBLE_DEVICES=N python experiments/probe_elf/analyze_residual_rank.py \
  --checkpoint converted/elf_b-owt-baseline_torch.pt \
  --states_dir results/exp07b_baseline \
  --output_dir results/exp12_baseline \
  --t_values 0.10,0.20,0.30,0.50,0.70
```

---

## 实验结果（Results）

**状态**: COMPLETED（2026-07-18）

**原始数据文件**：
- `results/exp12_baseline/residual_rank.json`
- `results/exp12_kd_cr/residual_rank.json`
- `results/exp12_kd2/residual_rank.json`

**关键数据（t=0.30）**：

| Checkpoint | G(t) | rank≤2 (wrong) | rank≤5 (wrong) | rank≤10 (wrong) | mean rank (wrong) |
|-----------|------|----------------|----------------|-----------------|-------------------|
| baseline  | 61.9% | 19.9% | 38.0% | 48.6% | 372              |
| kd-cr     | 89.3% | 44.9% | 73.0% | n/a   | 16               |
| kd2       | 87.9% | 41.5% | 69.3% | n/a   | 32               |

**关键发现**：

1. **kd-cr 的 mean rank for wrong positions = 16 vs baseline = 372**：当 kd-cr 预测错时，true token 平均排在第 16 位；而 baseline 错时，true token 平均排在第 372 位。差距超过 20 倍。

2. **kd-cr 的 rank≤5 率 = 73%**：在 kd-cr 预测错的 10.7% 位置中，有 73% 的情况 true token 在 top-5 内。这意味着 kd-cr 几乎是"差一点"，而不是"完全不知道"。

3. **baseline 的 rank≤5 率 = 38%**：baseline 错误预测中仍有 38% 在 top-5，说明 backbone 有一定的候选感知，但远不如 kd-cr 精确。

4. **t 依赖性**：rank 分布随 t 增大而迅速改善。t=0.70 时，所有 checkpoint 的 mean rank_wrong 均 < 100。

**全 t 值汇总（frac_correct 和 mean_rank_wrong）**：

| t | baseline G(t) | kd_cr G(t) | kd2 G(t) | baseline rank_wrong | kd_cr rank_wrong |
|---|--------------|------------|---------|---------------------|------------------|
| 0.1 | 1.96%   | 12.55%  | 12.32%  | 6651              | 1003             |
| 0.2 | 36.20%  | 58.49%  | 57.38%  | 1663              | 106              |
| 0.3 | 61.93%  | 89.26%  | 87.92%  | 372               | 16               |
| 0.5 | 75.55%  | 99.49%  | 99.00%  | 94                | 3                |
| 0.7 | 78.74%  | 99.84%  | 99.66%  | 74                | 5                |

**logit gap（top-1 logit − true token logit，衡量模型置信度）**：

| t | baseline | kd_cr | kd2 |
|---|----------|-------|-----|
| 0.1 | 111.4  | 15.6  | 13.9  |
| 0.2 | 47.2   | 4.2   | 5.1   |
| 0.3 | 22.3   | 0.8   | 1.1   |
| 0.5 | 13.3   | 0.0   | 0.1   |
| 0.7 | 11.3   | 0.0   | 0.0   |

**注**：kd_cr 在 t=0.3 时 logit_gap ≈ 0.8（几乎是"无置信度"状态），而 baseline = 22.3。这意味着 kd_cr 的 top-1 和 true token 在 t=0.3 几乎同分，decode branch 只需微调即可翻转到正确答案。

**论文启示**：
- 支持"KD 通过提升低 t 区间的预测质量（不仅是准确率，还有置信度/排名）来加速承诺"
- dec_sc 的作用可以解释为：在 kd-cr 预测错时（rank 16 左右），decode branch 提供额外 signal 将 true token 从 rank 16 推到 rank 1
- baseline 在 rank 372 时，dec_sc 不太可能有效（差得太远）

⚠️ **Selection bias 警告**：以上结果对比的是不同 checkpoint 在各自的 wrong positions 上的 rank distribution（baseline 38% wrong vs kd_cr 11% wrong）。这导致比较对象不同，无法直接说"在相同位置上 KD 的 rank 更好"。见 EXP-12v2 修正。

---

## EXP-12v2 结果（2026-07-22，Fixed Baseline-Wrong Set）

**状态**: DONE（2026-07-22）  
**脚本**: `experiments/probe_elf/probe_rank_analysis.py`  
**结果文件**: `results/exp12v2/rank_analysis.json`, `results/exp12v2_t030/rank_analysis.json`

**核心修正**: EXP-12 比较的是各 checkpoint 自己的 wrong set（baseline: 38%，kd_cr: 11%），存在 selection bias。EXP-12v2 固定 reference set（baseline T_stable never-commit，25.1%），对该相同集上比较所有 checkpoint。

### V2.1 Reference Set: baseline never-commit (25.1%, n=61,602)

这些是 baseline 在 EXP-16v2 中永远不能稳定读出的位置。

| t    | baseline MRR | kd_cr MRR | kd2 MRR | bl frac_correct | kd_cr frac_correct | kd2 frac_correct |
|------|:------------:|:---------:|:-------:|:---------------:|:------------------:|:----------------:|
| 0.10 | 0.0063       | 0.1132    | 0.1251  | 0.14%           | 6.96%              | 8.52%            |
| 0.20 | 0.1105       | 0.5884    | 0.5686  | 4.30%           | 50.17%             | 48.28%           |
| 0.30 | 0.1955       | 0.8781    | 0.8605  | 4.37%           | **82.55%**         | 80.46%           |
| 0.50 | 0.2685       | 0.9898    | 0.9817  | 2.50%           | **98.22%**         | 96.94%           |
| 0.70 | 0.3775       | 0.9959    | 0.9923  | 15.59%          | 99.38%             | 98.75%           |
| 1.00 | 0.7293       | 0.9963    | 0.9949  | 61.59%          | 99.45%             | 99.28%           |

**Logit gap（top1_logit − true_logit）**：

| t    | baseline | kd_cr  | kd2 |
|------|:--------:|:------:|:---:|
| 0.10 | 136.75   | 20.06  | —   |
| 0.20 | 81.97    | 5.74   | —   |
| 0.30 | **66.65** | **1.46** | — |
| 0.50 | 52.78    | 0.09   | —   |

**Median rank**:
- baseline @t=0.30: **12**; kd_cr: **1**
- baseline @t=0.10: 7017; kd_cr: 153

### V2.2 Reference Set: baseline wrong at t=0.30 (38.1%, n=93,685)

| t    | bl MRR | kd_cr MRR | bl correct | kd_cr correct | bl median_rank | kd_cr median_rank |
|------|:------:|:---------:|:----------:|:-------------:|:--------------:|:-----------------:|
| 0.30 | 0.1766 | 0.8248    | 0.00%      | **74.53%**    | 11             | 1                 |
| 0.50 | 0.5322 | 0.9927    | 38.37%     | **98.71%**    | 2              | 1                 |
| 1.00 | 0.8246 | 0.9976    | 75.21%     | 99.64%        | 1              | 1                 |

### V2.3 Key Findings（修正后无 selection bias）

1. **在 baseline 从未稳定读出的 25.1% 位置上，kd_cr 在 t=0.30 正确率 82.6%**（median_rank=1），而 baseline 只有 4.4%（median_rank=12）。这是在相同位置集合上的直接比较，无 selection bias。

2. **Logit gap 崩塌 45 倍**：在这些 never-commit 位置，baseline 的 mean_logit_gap=66.65 at t=0.30（true token 深埋），而 kd_cr=1.46（true token 几乎在首位）。

3. **KD 的改善是 decode interface 效应**（与 EXP-15v2 一致）：EXP-15v2 显示 unembed_bias 变化最大（R=2.59），而 backbone 变化较小（R≈0.22-0.34）。EXP-12v2 显示这些位置的 hidden state（经过相同 oracle 噪声）经 KD 的 decode head 后排名大幅提升。

4. **两种 reference set 结论一致**：无论用 never-commit 还是 wrong@t=0.30 定义 reference set，kd_cr 在 t≥0.30 时均大幅超越 baseline。

**论文可引用数字**：
- "on baseline's never-stably-committed positions (25.1%), kd_cr achieves 82.6% at t=0.30 vs. baseline 4.4% (median_rank 1 vs 12; mean logit gap 1.46 vs 66.65)"
- MRR improvement at t=0.30: 0.196 → 0.878 (4.5× on fixed reference set)
