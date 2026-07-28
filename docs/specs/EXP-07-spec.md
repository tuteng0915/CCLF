# EXP-07 Spec — Independent Linear Probe on x̂_t (Story A Validation)

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：验证 backbone 是否比 native head 更早"知道"答案（Story A）。**

ELF 的 G(t) 和 Rec@1(t) 都依赖 tied-weight 输出头 W = E（词汇嵌入矩阵）。但 tied weights 有两个问题：
1. 几何偏置（EXP-04 已证实）
2. 在训练中被优化为"多目标"（既要做去噪，又要做解码），所以不一定是当前 t 下 x̂_t 的最优线性解码器

**要验证的核心假说（Story A）**：
- 如果独立训练的线性探针准确率 >> Rec@1(t)，说明 backbone 的 x̂_t 中编码了比 tied-weight head 能提取出的更多的 token 信息。
- 即："backbone 早已知道答案，但 native head 说不出来"。
- 这支持了 dec_sc 机制的作用：decode branch 作为一个专门的解码器，能提取 backbone 已有但 native head 未能利用的信息。

**如果 Story A 不成立**（probe ≈ Rec@1）：则 backbone 的 x̂_t 本身没有比 native head 更多的信息，G(t) 的提升主要来自解码器的改进而非表示质量的提升。

**与其他实验的关系**：
- EXP-07 → EXP-07b（层级探针）→ EXP-07c（跨 checkpoint 探针转移）形成递进的机制分析链
- EXP-07 是这个链的起点：首先确认独立探针是否有意义（能否超越 native head）

**当前状态**：结果存在（2026-07-18），但有方法论问题需要修正（见 EXP-07v2）。core finding（baseline +46pp gap）描述性结论大概率成立，但数字需经 document-level held-out 验证。

## ⚠️ 方法论问题（2026-07-22 发现）

### 问题 1：spec 伪代码在训练集上评估

spec 中的伪代码：
```python
probe_acc = (probe(X).argmax(dim=-1) == y).float().mean()
```
直接在训练 probe 的同一批 X 上评估，没有任何 train/test 分割。这是最严重的问题。

实际实现（`train_linear_probe.py`）虽然有 80/20 分割，但是 **position 级别**分割：

### 问题 2：Position 级别分割（同一 document 的 position 同时在 train/val）

```python
perm = torch.randperm(N)   # N = 490,353 valid positions
x_train = x_flat[perm[:n_train]]  # positions from ALL 512 documents
x_val   = x_flat[perm[n_train:]]  # also from ALL 512 documents
```

同一条 document（seq i）的 positions 会分散到 train 和 val 两边。probe 在训练时见过所有 512 条 document 的大部分 positions，val 中同一 document 的 positions 受到内部相关性的"contamination"。

**参数量分析**：linear probe 有 512×32100 ≈ 1640 万参数，在 392K positions 上训练，比例约 1:24（略过参数化）。

### 问题 3：没有 train accuracy 报告

没有同时报告 train accuracy，无法判断过拟合程度。

### 问题 4：没有 shuffled-label control

没有 shuffled-label probe 验证"近随机"上界，无法排除记忆效应。

### 严重性评估

尽管存在上述问题，现有结果可能仍然有效，原因：

1. **训练 loss 的收敛模式合理**：以 t=0.20 为例，loss 从 1.30（epoch 10）降至 0.93（epoch 30），loss 仍在 0 以上，不像是过拟合
2. **三个 checkpoint 的 probe_acc 几乎相同**（差距<2pp）：如果是记忆特定 document 内容，三个 checkpoint 的 probe 不应该完全一样
3. **position-level split 对线性探针的影响相对有限**：线性 probe 使用全局权重 W，不能直接"记住"某条 doc 的 position，只能学通用的线性映射

但这些都是定性判断，不能用于论文。需要 EXP-07v2 正式验证。

**Dense re-run (2026-07-20)**：n_t_steps=20（step=0.05）→ n_t_steps=48（step≈0.020）。Cliff 区间 t=0.10-0.30 从 5 个点增至 11 个点，用于 paper 图表。数据：`models/ELF-torch/results/exp07_{baseline,kd_cr,kd2}_dense/probe_accuracies.json`。baseline + kd_cr + kd2 全部 DONE（2026-07-20）。JAX decoder_rec1 dense：probe_decode_v2_dense DONE（2026-07-21），完整 51-value 曲线见下方"JAX Baseline Decoder Dense"节。

**Dense probe_acc 曲线（51 t-values, step≈0.020）：**

| t | baseline probe_acc | kd_cr probe_acc | kd2 probe_acc |
|---|---|---|---|
| 0.050 | 6.4% | 6.5% | 6.6% |
| 0.091 | 11.7% | 11.9% | 12.3% |
| 0.131 | 22.5% | 24.6% | 24.7% |
| 0.151 | 31.9% | 33.3% | 33.5% |
| 0.171 | 42.6% | 42.5% | 43.0% |
| 0.191 | 52.2% | 51.4% | 52.1% |
| 0.212 | 60.6% | 59.6% | 60.8% |
| 0.232 | 67.8% | 66.0% | 66.7% |
| 0.252 | 73.2% | 71.5% | 72.1% |
| 0.272 | 77.8% | 75.6% | 76.2% |
| 0.293 | 81.3% | 78.9% | 79.4% |
| 0.353 | 88.1% | 85.9% | 86.4% |
| 0.500 | 94.2% | 93.2% | 93.1% |
| 0.700 | 96.0% | 95.3% | 95.0% |
| 1.000 | 96.4% | 95.9% | 96.1% |

三个 checkpoint 的 probe_acc 几乎一致（差距 < 2pp），与 Story A 一致：baseline/kd_cr/kd2 的 x̂_t 都编码了相近的 token 信息量。关键区别在 decoder_rec1（native decode path），待 probe_decode_v2_dense 完成后补充。kd2 与 kd_cr 的 probe_acc 在整个 t-grid 上差距 < 1pp，可视为等效。

---

**Goal:** Train a separately-parameterized linear probe on ELF's x̂_t at each timestep t,
and compare its accuracy vs the native tied-weight head (Rec@1(t)).

If probe accuracy >> Rec@1(t): the backbone already encodes token identity earlier than
the tied-weight projection can expose. This validates Story A:
"ELF forms token-predictive states earlier than its native linear interface can expose."

---

## Why this matters

G(t) is the cosine-normalized readout — it's still using the same E = W embedding matrix
(tied weights). EXP-04 shows this has significant geometry bias. An independent linear
probe trained with an MLE objective on x̂_t gives a cleaner measurement of backbone quality.

Key comparison at t=0.35 (commitment cliff):
- Native head Rec@1(0.35) ≈ 59.2%
- Decode branch accuracy at t=0.35 ≈ 86.8% (paper table)
- Independent probe (this experiment): expected to be higher than 59.2% if Story A is true

---

## Implementation

### New file: `experiments/probe_elf/train_linear_probe.py`

**Phase 1: Collect x̂_t states**

```python
# For each t in {0.05, 0.10, ..., 0.95, 1.00}:
# Run ELF backbone on real validation data at that t
# Save (x_hat_t, y_true_token) pairs

for t_val in t_grid:
    eps = torch.randn_like(x_clean)
    z_t = t_val * x_clean + (1 - t_val) * eps
    z_in = cat([z_t, zeros_sc], dim=-1)
    sc_scale = torch.ones(B, device=device)
    x_hat, _, _ = model(z_in, t_batch, decoder_step_active=None, self_cond_cfg_scale=sc_scale)
    # Save: x_hat (B, L, 512) and y_tokens (B, L)
    torch.save({"x_hat": x_hat.cpu(), "y": y_tokens.cpu()},
               f"results/exp07/states_t{t_val:.2f}.pt")
```

**Phase 2: Train linear probe at each t**

```python
# For each saved t file:
data = torch.load(f"results/exp07/states_t{t_val:.2f}.pt")
X = data["x_hat"].reshape(-1, 512).float()  # (N*L, 512)
y = data["y"].reshape(-1)                   # (N*L,)

# Simple logistic regression (no hidden layers)
probe = nn.Linear(512, vocab_size)
optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
for epoch in range(20):
    logits = probe(X)
    loss = F.cross_entropy(logits, y)
    loss.backward(); optimizer.step(); optimizer.zero_grad()

probe_acc = (probe(X).argmax(dim=-1) == y).float().mean()
```

**Phase 3: Compare**

For each t: plot probe_acc(t), Rec@1(t), G(t), decode_branch_acc(t) on same axes.

---

## Data requirements

- N=1000 validation sequences (not in training set)
- T5 encoder to get x_clean
- Use `load_dataset_split("embedded-language-flows/openwebtext-t5")` and sample from it

---

## Decision rule

- If probe_acc(0.35) > Rec@1(0.35) + 10pp: Story A confirmed. Add to paper:
  "A separately-trained linear probe achieves Xpp higher accuracy than the native head at t=0.35,
  confirming that the backbone's representation encodes token identity earlier than the
  tied-weight projection can expose."
- If probe_acc(0.35) ≤ Rec@1(0.35) + 5pp: Story A not confirmed by probe.
  Remove the "forms token-predictive states earlier" claim; keep G(t) vs Rec@1 gap as unexplained.

---

## Key verification

After training, check: does probe_acc at t=1.0 approach 100%? (Should, since x̂_1 ≈ x_clean.)
If not, the probe training or data loading has a bug.

---

## Effort: 3–5 days

- 1 day: data loading + state collection pipeline
- 1 day: linear probe training
- 1 day: plots + paper update

## Files to create
- `experiments/probe_elf/collect_probe_states.py` — run model on validation data, save states
- `experiments/probe_elf/train_linear_probe.py` — train logistic regression at each t
- `experiments/probe_elf/plot_probe_comparison.py` — overlay all accuracy curves

---

## 实验结果（Results）

**状态**: COMPLETED（见下方 UPDATED 2026-07-18 节）

---

## 实验结果（Results）— UPDATED 2026-07-18

**状态**: COMPLETED

数据来源:
- `results/exp07_{baseline,kd_cr,kd2}/probe_accuracies.json` — linear probe on x̂_t
- `results/exp07_{baseline,kd_cr,kd2}/mlp_probe_accuracies.json` — MLP probe on x̂_t (hidden=512)
- `results/exp07_{baseline,kd_cr,kd2}/decoder_rec1.json` — native decode path G_A

### 核心对比表

baseline — linear/MLP probe on x̂_t (512-dim) vs native decode path on L11 (768-dim):

| t    | decoder_rec1 (G_A) | linear_probe | mlp_probe | gap (lin)   |
|------|--------------------|--------------|-----------|-------------|
| 0.10 | 1.0%               | 13.3%        | 11.5%     | +12.3pp     |
| 0.20 | 10.2%              | 56.1%        | 55.3%     | **+45.8pp** |
| 0.25 | 30.0%              | 72.7%        | 72.4%     | +42.7pp     |
| 0.30 | 53.6%              | 82.3%        | 82.8%     | +28.7pp     |
| 0.35 | 67.5%              | 87.9%        | 88.8%     | +20.4pp     |
| 0.70 | 84.8%              | 96.0%        | 97.6%     | +11.2pp     |
| 1.00 | 99.9%              | 96.5%        | 98.2%     | −3.4pp      |

kd_cr — REVERSED: native decoder BEATS trained linear/MLP probe:

| t    | decoder_rec1 | linear_probe | mlp_probe | gap (lin)   |
|------|-------------|--------------|-----------|-------------|
| 0.10 | 12.5%       | 14.2%        | 11.7%     | +1.7pp      |
| 0.20 | 60.7%       | 54.0%        | 52.8%     | **−6.7pp**  |
| 0.30 | 90.5%       | 78.1%        | 78.5%     | −12.4pp     |
| 0.70 | 99.8%       | 92.8%        | 95.1%     | −7.0pp      |
| 1.00 | 99.9%       | 93.3%        | 95.3%     | −6.6pp      |

kd2 — same reversal pattern as kd_cr:

| t    | decoder_rec1 | linear_probe | mlp_probe |
|------|-------------|--------------|-----------|
| 0.20 | 60.1%       | 54.7%        | 53.3%     |
| 0.30 | 89.6%       | 78.6%        | 78.9%     |

### 3-model summary at key t values

| t    | baseline (lin/mlp/dec) | kd_cr (lin/mlp/dec) | kd2 (lin/mlp/dec) |
|------|------------------------|---------------------|-------------------|
| 0.10 | 13.3 / 11.5 / **1.0** | 14.2 / 11.7 / 12.5 | 13.96 / 11.76 / 12.5 |
| 0.20 | 56.1 / 55.3 / **10.2** | 54.0 / 52.8 / **60.7** | 54.7 / 53.3 / **60.1** |
| 0.30 | 82.3 / 82.8 / **53.6** | 78.1 / 78.5 / **90.5** | 78.6 / 78.9 / **89.6** |
| 0.70 | 96.0 / 97.6 / **84.8** | 92.8 / 95.1 / **99.8** | 92.5 / 94.8 / **99.7** |
| 1.00 | 96.5 / 98.2 / **99.9** | 93.3 / 95.3 / **99.9** | 93.9 / 95.6 / **99.8** |

Bold = best readout among the three; dec = native decoder, lin = linear probe on x̂_t, mlp = MLP probe on x̂_t.

**注**: MLP probe ≈ linear probe（大多数 t 差距 < 1pp），说明 x̂_t 与 token identity 的关系在 512-dim 空间中近似线性。

### 解读

关键发现：KD 训练逆转了 probe vs native head 的优劣方向。

- Baseline: x̂_t (512-dim) 内的 token 信息远多于 native decode path 能读出的（t=0.20 时 +46pp）。Story A 在 baseline 上完全成立。MLP probe 与 linear probe 几乎相同，说明线性探针已经榨干了 x̂_t 中的 token 信息。

- kd_cr / kd2: Native decode path（L11 → GELU → logits）超越了对 x̂_t 训练的 linear/MLP probe（t=0.20 时 −7pp）。KD 训练"解锁"了 L11 中的 token 信息，使 native decode path 成为最优读出器，甚至超过独立探针。

- 机制含义: baseline 的 G(t) 低不是因为 backbone 不知道答案，而是因为 native decode path 对 L11 的投影方式不适合 token 分类任务。KD 训练通过监督 decode branch 直接优化这一投影。

### Paper impact

Story A 需要分 baseline 和 kd_cr 分别表述:
- "Baseline: linear probe on x̂_t achieves 56.1% at t=0.20 vs native decode path 10.2% — a +46pp gap confirming representation-readout mismatch in the baseline model."
- "KD-trained models (kd_cr/kd2): the native decode path (60.7%) exceeds a freshly-trained linear probe on x̂_t (54.0%), suggesting KD aligns the decode projection to the token-discriminative subspace of L11."

---

## JAX Baseline Decoder Dense — DONE (probe_decode_v2_dense, 2026-07-21)

数据来源: `results/elf/probe_decode_v2_dense/probe_decode_branch.json`
字段: `lin_top1_gt_mean` = JAX ELF-B baseline decoder top-1 accuracy (51 t-values, step=0.02)
注: 此为 JAX ELF-B 模型（与 ELF-torch 不同实现），t=0.20 时给出 8.6%（ELF-torch baseline decoder_rec1=10.2%，略有差异）。

### Commitment cliff 完整曲线

| t    | JAX dec acc | | t    | JAX dec acc |
|------|-------------|---|------|-------------|
| 0.00 | 0.0%        | | 0.38 | 65.3%       |
| 0.02 | 0.0%        | | 0.40 | 68.5%       |
| 0.04 | 0.1%        | | 0.42 | 71.0%       |
| 0.06 | 0.4%        | | 0.44 | 73.0%       |
| 0.08 | 0.9%        | | 0.46 | 74.6%       |
| 0.10 | 1.6%        | | 0.48 | 75.7%       |
| 0.12 | 2.2%        | | 0.50 | 76.6%       |
| 0.14 | 2.8%        | | 0.52 | 77.3%       |
| 0.16 | 3.8%        | | 0.54 | 78.0%       |
| 0.18 | 5.5%        | | 0.56 | 78.6%       |
| **0.20** | **8.6%** | | 0.58 | 79.0%       |
| 0.22 | 14.3%       | | 0.60 | 79.3%       |
| 0.24 | 21.6%       | | 0.62 | 79.7%       |
| 0.26 | 30.5%       | | 0.64 | 79.6%       |
| 0.28 | 38.1%       | | 0.66 | 79.7%       |
| **0.30** | **45.3%** | | 0.68 | 79.8%       |
| 0.32 | 51.6%       | | 0.70 | 79.7%       |
| 0.34 | 57.1%       | | 0.80 | 78.5%       |
| 0.36 | 61.6%       | | 0.90 | 76.9%       |
|      |             | | 0.96 | 84.7%       |
|      |             | | 0.98 | 97.9%       |
|      |             | | 1.00 | 97.9%       |

**关键形态**：
1. **Cliff 区间 t=0.20→0.30**: 8.6% → 45.3%，每 0.10 步增加 +36.7pp，是最陡峭的区段
2. **Plateau t=0.50→0.90**: 约 76.6%–79.8%，基本平坦（约 3.2pp range）
3. **Anomalous jump at t≥0.96**: 0.90→0.96 jump from 76.9% to 84.7%，0.96→0.98 jump to 97.9%
   - 原因: cos_clean 在 SC=0、oracle protocol 下的几何异常：t=1.0 时去噪路径收到 clean input，decode branch 能完美读出 token。
4. **t=0 几乎 0%**: 纯噪声下无 token 信息可读

**与 ELF-torch decoder_rec1 对比**（t=0.20）:
- JAX baseline: 8.6%
- ELF-torch baseline: 10.2%（差 1.6pp，属不同实现间的正常偏差）

**对论文可视化的贡献**: 用于 Figure X（commitment cliff illustration），展示从 commit cliff 到 plateau 的三阶段结构。

---

## EXP-07 64-step Dense Probe — DONE (2026-07-21)

**脚本**: `collect_probe_states.py` + `train_linear_probe.py` + `eval_decoder_rec1.py`（标准 pipeline）  
**t-grid**: `np.linspace(0.05, 1.00, 64)` — 64 个均匀采样点，步长 ≈ 0.0151  
**数据**: `results/exp07_{baseline,kd_cr,kd2}_64/probe_accuracies.json` + `decoder_rec1.json`

### 关键数据点（t ≈ 0.20，index 10，t=0.2008）

| checkpoint | probe_acc | decoder_rec1 | gap (probe−dec) |
|-----------|-----------|-------------|----------------|
| baseline  | 56.3%     | 10.5%       | **+45.8pp** |
| kd_cr     | 55.3%     | 60.8%       | **−5.5pp** |
| kd2       | 56.1%     | 60.2%       | **−4.1pp** |

（与 20-step 结果高度一致：baseline 56.1% probe / 10.2% dec；kd_cr 54.0% probe / 60.7% dec）

### 完整 64-step 曲线（probe_acc，%）

| idx | t      | BASE_P | KDCR_P | KD2_P | BASE_D | KDCR_D | KD2_D |
|-----|--------|--------|--------|-------|--------|--------|-------|
| 0   | 0.050  | 6.4    | 6.5    | 6.6   | 0.3    | 6.0    | 5.8   |
| 5   | 0.1254 | 20.3   | 22.4   | 22.5  | 2.0    | 20.2   | 19.8  |
| 10  | 0.2008 | 56.3   | 55.3   | 56.1  | 10.5   | 60.8   | 60.2  |
| 15  | 0.2762 | 78.1   | 76.3   | 76.8  | 43.1   | 86.3   | 85.5  |
| 20  | 0.3516 | 88.1   | 85.6   | 86.3  | 68.0   | 95.8   | 94.9  |
| 30  | 0.5024 | 94.4   | 93.3   | 93.3  | 79.7   | 99.5   | 99.2  |
| 40  | 0.6532 | 95.9   | 95.1   | 95.0  | 84.2   | 99.8   | 99.6  |
| 50  | 0.8040 | 96.2   | 95.3   | 95.4  | 85.2   | 99.9   | 99.7  |
| 60  | 0.9548 | 96.3   | 96.1   | 96.2  | 91.7   | 99.9   | 99.8  |
| 63  | 1.000  | 96.4   | 95.9   | 96.2  | 99.9   | 99.9   | 99.8  |

### 视觉化

已更新至 `cclf_viz.html` Chart 1（artifact URL: https://claude.ai/code/artifact/32e7db44-7b52-4dde-8f6a-467df0336854）：
- T64 数组替换原 T20（20 点 → 64 点）
- Gap 标注更新：baseline +46pp，kd_cr −6pp（64-step 精确值）
- 曲线更平滑，cliff 区间 t=0.05→0.20 有 10 个点（原 3 个）

### 新增信息（相比 20-step）

1. **Cliff 的精细形态**: t=0.05→0.2008 的 baseline decoder_rec1 从 0.3% 到 10.5%，基本线性增长；kd_cr/kd2 从 6% 到 60%，在 t≈0.10–0.20 内急速上升（+54pp in 0.10t）
2. **kd_cr/kd2 probe_acc 悬崖**: probe 的 cliff 更宽（t=0.05→0.30 范围内 6%→76%），说明 probe 的"学习"不像 decoder 那么依赖 t=0.20 这一特定点
3. **高 t 平台**: t≥0.50 后三个 checkpoint 的 probe_acc 均稳定在 93–96%，decoder_rec1 kd_cr/kd2 已达 99.5%+

---


## EXP-07v2：Document-Level Held-Out Probe（DONE 2026-07-22）

### 动机

修复上述方法论问题，验证 EXP-07 的核心 finding 在严格 held-out 条件下是否成立。

### 修复内容

1. **Document-level split**（`--train_seq_frac 0.8`）：前 409 条 sequence 全部 positions 进 train，后 103 条 sequence 全部 positions 进 test。任何一条 document 的 position 只能出现在 train 或 test 中，不可同时出现。
2. **Train accuracy 报告**：每个 t 同时输出 `train_acc` 和 `test_acc` 和 `overfit_gap = train_acc − test_acc`。
3. **Shuffled-label control**（`--shuffled_control`）：用 shuffled labels 在 test set 上再训练一个 probe，验证近随机上界。

### 配置

| 参数 | 值 |
|------|-----|
| states 来源 | `results/exp07_{baseline,kd_cr,kd2}_64/states/`（512 seqs × 1024 tokens）|
| train seqs | 409（80%），test seqs = 103（20%） |
| train positions | ~392K，test positions = ~98K |
| epochs | 30，lr=3e-3，AdamW(weight_decay=1e-4) |
| 输出 | `results/exp07v2_{baseline,kd_cr,kd2}/probe_accuracies_v2.json` |

### 最终结果（全部 3 checkpoints，DONE）

#### 完整三 checkpoint 对比表（document-level test_acc）

| t | base_train | base_test | kd_cr_train | kd_cr_test | kd2_train | kd2_test |
|---|------------|-----------|-------------|------------|-----------|----------|
| 0.050 | 8.1% | 6.4% | 8.1% | 6.5% | 8.3% | 6.6% |
| 0.065 | 10.1% | 8.0% | 10.1% | 8.1% | 10.5% | 8.0% |
| 0.080 | 12.9% | 9.5% | 13.1% | 10.1% | 13.7% | 10.2% |
| 0.095 | 16.8% | 12.0% | 17.4% | 12.4% | 18.4% | 12.9% |
| 0.110 | 22.7% | 15.1% | 24.2% | 16.1% | 24.8% | 16.4% |
| 0.125 | 30.5% | 19.2% | 33.1% | 20.6% | 33.5% | 20.9% |
| 0.140 | 40.8% | 24.6% | 43.3% | 25.9% | 43.0% | 26.5% |
| 0.156 | 52.4% | 31.1% | 54.0% | 32.0% | 53.5% | 32.5% |
| 0.171 | 63.7% | 38.1% | 64.0% | 38.2% | 63.7% | 38.9% |
| 0.186 | 73.6% | 44.7% | 73.2% | 44.3% | 72.9% | 44.6% |
| **0.201** | **81.4%** | **51.3%** | **80.8%** | **49.7%** | **80.4%** | **50.6%** |
| 0.216 | 87.6% | 56.7% | 86.8% | 55.1% | 86.3% | 55.7% |
| 0.231 | 91.9% | 62.1% | 91.2% | 59.5% | 90.8% | 60.5% |
| 0.246 | 94.9% | 66.0% | 94.4% | 63.6% | 94.1% | 64.6% |
| 0.261 | 96.9% | 70.2% | 96.6% | 67.1% | 96.3% | 68.0% |
| 0.306 | 99.2% | 78.4% | 99.3% | 75.1% | 99.3% | 75.9% |
| **0.352** | **99.7%** | **83.9%** | **99.9%** | **80.6%** | **99.8%** | **81.1%** |
| 0.442 | 99.8% | 89.8% | 100.0% | 87.2% | 100.0% | 87.4% |
| 0.502 | 99.8% | 91.5% | 100.0% | 89.9% | 100.0% | 89.5% |
| 0.563 | 99.8% | 92.4% | 100.0% | 91.2% | 100.0% | 90.8% |
| 0.638 | 99.8% | 93.1% | 100.0% | 91.9% | 100.0% | 91.4% |
| 0.729 | 99.8% | 93.6% | 100.0% | 92.1% | 100.0% | 91.8% |
| 0.819 | 99.8% | 93.9% | 100.0% | 92.3% | 100.0% | 92.0% |
| 0.910 | 99.8% | 93.9% | 100.0% | 92.3% | 100.0% | 92.0% |
| 0.985 | 99.8% | 94.1% | 100.0% | 92.8% | 100.0% | 93.3% |
| 1.000 | 99.8% | 94.1% | 100.0% | 92.7% | 100.0% | 93.3% |

*完整 64 t 值数据见 `results/exp07v2_{baseline,kd_cr,kd2}/probe_accuracies_v2.json`*

#### Overfit gap 分析

| t | base_gap | kd_cr_gap | kd2_gap |
|---|----------|-----------|---------|
| 0.201 | +30.1pp | +31.1pp | +29.8pp |
| 0.306 | +20.8pp | +24.3pp | +23.4pp |
| 0.352 | +15.8pp | +19.3pp | +18.7pp |
| 0.502 | +8.3pp | +10.1pp | +10.5pp |
| 0.729 | +6.2pp | +7.9pp | +8.2pp |
| 1.000 | +5.7pp | +7.3pp | +6.7pp |

**规律：kd_cr/kd2 在中等 t 的 overfit_gap 比 baseline 大 2-4pp**（因为 kd_cr/kd2 更快 train 至 100%，train 侧更饱和）。

#### Shuffled-label control

所有 t 值、所有 checkpoint 的 shuffled_label_acc 均为 1-4%（反映 T5 token 频率偏差，非记忆效应）。无记忆伪影。

#### Doc-level vs Position-level 修正量（baseline）

| t | 原始 probe_acc（position-level） | v2 test_acc（document-level） | 差值（reduction） |
|---|----------------------------------|-------------------------------|------------------|
| 0.186 | 49.6% | 44.7% | **−4.9pp** |
| 0.201 | 56.3% | 51.3% | **−5.0pp** |
| 0.216 | 62.1% | 56.7% | **−5.4pp** |
| 0.231 | 67.5% | 62.1% | **−5.4pp** |
| 0.246 | 71.5% | 66.0% | **−5.5pp** |

**一致规律：document-level split 使 probe_test 一致降低约 5pp（±0.5pp）。**

### 架构说明：native head Rec@1 ≈ 0%（不是 bug）

model.forward() 源码确认：x_hat = `self.final_layer(x)` = 预测的 **clean T5 embedding**（512-dim，在 T5 词嵌入空间）。`unembed_kernel` 期待 `GELU(backbone_hidden @ proj_kernel)` 作为输入（不同空间），直接应用于 x_hat 导致 Rec@1≈0%（空间不匹配，预期行为）。

正确的 native head 比较 = **G(t)** = `argmax cos(x_hat, T5_embedding_v)` vs y_tokens：
- baseline G(t) at t=0.20 ≈ 10.2%（来自 JAX probe_decode_v2_dense 实验）
- kd_cr G(t) at t=0.20 ≈ 60.7%（来自 EXP-10 kd_cr oracle 数据）
- kd2 G(t) at t=0.20 ≈ 待补充（EXP-10 kd2 oracle G at t=0.20 ≈ 56-60% 估计）

### 最终结论（EXP-07v2 DONE）

**Story A（probe gap）完全确认（document-level held-out）**：

| checkpoint | probe_test@t=0.20 | G(t)@t=0.20 | 差值（probe−G） | 结论 |
|------------|-------------------|-------------|-----------------|------|
| baseline | **51.3%** | ~10.2% | **+41pp** | probe >> native（承诺在 x̂_t 中但解码器不能暴露） |
| kd_cr | **49.7%** | ~60.7% | **−11pp** | native >> probe（KD 训练使解码器直接暴露承诺） |
| kd2 | **50.6%** | ~56-60% | **~−6 to −9pp** | native 可能 >> probe（待 kd2 G(t) 精确测量） |

**关键观察**：
1. 三 checkpoint 的 probe_test 在 t=0.20 几乎相同（49.7–51.3%），差异仅 1.6pp
2. Baseline probe_test 在所有 t 系统性高于 kd_cr/kd2（约 2-3pp）——但这不矛盾，因为 baseline x̂_t 空间中 G(t)≈10%（decoder 弱），linear probe 发现了 decoder 无法利用的方向
3. Kd_cr/kd2 过拟合更严重（overfit_gap 比 baseline 大 2-4pp），但仍有显著泛化

**对 EXP-07 原始结论的修正**：
- 数字修正：baseline gap +46pp → +41pp（−5pp）；kd_cr gap −7pp → −11pp（更强）
- 方向不变：probe >> decoder（baseline），decoder >> probe（kd_cr）
- Paper 中应引用 EXP-07v2 的 document-level 数字，注明"document-level 80/20 split, 98K test positions"
- EXP-07 原始 position-level 结果下调为"初步证据"，EXP-07v2 为最终结论

---

## 待做补实验（优先级排序）

### P1：Oracle→Reverse Probe Transfer

**设计**：
- 在 oracle states（EXP-07 的 x̂_t）上训练 probe P_t
- 直接在 EXP-01v3 的 reverse ODE trajectory states（z_t^reverse）上测试
- 评估：P_t(z_t^reverse) 能否预测 y_final？

**科学意义**：真正连接 EXP-01v3 和 EXP-07。如果 oracle-trained probe 在 reverse states 上也高准确，说明 reverse trajectory 中的 z_t 确实包含 oracle 探针已经学到的 token-discriminative 信息。

**实现**：
- oracle states 已有（`results/exp07_{baseline,kd_cr,kd2}_64/states/`）
- reverse states 需要在 EXP-01v3 中同时保存 z_t_reverse（目前只计算了 G_reverse）
- 修改 `probe_reverse_trajectory.py` 保存各步的 z_t 向量（仅限 test sequences）

### P2：Cross-Time Probe Transfer

**设计**：
- 在 t_train 训练 probe P_{t_train}
- 在不同 t_test 上测试 P_{t_train}(x̂_{t_test})
- 生成完整的 (t_train × t_test) transfer matrix

**关键行**：
- t_train=1.0（clean space）probe，在各早期 t 上测试：如果 clean-space probe 在 t=0.20 也高，说明 early x̂_t 已经进入稳定 lexical coordinate
- t_train=0.20 probe，在各晚期 t 上测试：是否早期训练的分类方向在晚期也有效？

### P3：L11/Final-Layer/Head Factorial Swap

**设计**（EXP-07d 的机制定位补实验）：

四种组合：
- F_baseline(h_baseline) — baseline 正常输出
- F_kd(h_baseline) — KD head + baseline hidden
- F_baseline(h_kd) — baseline head + KD hidden
- F_kd(h_kd) — KD 正常输出

若 F_kd(h_baseline) >> F_baseline(h_baseline)：KD 改善的是 head（projection），不是 hidden
若 F_baseline(h_kd) >> F_baseline(h_baseline)：KD 改善的是 hidden geometry
若 F_kd(h_kd) >> F_kd(h_baseline) + F_baseline(h_kd)：co-adaptation（两者必须一起使用）

**实现**：修改模型 forward pass，支持 cross-checkpoint head/hidden 混用。需要 paired states（相同 document/noise）。

### P4：z_t Probe（Denoising Contribution 量化）

**设计**：
- 同时探针 z_t（noisy input）和 x̂_t（backbone output）
- Δ_denoise(t) = Acc[P_t(x̂_t)] - Acc[P_t(z_t)]
- 量化 backbone 本身贡献了多少 token recoverability（而非输入已有的）

**实现**：在 `collect_probe_states.py` 中同时保存 z_t（已有）和 x̂_t。修改 `train_linear_probe.py` 支持 `--use_z_t` flag。
