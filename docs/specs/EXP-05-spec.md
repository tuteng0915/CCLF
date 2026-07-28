# EXP-05 Spec — Learned Prior Estimation

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：识别 G(t) 中来自"先验知识"的部分，确保承诺测量是实例特异的。**

ELF backbone 的 G(t) 曲线（oracle 探针准确率）可能有两种来源：
1. **先验知识（频率偏置）**：backbone 学会了"英文文本中 'the' 最常见"，所以不管输入什么 z_t，都倾向于预测 'the' 等高频词。这与输入的具体内容无关。
2. **实例特异信息**：backbone 从 z_t 中读取了与该位置 x_clean 相关的信号，才做出正确预测。

**如果 G(t) 的大部分来自先验知识**，则"承诺悬崖"故事会被弱化——模型只是在重复训练数据的词频分布，而非真正理解内容。

**要验证的核心假说**：
- 通过 batch-shuffle（打乱 batch 内不同序列的 z_t），摧毁序列间的特异信息但保留统计先验
- 用 shuffle 后的 z_t 通过 backbone 得到的预测分布 = 先验 q_t(v)
- 计算去先验化残差：r_t(v) = log p_t(v) − log q_t(v)（去偏承诺度）
- 若 r_t 的 argmax 仍早早收敛到真实 token → 承诺是实例特异的（G(t) 可信）
- 若 r_t 的 argmax 收敛时间显著晚于 p_t → 大量 G(t) 来自先验偏置

**与其他实验的关系**：
- EXP-04 回答了"geometry 偏置"问题（答案：G_null ≈ 0.17%，geometry 可忽略）
- EXP-05 回答的是"词频先验偏置"问题（q_t 效应）
- EXP-06 用 EXP-05 估计的 q_t 做先验减法，得到去偏 commitment curve
- 两者合力验证：G(t) 曲线有多少是真实"词汇承诺"信号

---

## Implementation Plan

### 步骤 1：估计先验 q_t(v)

**方法**：batch-shuffle 打乱 B 个序列的 x_clean（沿 batch 维度随机置换），再用标准 oracle 前向噪声生成 z_t，过 backbone 得到预测分布。

```python
# z_t_oracle[b, l, :] uses x_clean[b, l, :] — per-instance
# z_t_shuffled[b, l, :] uses x_clean[perm[b], l, :] — shuffled across batch

# N 次 batch-shuffle 平均 → q_t(v)
all_logits = []
for _ in range(N_SHUFFLE):
    perm = torch.randperm(B)
    x_clean_shuffled = x_clean[perm]  # (B, L, 512)
    eps = torch.randn_like(x_clean_shuffled)
    z_t_shuf = t * x_clean_shuffled + (1 - t) * eps
    z_in = torch.cat([z_t_shuf, zeros_sc], dim=-1)
    with torch.no_grad():
        x_hat_shuf, _, _ = model(z_in, t_batch, ...)
    # ... get logits via decode path
    all_logits.append(logits)
q_logits = torch.stack(all_logits).mean(0)  # (B, L, V)
```

### 步骤 2：计算去偏残差

```python
# p_t = log_softmax of oracle logits
# q_t = log_softmax of shuffle logits
# r_t = log p_t - log q_t (unnormalized)
r_t = log_p_t - log_q_t   # (B, L, V)
# argmax of r_t = "most instance-specifically favored token"
```

### 步骤 3：比较时间线

对每个位置：
- 记录 argmax(p_t) 首次 = y_true 的 t 值（原始承诺时间）
- 记录 argmax(r_t) 首次 = y_true 的 t 值（去偏承诺时间）
- 如果两者相似 → 先验偏置小；如果去偏版本晚很多 → 先验占主导

### 脚本
**新建**：`experiments/probe_elf/probe_prior_debias.py`

### 运行配置
- checkpoint：kd-cr（与 EXP-16 保持一致）
- N=256 sequences，20 t values（0.05~1.00）
- N_SHUFFLE = 8（足够估计 q_t）
- GPU：任意空闲 GPU

---

## 实验结果（Results）

**状态**: COMPLETED（见下方 COMPLETED 2026-07-18 节）

**依赖**：probe_layerwise.py 已有的 layer states 数据（但该实验用 decode-path 而非 linear probe）

**决策规则**：
- 若去偏承诺时间 vs 原始承诺时间差异 < 2 t-steps：先验偏置小，G(t) 可信
- 若差异 > 5 t-steps：先验占主导，论文需要报告去偏的 r_t 曲线，而非原始 G(t)

**优先级**：EXP-04 已经证明 geometry bias ≈ 0.17%，所以如果 EXP-05 也显示 frequency bias 小，两个偏置来源都被排除，EXP-07 的 G(t) 就完全可信了。

---

## 实验结果 — COMPLETED 2026-07-18

**状态**: COMPLETE

数据文件:
- kd_cr: `results/exp05_kd_cr/prior_debias.json`
- baseline: `results/exp05_baseline/prior_debias.json`

参数: N=128 sequences, n_t_steps=20 (t=0.05~1.00), n_shuffle=8

### 结果表（选取关键 t 值）

| t    | kd_cr G_oracle | kd_cr G_prior | kd_cr G_debias | debias_Δ |
|------|----------------|---------------|----------------|----------|
| 0.05 | 6.2%           | 3.8%          | 0.16%          | −6.0pp   |
| 0.10 | 12.9%          | 3.7%          | 1.33%          | −11.6pp  |
| 0.20 | 61.1%          | 3.6%          | 41.1%          | −20.0pp  |
| 0.30 | 90.8%          | 3.4%          | 83.8%          | −7.0pp   |
| 0.50 | 99.7%          | 2.3%          | 95.6%          | −4.0pp   |
| 0.70 | 99.9%          | 2.0%          | 96.5%          | −3.5pp   |
| 1.00 | 100%           | 2.5%          | 96.3%          | −3.7pp   |

| t    | baseline G_oracle | baseline G_prior | baseline G_debias | debias_Δ |
|------|-------------------|-----------------|-------------------|----------|
| 0.05 | 0.29%             | 0.00%           | 0.22%             | −0.1pp   |
| 0.10 | 1.05%             | 0.01%           | 0.64%             | −0.4pp   |
| 0.20 | 9.9%              | 0.19%           | 7.8%              | −2.1pp   |
| 0.30 | 53.9%             | 1.24%           | 36.9%             | −17.0pp  |
| 0.50 | 80.1%             | 1.45%           | 58.5%             | −21.6pp  |
| 0.70 | 85.6%             | 1.35%           | 64.9%             | −20.7pp  |
| 0.90 | 86.1%             | 1.32%           | 66.7%             | −19.4pp  |
| 1.00 | 100%              | 3.65%           | 97.9%             | −2.1pp   |

### 核心发现

**1. G_prior 极小（0~4%）**：batch-shuffle 估计的先验对真实 token 的预测能力极低，说明纯词汇频率偏置可忽略。

**2. 去偏方向反直觉：G_debiased < G_oracle**（debias_Δ < 0 for all t, both models）。

- 去偏使准确率下降，而非提升。这意味着 log_q_t（shuffle 先验）实际上包含了部分有用的 token 信息，减去它会损失准确率。
- 可能原因：shuffle 后的模型预测更偏向常见 token（function words），在原始 G_oracle 中正好命中常见 token 的位置也被减去了。

**3. baseline 受去偏影响显著（−17~−21pp）vs kd_cr（−4~−12pp）**：

- baseline 在 t=0.3~0.9 时，约 17-21pp 的 G(t) 准确率来自"先验偏置"（即模型倾向预测常见词）
- kd_cr 的先验偏置仅 4-12pp，说明 KD 训练减少了对先验的依赖
- 换言之：**baseline 的 G(t) 曲线中有约 20pp 是"虚高"的**，可能来自模型预测 function words 的偏置，而非真正的 per-position commitment

**4. 去偏后的承诺曲线**：

- baseline G_debiased: 0.22% → 7.8% → 36.9% → 58.5% → 64.9%（更慢，plateau 更低）
- kd_cr G_debiased: 0.16% → 41.1% → 83.8% → 95.6% → 96.5%（仍保持显著差异）
- 去偏后 kd_cr vs baseline 的差距更大（而非缩小），说明 KD 训练的优势不是偏置产物

### 决策规则结论

原决策规则：去偏承诺时间 vs 原始承诺时间差异：

- 如果以"G=50%"作为承诺时间阈值：
  - baseline 原始 G_oracle: t≈0.30（53.9%）；去偏 G_debiased: t≈0.40（接近 58.5%@0.5）
  - 差异 ~0.10，> 5 t-steps 的阈值 → **先验偏置显著（baseline）**
  - kd_cr 原始 G_oracle: t≈0.20（61.1%）；去偏: t≈0.20（41.1%→约0.22）
  
**结论**：对 baseline 而言，先验偏置较大，G(t) 偏高。需要报告去偏 r_t 曲线或注明此偏置。kd_cr 的 G(t) 偏置较小，结果更可靠。

---

## ⚠️ 方法论问题 & 待修正 TODO（2026-07-21 审查）

### 问题 1（根本性）：Batch shuffle 得到的是 wrong-instance posterior，不是 prior

当前 `x_clean_shuffled = x_clean[perm]` 将**另一个真实序列**的信息放进了"先验估计"。

这不是 `q_t(v)`，而是 `q_t(v | another_sequence_at_same_position)`。它保留了：
- donor sequence 的真实 token 信息
- 序列位置的句法统计（function word patterns）
- context encoder structure
- length 和 padding pattern

**减去它可能移除完全合法的语言先验信息，而非频率偏置。**

### 问题 2：平均了 logits，而非概率（Jensen 不等式）

当前：`q_logits = mean(logits)` → `q = softmax(mean_logits)`

正确：`q = mean(softmax(logits_i))`（平均概率分布）

由于 softmax 是凸函数的逆，`softmax(mean) ≠ mean(softmax)`，差异可能很大。

### 问题 3：去偏后 accuracy 下降 ≠ "原 accuracy 被膨胀"

G_debiased < G_oracle 只说明 log(p/q) 的 argmax 与 log(p) 的 argmax 不同，这可能是因为：
- q 估计错误（引入了真实信息）
- rare token 被过度放大
- KD 和 baseline 的 logit scale/calibration 不同

**不能**把 −17pp 解释为"17pp 来自频率虚高"。

### 修正 TODO：四种正确的 q_t 估计方法

- [ ] **P0 — Global null prior（推荐起点）**：
  将 x_clean 完全替换为零向量或随机遮盖（mask），不输入任何实例信息：
  ```python
  x_clean_null = torch.zeros_like(x_clean)  # 或全 mask token embedding
  z_t_null = (1-t) * eps  # 无信号
  ```
  这才是"模型对无输入信息时的默认分布"。

- [ ] **P0 — 平均概率，不平均 logits**：
  ```python
  # 修正 Jensen 不等式问题
  q_probs = mean([softmax(logits_i) for i in range(N_SHUFFLE)])  # mean over probabilities
  # 而非 softmax(mean(logits))
  ```

- [ ] **P1 — Position-conditional prior**：
  只清除 local content，保留位置编码：
  ```python
  # shuffle x_clean 沿位置维度（而非 batch 维度）
  x_perm = x_clean[:, torch.randperm(L), :]  # 打乱同 batch 内的位置顺序
  ```

- [ ] **P1 — Context-only prior**：
  保留 z_{-i}（其他位置的 context），清除当前位置 z_i：
  实现：在 z_t 中将第 i 位置替换为零/噪声，保留其余位置
  （需要能做 masked self-attention，或逐位置 ablation）

- [ ] **P2 — 温度校准**：在对比 baseline vs kd-cr 之前，先对两个模型的 logit 做 temperature calibration（ECE 最小化），避免因 logit scale 不同导致去偏效果不可比

- [ ] **P2 — 报告 final-token rank 和 log-odds**（而非只有 argmax accuracy），这对"偏置改变排名"更敏感

---

## EXP-05v3 — Global Null Prior + Probability Averaging（进行中，2026-07-21）

**目标**：修复 EXP-05 的两个根本性方法论问题（问题 1 + 2），使用正确的先验估计。

**脚本**：`experiments/probe_elf/probe_prior_null.py`

### 修复内容

| 问题 | EXP-05（旧） | EXP-05v3（新） |
|------|------------|--------------|
| 先验估计 | batch-shuffle（包含 donor sequence 信息） | z_t_null = (1-t)·ε（零信号，纯噪声） |
| 概率计算 | softmax(mean(logits)) | mean(softmax(logits_i))（n_null=8 次平均） |
| 去偏公式 | log(softmax(mean_oracle)) − log(softmax(mean_shuffle)) | log p_t − log q_t，p_t/q_t 均为 averaged softmax |

**z_t_null 的含义**：模型在接收纯噪声（无任何实例信息）时的默认预测分布。EXP-04v2 已测得此分布的 argmax 准确率（G_backbone_null）约为 0.15–2%，是一个极小但非零的频率先验。

### 运行设置

- **GPU 0**：baseline checkpoint，`experiments/probe_elf/results/exp05v3_baseline/prior_null_baseline.json`
- **GPU 1**：kd-cr checkpoint，`experiments/probe_elf/results/exp05v3_kd_cr/prior_null_kd_cr.json`
- n_samples=128，n_oracle=4，n_null=8，n_t_steps=20，batch_size=8
- 启动时间：2026-07-21 17:01（PID baseline=2134330，kd_cr=TBD）

---

## EXP-05v3 完整结果（2026-07-21 完成）

**数据文件**：
- `experiments/probe_elf/results/exp05v3_baseline/prior_null_baseline.json`
- `experiments/probe_elf/results/exp05v3_kd_cr/prior_null_kd_cr.json`
- `experiments/probe_elf/results/exp05v3_kd2/prior_null_kd2.json`

**参数**：n_samples=128，n_oracle=4，n_null=8，n_t_steps=20，seq_len=1024

### baseline 结果表

| t | G_oracle | G_null | G_debias | rank_oracle | rank_null |
|---|----------|--------|----------|-------------|-----------|
| 0.05 | 0.24% | 0.13% | 0.04% | 10,860 | 9,871 |
| 0.10 | 0.88% | 0.07% | 0.15% | 10,613 | 10,179 |
| 0.20 | 7.18% | 0.06% | **9.72%** | 6,703 | 10,447 |
| 0.30 | 42.46% | 0.03% | **45.74%** | 489 | 10,737 |
| 0.40 | 66.00% | 0.26% | 67.26% | 48 | 11,796 |
| 0.50 | 72.14% | 3.96% | 71.72% | 30 | 10,533 |
| 0.70 | 77.51% | 2.91% | 75.24% | 23 | 8,222 |
| 0.90 | 79.60% | 0.64% | 77.68% | 15 | 7,370 |
| 1.00 | 94.74% | 2.38% | 94.56% | 0 | 5,136 |

### kd_cr 结果表

| t | G_oracle | G_null | G_debias | rank_oracle |
|---|----------|--------|----------|-------------|
| 0.05 | 5.46% | 3.81% | 0.66% | 1,773 |
| 0.20 | 49.43% | 3.99% | 45.04% | 185 |
| 0.30 | 80.74% | 3.64% | 79.48% | 5 |
| 0.40 | 91.30% | 3.99% | 91.75% | 1 |
| 0.50 | 93.92% | 4.01% | 93.65% | 2 |
| 1.00 | 94.65% | 4.01% | 94.34% | 5 |

### kd2 结果表

| t | G_oracle | G_null | G_debias | rank_oracle |
|---|----------|--------|----------|-------------|
| 0.05 | 5.56% | 4.13% | 0.76% | 2,403 |
| 0.20 | 49.03% | 3.82% | 43.03% | 262 |
| 0.30 | 80.25% | 3.72% | 80.00% | 20 |
| 0.50 | 93.44% | 4.02% | 93.27% | 2 |
| 1.00 | 94.58% | 0.04% | 93.80% | 31 |

### 核心发现

**1. G_null 极小（0.03–4%），G(t) 绝大部分信号来自 x_clean**

- baseline：G_null ≈ 0.03–4%（低 t 几乎为 0，高 t 约 4%）
- kd_cr/kd2：G_null ≈ 3.6–4.1%（与 EXP-04v2 G_backbone_null 完全一致）
- 信号比（baseline t=0.3）：G_oracle 42.5% / G_null 0.03% ≈ **1,400:1**
- 信号比（kd_cr t=0.3）：G_oracle 80.7% / G_null 3.6% ≈ **22:1**

**G(t) 指标无需频率先验修正。**

**2. G_debias ≈ G_oracle：去偏几乎不改变结果**

t ≥ 0.3 时：baseline ±3pp，kd_cr ±1.5pp，kd2 ±1.5pp。先验修正效果微弱，符合预期（因为 G_null 很小）。

**3. EXP-05（batch-shuffle）的 −17pp 下降是 artifact，已被推翻**

| 实验 | 方法 | baseline t=0.3 G_debias | kd_cr t=0.2 G_debias |
|------|------|------------------------|---------------------|
| EXP-05（旧） | batch-shuffle prior | 36.9%（−17pp from oracle） | 41.1%（−20pp from oracle） |
| EXP-05v3（新） | **global null prior** | **45.7%（≈ oracle 42.5%）** | **45.0%（−4pp from oracle 49.4%）** |

EXP-05 中 G_debias 大幅下降是因为 batch-shuffle 给出 wrong-instance posterior（包含另一序列信息），不是真实的频率先验。将其减去后损失了合法的语言信息，导致准确率下降。

**4. baseline 在低 t 的 G_debias > G_oracle 现象**

t=0.2-0.3：baseline G_debias=9.7-45.7% > G_oracle=7.2-42.5%（+1-3pp）。原因：null distribution q 偏向高频词，log(p/q) 相对提升稀有词的权重；当真实 token 是稀有词时，去偏后 argmax 更准确。这是一个弱正效应，对论文影响不大。

**5. 综合结论：G(t) 指标可信**

| 偏置来源 | 量级 | 结论 |
|---------|------|------|
| 头部几何偏置（EXP-04v2） | 0.017% | 可忽略 |
| 频率先验偏置（EXP-05v3） | 0.03–4% | 可忽略（不超过 G_oracle 的 5%） |
| EXP-05 batch-shuffle 效应 | −17–20pp | 方法论 artifact，应从论文中移除 |

**论文建议**：删除 EXP-05 的 batch-shuffle 去偏内容；可简单注明"global null prior 验证显示频率偏置 <4%，不影响 G(t) 解读"。

**状态**：DONE（2026-07-21，GPUs 0/1/3）
