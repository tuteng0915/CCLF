# EXP-06 Spec — Prior Subtraction (Debiased Commitment)

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：验证"去先验化"后的承诺曲线，决定论文是否需要报告去偏结果。**

EXP-05 估计了先验分布 q_t（backbone 在 batch-shuffled z_t 上的词频预测）。EXP-06 用这个估计做实际的"先验减法"，计算**去偏承诺度**：

r_t(v) = log p_t(v) − log q_t(v)

这个 r_t 是一个 pointwise mutual information（PMI）风格的量：它测量的是"与 x_clean 相关的信息量"超过"纯词频先验信息量"的部分。

**要验证的核心假说**：
- 如果 argmax(r_t) 的准确率曲线（去偏 G_debiased(t)）与原始 G(t) 形状相似 → 先验偏置不影响论文结论
- 如果 G_debiased(t) 的"悬崖"显著右移（更高 t 才出现） → 原始 G(t) 曲线被先验偏置拉升，论文需要报告去偏结果
- 如果 G_debiased(t) 根本无悬崖 → 承诺现象大部分是频率偏置的产物（重大 negative 发现）

**与 EXP-05 的关系**：EXP-05 是信号探测（先验有多强），EXP-06 是应用（先验减法后的曲线是什么）。两者可以在同一个脚本中完成。

---

## Implementation Plan

### 方法

1. 对每个 t，同时运行：
   - oracle 探针：z_t = t·x_clean + (1−t)·ε，得到 log p_t(v)
   - shuffle 探针（N_SHUFFLE 次平均）：log q_t(v)
2. 计算 r_t = log p_t − log q_t
3. 报告：
   - G(t) = acc(argmax p_t == y_true)，原始
   - G_debiased(t) = acc(argmax r_t == y_true)，去偏
   - q_bias(t) = acc(argmax q_t == y_true)，纯先验准确率

### 脚本

`experiments/probe_elf/probe_prior_debias.py`（与 EXP-05 同一脚本，增加 G_debiased 输出列）

### 输出格式

```json
[
  {"t": 0.10, "G_oracle": 0.126, "G_debiased": 0.118, "G_prior": 0.034},
  {"t": 0.20, "G_oracle": 0.585, "G_debiased": 0.539, "G_prior": 0.041},
  {"t": 0.30, "G_oracle": 0.893, "G_debiased": 0.871, "G_prior": 0.048},
  ...
]
```

---

## 实验结果（Results）

**状态**: COMPLETED（数据来自 EXP-05 results，2026-07-18）

**数据文件**：
- `results/exp05_baseline/prior_debias.json`
- `results/exp05_kd_cr/prior_debias.json`

### 去偏 G(t) 对比表

**baseline**：

| t | G_oracle | G_prior | G_debiased | debias_delta |
|---|----------|---------|------------|-------------|
| 0.10 | 1.05% | 0.01% | 0.64% | −0.41pp |
| 0.20 | 9.89% | 0.19% | 7.80% | −2.10pp |
| 0.30 | 53.9% | 1.24% | 36.9% | **−17.0pp** |
| 0.50 | 80.1% | 1.45% | 58.5% | **−21.6pp** |
| 0.70 | 85.6% | 1.35% | 64.9% | **−20.7pp** |
| 1.00 | 100.0% | 3.65% | 97.9% | −2.1pp |

**kd_cr**：

| t | G_oracle | G_prior | G_debiased | debias_delta |
|---|----------|---------|------------|-------------|
| 0.10 | 12.9% | 3.70% | 1.33% | −11.6pp |
| 0.20 | 61.1% | 3.62% | 41.1% | **−20.0pp** |
| 0.30 | 90.8% | 3.44% | 83.8% | **−7.0pp** |
| 0.50 | 99.7% | 2.28% | 95.6% | −4.0pp |
| 0.70 | 99.9% | 2.03% | 96.5% | −3.5pp |
| 1.00 | 100.0% | 2.50% | 96.3% | −3.7pp |

### 关键发现

1. **G_prior 很小但去偏影响大**：baseline 在 t=0.30 时 G_prior=1.2%，但 G_debiased 低了 17pp。原因：去偏不是简单地 G_oracle − G_prior，而是 log(p/q) 的 argmax 与 log(p) 的 argmax 不同——小量的频率先验会改变近似同分的候选词排名。

2. **baseline G(t) 被频率偏置膨胀约 17-22pp**（t=0.30-0.70 区间）：真实的"内容感知预测"准确率应用 G_debiased 报告。

3. **kd_cr 的膨胀在 t=0.30 后迅速减小**：t=0.30 只有 −7pp（vs baseline 的 −17pp）；t=0.70 时 −3.5pp。说明 KD 训练后模型的高 G(t) 更多来自内容感知，而非频率预测。

4. **kd_cr 在低 t（0.10-0.20）膨胀反而更大**（−11pp at t=0.10 vs baseline −0.4pp）：在极低信噪比时，kd_cr 也在靠频率预测，但因为基础准确率高，绝对膨胀量也大。

### 论文使用建议（执行 EXP-06 决策规则）

适用条件：G_debiased 在 t∈[0.2,0.4] 低于 G_oracle 7-20pp → **中等问题，需要在论文中说明**

建议：
- **报告两条曲线**：G_oracle（原始报告指标）和 G_debiased（去偏参考）
- **重点说明**：kd_cr 的高 G(t)（t>0.30 时 >90%）大部分是真实内容感知（去偏后 >84%），baseline 的 G_oracle≈85% 被膨胀约 20pp 至 G_debiased≈65%
- **定性结论不变**：即使用 G_debiased，kd_cr（84%+）vs baseline（37%）在 t=0.30 的差距仍然显著（47pp），"KD 加速 token 承诺"的结论仍然成立
- **保守声明**：在正文中说明 G(t) 包含词频信号，并在 appendix 报告去偏曲线

---

## ⚠️ 方法论问题 & 待修正 TODO（2026-07-21 审查）

EXP-06 直接继承了 EXP-05 的所有方法论问题。在 EXP-05 的 q_t 估计未修正之前，EXP-06 的去偏结果不可信。

### 具体问题

1. **当前 G_debiased 不能解释为"去除频率偏置后的真实 G(t)"**：因为减去的 q_t 本身包含实例信息（batch shuffle 问题），log(p/q) 的 argmax 是 PMI 风格的判断，会过度惩罚高频词。

2. **"baseline G(t) 被膨胀 17-20pp"的结论需要重新检验**：这个数字完全依赖 q_t 的正确估计。用 global null prior（零输入）重做后可能得到不同结果。

3. **baseline vs kd-cr 的对比受 logit calibration 影响**：两个模型的 logit 分布 scale 不同，去偏影响的大小差异可能是 calibration artifact 而非先验依赖程度的差异。

### 修正 TODO

- [ ] **P0**：等待 EXP-05 修正（global null prior + 平均概率）完成后，用新 q_t 重新计算 EXP-06 结果
- [ ] **P1**：用修正后的 q_t 重报去偏曲线；若 G_debiased 仍显著低于 G_oracle，才保留"baseline 有频率偏置"的结论
- [ ] **P2**：报告 final-token rank 分布（而非只有 argmax accuracy），rank 1→5 的分布变化更能体现先验影响
