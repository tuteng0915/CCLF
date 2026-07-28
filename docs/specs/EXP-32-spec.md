# EXP-32 / EXP-33 / EXP-34 Spec — DF Step Sweep + dec_sc × DF Interaction

## 实验背景

EXP-31/31b 发现 kd2 的 DF 增益极大（freeze_1.0：−48.9%），kd_cr DF 全面恶化。两个自然后续问题：
1. **EXP-32**：kd2 DF 收益如何随 ODE 步数扩展？
2. **EXP-33/34**：dec_sc（EXP-13 验证的最强推理改进）与 DF 是**互补**还是**竞争**关系？

---

## 实验设计

**EXP-32**（GPU 4，PID 298909，seed=456）：
- kd2 checkpoint，none + freeze_1.0，8/16 ODE steps
- 采样配置：spec32_kd2_steps.yml
- 输出：outputs/exp32_kd2_steps/

**EXP-33**（GPU 6，PID 298910，seed=123）：
- kd_cr checkpoint + dec_sc_mode="decode"，7 DF 条件（32 步）
- 采样配置：spec33_df_decsc.yml
- 输出：outputs/exp33_kd_cr_decsc/

**EXP-34**（GPU 2，PID 298911，seed=456）：
- kd2 checkpoint + dec_sc_mode="decode"，7 DF 条件（32 步）
- 采样配置：spec33_df_decsc.yml
- 输出：outputs/exp34_kd2_decsc/

所有实验：n_samples=256，ODE uniform schedule，GPT-2-large PPL。

---

## EXP-32 结果：kd2 DF 步数扩展

| steps | none | freeze_1.0 | Δ |
|-------|------|-----------|---|
| 8 | 688.11 | 615.30 | −10.6% |
| 16 | 602.64 | 486.80 | −19.2% |
| 32 | 282.52* | 144.42* | **−48.9%** |
(*EXP-31b)

DF 收益**超线性扩展**：8 步仅−10.6%，32 步−48.9%。

**解释**：tmin=0.7 门控意味着 DF 仅在 ODE 轨迹的高噪声部分生效（t≥0.7）。
- 8 步：约 2 步在 DF 区间，冻结机会极少
- 32 步：约 10 步在 DF 区间，每步冻结的干净位置为后续更多步骤提供上下文
- 超线性是因为冻结步数 × 受益步数 ∝ N²

---

## ⚠️ EXP-33 / EXP-34 — INVALID（退化输出）

**EXP-33 和 EXP-34 结果已全部作废。**

**根本原因**：`spec33_df_decsc.yml` 使用了 `dec_sc_mode: "decode"` 但**没有** `dec_sc_apply_t_min` 门控（等同于 `tmin=0.0`，即全范围 decode branch SC）。这与 EXP-13 中已知会产生退化的配置完全相同：

- 早期高噪声步骤（t≥0.7）中，decode branch 的 x̂_t 质量极差
- 无 tmin 门控 → decode branch 输出喂入自条件 → 正反馈循环
- 所有 checkpoint 均产生**退化重复文本**：
  - kd_cr：`"centre centre centre centre centre..."`
  - kd2：`"eight twenty eight twenty eight twenty..."`
  - baseline：`"AS. AS. AS. AS. AS. AS."`
- GPT-2 对重复序列给出人为偏低的 PPL，导致所有 PPL 数字**无效**

**正确做法**：需使用 `dec_sc_apply_t_min: 0.5`（参见 EXP-13v2）。但 EXP-13v2 显示即使 tmin=0.5，kd2 仍不稳定。

**上方 EXP-33/34 数字已从论文中完全移除。**

---

## 状态

- **EXP-32**: DONE — 2026-07-21，GPU 4，PID 298909
- **EXP-33**: INVALID — 无 tmin gate 的 dec_sc 产生退化文本，PPL 无效
- **EXP-34**: INVALID — 同上

---

## ⚠️ 方法论问题（2026-07-22 审查）

### 1. 仅三个数据点，且变量完全混淆

8/16/32 步的 DF 效果改善（10.6% → 19.2% → 48.9%）同时混淆了：
- solver accuracy（步数越多，baseline PPL 本身从 688→534→283 急剧改善，基线已变）
- DF 触发次数（更多步数 = 更多 freeze 机会）
- freeze 后继续演化的步数（剩余步骤数影响修正能力）
- 每步 state distribution（步数改变 Δt 和每步 noise level）
- baseline generation quality（none PPL 本身变化 2.4×）

三个 ratio 不能推断任何函数形式，更不能推出 benefit ∝ N²。

### 2. "超线性 / N²"说法没有证据

spec 的"超线性是因为冻结步数 × 受益步数 ∝ N²"是纯推测。从 3 个有噪声的 PPL ratio 推导幂律没有统计依据。需要至少 6-8 个数据点 + 控制变量才能拟合幂律。

### 3. none baseline 变化破坏了 ratio 解释

8步 none PPL=688，32步 none PPL=283。这 2.4× 的变化说明 PPL 大幅下降主要来自 solver 质量提升，而非 DF。DF ratio 用一个本身不稳定的基线归一化，会放大 DF 效果的数值。

### 4. 正确分离变量的实验设计

**固定 DF 干预次数**：在 8/16/32/64 步采样器中，仅在相同 log-SNR 的 3 个时间点应用 DF。若效果仍随步数增长，说明后续精炼步数重要，而非干预次数。

**固定后续受益步数**：只在最后 K 步内启用 DF（而非固定 t_min），测量 K 固定时不同总步数的效果。

**干预次数 sweep**：在相同 32-step 采样器中使用 1/2/4/8/10 次 DF，这才能真正检验累积作用是否线性/饱和/超线性。

### 5. 安全结论

> Under the tested kd2 configuration, DF's PPL benefit is strongly step-budget-dependent: 8 steps → 10.6%, 16 steps → 19.2%, 32 steps → 48.9% relative improvement. Whether this reflects DF effectiveness or solver-quality improvement in the baseline cannot be determined from these three data points.

**不能说**：DF 产生 N² 超线性收益；冻结步数 × 受益步数 ∝ N²。
