# EXP-09 Spec — Contextual Bootstrapping

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：测试已承诺的位置是否帮助未承诺的位置更早做出决定。**

EXP-16 显示：在 t=0.50 时，约 73.4% 的位置已经承诺（decode 正确）。EXP-08 假设这些位置的信号是从粗到细的。EXP-09 要回答：**已承诺位置的信息，是否作为"更干净的上下文"帮助了旁边的位置承诺更早？**

这是"上下文自举（contextual bootstrapping）"假说：
- 早期承诺的位置（通常是高频词、功能词、可预测内容词）变成了"已确定"的上下文
- 这些确定的上下文减少了周围位置的不确定性，让它们也更早做出决定
- 最终形成一个"传播效应"：承诺从语言中最确定的位置向外传播

**要验证的核心假说**：
- **H_bootstrap**：在 t < 0.50 时已承诺的位置（y*_i 已知），其邻域内的未承诺位置在 t=0.50 时的承诺率高于远距离未承诺位置的承诺率
- 控制：比较已承诺位置 ±k positions 内 vs 外的未承诺率

**具体实验设计**：

```python
# 对每个序列，在 t=T_commit 时：
# - 获取每个位置的承诺状态（已承诺 = True/False）
# - 对于未承诺位置，记录其"与最近已承诺位置的距离"
# - 在 T_next > T_commit 时，检查它们是否承诺了
# - 比较：近距离（d <= 5）vs 远距离（d > 10）的后续承诺率

committed_at_t = (decode_correct_at_t == True)  # (B, L) bool
# For each uncommitted position, find nearest committed neighbor
# Group by distance and compare subsequent commitment rate
```

**更强的设计（反事实）**：
- 运行 oracle，但**强制将部分随机位置设为 clean**（z_{t,i} = x_clean_i）
- 对比：被强制 clean 的位置邻域 vs 不被强制的位置邻域，哪个未承诺位置承诺更快
- 这需要修改 oracle 协议，不依赖真实轨迹

---

## Implementation Plan

### 基础版（观察相关性，不需要额外 GPU 实验）

1. 复用 EXP-16 per-position commitment timing 数据
2. 对每个位置对 (i, j)，计算：t*(i) = 位置 i 的承诺时间，dist(i, j) = |i - j|
3. 计算 Pearson corr(t*(j) | t*(i) < 0.30, dist(i,j) <= k) — 已早承诺邻居的条件下，j 的承诺时间
4. 与控制组比较（j 没有近距离早承诺邻居）

**脚本**：`experiments/probe_elf/probe_contextual_bootstrap.py`

### 强版（需要额外 oracle 实验）

修改 oracle 协议：为序列中的 p% 随机位置使用 t=1.0（clean 嵌入）而非标准 z_t

---

## 实验结果（Results）

**状态**: COMPLETED（2026-07-18）

**原始数据文件**：
- `results/exp09_kd_cr/contextual_bootstrap.json` + `commit_times_matrix.npy` + `y_tokens_ref.npy`
- `results/exp09_kd2/contextual_bootstrap.json` + `commit_times_matrix.npy` + `y_tokens_ref.npy`
- `results/exp09_baseline/contextual_bootstrap.json` + `commit_times_matrix.npy` + `y_tokens_ref.npy`

**脚本**：`experiments/probe_elf/probe_contextual_bootstrap.py`

参数：B=256 sequences, L=256 positions（使用 exp07b layer_states 的前 256 位置），
decode path: L11_hidden(768) → GELU(proj) → unembed → argmax vs y_tokens

### 承诺时间分布（Protocol A，decode path）

| t | kd_cr G(t) | kd_cr 首次承诺 | kd2 G(t) | baseline G(t) | baseline 首次承诺 |
|---|-----------|-------------|---------|--------------|----------------|
| 0.1 | 12.7% | 12.7% | 12.5% | 2.2% | 2.2% |
| 0.2 | 59.5% | 47.6% | 58.6% | 37.1% | 35.1% |
| 0.3 | 89.5% | 30.1% | 88.4% | 62.3% | 26.7% |
| 0.5 | 99.4% | 9.0% | 98.9% | 75.7% | 13.8% |
| 0.7 | 99.6% | 0.2% | 99.5% | 78.8% | 3.4% |
| 1.0 | 99.7% | 0.03% | 99.6% | 90.3% | 10.2% |
| never | 0.30% | — | 0.33% | 8.7% | — |

（kd_cr 与 kd2 非常相似。baseline 有 8.7% 的位置在所有 6 个 t 值下从未达到正确预测。）

### 空间自举（Contextual Bootstrapping）

**空间自举率**：对于每个"未承诺"位置，在距离 d 内有/无已承诺邻居时，下一步骤的承诺率

**kd_cr（B=256, L=256）**：

| 步骤 t→t | d=1 近/远 | d=5 近/远 | d=20 近/远 |
|---------|---------|---------|---------|
| 0.1→0.2 | 0.556/0.542 | 0.551/0.532 | 0.547/0.448 |
| 0.2→0.3 | 0.762/0.765 | 0.768/**0.297** | 0.768/**0.000** |
| 0.3→0.5 | 0.971/**0.545** | 0.971/**0.000** | 0.968/**0.000** |
| 0.5→0.7 | 0.834/**0.000** | 0.819/**0.000** | 0.769/**0.000** |
| 0.7→1.0 | 0.635/**0.000** | 0.589/**0.000** | 0.465/**0.000** |

**kd2（B=256, L=256）**（与 kd_cr 模式相同）：

| 步骤 t→t | d=5 近/远 | 注 |
|---------|---------|---|
| 0.1→0.2 | 0.538/0.531 | 微弱效应 |
| 0.2→0.3 | 0.745/**0.300** | **+45pp** |
| 0.3→0.5 | 0.936/**0.000** | **远组为空** |
| 0.5→0.7 | 0.753/**0.000** | — |

**baseline（B=256, L=256）**（模式与 kd_cr 显著不同）：

| 步骤 t→t | d=1 近/远 | d=5 近/远 | d=20 近/远 |
|---------|---------|---------|---------|
| 0.1→0.2 | **0.297/0.362** | 0.353/0.361 | 0.360/0.358 |
| 0.2→0.3 | 0.401/0.460 | 0.427/0.414 | 0.428/**0.017** |
| 0.3→0.5 | 0.382/0.401 | 0.388/**0.011** | 0.388/**0.000** |
| 0.5→0.7 | 0.163/0.133 | 0.163/**0.000** | 0.163/**0.000** |
| 0.7→1.0 | 0.578/0.426 | 0.576/**0.000** | 0.575/**0.000** |

注：baseline 在 t=0.1→0.2 时 d=1 近邻效应为**负**（0.297 < 0.362）——有承诺邻居反而不利于承诺。

### 关键发现与解读

**1. kd_cr/kd2 空间自举效应极强（t=0.2→0.3 时 +45pp@d=5）**：
- 在 t=0.2 时已承诺的 kd_cr 位置（约 60%），其 d=5 范围内的未承诺位置在 t=0.2→0.3 的承诺率为 76.8%
- 而 d=5 范围内无任何承诺邻居的位置承诺率仅 29.7%（+47pp 差距）
- 随着 t 增大，"远"组（无近邻承诺）的承诺率趋近零：在 d=5 下，t=0.3→0.5 时远组有效为零
- 解读：在 kd_cr/kd2 中，承诺是**高度局部聚集**的——远离已承诺区域的位置几乎停止承诺

**2. baseline 空间自举效应微弱甚至为负**：
- d=1 近/远在 0.1→0.2 时为 0.297/0.362（负效应，-6.5pp）
- baseline 的承诺更均匀分布，不依赖邻域已承诺状态
- 解读：baseline 的 decode path 在低 t 时主要依赖词频先验（从 EXP-05/06 已知），而词频不依赖局部空间上下文

**3. "远组为零"的解释**：
- t=0.3→0.5 时 kd_cr d=5 远组 = 0.000 主要因为样本量极小（已承诺 90% + 空间聚集 → 几乎没有"远离所有承诺位置"的未承诺位置）
- 不能直接解读为"远离承诺邻居的位置永远无法承诺"，而是"到 t=0.3 时所有未承诺位置都已有近邻"

**4. KD 训练改变了承诺的空间动力学**：
- kd_cr/kd2 的承诺从某些"核"位置开始向外传播（高空间聚集）
- baseline 的承诺更随机（词频驱动，无明显传播效应）

### 决策规则结论

✅ **强相关性（kd_cr/kd2 在 d=5 时 +45pp 差距）→ 加入论文 §4.x 作为"bootstrapping effect"**

但需注意：
- 效应在 kd_cr/kd2 上显著，在 baseline 上几乎不存在（甚至负效应）
- EXP-09 使用 Protocol A oracle 数据，空间相关性来自"同一序列内不同位置的噪声化状态的相关性"，不是真实生成轨迹（EXP-11 证明 Protocol B 的动力学与 Protocol A 差异大）
- 论文表述应为："在 oracle probe protocol 下，kd_cr 训练后的模型展现出显著的空间自举效应…"

### 对 EXP-08 的影响

per-position t* 矩阵（commit_times_matrix.npy）已保存，供 EXP-08 coarse-to-fine 分析使用：
- kd_cr: `results/exp09_kd_cr/commit_times_matrix.npy` — 形状 (256, 256)，值为 t_values 的索引（0-5）或 6（从未承诺）
- baseline: `results/exp09_baseline/commit_times_matrix.npy`
- kd2: `results/exp09_kd2/commit_times_matrix.npy`

---

## EXP-09v2 扩展：方向性分解（2026-07-20）

EXP-09v2 基于相同的 commit_times_matrix.npy 数据，分解了空间自举的方向性（func→content 还是 content→func）。

**关键发现**：EXP-09 的 +65pp 自举效应主要是 func→content 方向（kd_cr fc 峰值 = +65.7pp），不是对称效应。
- 这证明 EXP-09 观察到的空间自举是**功能词的早期承诺作为锚点**驱动的，而非单纯的空间位置邻近效应。
- LangFlow（EXP-28）的对称弱效应（+2.7pp vs +5.4pp）与 ELF KD 的单向强效应形成鲜明对比。

**详见**：`docs/specs/EXP-09v2-spec.md`

---

## EXP-09v3 结果（2026-07-22）— stable_k=3，fixed noise，exp07b_v2

**重要更新**：修复了三个 bug（固定噪声 seed=42、T5 tokenizer ▁ 前缀、K=3 consecutive stable）后的权威结果。

**状态**: COMPLETE（所有三个 checkpoint）

### Stable Commit Timing（K=3 consecutive correct steps）

| Checkpoint | never_commit | by t=0.10 | by t=0.20 | by t=0.30 | by t=0.50 |
|-----------|-------------:|----------:|----------:|----------:|----------:|
| baseline  | **24.8%**    | 2.0%      | 35.5%     | 61.2%     | 75.2%     |
| kd_cr     | **0.67%**    | 12.5%     | 59.3%     | 89.6%     | 99.3%     |
| kd2       | **1.07%**    | 12.5%     | 58.3%     | 88.5%     | 98.9%     |

t_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]（6 个点，exp07b_v2 fixed-noise states）

### 关键发现

1. **KD 将 never-commit 率从 24.8% 降至 0.67%（kd_cr）**：在 oracle Protocol A 下，KD checkpoint 几乎所有位置都能找到稳定承诺窗口。

2. **早期承诺大幅提前**：by t=0.20，kd_cr 59.3% 已稳定 vs baseline 35.5%。

3. **kd_cr ≈ kd2**：两者非常接近（kd2 never_commit=1.07%），说明这是 KD objective 共同驱动的效应。

### 与旧 EXP-09 的差异

旧 EXP-09 使用 first-hit t*（K=1），且有固定噪声 bug。旧结果中的 +65pp 近远差距在 v3 中不直接可比（v3 报告 cumulative commit fraction，而不是近远率差）。v3 的 stable_k=3 是更严格的标准，给出的 never_commit 数字更为保守和可靠。

### 输出文件

- `results/exp09v3_baseline/commit_times_matrix.npy` — 形状 (256, 256)，stable t* 索引（0–5 或 6 = never）
- `results/exp09v3_baseline/y_tokens_ref.npy`
- `results/exp09v3_baseline/contextual_bootstrap.json`
- `results/exp09v3_kd_cr/` — 同结构
- `results/exp09v3_kd2/` — 同结构

**EXP-08v2 已使用上述输出**（见 EXP-08 spec v2 结果节）。
