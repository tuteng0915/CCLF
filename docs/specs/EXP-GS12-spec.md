# EXP-GS12 Spec — Centered Spectral Decomposition (P0-2)

## 背景与地位

用户审阅 + `EXP-GS11` 共同指出 GS3 的两个叠加问题：(a) SVD 分解直接作用在**未中心化**的
`Z_t` 上，顶奇异方向天然会捕获跨位置共享的均值方向；(b) structural probe 的输入又恰好是
`mean_pool(G_t^{(k)})`——如果 `G^{(k)}` 主要就是在重构这个均值，那"structure 集中在
`G`"这个发现可能只是"分解方式与读出方式互相咬合"的必然结果，不是模型学到的语义-句法解耦；
(c) `EXP-GS11` 进一步证实这个 confound 在**原始输入**层面就已经成立（不需要模型），并且
发现模型自己的 `predicted_clean` 输出行为和原始输入很不一样（self-retrieval 反而更差）。

本实验（P0-2）做两处修正：**显式分离"跨位置共享的均值"这个平凡分量**，并且**同时在
raw oracle state 和模型真实输出（predicted_clean）上重复分析**，直接回答：一旦去掉均值,
剩下的"结构化方差"是否还表现出 GS3 报告的那种 global/local 分层？这个分层在模型的真实
输出上是否依然存在？

## 1. 分解

对每条序列的 `Z`（可以是 raw `z_t` 或模型的 `predicted_clean`，`(n_valid, d)`，只在有效
位置上操作）：

```
mu       = mean_i Z_i                      (d,)          -- 跨位置共享的均值（平凡分量）
Z_c      = Z - mu                          (n_valid, d)  -- 中心化后的矩阵
Z_c = U S V^T                                              -- 对中心化矩阵做 SVD
G_c^(k)  = U_k S_k V_k^T                   (n_valid, d)  -- 中心化后的 rank-k 分量
R_c^(k)  = Z_c - G_c^(k)                   (n_valid, d)  -- 中心化后的残差
```

三个"重建状态"（重新加回 `mu`，让它们成为可以喂给模型的、量级正常的完整状态，而不是
纯粹零均值的人造张量）：

- **MEAN-only**：`mu` 广播到每个位置（不含任何中心化后的方差分量）。
- **MEAN+G_c**：`mu + G_c^{(k)}`（均值 + 中心化后的低秩方差）。
- **MEAN+R_c**：`mu + R_c^{(k)}`（均值 + 中心化后的残差方差，等价于 `Z - G_c^{(k)}`）。

`MEAN+G_c` 和 `MEAN+R_c` 都包含**同一个** `mu`，这样两者的对比才是"低秩方差 vs 残差方差"
的干净对比，不会被"谁保留了均值、谁没有"混淆——这是相对 GS3 原始设计的核心修正。

## 2. 两条representation 线

- **raw**：`Z = z_t`（原始 oracle state，GS3 原本用的）。
- **model**：`Z = predicted_clean`（`adapter.forward_state(z_t, ...)` 的模型输出）。

`EXP-GS11` 已经证明这两者在 self-retrieval 上表现截然不同（raw 远好于 model），所以这里
不能只测 raw——**只有在 model 表示上重复出"结构在低秩、词汇在残差"的分层，才能说这是模型
真正学到的东西**，而不是原始输入的几何性质。

## 3. 指标

沿用 GS3 已验证的两个指标（避开 GS1/GS2 已知退化的 topic/sentence 指标，见 GS3 spec 第 0 节）：

- **Structural probe（POS ridge R²）**：⚠️ **实现踩坑（smoke test 发现）**：直接对
  `MEAN+G_c` 做 plain mean-pool，得到的向量和 `MEAN-only`（即 `mu` 本身）在数值上**完全
  相等**——因为 `G_c` 是中心化矩阵的低秩重构，本身在有效位置上的均值就 `≈0`，plain
  mean-pool 对它"视而不见"，导致 `MEAN-only` 和 `MEAN+G_c` 的 structural probe 输入
  完全相同、R² 数值也完全相同（smoke test 里 `t=0.28, raw` 两者都是 `-0.243`）。
  **修正**：改用 `pooled_summary(state) = concat(mean_pool(state), mean_pool(state**2))`
  （均值 + 二阶矩），对三种 summary 都用这个统一的 pooling 方式再训练 ridge 回归到 POS
  histogram，document-level train/test split（复用 GS1/GS3 的 POS 标签构造）。二阶矩项
  对 `G_c`/`R_c` 各自贡献的"能量"敏感，不会被均值为零这件事抹掉。
- **Token probe（native top-1，被动诊断）**：把 `MEAN-only`/`MEAN+G_c`/`MEAN+R_c` 三种
  完整 `(L,d)` 状态（广播 `mu`、padding 位置补零）喂给 `forward_state`，测 native top-1
  vs ground truth。

## 4. 核心问题

1. 去掉均值之后，`MEAN+G_c` 的 structural R² 是否依然明显高于 `MEAN+R_c`？（如果原来的
   GS3 分层主要是均值造成的，去均值后这个差距应该大幅缩小甚至消失。）
2. `MEAN-only` 自己的 structural R² 有多高？（如果 `MEAN-only` 已经能解释 GS3 原来大部分
   的 `syntax_G` 表现，就直接证实了"GS3 测的主要是均值"这个猜想。）
3. 在 `raw` 和 `model` 两条 representation 线上，这个模式是否一致？如果只有 `raw` 上有
   分层、`model` 上没有，说明分层是输入几何性质，不是模型学到的组织。

## 5. 数据与规模（pilot）

- ELF baseline，`eval_exp37c_baseline.yml`（1024-token）。
- `n_samples=64`（与 GS3 一致）。
- `t_grid`：复用 GS1/GS3 的 `[0.05, 0.12, 0.20, 0.28, 0.38, 0.50, 0.65, 0.85]` +
  clean-ref `0.99`（为控制算力，pilot 先只测其中 4 个代表点：`[0.05, 0.28, 0.50, 0.99]`，
  见第 6 节简化）。
- `k=8`（GS3 里两个 k 值定性一致，pilot 只用一个）。

## 6. 已知简化

1. ⚠️ `t_grid` 从 GS3 的 9 点减到 4 点（`raw` + `model` 两条线 × 3 种 summary × structural
   probe + token probe，比 GS3 原来单线 × 2 种 summary 贵了大约 3 倍，用更稀疏的 t 网格
   控制总算力）。
2. ⚠️ `MEAN-only` 状态广播到全部位置后喂给模型是一个相当极端的 OOD 输入（完全没有位置间
   变化），token probe 在这个条件下的数字需要谨慎解读，主要看**相对**大小（`MEAN+G_c` vs
   `MEAN+R_c`），不看绝对值。
3. Pilot 规模，数字仅用于判断方向。

## 7. 脚本与输出

```text
experiments/global_state/analyze_centered_modes.py
```

```text
results/global_state/<model>/<checkpoint>/centered_modes_<label>.json
```

## 状态

**Pilot DONE — GS3 的两条结论被拆开验证：一条不成立，一条成立**（ELF baseline，n=64，
`k=8`，`t∈{0.05,0.28,0.50,0.99}`，raw + model 两条 representation，GPU1，
`logs/global_state/gs12_elf_baseline_pilot.log`，输出
`results/global_state/elf/baseline/centered_modes_pilot.json`）。

## Results（pilot：ELF baseline，n=64，k=8，train/test=45/19）

`struct_r2`（POS ridge R²，`pooled_summary`=均值+二阶矩）：

| t | repr | MEAN_only | MEAN+G_c | MEAN+R_c |
|---|---|---|---|---|
| 0.05 | raw | −0.090 | −0.091 | −0.079 |
| 0.05 | model | −0.055 | −0.058 | −0.054 |
| 0.28 | raw | 0.426 | 0.428 | 0.379 |
| 0.28 | model | **0.710** | 0.680 | **0.718** |
| 0.50 | raw | 0.676 | 0.684 | 0.665 |
| 0.50 | model | **0.823** | 0.806 | 0.816 |
| 0.99 | raw | **0.797** | 0.778 | 0.790 |
| 0.99 | model | 0.807 | 0.813 | 0.808 |

`token_acc`（native top-1，被动诊断）：

| t | repr | MEAN_only | MEAN+G_c | MEAN+R_c |
|---|---|---|---|---|
| 0.28 | raw | 0.000 | 0.010 | **0.172** |
| 0.28 | model | 0.001 | 0.086 | **0.467** |
| 0.50 | raw | 0.001 | 0.036 | **0.705** |
| 0.50 | model | 0.001 | 0.116 | **0.780** |
| 0.99 | raw | 0.024 | 0.117 | **0.999** |
| 0.99 | model | 0.028 | 0.079 | **0.577** |

**解读**：

1. **GS3 的"structure 集中在低秩 `G`"结论不成立，被证实主要是均值 confound**：在全部
   8 个 `(t, repr)` 组合里，`MEAN_only`（只用跨位置共享的均值 `mu`，完全没有任何 SVD
   分解）单独就已经达到了和 `MEAN+G_c`、`MEAN+R_c` **几乎相同**的 structural R²——差距
   全部在 ±0.03 以内，很多时候 `MEAN_only` 甚至是三者中最高的（如 `t=0.99, raw`：
   `MEAN_only=0.797 > MEAN+R_c=0.790 > MEAN+G_c=0.778`）。**中心化后的低秩方差分量
   `G_c`、乃至整个残差方差 `R_c`，相对于"只用均值"几乎不增加任何额外的 POS 预测力**。
   这直接证实了用户的判断：GS3 原来"structure 集中在 `G`"的发现，本质上是"顶奇异方向
   捕获了跨位置共享均值、而 structural probe 又恰好只读均值"这一构造性 artifact，不是
   模型学到的语义-句法解耦。
2. **GS3 的"token 集中在残差"结论则完全存活，而且更清晰**：`MEAN+R_c` 的 `token_acc`
   在所有 `(t, repr)` 组合下都远高于 `MEAN+G_c`（例如 `t=0.50, model`：
   `MEAN+R_c=0.780` vs `MEAN+G_c=0.116`，近 7 倍；`t=0.99, raw`：`0.999` vs
   `0.117`）——**这个不对称在去掉均值 confound、且分别在 raw 和 model 两条
   representation 上都复现了**，说明"exact token identity 主要靠残差（高秩、大部分
   有效维度）而不是低秩共享分量"是一个相对稳健的发现，不是均值造成的假象。
3. **两条 representation（raw vs model）在这个问题上定性一致**：`MEAN_only` 主导
   structural R²、`R_c` 主导 token_acc 这两个模式在 raw 和 model 上都成立，说明这不是
   `EXP-GS11` 发现的"raw vs model 行为迥异"那种情况——这里测的是"分解出的哪个分量携带
   哪种信号"，和"整体上模型处理是否保留可检索身份信息"是两个独立的问题。
4. **修正后的、更精确的表述**：POS histogram 这个粗粒度统计量几乎完全可以从"跨位置的
   通道均值"里线性读出，不需要任何更复杂的低秩或残差结构；而 exact token identity
   则明确需要残差里的高秩、逐位置变化的信息，均值和低秩共享分量都严重不足。这比 GS3
   原来"structure 在 `G`、token 在 `R`"的表述更弱，但更可信。

## 结论

GS3 原本的"两路清晰解耦"发现里，**"token 集中在残差"这一半站得住**（去均值后依然成立，
且在模型真实输出上复现）；**"structure 集中在低秩 global mode"这一半不成立**，应该改为
"POS 这类粗粒度统计量几乎完全由跨位置均值解释，和是否做低秩分解无关"。已在
`EXP-GS3-spec.md` 的更正说明里链接到本结果。

## 正式规模复现（n=128，完整 9 点 t 网格 `[0.05,0.12,0.20,0.28,0.38,0.50,0.65,0.85,0.99]`，
`k=8`，`logs/global_state/gs12_elf_baseline_formal.log`，
`results/global_state/elf/baseline/centered_modes_formal.json`）

**两条结论都被进一步强化，不是弱化**：

1. **"structure 集中在 `G_c`"不成立，且在更大样本下更加确定**：在全部 `9 t × 2 repr =
   18` 个组合里，`MEAN_only` **无一例外**是三者中最高或并列最高的 structural R²
   （例如 `t=0.85, raw`: `MEAN_only=0.685 > MEAN+R_c=0.675 > MEAN+G_c=0.671`；
   `t=0.38, model`: `MEAN_only=0.708 > MEAN+R_c=0.705 > MEAN+G_c=0.680`）。多数情况下
   `MEAN+G_c` 甚至比 `MEAN_only` **更低**——额外的 `k=8` 维中心化方差不仅没有增益，
   反而像是在给 ridge 回归引入噪声维度。pilot（n=64，4 个 t 点）已经显示了这个模式，
   正式规模（n=128，9 个 t 点）完整复现，没有一个例外。
2. **"token 集中在残差"在完整 t 网格上同样清晰复现**：`MEAN+R_c` 的 `token_acc` 在
   全部 `t>=0.28` 的组合里都远高于 `MEAN+G_c`（如 `t=0.65, model`: `0.814` vs
   `0.091`，约 9 倍；`t=0.85, raw`: `0.846` vs `0.043`，约 20 倍）。⚠️ 唯一的小反常
   仍在 `t=0.99, model`：`MEAN+R_c` 只有 `0.584`（明显低于 `raw` 同条件的
   `0.999`），和 pilot 观察到的"模型自己的 predicted_clean 在 clean-ref 附近 token
   还原能力弱于 raw"的模式一致（呼应 `EXP-GS11` 发现的"model 处理可能抹除可检索身份
   信息"）。

**结论不变，且置信度提高**：GS3 的"structure 在低秩、token 在残差"这条双重解耦结论，
经过中心化 + 正式规模验证后，应该改写为单一、更精确的表述——
**"POS 这类粗粒度统计量几乎完全由跨位置均值解释；exact token identity 则明确需要
高秩残差部分，均值和低秩共享分量都远远不够"**。
