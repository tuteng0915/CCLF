# EXP-GS3 Spec — Low-Rank Global Mode Analysis

## 背景与地位

原始 doc 第 7 节 GLOBAL-3，P0 阶段第三项，也是最后一项 P0 实验。第三种独立方法学（前两种：
GS1 线性 probe accuracy，GS2 生成分支 consensus/entropy）：对每个 `Z_t`（每条序列的 `(L,d)`
矩阵）做 SVD，拆成 rank-k 的 "global mode" `G_t^{(k)}` 和残差 `R_t^{(k)} = Z_t - G_t^{(k)}`，
检验早期全局信息是否集中在低秩共享模式里——是否 `G_t^{(k)}` 比 `R_t^{(k)}` 更早/更强地携带
topic/token 信息，是否 `G_t^{(k)}` 比 `R_t^{(k)}` 更早变得和 clean 状态的对应分解"像"
（CKA 对齐）。

## 0. 汲取 GS1/GS2 的教训：不重复已知退化的指标

GS1 和 GS2 独立发现：**mean-pooled T5 latent 的 cosine 相似度 / KMeans 距离这一类"topic"
操作化在这个 embedding 空间下动态范围极窄、接近饱和**（GS1 的 `G_sent`：0.953→0.987；GS2 的
`C_topic`/`C_sent`：几乎全程 0.99+）。GS3 **不再用同一种退化指标**去测 `G_t^{(k)}` 的
topic 信号，而是只用两个已经证明有干净动态范围的指标：

- **Structural**：GS1 里唯一显示出清晰、大动态范围的探针（`G_syntax` ridge R²，从 t=0.05
  的 −0.207 到 clean 的 0.684），迁移过来用于 `G_t^{(k)}` / `R_t^{(k)}`。
- **Token**：GS1/GS2 都用的 native top-1 decode accuracy（把 `G_t^{(k)}`/`R_t^{(k)}`
  当作 backbone 的输入，跑一次前向，看模型自己的 decode 能恢复多少——这是原始 doc 第 8 节
  GLOBAL-4 "B. Preserve only global mode" 因果干预的被动版本，这里先用作诊断性读数，不做
  因果结论）。

不在 GS3 里重新引入 topic-cluster/sentence-cosine 探针，除非先解决 GS1/GS2 共同指出的
"cosine-on-mean-pooled-embedding 饱和"问题（留给后续，见 EXP-GS1/GS2-spec.md 的"下一步"）。

## 1. 分解与指标实现

对每条序列、每个 t、每个 `k ∈ {2, 8}`（原始 doc 建议 `{1,2,4,8,16}`；pilot 只取两端各一个
代表值控制算力，见第 4 节简化）：

- 只在 attention_mask 有效范围内做 SVD（`Z_t[valid_positions, :]`，形状 `(n_valid, d)`），
  避免 padding 零向量污染奇异值谱。
- `G_t^{(k)} = U_k Σ_k V_k^T`（rank-k 重构，形状和有效位置的 `Z_t` 一致），
  `R_t^{(k)} = Z_t_valid - G_t^{(k)}`；重新嵌回 `(L,d)` 时 padding 位置填零（不参与后续
  pooling/decode 的 mask 计算）。
- **Effective rank** `r_eff(t) = exp(-sum p_j log p_j)`，`p_j = sigma_j / sum(sigma)`，
  每条序列独立算，对样本取平均——这是不依赖 k 的纯诊断曲线，附带算出。
- **CKA 对齐**：`A_G(t) = CKA(G_t^{(k)}, G_clean^{(k)})`，`A_R(t) = CKA(R_t^{(k)},
  R_clean^{(k)})`，其中 `G_clean^{(k)}/R_clean^{(k)}` 是对 `x_clean`（而不是 `z_t`）做同样
  rank-k 分解得到的参考。用标准 linear CKA：
  `CKA(X,Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)`，`X,Y` 都是 `(n_valid, d)`，
  每条序列算一个标量，取样本平均。
- **Structural probe**：`masked_mean_pool` 后对 `G_t^{(k)}` 和 `R_t^{(k)}` 各自独立训练
  ridge（`g_t^mean` 换成 `G_t^{(k)}` 或 `R_t^{(k)}` 的 mean-pool，标签复用 GS1 的 POS
  histogram），文档级 train/test split。
- **Token probe（被动诊断）**：直接把 `G_t^{(k)}`/`R_t^{(k)}`（补零到全长 `(L,d)`）喂给
  `adapter.forward_state`（`sc_state=None`，即全零自条件，和 GS1/GS2 一致），native top-1
  vs ground truth。

## 2. 数据与规模（pilot）

- ELF baseline，`eval_exp37c_baseline.yml`（1024-token，与 GS1/GS2 一致）。
- `n_samples=64`（比 GS1 的 128 少，因为每个 t 现在要做 2 个 k 值 × 2 个分解
  （G/R）× 2 个指标（structural probe + token probe）+ CKA，单 t 的计算/统计成本显著高于
  GS1 的单一 mean-pool）。
- `t_grid`：复用 GS1 的 `[0.05, 0.12, 0.20, 0.28, 0.38, 0.50, 0.65, 0.85]` + clean-ref
  `t=0.99`，便于跨实验对比同一批 t 点上的信号。
- `k ∈ {2, 8}`。
- POS 标签、train/test split 独立于 GS1 重新构造（不同的 64-doc 子样本），不复用 GS1 的
  KMeans centroids（GS3 不用 topic 探针，不需要）。

## 3. 核心问题（对应原始 doc 第 7 节）

1. `A_G(t)` 是否明显早于 `A_R(t)` 上升？（低秩 global mode 是否比残差更早"变得像 clean 状态"）
2. `G_t^{(k)}` 的 structural probe R² 是否明显高于 `R_t^{(k)}` 在早期 t？
3. `R_t^{(k)}` 的 token probe（native top-1）是否明显滞后于 `G_t^{(k)}`？还是反过来
   （token 信息其实主要在残差里，而不是低秩共享模式里）？
4. `r_eff(t)` 随 t 如何变化——是否存在一个从"高有效秩（接近各向同性噪声）"到
   "低有效秩（信息集中在少数模式）"的转变，转变点是否和已知的 baseline commitment cliff
   （t≈0.20–0.30）对齐？

## 4. 已知简化

1. ⚠️ `k` 只取 `{2, 8}` 两点，不是原始 doc 建议的 `{1,2,4,8,16}` 五点；如果 pilot 显示
   `k` 敏感（G/R 的表现在 2 和 8 之间差异很大），需要补充中间值。
2. ⚠️ Structural/Token 探针沿用 GS1 已验证的操作化，**故意不测 topic/sentence**（见第 0 节
   理由）——GS3 无法回答"低秩模式是否更早支持 topic 恢复"这个原始 doc 最关心的问题之一，
   只能回答 structural 和 token 层面的对应问题。
3. ⚠️ SVD 分解和 CKA 都是**每条序列独立计算**（不是原始 doc 可能暗示的、对整个 batch 做统一
   低秩分解）；这是唯一在数学上合理的操作化（不同序列的 `Z_t` 没有共享的行/列结构可以拼在一起
   做 SVD），但需要明确这不是"跨序列共享的全局模式"，而是"每条序列自己的低秩子空间"。
4. Pilot 规模（64 样本），数字只用于判断信号方向。

## 5. 脚本与输出

```text
experiments/global_state/analyze_low_rank_modes.py
```

```text
results/global_state/<model>/<checkpoint>/low_rank_modes_<label>.json
```

## ⚠️ 重要更正（EXP-GS11，pooling confound）

`EXP-GS11` 证实：对**未经模型处理的原始 oracle state** `z_t` 做 mean pooling，本身就能在
`t=0.28` 达到近乎完美的 self-retrieval（不需要任何模型计算）。本实验的 `svd_decompose`
同样是直接作用在 raw `z_t` 上（不是模型 hidden state），`syntax_G`/`syntax_R` 探针的输入
是这个 raw 分解结果的 mean-pool。**这意味着下方"structural 信号集中在低秩 `G`"的发现，
很可能主要反映的是"raw 状态的顶奇异方向恰好携带更多可被简单线性统计量捕捉的信号"这一
数学性质，而不是模型学到的语义-句法解耦**——与用户审阅中提出的"uncentered SVD +
mean-pool 双重构造"批评完全吻合。GS4（因果干预、需要真正的 rollout）不受这个具体 confound
影响（因为它测的是"这个子空间能否驱动完整生成"，不是"raw 子空间能否被线性 probe 读出"），
其"低秩 global mode 单独不足以因果驱动"的结论依然成立，且和这次更正的方向一致（进一步说明
GS3 原本的"探针可读性"结论不能简单等同于模型的真实内部组织）。

**`EXP-GS12`（中心化 SVD 重做，P0-2）已经把这个猜想验证到底**：把跨位置均值显式分离后，
"structure 集中在低秩 `G`"这一半结论**不成立**——`MEAN_only`（只用均值，完全不做 SVD）
单独就能达到和 `MEAN+G_c`/`MEAN+R_c` 几乎相同的 structural R²，在 raw 和 model 两条
representation 上都是如此。但"token 集中在残差"这一半结论**在去均值后依然成立**，且在
模型真实输出（`predicted_clean`）上同样复现，是本实验里少数经受住重新检验的部分。
详见 `EXP-GS12-spec.md`。

## 状态

**Pilot DONE**（ELF baseline，n=64，1024-token，8 个 t 点 + t=0.99 clean-ref，k∈{2,8}，GPU1，
`logs/global_state/gs3_elf_baseline_pilot.log`，输出
`results/global_state/elf/baseline/low_rank_modes_pilot.json`）——**pilot 规模下信号最干净、
最内部一致的一次**，两个 k 值结果高度一致。

## Results（pilot：ELF baseline，64 序列，seq_len=1024，
t∈{0.05,0.12,0.20,0.28,0.38,0.50,0.65,0.85}+0.99(clean-ref)，train/test=45/19）

⚠️ pilot 规模，数字用于判断信号方向，但内部一致性（k=2 与 k=8 同向、且和已知 commitment
cliff 时序吻合）比 GS1/GS2 更强，值得作为后续正式规模验证的重点方向。

**k=2：**

| t | r_eff | A_G | A_R | syntax_G | syntax_R | token_G | token_R |
|---|---|---|---|---|---|---|---|
| 0.05 | 471.5 | 0.002 | 0.185 | −0.077 | −0.094 | 0.032 | 0.002 |
| 0.12 | 471.4 | 0.009 | 0.200 | 0.018 | −0.067 | 0.009 | 0.011 |
| 0.20 | 470.7 | 0.122 | 0.238 | 0.250 | −0.072 | 0.001 | 0.033 |
| 0.28 | 468.8 | 0.505 | 0.310 | 0.410 | −0.070 | 0.001 | 0.068 |
| 0.38 | 463.5 | 0.764 | 0.458 | 0.552 | −0.058 | 0.001 | 0.271 |
| 0.50 | 450.2 | 0.901 | 0.686 | 0.653 | −0.041 | 0.002 | 0.603 |
| 0.65 | 419.2 | 0.962 | 0.896 | 0.721 | −0.016 | 0.002 | 0.792 |
| 0.85 | 363.8 | 0.997 | 0.989 | 0.766 | 0.019 | 0.002 | 0.839 |
| 0.99 (clean) | 343.4 | 1.000 | 1.000 | 0.783 | 0.046 | 0.030 | 0.998 |

**k=8**（同样的定性模式，数值略有平移）：`A_G` 0.008→1.000，`A_R` 0.211→1.000（A_G 依然
全程领先）；`syntax_G` −0.082→0.789，`syntax_R` 全程 ≈−0.06 到 −0.09（几乎不随 t 变化，
一直贴着 0 附近，从未转正）；`token_G` 全程 ≤0.105（clean-ref 时也只有 0.105），`token_R`
0.003→0.993。

**解读**：

1. **三路清晰解耦，两个 k 值高度一致**：
   - **`A_G(t)` 系统性早于/领先于 `A_R(t)`**（例如 k=2, t=0.28: A_G=0.505 > A_R=0.310；
     t=0.50: A_G=0.901 > A_R=0.686）——这是原始 doc 第 7 节最核心的问题（"是否存在 A_G(t)
     明显早于 A_R(t) 上升"），**pilot 规模下明确、干净地支持**：低秩 global mode 比高秩残差
     更早"变得像 clean 状态的对应分解"。
   - **Structural signal（POS ridge R²）几乎完全集中在 `G_t^{(k)}` 里**：`syntax_G` 从
     −0.08 单调升到 0.78–0.79（clean-ref 时**超过 GS1 全量 mean-pooled `Z_t` 的 syntax R²
     0.684**——rank-2/rank-8 的低秩重构比完整的 mean-pooled 状态本身携带更多结构信号）；
     `syntax_R`（残差）**全程贴近 0，从未显著转正**（k=8 时甚至一直是负数）。structural
     information 几乎不在残差里。
   - **Token identity（native top-1 decode）几乎完全集中在 `R_t^{(k)}` 里**：`token_G`
     全程 ≤0.03–0.1（**即使在 clean-ref 输入时，只用低秩重构 backbone 也几乎解不出正确
     token**），`token_R` 的曲线（0.002→0.271→0.603→0.792→0.839→0.998）和 GS1 全量 `Z_t`
     的 `G_token` 曲线（0.002→0.672→0.761→0.808→0.836→1.000，t 点部分对齐）**形状高度相似**，
     说明全量状态的 token-decode 能力几乎全部来自残差部分，低秩 global mode 对 token 解码
     贡献可以忽略。
2. **`r_eff(t)` 全程都很大（343–472，相对 `min(n_valid,d)≈512` 的理论上限）**，说明
   `Z_t` 本身几乎是满秩的，不存在"少数几个方向主导"的直觉意义上的低秩结构；
   `r_eff` 随 t 增大缓慢下降（471→343），但降幅有限。**"low-rank global mode" 这个说法
   不能从"整体有效秩很低"的意义上成立**——而是说，即使在一个几乎满秩的高维状态里，人为截取
   最主要的 2 或 8 个方向，仍然能几乎完整地捕获 structural 信号、同时几乎不携带 token 信号。
   这比原始 doc 字面意义上的"早期信息集中在低秩共享模式"更精确：**不是"整体状态趋于低秩"，
   而是"结构信息本身是低秩的，可以和高秩的、携带 token 身份的部分几乎完全解耦"**。
3. ⚠️ **重要方法论警示（不影响上面的相对比较，但影响绝对数值解读）**：`token_G`/`token_R`
   是把合成的 `G_t^{(k)}`/`R_t^{(k)}` 直接喂给一个从未见过这种输入分布的 backbone
   （训练时只见过完整的 `Z_t`），因此绝对准确率可能同时受"真实信息含量"和"分布外输入导致
   decode 失效"两个因素影响——`token_G` 在 clean-ref 时仍然只有 0.03–0.1（而不是接近 1.0），
   如果低秩重构真的完整保留了"生成 clean token 所需的全部结构性/句法信息"，也不代表 backbone
   能在这种 OOD 输入下正常解码。**这个警示不会推翻"structure 集中在 G、token 集中在 R"这个
   相对论断**（因为 syntax probe 是独立训练的线性探针，不依赖 backbone 泛化到 OOD 输入），
   但会让"token_G≈0.03 说明低秩模式完全不携带 token 信息"这个更强的论断打个问号——更准确的
   表述是"backbone 的 native decode 机制无法从低秩重构里解码出 token，不代表这个重构在
   信息论意义上完全不含 token 信息"。
4. GS3 是三个 P0 pilot 里**信号最干净、内部最一致**的一个（k=2/k=8 定性一致，`A_G`/`syntax`/
   `token` 三个维度互相印证），且是三个实验里唯一**明确验证了原始 doc 提出的具体机制性问题**
   （A_G 早于 A_R）的一个，而不只是"topic 排序"这种更容易受退化指标污染的比较。

## 下一步

1. GS3 的核心发现值得推进到正式规模（512 样本，更密 t 网格，补充 k∈{1,4,16} 中间值）——
   这是当前 P0 三个 pilot 里最值得投入算力扩大验证的一个。
2. 用 GLOBAL-4（Global Mode Causal Intervention，`EXP-GS4`）把 GS3 这里的**被动诊断**
   （`token_G`/`token_R`）升级成**因果干预**：真正的"remove global mode / preserve only
   global mode / swap"实验，能排除第 4 节的 OOD 警示（如果 remove-global-mode 后从该状态
   *继续 ODE 采样*，而不是直接一次性 decode，能更公平地评估信息含量，因为继续采样让 backbone
   有机会把 OOD 输入"拉回"正常轨迹分布）。
3. 三个 P0 实验（GS1/GS2/GS3）目前收敛的工作假说：**structural information 是当前 pilot
   里唯一在所有三种方法学下都表现出"早、强、可靠"的信号**；"topic"层面的证据被 GS1/GS2 共同的
   mean-pooled-cosine 退化指标问题污染，暂时无法下结论；"token/lexical identity"确认是
   最晚确定的（GS1 的 G_token 曲线、GS2 的 C_lex 曲线、GS3 的 token_R 曲线三者独立指向同一个
   时序模式）。
