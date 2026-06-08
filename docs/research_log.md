# ELF Probing Research Log

> 记录从 probing 实验设计到方法方向的完整讨论过程。

---

## 1. 出发点

**项目**：在 ELF（Embedded Language Flows）基础上开发 Progressive Anchoring（渐进锚定）训练目标。

**核心命题**（来自 `proposal.md`）：
- ELF 在整个 flow 过程中只在 t=1 做一次 token 离散化（final-only discretization）
- 我们假设存在一个可学习的 lexical commitment schedule，让连续状态在适当时机以适当强度靠近 token anchor manifold
- 这个 schedule 可以通过额外的训练损失引导：$\mathcal{L} = \lambda(t)\mathcal{L}_\text{cont} + \mu(t)\mathcal{L}_\text{disc} + \beta(t)\mathcal{L}_\text{align}$

**但在设计新训练目标之前，必须先搞清楚 ELF 的内部动力学**：ELF 到底在什么时候、以什么方式自发地做出词汇决策？

---

## 2. ELF 架构关键细节

| 项目 | 值 |
|---|---|
| 模型 | ELF-B（105M params，depth=12，hidden=768，12 heads） |
| Latent 空间 | T5-small encoder 的 6 层输出空间（contextual embeddings） |
| Flow 公式 | $z_t = t \cdot x + (1-t) \cdot \varepsilon$，t=0 纯噪声，t=1 纯 clean |
| Latent normalization | `latent_std=0.2`，即 $x_\text{norm} = x_\text{T5} / 0.2$ |
| Self-conditioning | `self_cond_prob=0.5`，输入为 $[z_t, x_\text{pred}]$（拼接），无 self-cond 时置零 |
| Decode head | 仅在 20% 的训练步（decoder branch）激活：$\text{gelu}(\hat{x} W_\text{proj} + b) W_\text{unemb} + b'$ |
| Decode mode | `decoder_step_active=True` 强制在每个 t 计算 logits（probe 中使用） |

**关键区分**：
- $\hat{x}_t$（FinalLayer 输出）：在 **T5 contextual embedding** 空间（6 层 Transformer 输出）
- $E_\text{raw}$（T5 token embedding table）：在 **T5 input embedding** 空间（pre-transformer context-independent）
- 两者**不是同一个空间**，直接用 $\hat{x}_t \cdot E_\text{raw}^\top$ 作为 anchor distance 是错误的 proxy

---

## 3. Probing v1：基础 Anchor Emergence（已完成）

**脚本**：`experiments/probe_anchor/probe_anchor.py`  
**结果**：`experiments/probe_anchor/results_v1/anchor_probe.json`

**设置**：64 seq，τ=1.0，seq_len=256，21 t steps，n_noise=4

### 核心发现：ELF-B OWT 的 4-Phase Commitment Pattern

| 阶段 | t 范围 | 特征 |
|---|---|---|
| 1. Prior-dominated | 0–0.15 | H 峰值 0.49，top-5≈0，模型输出近似均匀分布 |
| 2. Commitment cliff | 0.15–0.25 | H 急降 5×，top-5 从 5% 跳到 37%，JSD 峰值 0.59 |
| 3. Stable plateau | 0.25–0.95 | H≈0.05–0.07，top-5≈85–90%，revision≈4–8% |
| 4. Final refinement | 1.0 | top-5 跳到 98%，19% 的位置 revision |

**关键结论**：ELF **自发地**在 t≈0.25 就完成大部分词汇决策，远早于 t=1。"stable-but-imperfect plateau"是 Progressive Anchoring 方法的作用目标区间。

### v2 验证结果（64 seqs，tau sweep）

tau sweep 验证（Q2）：top5_gt 在所有 tau 下几乎一致（0.320–0.368），说明 topk_recovery 对温度不敏感，4-phase 模式是真实信号而非温度 artifact。

| t | H | top1_gt | top5_gt | JSD |
|---|---|---|---|---|
| 0.00 | 0.118 | — | 0.005 | — |
| 0.15 | 0.494 | — | 0.050 | **0.537** ← JSD 峰 |
| 0.25 | 0.201 | — | 0.368 | 0.582 |
| 0.40 | 0.077 | — | 0.828 | 0.186 |
| 0.95 | 0.062 | — | 0.910 | 0.052 |
| 1.00 | 0.055 | — | **0.981** | 0.126 ← final spike |

**Q3 验证**：`topk_gt ≈ topk_final`（最大差 0.002），说明 ELF 的预测内部一致：它预测的 token 和它最终输出的 token 高度重合，不存在"途中乱预测、终点纠正"的情况。

---

## 4. Probing v3：扩展指标（已完成）

**脚本**：`experiments/probe_anchor/probe_anchor_v3.py`  
**结果**：`experiments/probe_anchor/results_anchor_v3/anchor_probe_v3.json`

### 新增指标

**Commitment state 分解**（使用熵阈值 H_thresh=0.1）：
- `committed_correct`：H < thresh 且 top-1 == gt（锁定且正确）
- `committed_wrong`：H < thresh 且 top-1 ≠ gt（锁定但错误）
- `uncommitted`：H ≥ thresh（尚未锁定）

**Transition matrix**（逐步纠错 vs 扰动）：
- `w2c`：上一步错误，这一步正确（纠错）
- `c2w`：上一步正确，这一步错误（扰动）
- `c2c`、`w2w`：稳定

**Noise agreement**：N 个 noise seed 在同一位置的 top-1 是否一致→衡量"已决定"vs"仍对噪声敏感"

**Entropy percentiles**：p10/p50/p90→检测 entropy 分布是否双峰（部分位置已决定，部分仍未定）

**Commitment time**：每个位置首次 H < thresh 的时刻→分布的均值/标准差

### Conjecture 5：几何 Anchoring 验证

**假设**：lexical commitment 不仅仅是 posterior 变尖，连续状态在几何上也应该靠近 token anchor manifold。

**三个几何指标**：
- $D_\text{soft}(t) = \|\hat{x}_t - p_t^\top E\|$（加权 anchor 期望距离）
- $D_\text{NN}(t) = \min_v \|\hat{x}_t - E_v\|$（最近邻 token 距离）
- $\text{Margin}(t) = \|\hat{x}_t - E_\text{2nd}\| - \|\hat{x}_t - E_\text{1st}\|$（分离 margin）

**支持 Conj.5**：三者都随 t 单调下降/上升，且与 entropy 下降同步。  
**反驳**：posterior 变尖但几何距离不变 → commitment 只是 decode head 的现象，不是连续状态的性质。

**注意**：目前 $E$ 仍用 T5 input embedding（非 contextual centroid），因此 $D_\text{soft}$ 和 $D_\text{NN}$ 是 proxy，相对趋势有意义，绝对值不可直接解释。

### v3 实验结果（tau=1.0，64 seqs）

| t | H | top1_gt | top5_gt | comm_c | comm_w | uncmt | w2c | c2w | agree |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.118 | 0.000 | 0.005 | 0.000 | 0.769 | 0.231 | — | — | 0.534 |
| 0.15 | 0.494 | 0.033 | 0.050 | 0.028 | 0.323 | 0.649 | 0.015 | 0.002 | 0.030 |
| 0.25 | 0.201 | 0.260 | 0.368 | 0.236 | 0.414 | 0.350 | 0.162 | 0.004 | 0.177 |
| 0.30 | 0.128 | 0.455 | 0.610 | 0.420 | 0.342 | 0.238 | **0.208** | 0.008 | 0.388 |
| 0.50 | 0.058 | 0.768 | 0.885 | 0.734 | 0.151 | 0.114 | 0.043 | 0.010 | 0.813 |
| 0.70 | 0.053 | 0.798 | 0.901 | 0.766 | 0.129 | 0.105 | 0.014 | 0.014 | 0.879 |
| 0.90 | 0.073 | 0.769 | 0.881 | 0.734 | 0.132 | 0.135 | 0.010 | 0.012 | 0.918 |
| 0.95 | 0.062 | 0.811 | 0.910 | 0.775 | 0.109 | 0.116 | 0.050 | 0.007 | 0.970 |
| 1.00 | 0.055 | **0.979** | 0.981 | 0.978 | 0.000 | 0.022 | **0.171** | 0.000 | 1.000 |

- commit_time mean = 0.019 ± 0.008（大部分位置在 t≈0.02 就锁定）
- never_committed = 0%（所有位置最终都锁定）
- **d_nn 和 margin 几乎恒定**（d_nn≈1216，margin≈71）→ 证实 space mismatch 问题：ELF latent 在 T5 contextual 空间，static embedding table 不是正确的距离参照系。Conjecture 5 在 ELF 上需要 contextual centroid 才能正确验证。

### 关键新发现

1. **committed_wrong 在 commitment cliff 之后依然大量存在**：t=0.25 时 comm_c=0.236，comm_w=0.414，说明大量位置"锁定"了错误 token。这是 Progressive Decode Correction 的核心目标：在 plateau 中修正这些错误锁定。
2. **w2c 在 t=0.30 达到峰值 0.208**，之后下降。c2w 始终很低（≤0.021），说明 ELF trajectory 几乎不会把正确预测变错（很稳定），但有相当数量的位置在 commitment cliff 之后逐步从错误→正确。
3. **Final jump（t=0.95→1.00）的 w2c=0.171**（17% 的位置在最后一步被纠正）对应 decode branch probe 的 G_traj+G_dec。

---

## 5. Decode Branch Probe（GPU 7 运行中）——最关键的新 probe

**脚本**：`experiments/probe_anchor/probe_decode_branch.py`

### 问题的起源

之前的 probe 把**两件事混在一起**：
1. Denoising latent 本身是否 token-readable（$\hat{x}_t$ 的质量）
2. Decode branch 能否把 latent contextualize-correct 成 token（decode head 的能力）

ELF 实际推理只在 t=1 用 decode branch。Probe A（forced decode at every t）测的主要是第 1 件事，但这并不等于 ELF 最终真正使用的路径。

### 两条 readout 路径

**Probe A — p_lin(t)**（已有，forced decode at denoising time）：
$$z_t \xrightarrow{\text{backbone}(t)} \hat{x}_t^{den} \xrightarrow{\text{decode head}} \text{logits}_{lin}$$

**Probe B — p_dec(t)**（新增，two-pass）：
$$z_t \xrightarrow{\text{backbone}(t)} \hat{x}_t^{den} \xrightarrow{\text{backbone}(t=1)} h_t^{dec} \xrightarrow{\text{decode head}} \text{logits}_{dec}$$

Pass 2 问的是："如果我们把 $\hat{x}_t^{den}$ 当作 t=1 的 clean input 交给模型，它最终会输出什么 token？"

### 核心 Gap 分析

**2×2 对照表**（回答 final revision spike 的来源）：

| 输入 | readout 方式 | 记号 | 问题 |
|---|---|---|---|
| $\hat{x}_{0.95}^{den}$ | lin（denoising time forced decode） | $p_{0.95}^{lin}$ | denoiser 在 t=0.95 能读出什么 |
| $\hat{x}_{0.95}^{den}$ | dec（backbone@t=1, then decode head） | $p_{0.95}^{dec}$ | decoder 能否提前修正 t=0.95 的输出 |
| $\hat{x}_{1.00}^{den}$ | lin | $p_{1.00}^{lin}$ | trajectory 到终点本身多好 |
| $\hat{x}_{1.00}^{den}$ | dec（ELF 实际输出） | $p_{1.00}^{dec}$ | ELF 最终实际输出 |

**三个 Gap**：

$$G_\text{dec}(t) = \text{top1}(p_t^{dec}) - \text{top1}(p_t^{lin}) \quad \text{（decode 修正能力）}$$

$$G_\text{traj} = \text{top1}(p_{1.0}^{lin}) - \text{top1}(p_{0.95}^{lin}) \quad \text{（trajectory 最后一步自身改进）}$$

$$G_\text{final} = \text{top1}(p_{1.0}^{dec}) - \text{top1}(p_{1.0}^{lin}) \quad \text{（t=1 时 decode head 还需补多少）}$$

### Decode Correction Residual

$$c_t = h_t^{dec} - \hat{x}_t^{den}$$

这是 decode branch 对 denoising output 施加的**几何 correction vector**。

分析方向：
- $\|c_t\|$ 随 t 的变化曲线（早期大/小？）
- w→c 位置的 $\|c_t\|$ 是否高于中位数（residual 能预测 correction？）
- 插值探针：$\tilde{x}_t(\gamma) = (1-\gamma)\hat{x}_t^{den} + \gamma h_t^{dec}$，$\gamma \in \{0, 0.25, 0.5, 0.75, 1.0\}$，看 top1_gt 是否单调递增

### 四种可能结果与研究含义

**Case 1：$p_t^{dec}$ 从很早就显著优于 $p_t^{lin}$**  
→ Decode branch 始终有纠错能力，只是 ELF 仅在终点用它  
→ 方法方向：**Progressive Decode Correction**——在 trajectory 中逐步注入 decode correction

**Case 2：$p_t^{dec}$ 只在 t > 0.4 后显著变好**  
→ 符合 4-phase story：早期噪声太大 decode 也救不了；crystallization 后 decode branch 开始发挥  
→ 方法方向：**Decode correction should be activated after lexical beliefs emerge**——残差门控，在 commitment cliff 之后才激活

**Case 3：$p_t^{dec}$ 和 $p_t^{lin}$ 几乎一样，只有 t=1 好**  
→ Final improvement 主要来自 t=1 的 input distribution 特殊，decode head 只适配 clean endpoint  
→ 含义：不能声称"decode correction can be used at every t"；需要更小心的 fine-tuning

**Case 4：$p_t^{dec}$ 早期反而更差**  
→ Decode branch 训练时只处理接近 clean 的 embedding，对中间 denoising output 不适配  
→ 含义：需要先训练 decode branch 接受 arbitrary-t 输入

### 实验结果（已完成，64 seqs）

| t | lin_top1 | dec_top1 | G_dec | lin_ce | dec_ce | res_norm |
|---|---|---|---|---|---|---|
| 0.00 | 0.000 | 0.002 | +0.002 | 20.67 | 16.77 | 18.52 |
| 0.25 | 0.260 | 0.345 | +0.084 | 13.93 | 9.04 | 20.48 |
| 0.30 | 0.455 | 0.700 | **+0.245** | 9.55 | 3.24 | 20.89 |
| 0.50 | 0.768 | 0.963 | +0.196 | 3.52 | 0.50 | 21.71 |
| 0.95 | 0.810 | 0.969 | +0.159 | 2.64 | 0.46 | 21.78 |
| 1.00 | **0.979** | 0.899 | **-0.080** | 0.39 | 0.69 | 12.73 |

**2×2 分解表**：

| 输入 | readout | top1_gt | CE |
|---|---|---|---|
| x̂_0.95 + lin | 0.810 | 2.635 |
| x̂_0.95 + dec | **0.969** | **0.464** |
| x̂_1.00 + lin | 0.979 | 0.385 |
| x̂_1.00 + dec（ELF 实际） | 0.899 | 0.687 |

**G_traj（lin 0.95→1.00）= +0.169**  
**G_dec（dec−lin @ t=0.95）= +0.159**  
**G_final（dec−lin @ t=1.00）= −0.080**（两遍 backbone 产生 artifact，不是真实 gap）

**插值探针 @ t=0.95**（γ=0 是纯 lin，γ=1 是纯 dec）：

| γ | top1_gt |
|---|---|
| 0.00 | 0.810 |
| 0.25 | 0.886 |
| 0.50 | 0.940 |
| 0.75 | 0.961 |
| 1.00 | 0.969 |

单调递增 → **decode residual c_t 是有效的修正方向**。

**w2c_corr @ t=1.0 = 0.906**：90% 的 wrong→correct 位置有超中位数的 ||c_t||。

### 结论：Case 1 确认

整个 plateau（t=0.30–0.95）decode branch 一直有能力提前修正，gap 持续在 +0.15–+0.25。ELF 只在 t=1 用 decode branch，这是一种资源浪费。

**方法方向锁定：Progressive Decode Correction**

核心思想：在 denoising trajectory 的中间时刻逐步注入 decode branch 的修正信号，而不是只在终点做一次。训练目标候选：

$$\mathcal{L}_\text{PDC}(t) = \left\|\hat{x}_t^{den} - \text{sg}(h_t^{dec})\right\|^2 \cdot \mu(t)$$

其中 $\mu(t)$ 在 commitment cliff（t≈0.25）之后才激活，在 t=1 降到 0（避免 artifact）。

---

## 6. LangFlow Probe 结果（v1 方向有误，v2 已修正运行中）

### v1 发现的 gamma 方向 bug

`probe_langflow.py` 原始版本 `gamma_from_t` 把 t=0 映射到 γ_min（CLEAN），t=1 映射到 γ_max（NOISY），即**反向**。修正后：

```python
# 正确：t=0 → γ_max（NOISY），t=1 → γ_min（CLEAN）
return gamma_max + t_grid * (gamma_min - gamma_max)
```

v1 结果仍然可以通过翻转 t 轴来读取单调指标（H、top1、d_nn、margin），**方向无关指标**：

| t_real（反转后）| H | top1_gt | d_nn | margin |
|---|---|---|---|---|
| 0.00（noisy） | 7.537 | 0.037 | 36.39 | 0.14 |
| 0.70 | 5.514 | 0.099 | — | — |
| 0.85 | 1.597 | 0.611 | — | — |
| 0.95 | 0.163 | 0.940 | — | — |
| 1.00（clean）| 0.025 | 0.989 | 33.62 | 1.01 |

**Conjecture 5 在 LangFlow 上验证通过**：
- Δd_nn = −2.77（几何距离收缩 ✓）
- Δmargin = +0.87（分离增大，非坍塌 ✓）

### LangFlow vs ELF 对比（方向无关指标）

| 特征 | ELF | LangFlow |
|---|---|---|
| Commitment cliff | t ≈ 0.15–0.25（早） | t ≈ 0.70–0.85（晚） |
| Stable plateau | ✓ 长（t=0.25–0.95） | ✗ 无（单调下降） |
| Final spike | ✓（w2c=0.171 @ t=1） | ✗ 无 |
| d_nn 趋势 | 无变化（space mismatch） | −2.77（geometry anchoring ✓） |
| margin 趋势 | 无变化（space mismatch） | +0.87（genuine anchoring ✓） |

**架构解释**：ELF final-only decode → early cliff + long plateau + final spike 三段式结构。LangFlow always-on supervision → 晚期、平滑、单调结晶，无 plateau，无 final spike。

### v2 结果（已完成，Jun 1 17:20）

| t | H | top1_gt | w2c | c2c | rev_jsd |
|---|---|---|---|---|---|
| 0.00 | 7.534 | 0.037 | nan | nan | nan |
| 0.25 | 7.545 | 0.040 | 0.010 | 0.001 | 0.008 |
| 0.50 | 7.110 | 0.039 | 0.006 | 0.001 | 0.206 |
| 0.75 | 4.948 | 0.171 | 0.086 | 0.015 | 0.491 |
| 0.85 | 1.597 | 0.614 | 0.299 | 0.256 | — |
| 0.90 | 0.595 | 0.821 | 0.260 | 0.542 | — |
| 0.95 | 0.163 | 0.939 | 0.147 | 0.795 | — |
| 1.00 | 0.025 | 0.989 | 0.054 | 0.940 | 0.043 |

**commit_time_mean = 0.909，never_committed = 3.2%**

**关键发现**：
- LangFlow cliff 在 t=0.75–0.90（entropy 从 4.95→0.60）
- w2c 峰值在 t=0.85（0.299），c2c 从 t=0.75 开始急升（0.015→0.256→0.542→0.795→0.940）
- rev_jsd 峰值在 t=0.75（0.491）——"颠覆性" JSD 在 cliff 区域最大
- 与 ELF 对比：**LangFlow 比 ELF 晚 47.9 倍** commit（0.909 vs 0.019）

---

## 7. Probe v4（ELF，新增 4 项指标，已完成 Jun 1 18:14）

### 新增指标设计

**A. Residual norm 分布**：`||r_t|| = ||x_hat_t - sum_v p(v|t) * E[v]||` 的 p10/p50/p90

**B. Temporal stability JSD 分布**：每个位置的 JSD(p_{t-1}, p_t) 的 p10/p50/p90

**C. Wrong commitment fate tracking**（fate 字段）

**D. Entropy bimodality coefficient**

### v4 结果（tau=1.0，64 seqs）

| t | H | top1_gt | res_p50 | stab_jsd_p50 | bimod | w2c | c2c |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.118 | 0.000 | 2668 | nan | 0.793 | nan | nan |
| 0.05 | 0.123 | 0.003 | 2684 | 0.026 | 0.800 | 0.000 | 0.000 |
| 0.10 | 0.448 | 0.015 | 2234 | 0.437 | 0.644 | 0.009 | 0.000 |
| 0.15 | 0.494 | 0.033 | 2062 | 0.674 | 0.604 | 0.015 | 0.007 |
| 0.20 | 0.376 | 0.086 | 2065 | 0.691 | 0.639 | 0.036 | 0.021 |
| **0.25** | **0.201** | **0.260** | 2057 | **0.693** | 0.737 | **0.162** | 0.052 |
| **0.30** | **0.128** | **0.455** | 2024 | **0.666** | 0.792 | **0.208** | 0.206 |
| 0.35 | 0.094 | 0.592 | 2023 | **0.058** | 0.814 | 0.164 | 0.401 |
| 0.40 | 0.077 | 0.683 | 2017 | 0.003 | 0.827 | 0.111 | 0.552 |
| 0.50 | 0.058 | 0.768 | 2022 | 0.000 | 0.845 | 0.043 | 0.714 |
| 0.75 | 0.057 | 0.794 | 2026 | 0.000 | 0.831 | 0.012 | 0.775 |
| 1.00 | 0.055 | 0.979 | 1894 | 0.003 | 0.989 | 0.171 | 0.808 |

**commit_time_mean = 0.019（h_frac_low@t=0 = 0.77，大量位置已在 t=0 时 H<0.1）**

**新发现**：
1. **Residual norm 先升后降**：p50 从 t=0→2668 升到 t=0.05→2684，随后在 t=0.10-0.15 急跌到 2062。这一 **residual norm 下降超前于 commitment cliff**（t=0.25-0.30），是 cliff 的先兆
2. **Temporal stability cliff 极其尖锐**：stab_jsd_p50 在 t=0.25 = **0.6931 ≈ log(2)**（最大可能 JSD！），在 t=0.35 骤降至 0.058（↓11×），t=0.40 接近 0——**锁定边界在 t=0.30-0.35**，且在 cliff 处预测不稳定性达到理论上限
3. **Bimodality 全程 > 5/9**（整个流轨迹都是双峰分布），BC@t=0 = 0.793，cliff 后持续升高至 0.989@t=1.0
4. **W→C 批量转换**：w2c 峰值在 t=0.25-0.30（0.162, 0.208），c2c 从 t=0.30 开始急升（0.206→0.401→0.552→0.714）

### Fate Tracking：错误承诺的命运

| source_t | n_wrong (均值) | traj_corrected | decode_corrected | stays_wrong |
|---|---|---|---|---|
| 0.25 | 69.7 | **73.9%** | 22.2% | 3.9% |
| 0.30 | 62.9 | 63.6% | 32.1% | 4.3% |
| 0.40 | 39.6 | 38.5% | **56.8%** | 4.7% |
| 0.50 | 30.6 | 24.0% | **71.8%** | 4.3% |

**关键规律**：
- **早期错误承诺（t=0.25）**：74% 在后续轨迹中自我修正，22% 靠 decode branch 修正。自愈能力强。
- **晚期错误承诺（t=0.50）**：仅 24% 自我修正，72% 依赖 decode branch 才修正。**decode branch 对晚期承诺不可或缺**。
- **不可修复率 ≈ 4%**：无论 source_t 如何，约 4% 的错误承诺永久错误。这是 ELF 的 hard error floor。
- **Progressive Decode Correction 的动机**：decode branch 在 t=0.40-0.50 已能修正 57-72% 的错误，将这个能力前移到轨迹中间，可以将错误减少 0.5-0.7× 而无需等到 t=1。

---

## 8. 当前 Probe 全图

```
Probe A (v2)          — 4-phase commitment，tau sweep，JSD             [已完成]
Probe B (v3)          — commitment state，transition matrix，几何指标  [已完成]
Probe C (decode)      — lin vs dec 两路分歧，Gap 分析，residual c_t    [已完成]
Probe D (LangFlow v1) — 跨模型对比（方向 bug，已用反转轴分析）          [已完成]
Probe D (LangFlow v2) — 修正方向后，正确的转移矩阵和 fate tracking      [已完成 Jun1 17:20]
Probe E (v4)          — 残差范数分布、temporal JSD 分布、fate、bimodality [已完成 Jun1 18:14]
Probe F (MDLM)        — AbsorbingState masked diffusion，熵/top1/JSD   [已完成 Jun2，commit_time=0.866]
Probe G (DUO)         — UniformState uniform-noise，corrupted/clean 分离 [已完成 Jun1（GPT-2 fallback）+ Jun2（真实 s-sahoo/duo 模型）]
Probe H (centroid)    — contextual centroid E[v]，compute_token_centroids [已完成（GPU 4）]
```

### SNR 分析（analyze_snr_gdec.py，Jun 1）

ELF SNR(t) = t²/(1-t)²，LangFlow SNR(t) = exp(-γ(t)) = exp(-16.05+13.45t)

| t | SNR_ELF | SNR_LF | ratio |
|---|---|---|---|
| 0.25 | 0.1111 | 0.00106 | **105×** |
| 0.50 | 1.0000 | 0.00424 | 236× |
| 0.75 | 9.0000 | 0.01694 | 531× |
| 0.85 | 32.11 | 0.04416 | 727× |

**结论**：ELF 在 t=0.25 时 SNR 比 LangFlow 高 105×，这直接解释了为什么 ELF 在 t=0.25 发生 commitment cliff 而 LangFlow 直到 t=0.85 才 cliff。

μ(t) 调度：0.9497·(t-0.25)^0.031·(0.95-t)^0.137（t∈[0.25,0.95]，G_dec 归一化后）

---

## 9. DUO（UniformState）Probe 结果（Jun 2）

**模型**：s-sahoo/duo（HuggingFace，trust_remote_code，GPT-2 tokenizer，vocab=50257）  
**配置**：n_samples=64，seq_len=128，n_t_steps=21，n_noise=4，alpha_min=1e-4，commit_thresh=0.1  
**结果文件**：`experiments/probe_anchor/results_duo_v1/duo_probe.json`

### 关键数值

| t | corr% | H_corr | top1_rec | sc_frac |
|---|---|---|---|---|
| 0.00 | 1.000 | 7.604 | 0.030 | 0.000 |
| 0.20 | 0.999 | 7.574 | 0.035 | 0.000 |
| 0.40 | 0.996 | 7.460 | 0.033 | 0.000 |
| 0.60 | 0.975 | 7.253 | 0.035 | 0.000 |
| 0.70 | 0.937 | 6.843 | 0.061 | 0.002 |
| 0.80 | 0.840 | 5.840 | 0.131 | 0.014 |
| 0.90 | 0.599 | 3.548 | 0.364 | 0.092 |
| 0.95 | 0.370 | 2.082 | 0.553 | 0.205 |

- **commit_time_mean = 0.904**（极晚，几乎在最后 5% 的 denoising 才发生）
- **never_committed_frac = 0.869**（87% 的 corrupted positions 全程熵不低于 0.1）
- **10% 恢复率达到 t = 0.80**（ELF 是 t≈0.15-0.20）

### 与 ELF 对比（AbsorbingState vs UniformState）

| 指标 | ELF (AbsorbingState) | DUO (UniformState) |
|---|---|---|
| 10% 恢复率时刻 | t≈0.15-0.20 | t≈0.80 |
| Commitment cliff | t≈0.20-0.35（极尖锐） | t≈0.80-0.95（缓慢） |
| commit_time_mean | ~0.22 | 0.904 |
| never_committed | ~0.02 | 0.869 |
| H 降至 < 1.0 的时刻 | t≈0.50 | t > 0.95 |

### 解释

DUO 的极晚 commitment 是噪声过程的内在属性：
- **AbsorbingState (ELF/MDLM)**：被噪声替换的 position 显式标记为 [MASK]，模型知道"哪里需要预测"→ 可以提前 focus → 早期 commitment
- **UniformState (DUO)**：被噪声替换的 position 是随机 token，模型无法区分"真实 token"和"随机 token"→ 必须全局保持不确定性 → 晚期 commitment

在 alpha_min=1e-4（lambda≈9.21）的 loglinear 调度下：
- t=0.70 时 alpha≈0.063，仍有 94% 的 token 是随机噪声
- 模型在近乎全随机的输入下无法做出任何有意义的 commitment

这个对比**支持 ELF 的 early commitment 结构是模型成功的关键机制**，而非 UniformState 这种"all-or-nothing"的晚期 commitment。

---

## 10. MDLM（AbsorbingState）Probe 结果 + 三模型对比（Jun 2）

**模型**：kuleshov-group/mdlm-owt（HuggingFace）  
**配置**：n_samples=64，seq_len=128，alpha_min=1e-4（loglinear），commit_thresh=0.1  
**结果文件**：`experiments/probe_anchor/results_mdlm_v1/mdlm_probe.json`

### MDLM 结果

| t | mask% | H_mask | top1_rec | sc_frac |
|---|---|---|---|---|
| 0.50 | 0.990 | 6.682 | 0.037 | 0.001 |
| 0.70 | 0.936 | 6.032 | 0.083 | 0.007 |
| 0.85 | 0.747 | 4.247 | 0.249 | 0.048 |
| 0.95 | 0.368 | 1.935 | 0.559 | 0.191 |

- **commit_time_mean = 0.866**
- **never_committed_frac = 0.851**
- **10% recovery at t = 0.75**

### 三模型对比

| 指标 | ELF（连续 flow） | MDLM（离散 AbsorbingState） | DUO（离散 UniformState） |
|---|---|---|---|
| 10% 恢复率时刻 | t≈0.15–0.20 | t=0.75 | t=0.80 |
| commit_time_mean | ~0.22 | 0.866 | 0.904 |
| never_committed | ~0.02 | 0.851 | 0.869 |
| H 降至 <1.0 的时刻 | t≈0.50 | t>0.95 | t>0.95 |

### 重要说明：调度差异

三个 probe 都使用 loglinear schedule（alpha_min=1e-4）：

| t | ELF SNR（$t^2/(1-t)^2$） | MDLM/DUO noise_rate（$1-\alpha_t$） |
|---|---|---|
| 0.25 | 0.111 | 99.9% |
| 0.50 | 1.000 | 99.0% |
| 0.75 | 9.000 | 90.0% |
| 0.90 | 81.00 | 60.2% |

**ELF 在 t=0.50 时 SNR=1.0，而 MDLM/DUO 在 t=0.50 时 99% 的 token 仍是噪声！**

这解释了为什么 MDLM 和 DUO 的 commitment 都发生在 t>0.80：在此之前，输入几乎完全是噪声，模型无法做出有信心的预测。ELF 的早期 commitment（t≈0.25）是其**线性 flow 的高 SNR 调度**的直接结果，不是模型架构差异。

### MDLM vs DUO 的微小差异

两者 recovery 曲线非常相似，但 MDLM 略好：
- t=0.75：MDLM=11.8% vs DUO=7.9%
- t=0.90：MDLM=39.1% vs DUO=36.4%

MDLM 的轻微优势符合理论：AbsorbingState 通过 [MASK] 标记明确知道"哪里需要预测"→ 可以利用 clean 位置的上下文；UniformState 需要同时推断"哪里被替换"和"应该是什么"。但这个差异很小，说明在相同 noise level 下两者的 commitment 机制大体相同。

---

## 11. 待做的 Probe

### 上下文敏感 Anchor（正确性修正）

当前用 $E_\text{raw}$（T5 input embedding）作为 anchor matrix，但 $\hat{x}_t$ 在 T5 contextual 空间，两者不同。正确做法：计算**contextual centroid**：
$$E[i] = \mathbb{E}_{\text{训练数据中 token } i \text{ 出现的位置}}\left[T5\text{-encoder-output}_i\right]$$

需要单独写 `compute_token_centroids.py`，用 OWT 数据跑 T5 encoder 收集统计量。

### LangFlow 对应 Probe

**LangFlow**（arXiv:2604.11748，PyTorch 实现）与 ELF 的主要区别：

| | ELF | LangFlow |
|---|---|---|
| Latent 空间 | T5 contextual embeddings | 可学习 token embedding（layernorm + √d） |
| 时间参数化 | t ∈ [0,1] 线性 | γ = log-SNR，learnable Gumbel proposal |
| 噪声公式 | $z_t = t \cdot x + (1-t)\varepsilon$ | $z = \alpha(γ)\cdot x + \sigma(γ)\cdot\varepsilon$ |
| Decoder | Factored unembedding head | 直接线性 + preconditioning skip |
| 框架 | JAX/Flax | PyTorch |
| Vocab | T5 (32100) | GPT-2 (50257) |

**LangFlow 的 geometric probe 更干净**：latent space IS token embedding space，所以 $D_\text{NN}$ 直接就是"到最近 token 的距离"，不存在 ELF 那样的 space mismatch。

LangFlow probe 适配要点：
1. t_grid 转 γ：`gamma_grid = model.proposal(t_uniform)` 或直接在 `[gamma_min, gamma_max]` 均匀采样
2. 噪声公式：$z = \alpha \cdot x + \sigma \cdot \varepsilon$，$\alpha = \sqrt{\sigma(-\gamma)}$，$\sigma = \sqrt{\sigma(\gamma)}$
3. Forward 直接返回 logits（无需 `decoder_step_active`）
4. $E$ = `model._get_embedding_matrix()`（已归一化）

---

## 12. 方法方向（待实验验证）

### 核心方法：Decode-Residual Guided Flow

根据 decode branch probe 的结果，可能的训练目标是：

$$\mathcal{L}_\text{residual}(t) = \left\| \hat{x}_t^{den} - \text{sg}(h_t^{dec}) \right\|^2 \cdot \mu(t)$$

其中 $\text{sg}$ 是 stop-gradient，$\mu(t)$ 是随 t 递增的调度函数（早期 t 权重低，commitment cliff 之后权重上升）。

这个损失引导 denoising output 逐步靠近 decode branch 的"理想化"输出，相当于把 decode correction residual $c_t$ 逐步内化进 denoising trajectory。

### 方法命名候选

- **Progressive Decode Correction**（如果 G_dec(t) 从很早就显著）
- **Decode-Residual Guided Flow**（如果 residual $c_t$ 能预测 w→c correction）
- **Annealed Lexical Commitment**（原始设想，更 general）

---

## 13. 待办

- [x] 等待 v3 probe（GPU 4）和 decode branch probe（GPU 7）完成，下载结果
- [x] 分析 2×2 表的结果，确定方法方向（Case 1 确认）
- [x] 写 LangFlow probe 脚本（`experiments/probe_anchor/probe_langflow.py`）
- [x] 修正 LangFlow gamma 方向 bug，上传服务器
- [x] 分析 LangFlow v1 结果（通过反转 t 轴），验证 Conjecture 5
- [x] 写 probe_anchor_v4.py（残差分布、temporal JSD、fate tracking、bimodality）
- [x] 修复 cuDNN 版本不匹配（9.1→9.23），重启 v4
- [x] 等待 LangFlow v2 probe 完成，下载并分析
- [x] 等待 v4 probe 完成，下载并分析 4 项新指标
- [x] 写 probe_mdlm.py（masked discrete diffusion，kuleshov-group/mdlm-owt）
- [x] 写 probe_duo.py（UniformState discrete diffusion，s-sahoo/duo）
- [x] 写 compute_token_centroids.py（contextual centroid E[v]）
- [x] 写 analyze_snr_gdec.py（SNR 对比 + μ(t) 调度推导）
- [x] 等待 MDLM probe 完成，下载并分析（results_mdlm_v1/）
- [x] centroid 计算完成（PyTorch T5-small，4096 texts，22250/32100 tokens），下载至 probe_anchor/token_centroids.npz
- [x] 完成 DUO (UniformState) probe，下载并分析（results_duo_v1/）
- [x] 撰写 probe findings 综述（docs/probe_findings_section.md）
- [x] 服务器代码整理至 /home/wjzhang/tt_workspace/CCLF/
- [ ] 用 contextual centroid 重跑 ELF anchor distance probe（替换 raw E_raw）
- [ ] 撰写方法部分（Revisable Lexical Anchoring 训练目标）
