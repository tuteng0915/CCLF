# Global State Formation in Continuous Language Flows  
## 从全局语义盆地到局部词汇结晶：实验设计文档

---

## 0. 核心问题

此前的实验主要从 **per-token / per-position** 角度研究连续语言模型中的 lexical commitment：

- 每个位置何时从高频默认 token 转向真实 token；
- true-token margin 如何增长；
- native decoder 何时越过离散决策边界；
- 某个位置的预测何时稳定。

但这些现象可能只是更高层过程的结果，而不是信息最初形成的地方。

连续语言模型在时刻 \(t\) 的完整状态是：

\[
Z_t=[z_{1,t},z_{2,t},\ldots,z_{L,t}]
\in\mathbb{R}^{L\times d}.
\]

因此，更基础的问题是：

> **模型是否先在整个序列状态中形成全局语义、主题、结构和关系模式，再由这些全局模式推动各位置的 lexical crystallization？**

本实验套件研究：

1. 全局语义是否早于 exact token identity 出现；
2. 整个序列何时进入一个稳定的生成 basin；
3. 全局结构如何影响局部 token evidence；
4. oracle 和 free-running trajectory 在全局状态形成上如何不同；
5. 全局阶段转移有哪些失败模式；
6. ELF 与 LangFlow 是否共享同一套 global-to-local 形成框架。

---

# 1. 核心假设

## H1：Global-before-local

模型首先形成全局主题和序列级语义，再形成结构骨架，最后确定具体 token：

\[
\boxed{
\text{global semantic basin}
\rightarrow
\text{structural scaffold}
\rightarrow
\text{local lexical evidence}
\rightarrow
\text{exact token}
}
\]

## H2：Globally informative but locally undecodable

在某些早期时刻，单个位置仍无法恢复真实 token，但完整序列状态已经能预测：

- 主题；
- 文本类型；
- 最终 sentence embedding；
- bag-of-words；
- 实体集合；
- 句法结构。

即：

\[
I(Y_{\text{global}};Z_t)>0
\]

但：

\[
I(y_i;z_{i,t})
\]

仍较低。

## H3：Global basin selection is collective

全局转移不是独立 token crossing 的简单平均，而是 attention 和联合 velocity field 驱动的 collective organization：

\[
v_i(Z_t,t)=v_i(z_{1,t},\ldots,z_{L,t},t).
\]

## H4：Global structure causally drives lexical crystallization

删除、交换或注入全局模式，会系统性改变后续局部 token evidence 和最终生成结果。

## H5：Free-running failure can occur at different levels

真实生成可能失败于：

1. 没有选择全局 basin；
2. 选择了错误 basin；
3. 全局正确但结构没有形成；
4. 结构正确但 lexical crystallization 失败；
5. 局部 token 过早锁定，反过来阻碍全局调整。

## H6：Cross-architecture generality

ELF 和 LangFlow 都可以用 global-to-local 框架描述，但它们在以下方面可能不同：

- 全局信息何时出现；
- categorical posterior 与 global state 的耦合程度；
- basin selection 的稳定性；
- global-to-local 信息传递的强度。

---

# 2. 全局阶段定义

## Stage A：Prior Cloud

特点：

- 整体状态主要由初始噪声、语言先验和 decoder bias 主导；
- branch continuation 在主题、结构和 token 层面都高度分散；
- sequence-level probe 很难预测最终生成属性；
- 不同位置之间缺乏稳定的 clean-like organization。

## Stage B：Global Basin Selection

特点：

- 多个 continuation 开始落入相似主题或语义区域；
- sentence embedding、topic、style 等全局属性逐渐稳定；
- exact token 仍然高度不确定；
- 可能出现多个词汇不同、但语义相近的 continuation。

例如：

```text
earthquake hit the city
storm damaged the capital
disaster struck the region
```

虽然 lexical realization 不同，但全局 basin 已经是“灾难新闻”。

## Stage C：Structural Scaffold Formation

特点：

- 主题已大致确定；
- 句法槽位、实体位置、短语边界和论元关系开始形成；
- 可以恢复：

```text
[ENTITY] suffered a [DISASTER] in [LOCATION]
```

但 exact token 尚未最终确定。

## Stage D：Lexical Crystallization

特点：

- 每个位置的候选集合显著收缩；
- true-token margin 越过 decoder decision boundary；
- exact token identity 逐步稳定；
- 局部 token 形成最终离散序列。

---

# 3. 全局阶段的操作化定义

对每个时间 \(t\)，定义四类可恢复性指标。

## 3.1 Global semantic recoverability

\[
G_{\mathrm{sem}}(t)
\]

衡量从完整 \(Z_t\) 中恢复最终序列的：

- sentence embedding；
- topic；
- text domain；
- style；
- entity set；
- bag-of-words。

## 3.2 Structural recoverability

\[
G_{\mathrm{struct}}(t)
\]

衡量：

- POS sequence；
- dependency adjacency；
- phrase boundary；
- token-order skeleton；
- relation matrix。

## 3.3 Local lexical recoverability

\[
G_{\mathrm{lex}}(t)
\]

衡量：

- per-position token probe；
- true-token rank；
- native top-1；
- stable-final token agreement。

## 3.4 Global branching uncertainty

\[
H_{\mathrm{global}}(t)
\]

从同一 \(Z_t\) 分叉多个 continuation，衡量：

- topic entropy；
- sentence-embedding cluster entropy；
- structural entropy；
- lexical entropy。

若 global-to-local 假设成立，应出现：

\[
\tau_{\mathrm{sem}}<\tau_{\mathrm{struct}}<\tau_{\mathrm{lex}}.
\]

其中：

\[
\tau_k=\min\{t:G_k(t)\ge\theta_k\}.
\]

阈值建议定义为达到 clean-state performance 的固定比例，例如 80%，不要跨任务使用同一个绝对阈值。

---

# 4. 共享实验协议

## 数据

主实验：

- OpenWebText validation；
- 512 sequences；
- sequence length = 128；
- padding 和特殊 token 单独处理；
- 所有 learned probes 使用 document-level train/validation/test split。

扩展实验：

- 长序列 \(L=512\) 或 \(1024\)；
- 多 domain 数据；
- 控制文本长度和主题分布。

## 时间网格

模型内部分析：

- 101 个 dense native timesteps；
- 同一 sequence 在所有 \(t\) 使用同一个 \(\epsilon\)。

跨模型分析：

- 在 ELF 和 LangFlow 重叠的 log-SNR 区间内；
- 使用 41 个 matched log-SNR 点；
- 不比较 nominal \(t\)。

## 统计

- sequence-level bootstrap；
- 95% confidence intervals；
- 干预实验至少 5 个 generation seeds；
- 多分支实验每个 state 至少 16 branches；
- 报告 mean、median、10/90 percentile；
- 所有 per-position 结果保留，以便与 global signals 对齐。

## 建议存储结构

```text
results/global_state/
  <model>/
    <checkpoint>/
      metadata.json
      tokens.npy
      mask.npy
      t_grid.npy
      logsnr_grid.npy
      states.zarr
      hidden.zarr
      logits.zarr
      velocity.zarr
      sc_state.zarr
      global_features.zarr
      branch_outputs/
```

---

# 5. GLOBAL-1 — Sequence-Level Probe Hierarchy

## 目的

验证完整序列状态是否比单位置状态更早携带全局信息。

## 输入表示

对完整状态 \(Z_t\) 构造多种 summary。

### A. Mean pooling

\[
g_t^{mean}=\frac{1}{L}\sum_i z_{i,t}.
\]

### B. Attention pooling

训练一个轻量 attention pooling：

\[
g_t^{attn}=\sum_i\alpha_i z_{i,t}.
\]

### C. Low-rank summary

对 \(Z_t\) 做 SVD，保留 top-\(k\) singular components：

\[
g_t^{svd}=\operatorname{vec}(U_k\Sigma_k).
\]

建议：

\[
k\in\{1,2,4,8,16\}.
\]

### D. Pairwise relational summary

构造：

\[
R_t=Z_tZ_t^\top
\]

或其低秩近似，用于捕捉位置间关系。

## 预测目标

### Global semantic targets

1. clean sentence embedding；
2. topic cluster；
3. document/domain label；
4. bag-of-words multi-label；
5. named-entity set；
6. final continuation cluster。

### Structural targets

1. POS histogram；
2. POS sequence；
3. dependency edge matrix；
4. phrase boundary；
5. sequence length；
6. entity-position pattern。

### Local lexical target

1. exact token identity；
2. native top-1；
3. per-position linear probe。

## Probe 设计

使用容量递增 ladder：

1. Ridge / linear probe；
2. 2-layer MLP；
3. lightweight transformer probe。

主结论优先依赖 linear / low-capacity probe。

所有 probe：

- document-level held-out test；
- 3–5 seeds；
- shuffled-label null；
- train-size scaling curve；
- equal-capacity controls。

## 关键指标

\[
G_{\mathrm{topic}}(t),\quad
G_{\mathrm{sent}}(t),\quad
G_{\mathrm{syntax}}(t),\quad
G_{\mathrm{token}}(t).
\]

计算：

\[
\tau_{\mathrm{topic}},
\quad
\tau_{\mathrm{syntax}},
\quad
\tau_{\mathrm{token}}.
\]

## 预期结果

支持 global-to-local：

\[
\tau_{\mathrm{topic}}<\tau_{\mathrm{syntax}}<\tau_{\mathrm{token}}.
\]

更强结果是：

- topic/sentence embedding 已可恢复；
- exact token probe 仍接近随机；
- mean/SVD summary 已有信息；
- 单位置状态仍无明显 token evidence。

## 失败解释

### 所有任务同时出现

可能没有明显 global-to-local hierarchy，而是整体 SNR 驱动。

### token 早于 topic

说明局部 lexical evidence 可能先出现，再被聚合成全局意义。

### 只有大型 probe 可恢复

可能是 probe 在训练集上重构复杂映射，不代表自然线性可读性。

## 脚本

```text
experiments/global_state/probe_sequence_hierarchy.py
experiments/global_state/analyze_probe_transition.py
```

---

# 6. GLOBAL-2 — Hierarchical Branch Consensus

## 目的

从动力学角度测量整个生成 basin 在什么时候被选定。

## 方法

从真实 trajectory 的多个时间点保存完整状态：

\[
S_t=(Z_t,\mathrm{SC}_t).
\]

对每个 state：

- 加归一化微扰；
- 或改变后续 stochasticity；
- 生成 \(K\) 条 continuation。

建议：

\[
K=16
\]

主实验可用：

\[
K=32.
\]

## 分支比较层级

### A. Global semantic agreement

- sentence embedding similarity；
- topic cluster agreement；
- document-type agreement；
- entity-set overlap；
- bag-of-words overlap。

### B. Structural agreement

- POS sequence similarity；
- dependency-tree similarity；
- phrase-boundary F1；
- sequence-length variance；
- entity-position agreement。

### C. Lexical agreement

- exact sequence match；
- per-position token agreement；
- edit distance；
- n-gram overlap。

## 定义层级 branch entropy

\[
H_{\mathrm{topic}}(t),\quad
H_{\mathrm{struct}}(t),\quad
H_{\mathrm{lex}}(t).
\]

也可定义：

\[
C_k(t)=1-\frac{H_k(t)}{H_k^{max}}.
\]

## 关键判断

若 \(H_{\mathrm{topic}}(t)\) 很早下降，但 \(H_{\mathrm{lex}}(t)\) 仍高，则说明：

> global basin 已确定，但 lexical realization 尚未确定。

若 topic 和 lexical entropy 同时下降，则 global 和 local 可能没有明显分层。

## Perturbation 规范

必须扰动完整状态：

\[
S_t=(Z_t,\mathrm{SC}_t).
\]

归一化：

\[
\delta=\eta\|Z_t\|_2\frac{u}{\|u\|_2}
\]

其中：

\[
\eta\in
\{10^{-4},3\times10^{-4},10^{-3},3\times10^{-3},10^{-2}\}.
\]

## 输出

```text
branch_consensus.json
branch_topic_clusters.npy
branch_structure_metrics.json
branch_lexical_metrics.json
```

## 脚本

```text
experiments/global_state/branch_global_consensus.py
experiments/global_state/analyze_branch_hierarchy.py
```

---

# 7. GLOBAL-3 — Low-Rank Global Mode Analysis

## 目的

检验早期全局信息是否首先出现在跨位置共享的低秩模式中。

## 分解

对每个：

\[
Z_t\in\mathbb{R}^{L\times d}
\]

做：

\[
Z_t=U_t\Sigma_tV_t^\top.
\]

定义 rank-\(k\) global component：

\[
G_t^{(k)}=U_{t,k}\Sigma_{t,k}V_{t,k}^\top.
\]

local residual：

\[
R_t^{(k)}=Z_t-G_t^{(k)}.
\]

建议：

\[
k\in\{1,2,4,8,16\}.
\]

## 指标

### Effective rank

\[
r_{\mathrm{eff}}(t)
=
\exp\left(-\sum_jp_j(t)\log p_j(t)\right),
\]

其中：

\[
p_j(t)=\frac{\sigma_j(t)}{\sum_l\sigma_l(t)}.
\]

### Global-mode clean alignment

\[
A_G(t)=\operatorname{CKA}(G_t^{(k)},G_{\mathrm{clean}}^{(k)}).
\]

### Residual clean alignment

\[
A_R(t)=\operatorname{CKA}(R_t^{(k)},R_{\mathrm{clean}}^{(k)}).
\]

### Probe performance

分别从：

- \(G_t^{(k)}\)；
- \(R_t^{(k)}\)；
- 完整 \(Z_t\)；

预测：

- topic；
- sentence embedding；
- syntax；
- exact token。

## 核心问题

是否存在 \(A_G(t)\) 明显早于 \(A_R(t)\) 上升？

是否：

- global component 早期可预测 topic；
- residual 后期才可预测 exact token？

## 重要控制

- mean-center across positions；
- remove positional embedding contribution；
- shuffled-position null；
- random low-rank subspace；
- matched-rank Gaussian matrix；
- different \(k\) sensitivity；
- sequence length controls。

## 脚本

```text
experiments/global_state/analyze_low_rank_modes.py
experiments/global_state/probe_global_residual.py
```

---

# 8. GLOBAL-4 — Global Mode Causal Intervention

## 目的

证明全局模式不是被动相关，而是后续 lexical crystallization 的因果驱动因素。

## A. Remove global mode

构造：

\[
Z_t^{-G}=R_t^{(k)}.
\]

从该状态继续采样。

测：

- topic drift；
- sentence embedding change；
- syntax degradation；
- lexical emergence delay；
- final quality。

## B. Preserve only global mode

构造：

\[
Z_t^G=G_t^{(k)}+\epsilon_{\mathrm{matched}},
\]

其中 local noise 保持与原 state residual 相同的均值和方差。

观察：

- topic 是否保留；
- style 是否保留；
- syntax 是否部分保留；
- exact token 是否丢失。

## C. Global-mode swap

对两个序列 \(A,B\)：

\[
Z_t^{swap}=G_t^A+R_t^B.
\]

判断最终：

- topic 更像 \(A\) 还是 \(B\)；
- lexical choice 更像 \(A\) 还是 \(B\)；
- structure 更像哪一方。

若 topic 跟随 \(G^A\)，local realization 部分跟随 \(R^B\)，则强支持 global/local disentanglement。

## D. Oracle global-mode injection

对 paired oracle 与 rollout：

\[
G_t'
=
(1-\lambda)G_t^{roll}
+
\lambda G_t^{oracle},
\]

\[
\lambda\in\{0,0.25,0.5,0.75,1.0\}.
\]

保持 local residual 不变：

\[
Z_t'=G_t'+R_t^{roll}.
\]

继续采样。

## 关键指标

- final topic agreement；
- sentence embedding similarity；
- syntax agreement；
- stable-token time；
- generation quality；
- degeneration rate；
- trajectory divergence。

## 失败解释

### Swap 无影响

SVD global mode 可能不是语义 global mode，只是数值主方向。

### Remove global mode 后 token 仍稳定

全局信息可能编码在高秩 relational structure，而非低秩 component。

### Oracle injection 不改善 rollout

free-running failure 可能主要来自 local dynamics 或 velocity field，而非 global basin selection。

## 脚本

```text
experiments/global_state/intervene_global_modes.py
```

---

# 9. GLOBAL-5 — Collective Coupling and Correlation Length

## 目的

判断阶段转移是否表现为集体组织化，而不是独立 token crossing 的平均。

## Per-position evidence increment

定义：

\[
\Delta m_i(t)=m_i(t+\Delta t)-m_i(t).
\]

计算不同距离的位置相关性：

\[
C_t(d)=\operatorname{Corr}(\Delta m_i(t),\Delta m_{i+d}(t)).
\]

## Correlation length

简单定义：

\[
\xi(t)=\sum_{d=1}^{D}\max(C_t(d),0).
\]

也可拟合：

\[
C_t(d)\approx A\exp(-d/\xi_t).
\]

## Global susceptibility

定义序列平均 evidence：

\[
\bar m(t)=\frac{1}{L}\sum_i m_i(t).
\]

跨 sequence 或 branch 计算：

\[
\chi(t)=L\cdot\operatorname{Var}(\bar m(t)).
\]

## 其他 collective metrics

- velocity direction covariance；
- attention graph modularity；
- position-state covariance；
- branch outcome variance；
- synchronized margin slope；
- mutual information across positions。

## 预期 collective transition 信号

在转移区附近：

- \(\xi(t)\) 上升；
- susceptibility 达到峰值；
- 多个位置 margin slope 同步；
- branch-level global variance 先升后降；
- 后续 global entropy 快速下降。

## 重要控制

- shuffle positions；
- shuffle sequences；
- frequency-matched positions；
- remove function words；
- fixed token-type composition；
- compare to independent-position denoiser baseline。

## 脚本

```text
experiments/global_state/analyze_collective_coupling.py
```

---

# 10. GLOBAL-6 — Competing Global Basins

## 目的

直接测量全局 basin selection 和 bifurcation。

## 数据构造

选择主题明显不同、但长度和句式接近的 clean sequences：

- disaster news；
- election news；
- sports report；
- dialogue；
- scientific explanation。

构造：

\[
X^A,\quad X^B.
\]

## State interpolation

在 matched SNR 下构造：

\[
Z_t(\lambda)=\lambda Z_t^A+(1-\lambda)Z_t^B,
\]

其中：

\[
\lambda\in\{0,0.1,\ldots,1.0\}.
\]

也可只混合 global mode：

\[
Z_t(\lambda)
=
\lambda G_t^A+(1-\lambda)G_t^B+R_t.
\]

## 继续采样

从每个 \(\lambda\)：

- 运行多个 branches；
- 分类最终输出属于 basin A、basin B 或其他；
- 测 exact-token variability。

定义：

\[
P_A(\lambda,t),\quad P_B(\lambda,t).
\]

## 核心问题

- 是否存在明显 bifurcation point？
- 随 \(t\) 增大，basin boundary 是否变陡？
- topic basin 是否早于 lexical realization 稳定？
- ELF 和 LangFlow 的 basin sharpness 是否不同？

## 输出

```text
basin_transition_curves.json
basin_branch_outputs.jsonl
```

## 脚本

```text
experiments/global_state/probe_competing_basins.py
```

---

# 11. GLOBAL-7 — Oracle vs Free-Running Global Alignment

## 目的

把 oracle–rollout gap 从 per-token readout 提升到全局状态形成层面。

## Paired construction

对每次真实生成：

1. 保存初始 noise；
2. 保存完整 rollout trajectory；
3. 编码最终生成文本为 clean endpoint；
4. 用同一 noise 构造 paired oracle path。

得到：

\[
Z_t^{oracle},\quad Z_t^{rollout}.
\]

## 比较指标

### Global semantic alignment

- sentence embedding；
- topic cluster；
- entity set；
- bag-of-words；
- style。

### Structural alignment

- POS sequence；
- dependency relation；
- phrase boundaries；
- relation matrix。

### Geometric alignment

- CKA；
- low-rank subspace distance；
- effective rank；
- cross-position covariance；
- attention structure。

### Dynamical alignment

- velocity disagreement；
- SC disagreement；
- global-mode drift；
- branch entropy。

## 可能的 failure type

### A. Global indecision

rollout 迟迟不进入稳定 semantic basin。

### B. Wrong-basin selection

rollout 很早进入错误 basin。

### C. Fragmented basin

不同位置对应不一致的局部主题。

### D. Scaffold failure

主题大致正确，但结构不稳定。

### E. Lexical crystallization failure

全局和结构正确，但 exact token 不稳定。

## 因果插值

只插值 global component：

\[
G_t(\lambda)
=
(1-\lambda)G_t^{roll}
+
\lambda G_t^{oracle}.
\]

保持 \(R_t^{roll}\) 不变，然后继续采样。

如果 global injection 提升后续 lexical consistency，说明 free-running error propagation 的一部分来自 global basin formation failure。

## 脚本

```text
experiments/global_state/compare_oracle_rollout_global.py
experiments/global_state/interpolate_global_oracle_rollout.py
```

---

# 12. GLOBAL-8 — Global-to-Local Causal Chain

## 目的

直接检验：

\[
\text{global state}
\rightarrow
\text{local evidence}
\rightarrow
\text{exact token}.
\]

## 方法

选择在早期 global probe 已经较强、但 token probe 仍较弱的时间点。

对 global representation 做小干预：

1. toward correct topic；
2. toward incorrect topic；
3. orthogonal control；
4. random same-norm direction。

然后测目标位置：

- true-token margin；
- residual evidence；
- boundary-crossing time；
- stable-final time；
- final token accuracy。

## Global direction construction

### Topic classifier direction

若线性 topic probe 为 \(W_{\mathrm{topic}}\)，则：

\[
u_{\mathrm{topic}}
=
W_{\mathrm{correct}}
-
W_{\mathrm{wrong}}.
\]

### Sentence embedding direction

使用 clean sentence embedding regression 的 gradient direction。

### Oracle global-mode direction

\[
u_G=G_t^{oracle}-G_t^{roll}.
\]

## 干预

\[
Z_t'=Z_t+\alpha u_G,
\]

其中：

\[
\alpha\in\{-1,-0.5,-0.25,0.25,0.5,1\}.
\]

## 支持 causal chain 的结果

- 正确 global direction 提前 local true-token margin；
- 错误 global direction 将 evidence 推向语义一致的错误 token；
- orthogonal control 无明显作用；
- effect 在多个位置上协调出现。

## 脚本

```text
experiments/global_state/intervene_global_to_local.py
```

---

# 13. GLOBAL-9 — Minimal Global Contrast Sets

## 目的

用可控数据判断全局属性如何影响局部 lexical formation。

## 构造方式

设计句子对，仅改变全局语义或 discourse frame，但保留局部词形和句法环境。

例如：

```text
After the tremor, the city declared a state of ...
After the election, the city declared a period of ...
```

或：

```text
The report described widespread damage across ...
The report described widespread support across ...
```

目标 token 位于相同位置，但 global frame 改变。

## 实验

- 固定 noise；
- 比较两条 oracle path；
- 追踪 global probe 输出；
- 追踪 target token margin；
- 做 global-context swap；
- 做 local-window-only control。

## 判断

如果 global frame 在 lexical evidence 出现前已可恢复，并因果改变 target margin，则支持 global-to-local conditioning。

## 脚本

```text
experiments/global_state/build_global_minimal_pairs.py
experiments/global_state/probe_global_minimal_pairs.py
```

---

# 14. GLOBAL-10 — Global Failure Predictors

## 目的

建立可解释的全局 failure taxonomy，并为后续 method 设计提供目标。

## failure labels

1. Successful global-to-local transition；
2. Global indecision；
3. Wrong-basin selection；
4. Fragmented basin；
5. Scaffold failure；
6. Lexical crystallization failure；
7. Premature lexical locking；
8. Oracle-success / rollout-failure。

## predictor features

- early global probe confidence；
- topic entropy；
- branch consensus；
- effective rank；
- global-mode alignment；
- correlation length；
- velocity covariance；
- SC norm；
- oracle–rollout global distance；
- token frequency；
- surprisal；
- sequence length；
- domain。

## 分析

优先使用：

- multinomial logistic regression；
- survival analysis；
- sequence-grouped cross-validation；
- calibrated probability；
- feature ablation。

## 目的

回答：

> 什么样的序列和轨迹会卡在不同阶段？

并据此决定 method 应该：

- 帮助 basin selection；
- 增强 scaffold；
- 延迟 lexical locking；
- 还是训练 local recovery。

## 脚本

```text
experiments/global_state/analyze_global_failures.py
```

---

# 15. ELF / LangFlow 统一 Adapter

建议实现：

```python
class GlobalFlowAdapter:
    def encode_clean(self, token_ids, attention_mask):
        ...

    def make_oracle_state(self, clean_state, epsilon, t):
        ...

    def forward_state(
        self,
        state,
        sc_state,
        t,
        capture_hidden=False,
        capture_attention=False,
    ):
        \"\"\"
        Returns:
            logits
            predicted_clean
            velocity
            hidden_states
            attentions
            next_sc_state
        \"\"\"
        ...

    def solver_step(self, state, sc_state, t, t_next):
        ...

    def native_logsnr(self, t):
        ...

    def clone_full_state(self, state, sc_state):
        ...

    def decode_tokens(self, predicted_clean, hidden_states=None):
        ...
```

文件：

```text
experiments/global_state/adapters/elf_adapter.py
experiments/global_state/adapters/langflow_adapter.py
```

所有全局分析脚本只调用 adapter，不直接依赖模型内部实现。

---

# 16. 推荐执行顺序

## P0：确认 global-before-local

1. GLOBAL-1 Sequence-Level Probe Hierarchy；
2. GLOBAL-2 Hierarchical Branch Consensus；
3. GLOBAL-3 Low-Rank Global Mode Analysis。

这三项回答：

- 全局信息是否更早出现；
- basin 是否早于 token 确定；
- 全局信息是否集中在低秩联合模式。

## P1：验证因果作用

4. GLOBAL-4 Global Mode Intervention；
5. GLOBAL-8 Global-to-Local Causal Chain；
6. GLOBAL-9 Minimal Global Contrast Sets。

## P1：连接 free-running failure

7. GLOBAL-7 Oracle vs Free-Running Global Alignment；
8. GLOBAL-10 Global Failure Predictors。

## P2：真正的 collective phase transition

9. GLOBAL-5 Collective Coupling；
10. GLOBAL-6 Competing Global Basins。

---

# 17. 最小可行实验包

## MVP-A：Global-before-local

只做：

- topic probe；
- sentence embedding regression；
- POS / syntax probe；
- exact-token probe。

目标：

\[
\tau_{\mathrm{topic}}
<
\tau_{\mathrm{syntax}}
<
\tau_{\mathrm{token}}.
\]

## MVP-B：Global commitment

加入 branch consensus：

\[
H_{\mathrm{topic}}(t)<H_{\mathrm{lex}}(t)
\]

在较早阶段下降。

## MVP-C：Global causality

加入 global-mode remove / swap：

\[
G^A+R^B.
\]

若 topic 随 \(G^A\)、lexical realization 部分随 \(R^B\)，则形成非常强的机制证据。

---

# 18. 最终论文级假设

## Global Semantic Emergence

Global semantic information becomes recoverable from the joint continuous state before exact token identities become recoverable from individual positions.

## Basin-before-token

The generation trajectory enters a relatively stable semantic basin before its lexical realization is fixed.

## Structural Mediation

A relational scaffold forms between semantic basin selection and exact lexical crystallization.

## Causal Global-to-Local Influence

Global components causally shape subsequent local token evidence and crossing times.

## Non-guaranteed Transition

Global basin selection, scaffold formation, and lexical crystallization can fail independently.

## Architecture-dependent Coupling

ELF and LangFlow share the same global-to-local organization framework but differ in how strongly global state, categorical readout, and sampling dynamics are coupled.

---

# 19. 最核心的研究表述

之前的问题是：

> 每个位置如何从高频词跳到正确 token？

更完整的问题是：

> **连续语言模型如何把一个无结构的高维噪声场，组织成一个全局一致、结构化、最终可离散化的语言序列？**

对应的机制假设是：

\[
\boxed{
\text{prior cloud}
\rightarrow
\text{global basin}
\rightarrow
\text{structural scaffold}
\rightarrow
\text{lexical crystallization}
}
\]

这套框架中的关键不只是“token 何时 commit”，而是：

- 全局语义何时形成；
- 全局 basin 是否稳定；
- 结构如何协调多个位置；
- lexical commitment 是否只是全局组织化的最终表面结果。
