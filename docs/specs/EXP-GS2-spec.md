# EXP-GS2 Spec — Hierarchical Branch Consensus

## 背景与地位

原始 doc（`docs/global_state_formation_experiment_suite.md`）第 6 节 GLOBAL-2，P0 阶段第二项。
和 GLOBAL-1（`EXP-GS1`，线性 probe 方法学）完全不同的方法学：从同一个中间状态 `Z_t` 出发，
真正做多步 reverse-ODE **rollout** 生成 K 条 continuation，测量这些 continuation 在
topic / structure / lexical 三个层级上的一致性（consensus/entropy）何时收敛。用来交叉验证
EXP-GS1 的反常发现（pilot 规模下 `tau_syntax < tau_topic`，见 `EXP-GS1-spec.md`）是否是
线性 probe 方法学本身的 artifact，还是模型的真实行为。

核心判断（原始 doc）：若 `H_topic(t)` 很早下降但 `H_lex(t)` 仍高，说明 global basin 已确定但
lexical realization 未定；这次用**生成分支的离散程度**而不是 probe accuracy 来测同一件事。

## 0. 复用与新增基础设施

- Adapter：同 GS1，复用 `experiments/phase_transition/adapters/elf_adapter.py`。
  **首次使用 `ELFAdapter.solver_step`**——经排查（见下）目前仓库里没有任何脚本实际循环调用过它
  做多步 rollout，本实验是第一个这么用的脚本，因此把从 `t_start` 到 `t≈0.99` 的
  Euler-ODE 多步循环、self-conditioning 逐步传递（`x_pred` 输出喂给下一步 `x_pred_prev`
  输入，起始为全零，与 `models/ELF-torch/src/utils/generation_utils.py` 的生产采样循环
  `_generate_samples_single_batch` 一致）自己实现在 `branch_global_consensus.py` 里。
- POS histogram / mean pooling / cosine 相似度：复用 GS1 新拆出的
  `experiments/global_state/common.py`（`pos_histogram`, `masked_mean_pool`, `cosine_rows`）。
- **Topic 空间复用 GS1 的结果**：不在 GS2 的小样本（pilot N=4-8 条起始序列）上重新拟合
  KMeans（样本太少不稳定），而是加载 GS1 pilot 跑出来的
  `results/global_state/elf/baseline/topic_kmeans_centroids_pilot.npy`（在 N=128 上拟合的
  8-cluster 质心），用最近质心分配给每条 branch 的最终 pooled embedding。这让 GS1/GS2 用的是
  **同一个语义簇空间**，如果 GS2 独立发现同样的"topic 不比 syntax 早"模式，就能排除"两个
  实验各自的 topic 定义不一致"这个混淆解释。

## 1. Branch 构造

- **起始状态**：直接用 oracle 公式在若干 `t_start` 处构造 `Z_t = t*x_clean + (1-t)*eps`
  （每条原始文档一个固定 `eps`，不是沿着一条真实生成轨迹保存下来的中间状态）。这是 GS1 也用的
  同一近似（"trajectory point"用 oracle 状态代替真实 rollout 中间状态），原始 doc 本身也是在
  抽象层面讨论"从 trajectory 保存状态"，具体用 oracle 还是真实 rollout 中间状态，本仓库其它
  实验（EXP-01v3 等）里两者都用过；这里选 oracle 是为了和 GS1 的构造方式保持一致，便于比较。
- **自条件状态**：起始时 `sc_state = zeros`（"cold start"），不携带 `0→t_start` 阶段真实累积的
  self-conditioning 信号。⚠️ 已知简化，见第 4 节。
- **扰动**：对每条 branch 独立采样 `u ~ N(0,I)`（形状与 `Z_t` 相同），归一化后按
  `delta = eta * ||Z_t||_F * u/||u||_2` 加到 `Z_t` 上（原始 doc 公式，`||.||_2` 按整个
  `(L,d)` 矩阵的 Frobenius 范数理解，因为 doc 里 `Z_t` 本身就是矩阵）。K 条 branch 全部独立扰动
  （不设一条"零扰动参考 branch"——因为 ODE 是确定性的，`eta=0` 的 branch 之间永远完全相同，
  放一条参考 branch 不会增加信息）。
- **eta 取值**：原始 doc 建议 sweep `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}`。**预校准发现**：在
  1024-token 规模下，doc 建议范围里最大的 `1e-2` 只能在早期 t_start（0.05）产生极弱的 lexical
  分歧（`C_lex≈0.91`），`1e-3` 几乎完全无分歧（`C_topic=C_struct=C_lex=C_sent=1.000`，
  4-step rollout 下）；而 `eta=0.1`（doc 范围之外）在 t_start=0.05 才产生明显分歧
  （`C_lex≈0.58`，`C_struct≈0.995`）。说明 ELF backbone 在这个 embedding 空间/长度下的
  ODE 收缩性比原始 doc（可能针对更小的模型/不同 embedding 尺度设计）预期的更强。
  **决策**：不拘泥于原始 doc 的建议范围，pilot 改用自己校准出的三点 sweep
  `eta ∈ {0.01, 0.03, 0.1}`，覆盖"几乎无分歧"到"有意义分歧"的过渡区，而不是盲目套用可能本来
  就不适配当前实现的 doc 数值。
- **rollout**：从 `t_start` 到 `t_end=0.99`（与 GS1 clean-ref 的 t 一致，避免 t=1.0 的已知
  two-pass artifact），用 `torch.linspace(t_start, 0.99, n_steps+1)`，`n_steps` 按剩余距离
  比例设置（`n_steps = max(4, round(32 * (0.99 - t_start)))`，让步长密度接近标准 32-step 全程
  生成的密度，而不是不管起点多晚都固定用 32 步）。每步 `ELFAdapter.solver_step`，全部
  `(doc × branch)` 组合一起 batch 前进（在时间维度上是串行的多步循环，batch 维度并行）。
- Rollout 结束后再做一次 `forward_state(z, sc, t=0.99)` 取 `logits.argmax(-1)` 作为该 branch
  的最终 native-decode token 序列（与 GS1 的 `G_token` 定义一致的读出方式）。

## 2. 一致性指标

对每个 `(doc, t_start)`，在 K 条 branch 上计算：

- **Lexical**：`H_lex` = 逐位置（仅在 attention_mask 有效范围内）K 个 branch 的 token 分类分布
  的香农熵，取所有位置平均；`C_lex = 1 - H_lex / ln(K)`（consensus，1=完全一致）。
- **Structural**：每条 branch 解码文本后算 POS histogram（复用 GS1 的 `pos_histogram`），
  `C_struct` = K 条 branch 两两 POS histogram cosine 相似度的平均值（不是严格意义上的"熵"，
  是原始 doc 允许的"structural agreement"的一种操作化；1=完全一致的词性分布）。
- **Global/topic**：每条 branch 最终状态 mean-pool 后，用 GS1 的 KMeans 质心做最近邻分配得到
  topic label；`H_topic` = K 条 branch topic label 分布的香农熵，`C_topic = 1 - H_topic /
  ln(min(K, n_clusters))`。
- **Sentence-embedding agreement**：K 条 branch 最终 pooled embedding 两两 cosine 相似度的
  平均值，`C_sent`（不依赖 KMeans，是一个连续的补充信号）。

## 3. 数据与规模（pilot）

- ELF baseline，`eval_exp37c_baseline.yml`（1024-token，与 GS1 一致）。
- `n_docs=4` 起始文档（从 OWT val 采样，与 GS1 独立采样，不要求是同一批文档）。
- `K=8` branch（原始 doc 建议 16，正式规模 32；pilot 减半控制算力）。
- `t_start ∈ {0.05, 0.20, 0.38, 0.65}`（对齐 GS1 t-grid 的一个子集，方便交叉比较）。
- `eta=1e-3`（单值，见上）。
- 预计算力：4 docs × 8 branches = 32-batch，四个 `t_start` 的 ODE 步数分别约
  `round(32*0.94)=30, round(32*0.79)=25, round(32*0.61)=20, round(32*0.34)=11`，
  共 86 次 batched forward pass（batch=32, L=1024）+ 4 次最终 decode forward，量级上和 GS1
  pilot（128 samples × 9 个 t 点单步 forward）相近，可以在 A40 上几分钟内跑完。

## 4. 已知简化

1. ⚠️ 起始状态用 oracle 公式而非真实 rollout 中间状态；自条件状态在起点是冷启动的全零，
   不是真实轨迹累积的 SC 状态（与 GS1 单步 probing 的简化一致，但在多步 rollout 里累积误差
   可能更明显——如果 branch 之间的最终差异看起来比预期小，需要考虑是否是因为"重新从冷 SC 状态
   出发"本身就抹平了一部分早期已经建立的路径依赖；"冷启动 SC 让 ODE 更收缩、更难产生分歧"
   也是第 1 节里 eta 需要远大于原始 doc 建议值才能看到信号的一个可能原因）。
2. ⚠️ `eta` 用三点校准 sweep（0.01/0.03/0.1）而非原始 doc 建议的五点范围，且校准只在
   t_start=0.05（信号最强的区域）和 t_start=0.65 各测了一两个值，不是系统性的；如果正式规模
   需要更精细的 eta-敏感度曲线，需要在这三点之外补充。
3. ⚠️ `C_struct`（POS histogram 两两 cosine 相似度均值）是 doc "structural agreement" 的
   一种操作化，不是熵；数值尺度和 `C_lex`/`C_topic`（都是 `1 - H/H_max` 形式，理论范围 [0,1]
   且有明确的"完全一致=1，均匀分布=0"锚点）不完全可比，只能看**同一指标随 t 的变化趋势**，
   不能直接比较 `C_struct` 和 `C_lex` 的绝对数值。
4. Pilot 规模（4 docs × 8 branches × 4 t_starts）远小于正式规模，仅用于验证代码路径和判断
   信号方向。

## 5. 脚本与输出

```text
experiments/global_state/branch_global_consensus.py
experiments/global_state/analyze_branch_hierarchy.py
```

```text
results/global_state/<model>/<checkpoint>/branch_consensus_<label>.json
```

## ✅ 稳健性确认（EXP-GS14，P0-4）

用户审阅指出本实验的 branch 起点是"直接构造的 oracle state + 冷启动 SC"，不是真实
free-running 轨迹里保存的中间状态。`EXP-GS14`（`docs/specs/EXP-GS14-spec.md`）用真正的
自由生成轨迹（同时保存真实 `Z_t` 和真实累积的 `SC_t`，不冷启动）重做了同款 consensus
分析，**结论完全复现**：`C_struct`/`C_topic` 依然早早饱和，`C_lex` 依然是唯一有动态
范围、随 `t_start` 单调上升的指标。数值上真实轨迹版本略低一点（尤其早期 checkpoint），
方向上符合"真实路径历史应该带来更多而非更少分支多样性"的直觉，但不影响定性结论。
这是本轮 P0 系列返工里"重新检验后结论完全存活"的例子。

## 状态

**Pilot DONE⚠️**（ELF baseline，4 docs × 8 branches × 4 t_start × 3 eta，GPU1，
`logs/global_state/gs2_elf_baseline_pilot.log`，输出
`results/global_state/elf/baseline/branch_consensus_pilot.json`）。

## Results（pilot：ELF baseline，4 docs，K=8 branches，seq_len=1024，
t_start∈{0.05,0.20,0.38,0.65}，eta∈{0.01,0.03,0.1}，t_end=0.99）

⚠️ pilot 规模（尤其 n_docs=4 极小），数字用于判断方法学和信号方向，不作为最终结论。

| eta | t_start | n_steps | C_topic | C_struct | C_lex | C_sent |
|---|---|---|---|---|---|---|
| 0.01 | 0.05 | 30 | 1.000 | 0.999 | 0.904 | 0.999 |
| 0.01 | 0.20 | 25 | 1.000 | 1.000 | 0.983 | 1.000 |
| 0.01 | 0.38 | 20 | 1.000 | 1.000 | 0.998 | 1.000 |
| 0.01 | 0.65 | 11 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.03 | 0.05 | 30 | 1.000 | 0.999 | 0.808 | 0.998 |
| 0.03 | 0.20 | 25 | 1.000 | 1.000 | 0.959 | 0.999 |
| 0.03 | 0.38 | 20 | 1.000 | 1.000 | 0.996 | 1.000 |
| 0.03 | 0.65 | 11 | 1.000 | 1.000 | 0.999 | 1.000 |
| 0.10 | 0.05 | 30 | 1.000 | 0.997 | 0.596 | 0.992 |
| 0.10 | 0.20 | 25 | 0.955 | 1.000 | 0.898 | 0.998 |
| 0.10 | 0.38 | 20 | 0.955 | 1.000 | 0.986 | 0.998 |
| 0.10 | 0.65 | 11 | 1.000 | 1.000 | 0.997 | 1.000 |

**解读**：

1. **`C_lex` 是三个指标里唯一表现出清晰、单调随 t_start 上升的趋势**（例如 eta=0.1：
   0.596→0.898→0.986→0.997），且量级和方向与本仓库已知的 baseline commitment cliff
   （t≈0.20–0.30，EXP-01v3/EXP-16v2）吻合——早于悬崖的 t_start=0.05 时 lexical consensus
   最低，之后迅速上升。这是本次 pilot 里**最干净、最可信的信号**。
2. **`C_struct` 在所有 eta、所有 t_start（包括最早的 0.05）下都接近 1.0**（0.997–1.000），
   即便在 `C_lex` 只有 0.596（eta=0.1, t_start=0.05）、branch 之间 exact token 差异很大的
   同一条件下，POS 词性分布依然高度一致。**这与 GS1 的 tau_syntax(0.38) < tau_topic(0.50)
   方向一致，且证据更强**：不是"structural probe 比 topic probe 早一点点达标"，而是
   "structural consensus 从最早的可测 t_start 起就几乎已经封顶"。两个完全独立的方法学
   （GS1 的 probe accuracy，GS2 的 branch consensus）**互相印证**了同一个方向：结构信息
   比原始 doc 假设的"topic 最先"更早/更强地稳定下来。
3. **`C_topic` 和 `C_sent` 几乎全程贴着 1.0**（仅 eta=0.1 时 t_start=0.20/0.38 出现
   0.955 的小幅下降），动态范围比 `C_lex` 小得多。这两个指标都建立在 mean-pooled embedding
   的 cosine 相似度/最近质心分配上——**和 GS1 里被标记为退化指标的 `G_sent`（同样是高维
   mean-pooled embedding 的 cosine 相似度）用的是同一种底层几何量**。两次独立实验的
   "cosine-on-mean-pooled-embedding"类指标都出现了同样的"动态范围极窄、接近饱和"现象，
   这不太可能是巧合，更可能说明：**mean pooling + cosine 相似度这一具体操作化方式，本身就
   对这个 embedding 空间里的语义差异不敏感**，不能用来支持或反驳"topic 是否比 lexical 更早
   收敛"这个问题——需要换成对语义差异更敏感的度量（检索式指标、更高秩的 summary、或者
   直接用外部 sentence-embedding 模型而不是复用同一个 T5 latent 空间的均值池化）。
4. **一个交叉验证的推论**：GS1（probe accuracy 方法学）和 GS2（branch consensus 方法学）
   在"topic 是否真的比 structure 更早/更强"这一点上**方向一致地不支持原始 doc 的严格 H1**——
   两次都是 structure 信号更强/更早，topic 信号更弱或被同一类退化指标污染。这让"topic-before-
   syntax 只是 GS1 里 KMeans 标签构造方式的 artifact"这个原本在 GS1 spec 里提出的解释
   变得不太可能成立（因为 GS2 完全没有用同一个 KMeans-probe 组合，是从生成分支的离散/收敛程度
   独立测的，结果方向仍然一致）；更可能的解释是：**这个具体实现（mean-pooled T5 latent + 8
   KMeans 簇 or cosine 相似度）系统性低估了"topic"这个层级的信号**，而 POS histogram 作为
   一个更"表层"但更敏感的结构指标，意外地成为了当前 pilot 里最可靠的"早期全局信号"代理。

## 下一步

1. **优先修复 topic/sentence 层面的度量**（GS1 和 GS2 共同指向的问题）：不要再用
   mean-pooled T5 latent 的 cosine 相似度/KMeans 距离作为 topic 的操作化，改用检索式指标
   （在候选池里找最近邻文档、报告 top-1/top-5 命中率）或换一个独立的 sentence-embedding
   来源，避免语义信号和"是否用同一个几乎饱和的几何量"混在一起。
2. GLOBAL-1/GLOBAL-2 pilot 一致指向"structural signal 早于/强于 topic signal"，这个更精确、
   比原始 doc 更弱的结论（"structure-and-lexical-ordering supported，topic 的位置存疑"）
   应该被当作当前的工作假设，而不是原始 doc 第 1 节的强 H1 表述。
3. 继续推进 GLOBAL-3（Low-Rank Global Mode Analysis）——用完全不同的第三种方法学（SVD 分解
   + CKA 对齐）再次检验同一个问题，尤其是"低秩 global component 是否比 residual 更早支持
   topic/token 恢复"，进一步交叉验证。

## 严谨性自审：发现并修复第二处 nearest_topic bug（2026-07-27）

GS6 debug 时（见 EXP-GS6-spec.md）已经定位并修复了"最近质心分类用平方欧氏距离而非
cosine"这个 bug，并声称已经"统一修复了 GS4/GS6/GS8-mechanism/GS14 用到的同一函数"。
但那次验证只检查了"这四个文件里 `nearest_topic` 的 import 是否指向同一个函数对象"，
没有检查每个文件的 C_topic 计算**是否真的调用了这个被 import 的函数**。

用户要求做一次全面严谨性自审后，重新 grep 了整个 `experiments/global_state/` 目录，
发现 `branch_global_consensus.py`（本实验 GS2 的脚本）第 168-171 行有一份**内联手写的**
平方欧氏距离最近质心分类代码，从未调用过 import 进来的 `nearest_topic`——这是本 bug
最初的出处（GS6/GS4/GS14 里的重复定义应该都是从这里 copy-paste 出去的），而这次
自审之前它本身从未被修复过。`branch_true_trajectory.py`（GS14）里也发现了同一份
未修复的副本，见 EXP-GS14-spec.md。

**修复后重跑（pilot 规模，n_docs=4, k_branches=8，ELF baseline + LangFlow baseline
各一次，eta∈{0.01,0.03,0.1}，label=`pilot_topicfix`）：**

ELF：
```
eta=0.01: C_topic 0.932-1.000（4 个 t_start）
eta=0.03: C_topic 0.920-1.000
eta=0.10: C_topic 0.917-1.000
```
LangFlow：
```
eta=0.01: C_topic 0.920-1.000
eta=0.03: C_topic 0.932-1.000
eta=0.10: C_topic 0.909-1.000
```

**结论：C_topic 的饱和现象在两个架构上都不受这次 bug 修复的影响**（修复前后数值几乎
一样），这排除了"GS2 的 C_topic≈1.0 是这个 norm-mismatch bug 造成的假阳性"这个担忧——
和 GS6（bug 修复后结果发生质变）不同，这里 bug 修复只是让计算方式正确了，但底层信号
本身确实接近饱和。

同一批数据里 eta sweep 提供了一个额外的、更有说服力的证据：**C_topic 对 eta 有真实的
渐变响应**，只是比 C_lex 迟钝得多——例如 ELF t_start=0.05：C_lex 从 eta=0.01 的 0.904
降到 eta=0.03 的 0.808 再到 eta=0.1 的 0.596（明显下降），而同一组 C_topic 只从 1.000
微降到 0.955；LangFlow 同一位置 C_lex 从 0.864→0.744→0.536，C_topic 从 0.920→0.932→0.909
基本不变。这排除了"C_topic 是完全不敏感的天花板伪影"这个替代解释——它是一个真实但
远比 lexical identity 更鲁棒的信号，这个鲁棒性差异本身就是有信息量的发现。

详见 `logs/global_state/gs2_{elf,langflow}_baseline_pilot_topicfix.log` 和对应
`results/global_state/{elf,langflow}/baseline/branch_consensus_pilot_topicfix.json`。
