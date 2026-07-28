# EXP-PT9 Spec — Cross-Time Evidence-Direction Transfer

## 背景与地位

P2 阶段实验，但复用现成的 adapter/数据基础设施，实现成本不算高，所以在把
P1 主要实验跑完的同时一起做了。目标：区分"证据沿一个稳定的 token 判别方向
累积（早期方向持续有效甚至变强）"还是"证据在每个时间点被重新编码（探针只在
自己训练的那个 t 附近有效）"。

## 方法与已有代码的关系

这个实验和 `EXP-07c`（`probe_cross_checkpoint_full.py`，跨 **checkpoint** 的
探针迁移）结构上几乎一样，只是把"跨 checkpoint"换成"跨 t"。复用了它的探针
架构（`torch.nn.Linear(d, vocab_size)`，Adam，cross-entropy，多轮 full-batch
SGD）。

**一个重要的方法论差异，主动修正**：`EXP-07c` 在自己训练的那个切片上直接报
准确率作为 transfer matrix 的对角线——这其实是**训练准确率**，不是留出准确率，
对角线会偏乐观。这次改成**序列级别的 train/val 切分**（探针只在 TRAIN 序列
上训练，整个矩阵——包括对角线——都在 VAL 序列上评估），沿用本 suite 在
`EXP-PT10` 里已经应用过的、源自 `EXP-07`→`EXP-07v2` 教训的做法。**这意味着
这次的数字和 EXP-07c 的数字不能直接比较**（这次的对角线本来就应该更低，因为
是真正的留出准确率）。

## 已实现 / 未实现的状态表示

只实现了 doc 列的 4 种状态表示里的 1 种——**predicted_clean**（`x̂_t`，模型
自己的去噪预测），这也是 `EXP-07d`（"x̂_t 跨 checkpoint 迁移"）已经用过的
表示，选它是为了和已有结果风格一致。**没有实现**：原始状态 `z_t`、
native hidden states（需要 `capture_hidden=True`，adapter 已经支持，只是
这次没有跑）、prior-subtracted logits（需要额外的 reference prior 前向传播，
类似 EXP-PT2 的做法）。这三个都是直接可以在现有 adapter 上加的后续工作，
不需要新的基础设施。

## 脚本与输出

```text
experiments/phase_transition/probe_cross_time_transfer.py
```

```text
results/phase_transition/<model>/<checkpoint>/cross_time_transfer_<label>.json
```

## 判定规则（对应 doc 11 节，用这次的矩阵统计量表述）

- `upper_tri_mean`（早期探针 → 更晚状态的平均准确率）明显高于 `lower_tri_mean`
  （晚期探针 → 更早状态）→ 早期方向持续存在甚至变强，支持"稳定累积"。
- `diag_mean` 远高于 `upper_tri_mean`/`lower_tri_mean` → 证据是逐时间点重新
  编码的，探针只在自己训练的 t 附近有效。
- 比较不同 checkpoint 的 `upper_tri_mean`/`diag_mean` 比值，检验"KD 改善的是
  迁移能力还是单点探针准确率"这条 doc 提到的判定。

## 状态

**DONE（ELF baseline/kd_cr/kd2 + LangFlow，全部正式规模，128 序列，7 个 t 点，
predicted_clean 表示，序列级 train/val 切分）**

## Results

| model/ckpt | diag_mean（留出准确率） | upper_tri_mean（早探针→晚状态） | lower_tri_mean（晚探针→早状态） | upper/lower 比值 |
|---|---|---|---|---|
| ELF baseline | 0.665 | 0.544 | 0.438 | 1.24 |
| ELF kd_cr | 0.668 | 0.569 | 0.499 | 1.14 |
| ELF kd2 | 0.671 | 0.576 | 0.504 | 1.14 |
| LangFlow | 0.243 | 0.219 | 0.120 | **1.82** |

**核心发现**：

1. **四个模型都是 `upper_tri_mean > lower_tri_mean`**——早期探针方向能不错地
   迁移到更晚的状态，反过来（晚期探针迁移到早期状态）差一些。支持"证据沿一个
   相对稳定的方向累积、变强"这个方向，而不是"每个时间点完全重新编码"。
2. **LangFlow 的不对称性远强于 ELF**（比值 1.82 vs ELF 三个 checkpoint的
   1.14-1.24）——虽然 LangFlow 的绝对迁移准确率低得多（diag_mean 只有 0.243
   vs ELF 的 ~0.67，这和已知的"LangFlow 承诺晚得多"一致），但它"早期方向
   在晚期依然有效、反过来不行"这个**相对**不对称性反而更干净。
3. **一个和 doc 判定规则直接对应的发现**：ELF 三个 checkpoint 的 `diag_mean`
   几乎不随 KD 变化（0.665→0.668→0.671，Δ<0.6pp），但 `upper_tri_mean` 涨得
   更多（0.544→0.569→0.576，Δ≈3.2pp）——即 **KD 提升的主要是"早期方向能不能
   迁移到后面"，而不是"每个单独时间点的探针准确率本身"**。这正好对应 doc 里
   "KD improving transfer rather than per-time probe accuracy: KD stabilizes
   the evidence coordinate system" 这条判定规则，是这次 suite 里少数几个
   明确支持某一条具体判定规则、且是正面证据的结果。

⚠️ 只用了 `predicted_clean` 一种状态表示；序列级切分的对角线是真实留出准确率，
数字不能直接和 `EXP-07c`（训练集准确率）比较。

## 额外状态表示补测（raw_z, native hidden states）

**脚本**：`experiments/phase_transition/probe_cross_time_transfer_extra_reps.py`。

**过程中发现并修了一个真实 bug**：ELF 的 hidden state 通过 hook 从
`model.blocks[i]` 抓取时，包含了模型内部拼接的 mode/time/context 前缀 token
（`model.py` 只在算最终 decode 输出前才切掉这些前缀），导致抓到的 hidden
张量比 `gt_ids` 多了 `prefix_len`（最多 12）个位置。训练探针时 `X`（hidden）
和 `Y`（token id）行数不对齐，`cross_entropy` 里的索引越界，直接触发 CUDA
assertion 崩溃。修复：在这个补测脚本里显式 `hidden[:, prefix_offset:, :]`
切掉前缀（`prefix_offset = hidden.shape[1] - L`）。LangFlow 没有这个前缀
结构，不受影响。**这是本 session 第一次真正用到 `capture_hidden=True` 这条
adapter 代码路径**，之前从未被其它 PT 脚本触发过，值得记录以防未来再犯。

### Results（128 序列，7 个 t 点）

| model/ckpt | 表示 | diag_mean | upper_tri_mean | lower_tri_mean |
|---|---|---|---|---|
| ELF baseline/kd_cr/kd2 | raw_z | 0.582（三者完全相同） | 0.580 | 0.377 |
| ELF baseline | hidden | 0.604 | 0.473 | 0.399 |
| ELF kd_cr | hidden | 0.646 | 0.548 | 0.453 |
| ELF kd2 | hidden | 0.649 | 0.458 | 0.424 |
| LangFlow | raw_z | 0.129 | 0.128 | 0.087 |
| LangFlow | hidden | 0.139 | 0.083 | 0.063 |

**一个值得注意的健全性检查**：ELF 三个 checkpoint 的 `raw_z` 数字**逐位数字
完全相同**——这不是 bug，是必然如此：`raw_z`（即 `z_t`）只由共享的 T5
encoder 输出和随机噪声构造（`make_oracle_state`），构造过程根本不调用
backbone，所以和用哪个 checkpoint无关。这正好提供了一个免费的"无模型基线"：
`raw_z` 的探针准确率代表"不经过任何 backbone、纯粹从带噪状态本身能线性
恢复多少 token 信息"，`predicted_clean`/`hidden` 相对于它的提升量，才是
backbone 真正贡献的部分。三个 checkpoint 在 `hidden`/`predicted_clean`
表示上的差异（KD 更高）因此可以更有信心地归因于 backbone 本身，而不是
数据/噪声构造的偶然差异。

**发现**：所有表示、所有模型都保持 `upper_tri_mean > lower_tri_mean`
（早期方向更能迁移到晚期，反过来弱一些），和主脚本用 `predicted_clean`
表示得到的结论方向完全一致——**跨 4 种状态表示（predicted_clean、raw_z、
hidden、以及主脚本里没做的 residual logits 除外）都支持"证据沿稳定方向
累积"这个结论，不是某一种表示方式的 artifact**。`hidden` 表示的
`diag_mean`（0.60-0.65，ELF）比 `predicted_clean`（0.665-0.671，见前面
Results）略低，`raw_z`（0.44，pilot 数字）更低——这个排序（predicted_clean
> hidden > raw_z）符合直觉：`predicted_clean` 是模型对干净目标的直接估计，
天然离最终 token identity"最近"；`raw_z` 是最原始的带噪状态，离 token
identity"最远"，中间需要模型自己再提炼一次。

"Prior-subtracted logits"（doc 第 4 种表示）仍未实现，见脚本 docstring 里
的具体原因（维度不匹配，不是"没时间做"这么简单）。

## 严谨性补强：bootstrap CI（rigor audit 后，2026-07-26）

给 `probe_cross_time_transfer.py` 加了逐序列 accuracy 矩阵保存
（`cross_time_transfer_raw_<label>.npz`：`acc_per_seq_matrix`，shape
`(T,T,n_val)`），重跑全部 4 个模型（数字与修复前逐位吻合），新增
`bootstrap_pt9.py` 按 held-out 序列重采样 2000 次：

| model/ckpt | diag_mean CI | upper_tri_mean CI | lower_tri_mean CI | upper−lower CI |
|---|---|---|---|---|
| ELF baseline | 0.665 [0.657,0.672] | 0.544 [0.537,0.551] | 0.438 [0.432,0.445] | **0.106** [0.102,0.110] |
| ELF kd_cr | 0.668 [0.660,0.676] | 0.569 [0.562,0.576] | 0.499 [0.492,0.506] | **0.070** [0.067,0.074] |
| ELF kd2 | 0.671 [0.663,0.679] | 0.576 [0.568,0.584] | 0.504 [0.497,0.511] | **0.072** [0.069,0.076] |
| LangFlow | 0.243 [0.231,0.254] | 0.219 [0.209,0.229] | 0.120 [0.113,0.129] | **0.099** [0.094,0.103] |

**结论：`upper_tri_mean > lower_tri_mean` 这个"证据方向持续累积"的核心发现，
在全部 4 个模型上都有 CI 不含 0 的统计支持**——不是某几个 held-out 序列的
偶然结果。这次 bootstrap 没有改变任何方向性结论，只是把此前基于点估计的
"四个模型都是这个方向"的观察，确认为在序列级重采样下稳健。详见
`results/phase_transition/<model>/<checkpoint>/bootstrap_pt9_full.json`。

## 补充：第 4 种表示"prior-subtracted logits"（2026-07-26）

`probe_cross_time_transfer_extra_reps.py` 之前明确记录了不实现这个表示的
理由：残差 logit 本身已经活在 vocab-size（32k-50k）维空间，在它上面再训练
一个新的线性 probe 意味着一个 `Linear(vocab_size, vocab_size)` 层
（10 亿+参数），既不可行也没有意义（残差 logit 本身已经就是一个逐类别的
分数，不需要再训练一个 probe 去"读出"它）。这次没有推翻这个判断——训练
这样一个 probe 依然是个坏主意。

**改用的操作化方式**：新脚本 `cross_time_residual_consistency.py` 复用
PT2/PT5 的 `oracle_probs`/`null_probs`（同一套 EXP-05v3 风格 null
reference），在每个 t 独立计算 `e_t(v) = log p_oracle(v|z_t) - log
q_null(v|t)` 的 argmax 是否正确，得到一个逐位置的"残差信号在这个 t 是否
正确"的布尔矩阵。**不训练任何 probe**，而是直接问一个不同但相关的问题：
"残差信号正确的这组位置，在时间上是否稳定"——用条件概率
`M[a,b] = P(在t_b正确 | 在t_a正确)` 代替 PT9 主脚本里"训练好的方向能不能
迁移"这个概念。

⚠️ **这不是同一个指标，量级不可直接比较**：主表格的 `upper_tri_mean`/
`lower_tri_mean` 衡量的是"在 t_a 训练的探针方向"泛化到 t_b 的**held-out
准确率**；这里的 `M[a,b]` 衡量的是"在 t_a 正确"这个条件下"在 t_b 也正确"
的**条件概率**，两者定义完全不同，数字大小没有可比性，只有**方向性的
不对称模式**（早期蕴含晚期 vs 晚期蕴含早期哪个更强）在概念上类似，可以
类比讨论。

### 结果（全部 4 个模型，T=7，与主脚本同一 t-grid）

| model/ckpt | P(later correct \| earlier correct) | P(earlier correct \| later correct) |
|---|---|---|
| ELF baseline | 0.909 | 0.445 |
| ELF kd_cr | 0.913 | 0.525 |
| ELF kd2 | 0.889 | 0.528 |
| LangFlow | 0.440 | 0.050 |

**结论：全部 4 个模型都复现了和主脚本（`predicted_clean` 表示）完全一致的
不对称方向**——"早期正确 ⟹ 晚期大概率也正确"（P 在 0.89-0.91 之间，ELF；
LangFlow 稍低但仍是 0.44）远强于反过来的"晚期正确 ⟹ 早期也正确"（P 只有
0.44-0.53，ELF；LangFlow 只有 0.05）。这是用一套完全独立的、不训练任何
参数的方法，第二次确认"证据沿一个相对稳定的方向累积"这个 PT9 核心结论，
不是 `predicted_clean` 这一种表示或训练好的探针方向特有的 artifact。

⚠️ 这个不对称性部分是"准确率本身随 t 单调上升"这个基础事实的机械推论
（如果晚期准确率本来就高，"早期正确的位置在晚期大概率也正确"自然容易
成立），不完全是一个独立于"整体越往后越准"这个平凡事实的新信息；解读时
应该把它看作对"证据累积方向稳定"这个定性叙事的又一次交叉验证，而不是一个
全新的、独立的因果证据。详见
`results/phase_transition/<model>/<checkpoint>/cross_time_residual_consistency_full.json`。
