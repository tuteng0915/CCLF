# EXP-GS1 Spec — Sequence-Level Probe Hierarchy（Global-State-Formation 系列首个实验）

> 📌 本实验的发现已被整合进 `docs/global_state_formation_synthesis.md`（综合解读文档，
> 第 1 节）。该文档是当前对整个 GS 系列最新、最可信的解读，建议先读那份文档再看本
> spec 的细节。

## 背景与地位

`docs/global_state_formation_experiment_suite.md`（下称"原始 doc"）提出 GLOBAL-1–GLOBAL-10，
核心假设 H1（global-before-local）：模型先在完整序列状态 `Z_t` 中形成全局语义，再形成结构骨架，
最后确定 exact token。GLOBAL-1（原始 doc 第 5 节）是 P0 阶段第一项、也是 MVP-A（原始 doc 第 17
节）的全部内容：训练容量递增的 probe，从 `Z_t` 的 summary 里恢复 topic / sentence embedding /
POS-syntax / exact token，检验

```
tau_topic < tau_syntax < tau_token
```

本 spec 记录把抽象协议落地为代码时的工程决策，命名沿用 `phase_transition` 系列的 `EXP-PTx`
惯例，改用前缀 `EXP-GSx`（Global-State）对应原始 doc 的 `GLOBAL-x`。

## 0. Adapter 复用（不重新实现）

原始 doc 第 15 节要求 `experiments/global_state/adapters/{elf_adapter,langflow_adapter}.py`，
接口（`encode_clean` / `make_oracle_state` / `forward_state` / `solver_step` / `native_logsnr` /
`clone_full_state`）与 `experiments/phase_transition/adapters/` 里已经跑通、被 PT1/PT2/PT5
验证过的 `FlowModelAdapter` 几乎逐字相同（原始 doc 显然是照抄了 PT 系列 doc 的同一节）。

**决策**：不复制/不重写。`experiments/global_state/*.py` 直接把
`experiments/phase_transition/` 加入 `sys.path` 并 `from adapters.elf_adapter import ELFAdapter`
（与 PT 系列脚本完全相同的 import 方式）。唯一的差异方法名 `clone_full_state` vs 现有的
`full_state_clone`——两者语义相同，直接用现有名字，不做无意义的重命名包装。

## 1. GLOBAL-1 targets 的落地选择

原始 doc 列了 6 个 global-semantic targets、6 个 structural targets、3 个 local-lexical
targets，并建议 4 种 summary representation（mean / attention-pool / low-rank SVD / pairwise
relational）和 3 级 probe capacity（linear / MLP / transformer probe）。全部实现在 P0 阶段代价
过高；本 spec 选取一个**能验证核心假设、且每个 target 都可以只用仓库里已有依赖（torch, sklearn,
nltk）实现**的最小子集，其余留待 GLOBAL-1 v2（如果 pilot 显示信号存在，值得投入更贵的 target）。

| 层级 | 原始 doc 选项 | 本次落地 | 理由 |
|---|---|---|---|
| Global semantic | sentence embedding / topic / domain / BoW / entity / cluster | **(a)** topic cluster：对 clean pooled embedding 做 `KMeans(k=8)`（无监督标签，一次性算出，不依赖外部语料标注）；**(b)** sentence embedding regression：直接以 clean pooled T5 embedding 向量为回归目标 | 不需要外部 sentence-embedding 模型（环境里没有 `sentence-transformers`）；KMeans 标签是自包含的、可复现的"文档簇"代理，NER/BoW/domain 需要额外标注或语料统计，留到 v2 |
| Structural | POS histogram / dependency / phrase boundary / relation matrix | POS histogram（8 个粗粒度词性桶的归一化计数），用 `nltk averaged_perceptron_tagger_eng`（已确认可下载，见下）对 detokenize 后的文本做词性标注 | dependency parsing / phrase boundary 需要 spaCy（环境未安装，且引入新依赖超出 P0 范围）；POS histogram 是 doc 结构目标列表里最便宜、纯 statistics 的一项，且是文档级（不需要 per-position 对齐 T5 subword ↔ 单词，回避了 tokenizer 边界不对齐的已知坑） |
| Local lexical | per-position probe / true-token rank / native top-1 / stable-final agreement | **native top-1 accuracy**（直接用 `adapter.forward_state` 的 logits），不训练额外的 32128-way 线性分类器 | native top-1 accuracy 正是本仓库从 EXP-01 起反复使用、已验证稳定的 `G(t)` 定义；doc 本身把它列为局部可恢复性的 4 个可接受操作化之一。训练一个从头开始的 32k-way per-position 线性探针在小 N 下既昂贵又不会比模型自己的 decode 更可信，属于不必要的重复工作 |

Summary representation：只用 **Method A（mean pooling）**，doc 自己也说"主结论优先依赖
linear / low-capacity probe"，mean pooling + ridge/logistic 就是这个 ladder 里能力最低的一级，
最适合做 P0 pilot 的第一刀。SVD / attention-pool / pairwise-relational summary 留给 GLOBAL-3
（本来就是 GLOBAL-3 的主题，这里不重复）。

Probe capacity：只用 **Ridge / LogisticRegression**（sklearn，闭式或凸优化，无需额外训练循环）。

## 2. 阈值与 tau 定义

按原始 doc 第 3 节："阈值建议定义为达到 clean-state performance 的固定比例，例如 80%，不要跨
任务使用同一个绝对阈值"。本实现：

- `G_topic(clean)` / `G_syntax(clean)`：用 clean pooled embedding 本身（不经过 backbone，
  t=1 极限）作为 probe 输入直接训练同一 probe，得到线性可分性的上限。
- `G_sent(clean) = 1.0`（回归目标就是 clean pooled embedding 本身，定义上的上确界）。
- `G_token(clean)`：在 `t=0.99`（不用精确 1.0，见 EXP-PT1 对 decode branch 在 t=1.0 处已知
  two-pass artifact 的记录；这里不涉及 decode branch，但沿用同一"避免恰好 1.0"的保守习惯）
  处跑一次 native top-1 accuracy 作为上限。

`tau_k = min{t in t_grid : G_k(t) >= 0.8 * G_k(clean)}`；若整个 grid 都没达到，记为 `None`
（"never reaches 80% of clean performance within pilot grid"），不做外推。

## 3. 数据与规模

- OWT val split，T5 tokenizer。**原计划**复用 `eval_exp36_baseline.yml`（`max_length=128`），
  与原始 doc 第 4 节"主实验：512 sequences, seq_len=128"的长度设定一致；**实测发现该配置对
  oracle probing 无效**：即使直接喂 `x_clean`（不加噪声）在 t≈1 时 native top-1 accuracy 也只有
  ~24%（诊断脚本见下），而已验证的 `eval_exp37c_baseline.yml`（`max_length=1024`，EXP-PT1/PT2
  同款）在同一 checkpoint、t=0.95 下给出 86.5%，与 EXP-PT1 pilot 表的 81.87%@t=0.95 量级吻合。
  说明 `eval_exp36_baseline.yml` 这类 128-length config 是为 DF/dec_sc **生成**实验准备的
  （EXP-33/34/35/36 只用它做采样，从不做 oracle probing），不能直接套用在需要精确 oracle-state
  forward pass 的探针实验上（具体原因未深挖——不影响本 spec 的可用结论：**改用已验证的
  `eval_exp37c_baseline.yml`，`max_length=1024`**）。这意味着实际 pilot 的 seq_len 是 1024
  而非 doc 建议的 128，与本仓库其余所有 oracle-probe 类实验（EXP-01v3/05v3/16v2/PT1/PT2）保持
  一致，是刻意的一致性选择。
- **噪声协议**：原始 doc 第 4 节明确"同一 sequence 在所有 t 使用同一个 epsilon"——对每条序列只
  采样一次 `eps ~ N(0,I)`，所有 t 复用同一份，构成一条 oracle path（不是 PT2 那样每个 t 独立
  重采样多个 noise seed 做平均）。这与 GLOBAL-1 的定位一致（doc 讨论的是"同一条轨迹上全局信息
  何时出现"，不是跨噪声种子平均后的期望）。
- Pilot 规模：`n_samples=128`（1024-token 序列，比 128-token 更贵，故比最初计划的 240 略降），
  `n_t_steps=8`，`t_grid` 在已知承诺悬崖（baseline t≈0.20–0.30，见 EXP-01v3/EXP-16v2）附近加密：
  `[0.05, 0.12, 0.20, 0.28, 0.38, 0.50, 0.65, 0.85]`。只跑 ELF baseline。
  比 EXP-PT1 的"纯验证代码路径"pilot（n=16）更大，因为 KMeans/POS 标签本身需要足够样本才稳定，
  但仍明显小于原始 doc 建议的 512 样本 / 101 密集 t 网格的正式规模。

## 4. 已知简化（需在论文/后续 spec 中保留）

1. ⚠️ topic label 由 KMeans(k=8) 对 **同一批** clean pooled embedding 无监督聚类得到，
   不是外部人工标注的主题标签；`G_topic` 衡量的是"probe 能否恢复模型自己隐含的语义簇结构"，
   不是"probe 能否恢复人类定义的新闻/对话/科普类别"（那需要标注语料，留给正式规模 + 外部
   topic 数据集，如 doc 第 10 节 GLOBAL-6 用到的 disaster/election/sports/dialogue/science
   五类对照集）。
2. ⚠️ POS histogram 是文档级词性分布，不是原始 doc 结构目标里的 dependency tree / phrase
   boundary；也不做 per-position 的 T5-subword ↔ 词对齐（GLOBAL-1 目标本来就是 document-level
   summary，不需要 per-position 对齐）。
3. ⚠️ `G_token` 用 native top-1（模型自己的 decode 输出），不是从 `Z_t` 训练的独立线性探针；
   这让 G_token 的比较对象和 G_topic/G_syntax（都是从同一个 `g_t^mean` 训练的 probe）不是完全
   同源的"探针能力"比较——如果 `tau_token` 明显晚于 `tau_topic/tau_syntax`，需要在解读时注意
   native top-1 本身包含了模型的全部非线性能力（远强于线性 probe），这其实是**对 H1 更保守
   （不利）的比较**：如果连能力更强的 native decode 都比线性 probe 更晚"达标"，H1 的证据会更强；
   反过来如果 native decode 更早达标，则不能说明 token identity 真的比 topic 更难恢复，
   只能说线性 probe 能力不够——这一点必须在结果解读里显式说明。
4. Pilot 规模（240 样本，8 个 t 点）不是原始 doc 建议的正式规模，数字可用于判断信号方向，
   但正式结论需要在 pilot 验证代码路径无误后，扩大到 512 样本 + 更密 t 网格重跑。
5. 只跑 ELF baseline；kd_cr / kd2 / LangFlow 对比留待本 spec 后续更新。

## 5. 脚本与输出

```text
experiments/global_state/probe_sequence_hierarchy.py
experiments/global_state/analyze_probe_transition.py
```

```text
results/global_state/<model>/<checkpoint>/probe_hierarchy_<label>.json
```

## ⚠️ 重要更正（EXP-GS11，pooling confound）

`EXP-GS11`（`docs/specs/EXP-GS11-spec.md`）用检索式指标直接证实：本实验的 probe 输入
`g_t^mean = masked_mean_pool(z_t, mask)` 是**未经模型处理的原始 oracle state**；只对
`z_t` 做位置平均（完全不经过模型）就已经能在 `t=0.28` 达到 100% self-retrieval
accuracy（`L_eff=32` 时即如此），且在极早的 `t=0.05` 随 `L_eff` 单调上升，符合纯粹的
`1/sqrt(L)` 噪声平均直觉。**因此本实验下方报告的"早期 global 信号"（`G_topic`、
`G_sent`）应被理解为"对原始噪声输入求位置均值这一操作本身的统计性质"，而不是"模型早期
已经形成了全局语义组织"**——GS1 的数字本身没有错，但下方 Results/解读部分（尤其
第 2、3 条）里"模型在早期就能恢复全局信息"这一措辞需要按此更正理解。详见 GS11 spec 的
Results 和"结论与后续行动"。

## 状态

**Pilot DONE⚠️**（ELF baseline，n=128，1024-token，8 个 t 点 + t=0.99 clean-ref，GPU1，
`logs/global_state/gs1_elf_baseline_pilot.log`，输出
`results/global_state/elf/baseline/probe_hierarchy_pilot.json`）。

## Results（pilot：ELF baseline，128 序列，seq_len=1024，
t∈{0.05,0.12,0.20,0.28,0.38,0.50,0.65,0.85}+0.99(clean-ref)，train/test=90/38 document-level）

⚠️ pilot 规模，数字可用于判断信号方向，不作为最终结论（见第 4 节已知简化）。

| t | G_topic (acc, k=8) | G_syntax (ridge R²) | G_syntax (cos) | G_sent (cos) | G_token (native top-1) |
|---|---|---|---|---|---|
| 0.05 | 0.132 | −0.207 | 0.986 | 0.953 | 0.002 |
| 0.12 | 0.158 | 0.110 | 0.989 | 0.963 | 0.015 |
| 0.20 | 0.395 | 0.381 | 0.993 | 0.972 | 0.076 |
| 0.28 | 0.500 | 0.524 | 0.994 | 0.977 | 0.352 |
| 0.38 | 0.526 | 0.609 | 0.995 | 0.981 | 0.672 |
| 0.50 | 0.632 | 0.654 | 0.996 | 0.984 | 0.761 |
| 0.65 | 0.737 | 0.677 | 0.996 | 0.986 | 0.808 |
| 0.85 | 0.737 | 0.685 | 0.996 | 0.987 | 0.836 |
| 0.99 (clean-ref) | 0.763 | 0.684 | — | 0.987 | 1.000 |

`tau_k = min{t : G_k(t) >= 80% * G_k(clean)}`（`analyze_probe_transition.py` 输出）：

- `tau_syntax = 0.380`（threshold 0.547，clean R²=0.684）
- `tau_topic = 0.500`（threshold 0.611，clean acc=0.763）
- `tau_token = 0.650`（threshold 0.800，clean acc=1.000）

即 `tau_syntax(0.38) < tau_topic(0.50) < tau_token(0.65)`。

**解读**：

1. **H1 严格排序（tau_topic < tau_syntax < tau_token）在 pilot 规模下不成立**——在同一线性
   probe 能力下（G_topic 和 G_syntax 都是从同一个 mean-pooled `g_t` 训练的 probe，capacity
   完全匹配），POS histogram 反而比 topic cluster **更早**达到 80% 阈值（0.38 vs 0.50）。这与
   原始 doc 的核心假设方向相反，需要如实报告，不能因为"想要 H1 成立"而弱化这个结果。
   一个可能的解释：本 pilot 用 KMeans 对 clean pooled embedding 聚类得到的"topic"标签，
   可能本身就混合了文体/句法信息（pooled T5 embedding 对整体句法风格很敏感），使得"topic"
   probe 实际上并不是一个纯语义信号，和"syntax" probe 在特征来源上有重叠，不是真正独立的
   两个层级——这是**本 pilot topic 标签构造方式的已知弱点**（见第 4 节简化 1），并非必然反驳
   H1，需要用外部标注的 topic（如原始 doc GLOBAL-6 的 disaster/election/sports/dialogue/science
   五类对照集）复核。
2. `tau_token(0.65)` 确实晚于 `tau_topic` 和 `tau_syntax`——"local lexical 最后确定"这部分
   方向上和 H1 一致，但 G_token 用的是 native top-1（能力远强于线性 probe，见第 4 节简化 3），
   所以这个比较**对 H1 不公平地有利**（能力更强的读出方式理应更晚打平 80% 阈值这件事本身信息量
   有限）。不能仅凭这一点得出"local lexical 确实最晚形成"的强结论。
3. **G_sent（cosine 到 clean pooled embedding 的 ridge 回归）几乎从 t=0.05 起就已经很高
   （0.953）且动态范围极窄（0.953→0.987，仅 3.4pp）**，明显是一个退化指标：高维向量间的
   cosine similarity 容易被"沿 t 缩放"这类平凡几何关系（`z_t = t*x_clean + (1-t)*eps` 中
   `t*x_clean` 项本身就和 `x_clean` 高度共线）撑高，而不是反映真实的语义可恢复性。
   **G_sent 在当前实现下不能作为可信的 global-semantic 信号**，不应该拿来支持或反驳 H1；
   需要换成更敏感的度量（比如先做 per-t 内部的 rank/retrieval 指标而非绝对 cosine，或者
   对 `g_t` 做 t-conditional 的去均值/去缩放归一化后再算 cosine）才能用。这是本次 pilot
   最重要的方法论发现之一，必须在正式规模重跑前修复，否则正式规模的 G_sent 数字同样不可信。
4. `G_topic` 和 `G_syntax` 在 t<0.2 时都已经明显高于随机水平（chance ≈ 1/8=12.5% for topic；
   `G_syntax` 的 R² 在 t=0.05 时为负说明还没有信号，t=0.12 转正），说明**某种全局/结构信息确实
   比 exact-token identity 更早出现**——这与 doc 的粗粒度主张（存在 global-before-token 的时序
   差）方向一致，只是 topic vs syntax 谁更早这一更细的排序，在本 pilot 下是反的。

## 下一步（需要用户确认规模/优先级后再执行）

1. **优先修复 G_sent 度量**（第 4 节简化补充第 6 点）：当前实现不可信，正式规模重跑前必须换成
   有更大动态范围的版本（如 t-conditional 归一化后的 cosine，或改成检索式指标：在 test set 内
   用预测向量找最近邻 clean embedding，报告 top-1/top-5 retrieval accuracy）。
2. 用外部 topic 标注（而非 KMeans 自诱导标签）复核 tau_topic vs tau_syntax 的排序，排除
   "topic 标签本身混入句法信息"这个混淆解释。
3. 若以上两点修复后排序方向不变，需要如实在论文里报告"topic-before-syntax 假设不成立，
   但 global/structural-before-token 粗粒度排序成立"这个更弱、更精确的结论，而不是原始 doc
   第 1 节的强 H1 陈述。
4. 决定是否推进 GLOBAL-2（Hierarchical Branch Consensus，用生成分支而非线性 probe 独立验证
   同一问题）——用一个完全不同的方法学（branch entropy 而非 probe accuracy）交叉验证，
   对判断"topic-before-syntax 是否是 probe 构造方式的 artifact"特别有价值。

## 6. 已知简化（补充）

6. ⚠️ `G_sent`（cosine 到 clean pooled embedding 的 ridge 回归预测）在 pilot 中动态范围极窄
   （0.953–0.987），怀疑是高维向量共线性造成的退化指标，不建议在当前实现下引用其数字或用于
   支持/反驳任何假设，见上方 Results 解读第 3 点。
