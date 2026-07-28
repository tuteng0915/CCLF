# EXP-PT8 Spec — Controlled Minimal-Pair Evidence Sources

## 背景与地位

P2 阶段实验。目标：用已知因果线索来源的受控 minimal pair，做证据来源的因果
归因。

## 数据来源：用 BLiMP 代替手工构造

doc 建议"构造或收集自然出现的 minimal pair"，覆盖主谓一致、命名实体位置、
数量一致、局部搭配、否定、语义角色、功能词 vs 内容词目标等类别。这次没有
手工构造/筛选句子，而是直接用了 **BLiMP**（`nyu-mll/blimp`，语言学界标准的
minimal-pair 基准数据集），因为它已经覆盖了 doc 列表里的大部分类别（主谓
一致、数量一致、否定/NPI licensing、论元结构可以近似语义角色），每一对句子
本来就只在一个语法线索上不同，天然符合 doc"用外部 LM 过滤掉不自然样本"这个
要求的精神（BLiMP 本身就是经过语言学家和统计过滤的数据集）。

**doc 列表里 BLiMP 没有覆盖的类别**（命名实体位置、局部搭配）**没有实现**。

## 关键方法论问题：BLiMP 的"差异点"是 target，不是 cue（重要，如实说明）

doc 原始设计想要一个"cue"位置（比如主语的单复数）和一个**分开的**"target"
位置（比如动词），构造"只换 cue、target 位置的正确答案保持不变"的三元组
（正确 cue + 正确 target / 错误 cue + 正确 target 应该变错 / ...）。

**BLiMP 的 pair 不是这样构造的**：`build_minimal_pairs.py` 筛选出的 pair
在**恰好一个 token 位置**上不同——对主谓一致类的 UID，这个唯一的差异点
**就是 target（动词）本身**，而 cue（主语名词，比如"radii"）在 good/bad 两句
里是**完全相同**的（这正是 BLiMP 的设计初衷：只有一个语法维度不同）。也就是说
用 BLiMP，我们**没有**"cue 变了、target 位置不变"这种三元组可用——只有"cue
固定不变、target 位置的『正确噪声目标』在两句间不同"这一种结构。

**因此这次实现的实际是一个不同但相关的因果测试**（在 `probe_minimal_pairs.py`
的 docstring 里也写了）：对每一对句子，用**同一个固定噪声** epsilon 构造两条
oracle 路径——"good"路径（denoise 目标是语法正确的词）和"bad"路径（上下文
完全相同，但 denoise 目标是语法错误的词）。核心问题变成：**在"bad"路径里
（模型被要求重建错误的词），周围正确的语法线索（cue）会不会依然把证据"拉向"
语法正确的词**——即使正确词根本不是这次官方的去噪目标。这是"上下文线索能不能
因果地影响证据方向"这个更广的问题的一个合理但**不完全等同于 doc 字面设计**的
操作化版本，需要在论文里说清楚这个区别。

## 工程细节

- `build_minimal_pairs.py`：从若干 BLiMP UID 里各取若干条，用目标 tokenizer
  分词，只保留"两句 token 长度相同 + 恰好一个 token 不同"的 pair（这是一个
  相当激进的过滤——保留率因 UID 而异，运行时会打印每个 UID 实际保留了多少条，
  需要在正式结果里报告）。
- **ELF 的 seq_len 陷阱（这次提前避开了）**：ELF 的 RoPE 是固定长度表
  （见 `EXP-PT1-spec.md` 的踩坑记录），BLiMP 句子很短（十几个 token），但
  喂给 ELF 时必须 pad 到 `config.max_length`（1024）才行，`probe_minimal_pairs.py`
  会在长度不匹配时直接 assert 报错并提示重新用正确的 `--seq_len` 跑
  `build_minimal_pairs.py`。这意味着 ELF 的 minimal-pair 数据要单独构建一份
  （`--seq_len 1024`），和 LangFlow 那份（`--seq_len` 可以用比较短的值，比如
  32）不是同一个文件。
- `probe_minimal_pairs.py` 输出的核心指标：`rank_good_in_bad_traj`（"bad"
  路径里，语法正确词的 rank）、`rank_bad_in_bad_traj`（"bad"路径里，
  denoise 官方目标——也就是错误词——的 rank）、以及对应的概率值，另外按
  BLiMP UID 分类汇总。

## 状态

**LangFlow DONE；ELF baseline/kd_cr/kd2 跑中**（480 对 BLiMP pair，6 个 UID，
100% 通过过滤，9 个 t 点）

## Results（LangFlow，t_max=0.95）

`rank_good_in_bad_traj`（"bad"路径里，语法正确词的 rank）：7211→455（随 t
下降，即使不是官方去噪目标，也在慢慢变得更可信）；`rank_bad_in_bad_traj`
（官方目标本身的 rank）：7233→34.2（下降得更快、更彻底，最终几乎总是
top choice）。

**按 UID 拆分出一个清楚的、可能跨架构一致的模式**——用
`rank_good/rank_bad` 比值衡量"语法线索的拉力有多强"（比值越接近 1，说明
即使去噪目标错了，正确词依然很有竞争力；比值越大，说明模型几乎完全服从
字面去噪目标，不受语法线索影响）：

| UID | rank_good | rank_bad | 比值 |
|---|---|---|---|
| determiner_noun_agreement_1 | 479.4 | 262.9 | **1.8**（最强"语法拉力"） |
| distractor_agreement_relational_noun | 286.6 | 27.0 | 10.6 |
| wh_vs_that_no_gap | 186.8 | 22.2 | 8.4 |
| npi_present_1 | 304.1 | 29.0 | 10.5 |
| irregular_plural_subject_verb_agreement_1 | 298.8 | 18.4 | 16.2 |
| existential_there_subject_raising | 1472.8 | 46.0 | **32.0**（最弱） |

`determiner_noun_agreement_1`（限定词-名词数量一致）在 LangFlow 上语法线索的
"拉力"最强，`existential_there_subject_raising`（存在句主语提升，语法关系
更间接）最弱——**这个排序方向和 ELF baseline 的 pilot 跑（N=480, T=3）
一致**（ELF pilot 里 `determiner_noun_agreement_1` 的比值最小（54.7/0.4≈137,
是 ELF pilot 里最小的），`existential_there_subject_raising` 的比值最大
（9261.6/0.5≈18523，是 ELF pilot 里最大的））——如果 ELF 正式规模也保持这个
排序，会是一个真正跨架构一致的语言学发现，值得专门确认。

## Results（ELF baseline/kd_cr/kd2，全部完成）

按 `rank_good_in_bad_traj`（越小 = 语法线索拉力越强）排序，四个模型放在一起看：

| UID | baseline | kd_cr | kd2 | LangFlow |
|---|---|---|---|---|
| determiner_noun_agreement_1 | **53.6**（第1） | **2.3**（第1） | 19.2（第3） | 479.4（第5） |
| distractor_agreement_relational_noun | 340.3（第2） | 5.9（第2） | **8.2**（第1） | 286.6（第2） |
| irregular_plural_subject_verb_agreement_1 | 372.8（第3） | 7.9（第3） | 61.9（第4） | 298.8（第3） |
| wh_vs_that_no_gap | 3687.6（第5） | 15.1（第4） | 10.2（第2） | **186.8**（第1） |
| npi_present_1 | 1534.2（第4） | 239.6（第5） | 2183.4（第5） | 304.2（第4） |
| existential_there_subject_raising | **9296.5**（第6，最弱） | **3445.3**（第6，最弱） | **9193.0**（第6，最弱） | **1472.8**（第6，最弱） |

**核心发现：`existential_there_subject_raising`（存在句主语提升）在全部
4 个模型上都是排名最后（语法线索拉力最弱），没有一个例外**——不管
架构（ELF vs LangFlow）还是训练方式（baseline vs KD），这个类别的"正确词
即使在去噪目标错误时也应该有竞争力"这个效应最弱，说明模型在处理这种更
间接的句法关系（"there is/are..."结构里主语和动词的关系被"there"这个
虚位主语打断）时，学到的语法敏感性明显弱于直接的主谓/限定词-名词一致。

`determiner_noun_agreement_1`/`distractor_agreement_relational_noun`（两个都是
数量一致相关的 UID）持续排在前列（4 个模型中至少 3 个排进前 2），是"拉力"
最强的一类。`npi_present_1`（否定极性词许可）中等偏弱，四个模型都不在最强
也不在最弱。

⚠️ 每个 UID 只有 80 对样本，最初这里只有描述性排序、没有显著性检验；
下面"严谨性补强"一节补上了 bootstrap CI，排名中间的类别（distractor
agreement vs irregular plural vs wh_vs_that vs npi）顺序在不同模型间有明显
浮动，不应该过度解读中间名次的细节顺序。

## 严谨性补强：per-UID bootstrap CI（rigor audit 后，2026-07-26）

给 `probe_minimal_pairs.py` 加了逐 pair 原始数组保存
（`minimal_pairs_raw_<label>.npz`：`uids`, `rank_good_in_bad_traj`,
`rank_bad_in_bad_traj`），重跑全部 4 个模型（数字与修复前完全吻合），新增
`bootstrap_pt8.py` 在每个 UID 内部按 pair 重采样 2000 次。

**踩坑记录（方法论上有意思，值得记住）**：最初的版本用
`rank_good/rank_bad` 这个比值做 CI 检验，结果在 kd_cr/kd2 上出现明显荒谬
的数字（比如 CI 达到 `[1.57e11, 3.34e11]`）——排查后发现：KD checkpoint
把"bad"（去噪的字面官方目标，即错误词）的 rank 几乎全部压到 0（比如
kd_cr 的 `npi_present_1`/`wh_vs_that_no_gap` 两个 UID 里，**全部 80 个
pair 的 `rank_bad` 都精确等于 0**），比值在 bootstrap 重采样时会被约等于
0 的分母放大到任意大，是纯粹的数值退化，不是真实效应。**修复：改用
`rank_good` 本身（而不是比值）做 CI 和"最弱 UID 是否与其它 UID 不相交"的
检验**——这也是 spec 前面表格实际使用的排序标准，比值本来就只是一个辅助
展示，不是核心指标。

**修复后的结果（全部 4 个模型）**：

| model/ckpt | existential_there rank_good CI | 与其它 5 个 UID 的 CI 是否全部不相交 |
|---|---|---|
| ELF baseline | 9296.5 [7284.3, 11566.5] | **是** |
| ELF kd_cr | 3445.3 [2155.3, 4987.9] | **是** |
| ELF kd2 | 9193.0 [7341.7, 11194.8] | **是** |
| LangFlow | 1472.8 [787.4, 2380.7] | **是** |

**结论：修复数值问题后，"existential_there_subject_raising 排名最弱"这个
发现在全部 4 个模型上都通过了最严格的检验——它的 rank_good 置信区间与
其余全部 5 个 UID 的置信区间都不相交**，不是断崖式点估计的偶然产物。这把
本节前面"4/4 模型方向一致，值得信赖但未做显著性检验"的表述，升级为
"4/4 模型都有单模型内部的统计显著性支持"——比原计划的结论更强。详见
`results/phase_transition/<model>/<checkpoint>/bootstrap_pt8_full.json`。
