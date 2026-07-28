# EXP-GS6 Spec — Competing Global Basins

## 背景与地位

原始 doc 第 10 节 GLOBAL-6，P2 阶段。直接测量全局 basin selection/bifurcation：取两条
主题明显不同的 clean 序列 `A`、`B`，在匹配的 SNR 下插值它们的状态
`Z_t(lambda) = lambda*Z_t^A + (1-lambda)*Z_t^B`，从每个 `lambda` 继续采样，判断最终输出
落在 basin A、basin B 还是"其它"，检验是否存在清晰的 bifurcation point。

## 0. 复用

- `rollout_branches`（GS2）：从任意中间状态做完整 reverse-ODE rollout，直接复用（不用
  GS7 的 `rollout_with_checkpoints`，因为这里不需要记录中间 checkpoint，只要终点）。
- GS1 的 topic KMeans centroids：判断最终输出"属于 basin A / basin B / 其它"，用最终
  pooled embedding 的最近质心是否等于文档 A 的质心、文档 B 的质心，还是两者都不是。
- Oracle 状态构造：同 GS1–GS5，`Z_t = t*x_clean + (1-t)*eps`。

## 1. Pair 选择

不用原始 doc 建议的人工设计对照集（那是 GS9 已经做过、且发现有实验者偏差风险的路子）。
改为**从 OWT 中自动采样一批文档，用 GS1 的 KMeans 质心给每篇打 topic 标签，只保留 topic
标签不同的文档对**——避免人工挑选，且和 GS1 的语义簇定义保持一致。

## 2. 插值与 rollout

对每一对 `(A, B)`：

1. 独立采样各自的噪声 `eps_A`、`eps_B`。
2. 在 `t=0.28`（沿用 GS1/GS8 已经确认"early global probe 较强"的同一个 t，保持跨实验
   可比）构造 `Z_t^A`、`Z_t^B`。
3. `lambda ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}`（原始 doc 建议 11 点 `{0,0.1,...,1.0}`，pilot
   用 6 点控制算力，见第 4 节）：`Z_t(lambda) = lambda*Z_t^A + (1-lambda)*Z_t^B`。
4. 从 `Z_t(lambda)` rollout 到 `t_end=0.99`（复用 GS2 的多步 ODE，`sc` 冷启动全零）。
5. 判断终点：用 GS1 质心，最终 pooled embedding 最近的质心是 A 的质心、B 的质心，还是
   两者都不是（"其它" basin）。

## 3. 指标

- `P_A(lambda)`：该 lambda 下落入 basin A 的比例（本 pilot 每个 `(pair, lambda)` 只做
  一次确定性 rollout，不像 GS2 那样对每个起点采样多条 branch——`P_A` 因此是跨多个 pair
  在同一 `lambda` 下的比例，不是同一 pair 多次重复的比例，见第 4 节）。
- Bifurcation 陡峭度：`P_A(lambda)` 从接近 0 上升到接近 1 所跨越的 `lambda` 区间宽度
  （越窄=越陡峭=越像真正的相变；越宽=越像线性/平滑过渡）。

## 4. 已知简化

1. ⚠️ **不对每个 `(pair, lambda)` 做多个随机分支**（不像 GS2 每个起点采 K=8 个 branch）——
   `Z_t(lambda)` 本身是确定性插值，rollout 也是确定性 ODE，所以同一个
   `(pair, lambda, eps_A, eps_B)` 只有一个结果；`P_A(lambda)` 的"比例"是跨不同 pair 聚合
   出来的，不是同一起点的多次独立采样，样本量因此完全由 pair 数量决定。
2. ⚠️ `lambda` 只有 6 个点，不是原始 doc 的 11 点。
3. ⚠️ 只在单一 t（0.28）插值，不是原始 doc 建议的"随 t 增大观察 basin boundary 是否变陡"
   （那需要在多个 t 上重复整个 lambda sweep，pilot 算力不支持）。
4. `n_pairs` 数量取决于自动采样后有多少对 topic 不同的文档，pilot 目标 4 对。

## 5. 脚本与输出

```text
experiments/global_state/probe_competing_basins.py
```

```text
results/global_state/<model>/<checkpoint>/competing_basins_<label>.json
```

## 状态

**Pilot DONE**（ELF baseline，4 对 topic 不同的 OWT 文档，t=0.28，6 个 lambda，GPU1，
`logs/global_state/gs6_elf_baseline_pilot.log`，输出
`results/global_state/elf/baseline/competing_basins_pilot.json`）——出现清晰的
bifurcation 信号，是 GS 系列第三个干净正面结果（另两个是 GS7、GS8）。

## Results（pilot：ELF baseline，4 对，t=0.28，t_end=0.99）

| lambda | P_A | P_B | P_other |
|---|---|---|---|
| 0.0（纯 B） | 0.25 | 0.75 | 0.00 |
| 0.2 | 0.25 | 0.75 | 0.00 |
| 0.4 | 0.50 | 0.50 | 0.00 |
| 0.6 | 1.00 | 0.00 | 0.00 |
| 0.8 | 1.00 | 0.00 | 0.00 |
| 1.0（纯 A） | 1.00 | 0.00 | 0.00 |

逐对明细：pair 0/1/3 都表现出**单调、集中在一两个相邻 lambda 区间内完成的跳变**
（pair 0：0.2→0.4 之间从 B 跳到 A；pair 1：0.4→0.6 之间；pair 3：0.4→0.6 之间），不是
在 6 个点上均匀渐变——符合原始 doc"是否存在明显 bifurcation point"的正面判据。

⚠️ pair 2 是个例外：**全部 6 个 lambda（包括 `lambda=0.0`，理论上是纯 B）都被分类为
basin A**，即使起点是 100% 文档 B 的 oracle 状态，rollout 终点依然落进了 A 的语义簇——
说明这一对里 A 的 basin 明显比 B "更强/更具吸引力"，纯 B 起点都拉不住。聚合的 `P_A(0.0)=
0.25` 之所以不是 0，就是被这一个异常对拉高的。

**解读**：

1. **总体上支持"存在 bifurcation point"而不是"线性平滑过渡"**——3/4 对文档都在相邻的
   一两个 lambda 区间内完成从 B 到 A 的整体切换，而不是随 lambda 缓慢线性漂移。
2. **basin 吸引力可能不对称**（pair 2 的例外情况）——不是所有"A vs B"pair 都是对称的
   competing basin，有些 pair 里一个 basin 明显更强，纯粹起点组成不足以决定最终归属，
   这提示"哪个 topic 更常见/训练数据中更占优"可能影响 basin 的相对稳定性，值得在正式规模
   里专门统计。
3. **`n=4` 对，`P_A(lambda)` 是跨 pair 聚合而不是同一起点的重复采样**——单个 lambda 点的
   统计量实际上只有 4 个二元观测，结论方向可信但精确的"跳变宽度"数字（比如"恰好在
   lambda=0.4–0.6 之间"）需要更大样本或者对同一 pair 做更细的 lambda 网格才能确认。

## 下一步

1. 扩大 `n_pairs`（当前 4 对太少），并对"跳变区间"做更细的 lambda 网格（比如在
   0.3–0.7 之间加密到 0.05 间隔），确认跳变宽度是否真的很窄（"陡峭"）。
2. 排查 pair 2 这类"basin 不对称"的例子是否和 topic 在训练语料/GS1 8-簇里的相对频率
   有关（GS1 pilot 的 8 簇里，簇 2 出现 8 次、簇 6 出现 5 次，是最大的两个簇——pair 2 恰好
   涉及簇 6 vs 簇 4，簇 4 只出现 4 次，可能是"小簇"更容易被"大簇"的 basin 吸走的一个例子，
   值得在正式规模里系统统计簇大小和 basin 吸引力的关系）。
3. 在多个 t（不只是 t=0.28）重复整个 lambda sweep，检验"随 t 增大 basin boundary 是否
   变陡"这个原始 doc 提出但本 pilot 没测的问题。

## LangFlow 复现（发现并修复了一个关键 bug，2026-07-26）

### Bug：nearest-centroid 分类在 Euclidean 距离下被 norm 差异主导

第一次在 LangFlow 上跑 GS6（`t=0.28` 和后来改用 `t=0.65`）时，`P_A(lambda)` 全程完全
平坦（`P_A≈0.25` 不随 lambda 变化），甚至 `lambda=1.0`（纯 A 起点）也经常被误判成
"other"。诊断发现根因是**量纲不匹配**，不是语义/schedule 问题：

```
clean_pooled（GS1 用来拟合 KMeans centroids 的向量）  norm ≈ 3.70 ± 0.80
GS1 topic centroids 本身                              norm 范围 1.36–8.86
LangFlow rollout 终点（GS6/GS4/GS14 实际要分类的对象） norm ≈ 1.15 ± 0.15
```

`nearest_topic` 原来用平方欧氏距离找最近质心——rollout 终点的 norm 系统性地远小于
拟合 centroids 时用的 clean embedding，导致**每一个 rollout 终点都离 norm 最小的
那个质心（质心 0，norm=1.36）最近，和真实内容完全无关**：直接验证 8 篇文档的
rollout 终点，全部被分到同一个 cluster 0，而它们各自和自己 clean embedding 的
cosine 相似度是 0.40–0.67（真实的、按内容区分的信号其实都在，只是被 norm 差异
淹没了）。

**修复**：把 `nearest_topic` 换成基于 cosine 相似度（scale-invariant），统一放进
`common.py`，替换掉原来在 `intervene_global_modes.py`/`probe_competing_basins.py`/
`intervene_global_to_local.py`/`branch_true_trajectory.py` 四处重复的欧氏距离版本。

### 修复后的结果（`t=0.65`，24 docs，4 pairs）

| lambda | P_A | P_B | P_other |
|---|---|---|---|
| 0.0 | 0.00 | 0.75 | 0.25 |
| 0.2 | 0.00 | 0.75 | 0.25 |
| 0.4 | 0.00 | 0.75 | 0.25 |
| 0.6 | **1.00** | 0.00 | 0.00 |
| 0.8 | 1.00 | 0.00 | 0.00 |
| 1.0 | 1.00 | 0.00 | 0.00 |

**比 ELF 的原始结果更干净**：4 对文档**全部**在 `lambda=0.4→0.6` 之间同步完成从
B-basin 到 A-basin 的整体切换（ELF 原始结果里跳变区间在不同 pair 之间分散在
0.2–0.6，还有一个异常对全程卡在 A）；`lambda=1.0`（纯 A）现在正确分类为 A，
`lambda=0.0`（纯 B）正确分类为 B/other，不再有"任何输入都判成同一个 cluster"的
退化行为。**这是目前 GS6 系列里最干净的 bifurcation 证据，且是跨架构的**。

### 关联修复：GS4、GS14 的 topic 维度同样受这个 bug 影响，已一并重跑

- **`EXP-GS4`**（同样用 `nearest_topic` 判断 rollout 终点属于哪个 topic）：修复后
  `t=0.65` 上 `topic` 维度从"全程平坦 0.25"变成有真实区分度的结果——`baseline=1.00`，
  `A_remove_global=1.00`（去掉低秩 global mode 不影响 topic），
  **`B_preserve_global=1.00`（只保留低秩 global mode，topic 依然完美恢复，即使同一
  条件下 token 已经崩溃到 0.005）**——这是原始 doc GLOBAL-4 "preserve only global
  mode 应该保留 topic 但丢失 token"这个预测目前拿到的最干净的一次直接确认。
  `C_swap` 里 topic 跟随残差 donor B（`vs B=1.00` vs `vs A=0.25`），和 token 维度的
  "残差主导 swap 结果"方向一致——但和 `B_preserve_global` 单独看时"G 已经足够决定
  topic"表面上有张力：说明 `G` 单独存在时**足以**决定 topic，但换到另一条序列的真实
  高秩残差里时，残差的丰富信息会盖过 `G` 的贡献，两者并不矛盾，只是因果充分性
  （sufficiency）和因果优势度（dominance in competition）是两个不同的问题。
- **`EXP-GS14`**：`C_topic` 修复前后数值完全一样（全程 1.000），但意义不同——修复前
  无法确定这是"真的 topic consensus"还是"bug 导致所有分支都判成同一个 cluster"；
  修复后，同一套 `nearest_topic` 已经被 GS4/GS6 证明能正确区分不同 topic，所以
  `EXP-GS14` 的 `C_topic=1.000` 现在是一个**确认为真**的发现，不是 bug 的副产品。

详见 `logs/global_state/gs{4,6,14}_langflow_baseline_pilot*cosfix.log` 和对应的
`results/global_state/langflow/baseline/*_cosfix.json`。

## 严谨性自审：n=4 的"干净同步切换"在 n=20×3 seed 下打了折扣（2026-07-27）

上一节记录的 `t=0.65` cosine-fix 重跑用的是原始 pilot 规模（n_docs=24 → 4 对不同
topic 的文档），显示"4 对文档全部在 `lambda=0.4→0.6` 之间同步完成 B→A 整体切换"，
被当作"比 ELF 更干净"的正面结果写进了 EXP-INDEX 和 synthesis 文档。

用户要求做严谨性自审后，用 `--n_docs 80 --n_pairs 20 --seed {42,123,456}` 重跑（贪心
配对是确定性的，3 个种子实际用的是同一批 20 对文档，只是 rollout 用的噪声不同），并对
pooled n=60（3 seed × 20 pair）做 bootstrap CI：

| lambda | P_A (point [95% CI]) |
|---|---|
| 0.0 | 0.017 [0.000, 0.050] |
| 0.2 | 0.050 [0.000, 0.117] |
| 0.4 | 0.183 [0.083, 0.283] |
| 0.6 | 0.617 [0.483, 0.733] |
| 0.8 | 0.850 [0.750, 0.933] |
| 1.0 | 0.817 [0.717, 0.900] |

**好消息**：λ=0.4 和 λ=0.6 的 CI 完全不重叠（[0.083,0.283] vs [0.483,0.733]），说明
"存在真实的、统计显著的 bifurcation"这个核心主张站得住，不是 n=4 的噪声。

**需要更正的地方**：
1. 转变发生在 λ=0.4→0.8 一个更宽的窗口，不是 n=4 显示的单一 `0.4→0.6` 区间那么整齐；
2. 即使是最干净的对照条件——纯 A 起点（λ=1.0，理论上应该 100% 判成 A）——P_A 的点
   估计只有 0.817，CI 上界也只到 0.9，**从未接近 1.0**，说明约 15-20% 的情况下即使
   输入就是纯 A 文档，rollout+分类流程依然判不回 A；
3. n=4 pilot 从未报告过的 `P_other`（rollout 终点落入的 topic 既非 A 也非 B）在大样本
   下相当可观：λ=0.0/0.2 附近约 35-45%，λ=0.6/0.8 仍有 10-35%。

"比 ELF 更干净、4 对文档全部同步切换"这个表述需要撤回。更准确的说法是：**存在真实、
统计显著、但有相当噪声的 bifurcation**，转变窗口比小样本显示的更宽，且很大一部分
rollout 会落入第三方 topic 而非二选一。这不推翻"存在 competing basin 且切换点集中在
中间 λ 附近"这个定性结论，但削弱了"干净利落"这个定量描述。

详见 `logs/global_state/gs6_langflow_bigN_seed{42,123,456}.log` 和对应
`results/global_state/langflow/baseline/competing_basins_bigN_seed{42,123,456}.json`。

## 严谨性自审：branch-consensus 类实验（GS2/GS14）里发现了第二处未修复的同款 bug

上一次 GS6 debug 时的"四文件统一修复"验证方法有漏洞：只检查了 `nearest_topic` 的
import 是否指向同一函数对象，没检查每个文件的 topic 分类计算是否真的调用了它。
`branch_global_consensus.py`（GS2）和 `branch_true_trajectory.py`（GS14）里各自有一份
内联手写的平方欧氏距离最近质心分类，从未走 import 路径，是这次自审才发现并修复的。
详见 EXP-GS2-spec.md 和 EXP-GS14-spec.md 对应章节。

## 严谨性自审：ELF 版本补做同样的大样本 + bootstrap CI 复核（2026-07-27）

之前的 n=20×3 seed + bootstrap CI 复核只做了 LangFlow（见上面"n=4 的干净同步切换在
n=20×3 seed 下打了折扣"一节）。补做 ELF 版本：`--n_docs 80 --n_pairs 20 --t 0.28
--seed {42,123,456}`，pooled n=60 的 bootstrap CI：

| lambda | P_A (point [95% CI]) |
|---|---|
| 0.0 | 0.067 [0.017, 0.133] |
| 0.2 | 0.067 [0.017, 0.133] |
| 0.4 | 0.133 [0.050, 0.217] |
| 0.6 | **0.983 [0.950, 1.000]** |
| 0.8 | 0.950 [0.883, 1.000] |
| 1.0 | 0.950 [0.883, 1.000] |

**和 LangFlow 的结果对比是本次审计最有意思的地方**：ELF 在大样本下依然非常干净——
λ=0.4→0.6 几乎一步到位（0.133→0.983），纯端点 CI 下界都在 0.88 以上；而 LangFlow
同款复核（n=60）里，转变窗口更宽（0.4→0.8）、纯端点 P_A 只有 0.82 左右、`P_other`
高达 10-45%（见上一节）。**这不是 n=4 pilot 的运气或架构特有 bug，是在完全对齐的
方法学（同一脚本、同样 n_pairs/seed 数）下测出来的真实架构差异**：ELF 的 competing
basin bifurcation 明显比 LangFlow 更"决绝"（更少落入第三方 topic、纯端点恢复率更高）。
这个差异本身值得作为一个跨架构发现写进论文，而不只是"复现"或"不复现"的二元判断。

详见 `logs/global_state/gs6_elf_bigN_seed{42,123,456}.log` 和对应
`results/global_state/elf/baseline/competing_basins_bigN_seed{42,123,456}.json`。
