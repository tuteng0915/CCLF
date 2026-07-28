# EXP-GS4 Spec — Global Mode Causal Intervention

## 背景与地位

原始 doc 第 8 节 GLOBAL-4，P1 阶段第一项（"验证因果作用"）。GS3 用**被动诊断**
（把 `G_t^{(k)}`/`R_t^{(k)}` 直接喂给 backbone 做单次 decode）发现 structural 信号集中在
低秩 global mode、token 信号集中在残差；但 GS3 spec 第 4 节已经标注了一个重要警示：单次
被动 decode 是喂给 backbone 一个它训练时从未见过的分布外输入，`token_G≈0.03`（即使在
clean-ref 时）既可能反映"低秩模式真的不含 token 信息"，也可能只是"OOD 输入让 decode 失效"。

GS4 用**因果干预 + 完整 rollout**代替单次被动 decode：把 `G_t^{(k)}`/`R_t^{(k)}` 当作
reverse-ODE 的**起点**（复用 GS2 已经验证过的 `rollout_branches` 多步积分机制），让 backbone
有机会在几十步 ODE 积分里把这个分布外起点"拉回"正常轨迹分布，再看最终结果——这是对 GS3
警示的直接回应，也是原始 doc 本身要求的"从该状态继续采样"协议。

## 0. 复用

- Adapter、`rollout_branches`（直接从 `experiments/global_state/branch_global_consensus.py`
  import，不重写）：GS2 已经实现并验证过多步 Euler-ODE rollout（self-cond 冷启动全零，
  `t_start→t_end=0.99`，步数按剩余距离比例缩放）。
- `svd_decompose` / `pad_to_full`（从 `experiments/global_state/analyze_low_rank_modes.py`
  import）：GS3 已验证的 per-sequence SVD 分解，`k=8`（GS3 里 syntax 信号最强、且和 k=2
  定性一致的取值）。
- `pos_histogram` / `masked_mean_pool` / `decode_text`（`common.py`）。
- Topic 判定复用 GS1 的 KMeans centroids（`topic_kmeans_centroids_pilot.npy`），但**这次不是
  测"K 条 branch 之间是否一致"（GS2 的问题），而是测"最终结果的 topic 是否等于原始文档的
  真实 topic"**——是一个新的、更直接的正确性问题，不是重复 GS2。

## 1. 四个条件

对每条起始序列，在若干 `t_start` 处构造 oracle `Z_t`，做 rank-`k=8` SVD 分解，然后从四种
不同的起点分别做完整 rollout 到 `t_end=0.99`：

- **Baseline**（对照）：直接从 `Z_t`（未分解）rollout——原始 doc 里隐含的"不干预"参照组，
  用来确认 rollout 机制本身在这批文档/这些 t_start 上是正常工作的（如果 baseline 本身都
  恢复不出原文档的 topic/token，后面三个条件的比较就没有意义）。
- **A（Remove global mode）**：从 `Z_t^{-G} = R_t^{(k)}` rollout（原始 doc 8.A）。
- **B（Preserve only global mode）**：从 `Z_t^G = G_t^{(k)} + \epsilon_{matched}` rollout，
  其中 `\epsilon_{matched}` 是逐通道均值/方差匹配 `R_t^{(k)}`（同一条序列自己的残差统计量）
  的高斯噪声（原始 doc 8.B）。
- **C（Global-mode swap）**：把 batch 内的序列按固定 derangement（`i -> (i+1) mod N`）两两
  配对成 `(A,B)`，从 `Z_t^{swap} = G_t^A + R_t^B` rollout（原始 doc 8.C）。

（原始 doc 8.D 的 oracle-vs-rollout 插值需要一条真实 free-running 轨迹作为 "roll" 参照，
本仓库目前的 oracle-only 协议不直接提供；留给 GLOBAL-7/EXP-GS7，这里不做。）

## 2. 指标

对每个条件的 rollout 终点 `z_final`（配合最终一次 `forward_state(z_final, sc_final,
t_end)` 取 native top-1 tokens）：

- **Final topic agreement**：`z_final` mean-pool 后用 GS1 centroids 找最近簇，是否等于
  原始 *clean* 文档（该条件所用的"目标"文档——条件 C 里，判断是等于 A 的簇还是 B 的簇）
  mean-pool 后的簇。报告 agree-with-A / agree-with-B 比例（条件 C）或 agree-with-self 比例
  （baseline/A/B）。
- **Structural preservation**：`z_final` 解码文本的 POS histogram 与目标文档（同上，C 条件
  下分别对 A、B）clean POS histogram 的 cosine 相似度。
- **Final token accuracy**：`z_final` 的 native top-1 tokens 与目标文档 ground-truth token
  的逐位置 accuracy（C 条件下同样分别对 A、B 各报一个）。

## 3. 数据与规模（pilot）

- ELF baseline，`eval_exp37c_baseline.yml`（1024-token）。
- `n_docs=4`（与 GS2 一致，配对做 derangement `i -> (i+1)%4`）。
- `t_start ∈ {0.05, 0.38}`（一个远早于 cliff、一个接近 cliff 中段）。
- `k=8`。
- 每个 (t_start, 条件) 组合 batch=4（不需要像 GS2 那样对每条文档采样 K 个 branch，
  GS4 每条文档/每个条件只需要一次确定性 rollout）。

## 4. 已知简化

1. ⚠️ 不含原始 doc 8.D（oracle injection，需要 free-running 轨迹）。
2. ⚠️ `\epsilon_{matched}`（条件 B）是逐通道均值/方差匹配的**独立高斯采样**，不保留
   `R_t^{(k)}` 的任何结构（协方差、跨位置相关性）——这是"用同分布均值/方差的白噪声代替
   残差"的最简操作化，原始 doc 8.B 的表述本身也只要求"局部噪声保持与原 state residual
   相同的均值和方差"，与此一致。
3. ⚠️ 条件 C 的配对是固定的 `i -> (i+1) mod N` derangement，不是随机采样多个 pairing——
   pilot `n_docs=4` 太小，无法支撑多 pairing 的统计。
4. Pilot 规模（4 docs × 2 t_start），数字仅用于判断方向和验证 rollout-from-decomposed-state
   代码路径是否正确。

## 5. 脚本与输出

```text
experiments/global_state/intervene_global_modes.py
```

```text
results/global_state/<model>/<checkpoint>/intervene_global_modes_<label>.json
```

## 状态

**Pilot DONE⚠️**（ELF baseline，4 docs，t_start∈{0.05,0.38}，k=8，GPU1，
`logs/global_state/gs4_elf_baseline_pilot.log`，输出
`results/global_state/elf/baseline/intervene_global_modes_pilot.json`）。

## Results（pilot：ELF baseline，4 docs，seq_len=1024，k=8，t_end=0.99）

⚠️ n_docs=4 极小（`topic_match` 只是 0/4–4/4 的计数），数字仅用于判断方向。

**t_start=0.05**：baseline 本身 `topic=0.00`（4 篇全部没恢复对原始 topic），所有条件的
`token≈0.01–0.02`。**这个 t_start 在当前 pilot 设置下不具诊断性**——连未经干预的对照组都
恢复不出正确 topic，说明"从 t=0.05、self-conditioning 冷启动"这个 rollout 协议本身在这个
极早起点上就已经很脆弱（大概率是 EXP-GS2/GS4 共同标注的"冷启动 SC"简化在如此早的起点上
影响更大），无法用来比较 A/B/C 条件之间的差异，不纳入结论。

**t_start=0.38**（baseline 对照组本身工作正常，可用于比较）：

| condition | compare_to | topic (n=4) | struct_cos | token_acc |
|---|---|---|---|---|
| baseline | self | 0.75 (3/4) | 0.999 | 0.859 |
| A_remove_global | self | 0.50 (2/4) | 0.972 | **0.171** |
| B_preserve_global | self | **0.00 (0/4)** | 0.917 | **0.050** |
| C_swap | vs A_donor（G 来源） | 0.50 (2/4) | 0.996 | **0.014** |
| C_swap | vs B_donor（R 来源） | **1.00 (4/4)** | 0.999 | **0.763** |

**解读**：

1. **Baseline 对照组工作正常**（topic 3/4、token 0.859，和未受扰动的 rollout 应有的表现
   一致），说明 A/B/C 三个条件之间的差异是真实的干预效应，不是 rollout 机制本身失效。
2. **条件 A（只留残差 R^(8)）**：token accuracy 从 0.859 暴跌到 0.171（相对下降
   ~80%），topic 从 0.75 降到 0.50（相对下降 ~33%）——**token 受到的伤害明显比 topic 大**，
   方向上支持"low-rank global mode 对 token/lexical 的因果贡献比对 topic 更大"，但两者
   都受损，不是"topic 完全不受影响"的干净结果。
3. **条件 B（只留低秩 global mode + 匹配噪声）**：topic 归零（0/4），token 也几乎归零
   （0.050）——**低秩 global mode 单独不足以让模型正确恢复 topic，更不用说 token**。这和
   GS3 里"低秩重构的线性 probe 能恢复 78% 以上的 POS R²"看起来矛盾，但其实揭示了一个重要的
   区别：**GS3 测的是"这个子空间对一个独立训练的线性 probe 有多大预测力"，不是"这个子空间
   单独是否足以驱动模型自己的生成过程"**。`k=8` 只是从有效秩 ~460–470 的几乎满秩状态里
   截出的一个极薄的切片；线性 probe 可以从这个切片里榨出不成比例的结构信号（可能是因为
   POS 分布这种粗粒度目标本身对"任意一个和它相关的子空间"都不难线性可分），但这不代表模型
   自己的生成机制只需要这个切片就能正常工作——把其余 ~450+ 个有效维度换成匹配矩、方差的
   纯噪声后，模型自己的 decode 完全失效。**这是本次 pilot 里最重要的方法论纠正**：
   被动 probe 发现的"信息在哪"和因果干预发现的"驱动力在哪"可能不是一回事。
4. **条件 C（swap，`G^A + R^B`）**：无论 topic 还是 token，最终结果都**明显更像 B（残差
   来源）而不是 A（global mode 来源）**——vs B_donor 的 topic 命中率 4/4、token accuracy
   0.763，vs A_donor 只有 2/4 和 0.014。这进一步印证第 3 点：在 `k=8` 这个操作化下，
   残差 `R^{(8)}` 几乎承载了文档的完整身份（因为它仍然是原状态里 98%+ 的有效维度），
   而 `G^{(8)}` 单独换到另一条序列上几乎不能把结果"拉向" A。`struct_cos` 在这个条件下对
   A/B 两个donor都同样高（0.99+），**没有区分度**，和 GS1/GS2 已经指出的"某些 cosine 类
   指标动态范围过窄"是同一类问题，这里不能用它做判断，只能靠 `topic_match`/`token_acc`。
5. **与 GS3 的关系，如实报告的张力**：GS3（被动诊断）发现"structure 集中在 G、token 集中
   在 R"，本 pilot（因果干预）发现"G 单独不足以驱动任何东西（topic 或 token 都不行），
   R 单独/换到别的序列上几乎能保留大部分身份"。两者并不矛盾，但强调的是不同的问题——
   GS3 回答"哪个子空间对探针更有预测力"，GS4 回答"哪个子空间对生成过程更有因果必要性/
   充分性"，在 `k=8`（只占约 1.7% 的有效维度）这个操作化下，两个问题的答案明显不同，
   这个区别本身就是需要写进论文的重要发现，而不是矛盾需要消解。

## 下一步

1. `t_start=0.05` 需要先解决冷启动 SC 在极早起点失效的问题（比如允许从真实累积的 SC
   状态开始，而不是全零），否则这个 t_start 在当前实现下无法使用。
2. 试更大的 `k`（比如 GS3 建议的 16，或者更大，比如 64/128）——如果 `k=8` 太薄导致条件 B
   必然失败，提高 `k` 可能会看到"topic 恢复但 token 仍然缺失"这个更符合原始 doc 预期的
   过渡区间，找到"多大的 k 才能让 global mode 单独具备因果充分性"。
3. 扩大 `n_docs`（当前 4 太小，`topic_match` 统计噪声很大）和 `t_start` 覆盖更多贴近
   已知 commitment cliff（0.2–0.3）的点。
4. 把这个"被动 probe 发现的信息位置 vs 因果干预发现的驱动力位置可能不一致"的教训，
   同样用于审视 GS1/GS2 的发现——尤其是"structure 早于 topic"这个排序，需要一个对应的
   因果版本（比如 GLOBAL-8/EXP-GS8）才能确认是真实的因果时序，而不只是探针可读性的时序。

## 严谨性自审：n=4 headline 数字在大样本 + bootstrap CI 下打折扣（2026-07-27）

`nearest_topic` bug 修复后（见 EXP-GS6-spec.md），LangFlow 上用 n_docs=4（原 pilot 规模）
+ t_start=0.65 重跑曾显示一组"完美"数字：`baseline=1.00, A_remove_global=1.00,
B_preserve_global=1.00`（token_acc 上 B_preserve_global≈0.005 相对 baseline≈0.555 有
巨大差距），当时被写成"GLOBAL-4 预测的最干净确认"。

用户要求做严谨性自审后，用 `--n_docs 16 --seed {42,123,456}`（3 个种子，t_start 只保留
0.65，去掉已知不具诊断性的 0.05）重跑，并用新加的 `bootstrap_ci`（`common.py`）对
pooled n=48（3 seed × 16 docs）做 2000-resample bootstrap CI：

| condition | topic_match (point [95% CI]) |
|---|---|
| baseline | 0.854 [0.750, 0.938] |
| A_remove_global | 0.875 [0.771, 0.958] |
| B_preserve_global | 0.812 [0.688, 0.917] |

**三者的 CI 完全重叠，统计上无法区分。** 这意味着 n=4 时看到的"三者都恰好=1.00"是小样本
巧合，不能支持"G 单独排他性地承载 topic 信息"这个更强的因果说法——更准确的说法是：
topic 信息在完整状态、去掉 G 的残差、只保留 G 三种重构里都大致同等程度可恢复，是一种
冗余而非 G 独占。

**token_acc 维度的不对称性完整存活，且不需要 CI 就能看出**（3 个 seed 一致）：

| condition | token_acc (3 seed: 42/123/456) |
|---|---|
| baseline | 0.567 / 0.575 / 0.565 |
| A_remove_global | 0.538 / 0.541 / 0.525 |
| B_preserve_global | 0.006 / 0.007 / 0.008 |

`B_preserve_global` 的 token_acc 比另外两个条件低了两个数量级，三个种子几乎一模一样——
这是本实验里唯一在大样本下依然干净、可信、值得写进论文的结论：**token identity 几乎
完全依赖残差，去掉残差（只留低秩 G）后 token 恢复能力崩溃，而 topic 恢复能力不受明显
影响**。C_swap 方向性结论（swap 后 topic 跟随残差 donor 而非 global-mode donor）也复现：
vs A_donor topic=0.19-0.31，vs B_donor topic=0.69-0.88（3 seed）。

详见 `logs/global_state/gs4_langflow_bigN_seed{42,123,456}.log` 和对应
`results/global_state/langflow/baseline/intervene_global_modes_bigN_seed{42,123,456}.json`。

## 严谨性自审：ELF 版本补做同样的大样本 + bootstrap CI 复核（2026-07-27）

之前只对 LangFlow 版本做了 n=16×3 seed 复核（见上一节），ELF 版本仍停留在 n=4 pilot
（"双重标准"，见 EXP-INDEX.md 第六条跨架构教训）。补做同样的复核：
`--n_docs 16 --seed {42,123,456}`（t_start 用 ELF 原生的 0.05/0.38，其中 0.38 是
"对照组正常"的诊断性 t_start），对 pooled n=48（3 seed × 16 docs）在 `t_start=0.38`
上做 bootstrap CI：

| condition | topic_match (point [95% CI]) |
|---|---|
| baseline | 0.917 [0.833, 0.979] |
| A_remove_global | 0.292 [0.167, 0.417] |
| B_preserve_global | 0.229 [0.125, 0.354] |

**比 n=4 pilot 更完整的发现**：n=4 时只强调了"`B_preserve_global`（只留 G）topic 归零，
G 单独不足以因果驱动生成"。大样本下看到的是——`A_remove_global`（只留 R，去掉 G）的
topic_match 同样很低（0.292），和 `B_preserve_global`（0.229）的 CI 几乎完全重叠，
两者都远低于 baseline（0.917）。**在 ELF 上，G 和 R 单独都不足以恢复 topic，必须两者
兼备**——不是"topic 信息集中在 G 之外的某处"，而是"topic 恢复依赖完整状态的某种协同，
拆开后信息本身还在但没法正常被 decode 恢复"。

**这和 LangFlow 同款大样本结果形成一个值得写进论文的架构差异**：LangFlow 上
`baseline`/`A_remove_global`/`B_preserve_global` 三者统计不可区分（见上一节），topic
信息在三种重构里都大致同等程度可恢复；ELF 上则是 baseline 远高于另外两者、A/B 又
彼此不可区分。两个架构都不支持"G 排他性地承载 topic"这个最初的强说法，但差异本身
（"完整状态依赖" vs "冗余可恢复"）可能反映了两种模型 backbone 处理分解态 OOD 输入的
方式不同，值得在 discussion 里提一句。

详见 `logs/global_state/gs4_elf_bigN_seed{42,123,456}.log` 和对应
`results/global_state/elf/baseline/intervene_global_modes_bigN_seed{42,123,456}.json`。
