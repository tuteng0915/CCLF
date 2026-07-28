# EXP-PT4 Spec — Causal Context-Source Ablation（有范围限定的实现）

## 背景与地位

P1 阶段实验。目标：确定最早的样本特定证据来自哪里——目标位置自身的噪声信号、
局部上下文、还是全局上下文。

## 为什么不是逐位置精确实现（关键工程决策）

doc 原文的协议本质上是**逐位置**的："keep the target position state unchanged
and intervene only on other positions"——这意味着对每一个候选目标位置，都需要
一次"只修改除它以外所有位置"的前向传播。严格实现有两条路：

1. **每个目标位置单独跑一次前向**——对 ELF（L=1024）完全不可行（一个 t、一个
   条件就要 1024 次前向）。
2. **用 per-query 的 attention mask**（形状 (B,L,L)，不同 query 位置有不同的
   可见 key 集合）——技术上可行：确认了 `scaled_dot_product_attention` 底层
   支持 3D mask，但 ELF `model.py` 里 prefix token（mode tokens/context tokens）
   的拼接逻辑目前只处理 2D `(B,S)` mask（`torch.cat([mode_mask, attention_mask],
   dim=1)`），要支持 3D mask 需要修改这段共享库代码。**LangFlow 的 attention
   实现（`regular_attention_multi_headed`）完全不接受任何 mask 参数**，需要更大
   改动。

两条路都要么不可行、要么需要动共享的、被几十个已完成实验依赖的核心模型代码。
权衡之下，选择了**不修改任何共享模型代码**，改用一个只在
`experiments/phase_transition/` 内部实现的**值层面（value-level）代理方案**
（见下）。这是本 spec 里最大的一处简化，需要在论文里非常明确地说明。

## 代理方案：稀疏探针位置 + 跨序列内容替换

- 选一批间距足够大（`> 2 * max(radius)`，默认 40，覆盖测试的所有半径
  0/1/2/4/8/16）的**探针位置**，保证不同探针的局部窗口不会重叠。
- **local_window(r)**：探针半径 r 以内的位置员保留真实值，半径外的位置替换成
  **另一条随机序列在相同位置的值**（cross-sequence content swap，复用
  `EXP-PT1` Reference B 的 derangement 机制）。指标只在探针位置读出。
- **global_only(r)**：上面的补集——半径 r 以内替换掉，远处保留真实值。
- **no_context** = `local_window(r=0)`：探针位置自己保留，其余全部替换。
- **full_context**：不做任何替换，在同样的探针位置读出，作为公平对照。
- **within_sequence_shuffle** / **cross_sequence_swap**：全局版本，直接复用
  `EXP-PT1` Reference C / Reference B 的机制，在**全部**位置读出（这两个条件
  本来就没有"局部"结构，不需要探针）。

这个代理方案**不是**"目标位置在别的位置被替换时保持完全不变"的精确复现——
`cross_sequence_swap`/`shuffle` 是全局操作，`local_window`/`global_only` 虽然
探针位置的自身值确实保持真实（这一点是精确的），但"替换来源"用的是跨序列内容
而不是 doc 建议的多种替换源之一，且每次替换是**同一个**目标序列（用同一个
derangement 排列），不是逐位置独立采样替换源。优点：整个 t-grid × 6 个半径 ×
2 个方向只需要 `2*len(RADII)+3` 次前向传播（而不是 `L * len(RADII)`），在 ELF
规模下也能负担得起。

**没有实现**：doc 条件 7（oracle-clean context substitution）和条件 8
（wrong-but-grammatically-matched context substitution）——都需要额外的"干净
替代文本"或语法性判断工具，这次没有接入，标记为后续 gap。

## 指标简化

doc 要求的指标里，"shift in residual margin"、"shift in tau_e"、
"shift in tau_b" 需要完整的 EXP-PT2 式多 t 转变时间分析，对每个条件都做一遍
计算量很大。这次只计算了**每个条件在每个 t 上的探针位置 accuracy 和平均
rank**（`G_probe`、`rank_probe_mean`），把"转变时间"简化成了"看曲线在哪个 t
附近从低到高"，没有做正式的 `tau_b(condition)` 拟合。如果某个条件的结果特别
值得深挖，可以用这里存的每-t 数字再接一次 EXP-PT2 式的转变时间分析。

也没有单独做"function-word vs content-word 细分"和"frequency/surprisal
matched 分析"——这两个可以在已有的 `gt_ids` 上事后加，标记为后续工作。

## 脚本与输出

```text
experiments/phase_transition/intervene_context.py
```

```text
results/phase_transition/<model>/<checkpoint>/context_ablation_<label>.json
```

## 判定规则（对应 doc 6.4 节，用这次的代理指标重新表述）

- `local_window(r)` 的 `G_probe` 接近 `full_context` → 局部窗口半径 r 已经
  基本足够，证据主要是局部的。
- `global_only(r)` 的 `G_probe` 接近 `full_context` → 说明**破坏局部、保留远处**
  几乎不影响预测，证据不靠局部，可能靠位置自身信号或远处；结合
  `local_window` 的结果一起看才有意义。
- `no_context`（=`local_window(0)`）的 `G_probe` 明显低于 `full_context` →
  上下文（不管远近）确实提供了因果证据；如果 `no_context` 都还能维持较高
  accuracy，说明证据主要来自目标位置自身的噪声状态。
- `cross_sequence_swap` 大幅破坏 accuracy → 证据是样本特定的，不是通用先验。

## 状态

**DONE（ELF baseline/kd_cr/kd2 + LangFlow，全部正式规模，128 序列，11 个 t 点）**

## Results（t_max 处的探针 accuracy，即 t≈0.95）

| condition | ELF baseline | ELF kd_cr | ELF kd2 | LangFlow (t≈0.95) |
|---|---|---|---|---|
| full_context | 0.824 | 0.930 | 0.930 | 0.839 |
| local_window_r0（=no_context） | 0.125 | 0.124 | 0.094 | 0.255 |
| local_window_r1 | **0.821** | **0.922** | **0.920** | 0.555 |
| local_window_r2 | 0.835 | 0.929 | 0.928 | 0.630 |
| local_window_r8 | 0.839 | 0.930 | 0.930 | 0.742 |
| local_window_r16 | 0.835 | 0.930 | 0.930 | 0.813 |
| global_only_r0（只破坏自身） | **0.438** | **0.761** | **0.788** | **0.151** |
| global_only_r1+（破坏半径1，保留远处） | ~0.01 | ~0.01 | ~0.01 | ~0.01-0.02 |
| within_sequence_shuffle | 0.007 | 0.009 | 0.009 | 0.010 |
| cross_sequence_swap | 0.009 | 0.009 | 0.009 | 0.007 |

**核心发现**：

1. **ELF 三个 checkpoint 的证据高度局部**：`local_window_r1`（只保留紧邻一个
   token 的真实上下文）就已经恢复了 `full_context` 几乎全部的 accuracy
   （baseline 0.821 vs 0.824；kd_cr/kd2 更是几乎无损）。半径再大也不再有明显
   提升——局部窗口半径 1-2 对 ELF 而言就已经**足够**。
2. **LangFlow 需要大得多的窗口才能恢复**：`r1` 只恢复到 0.555（vs full
   0.839），要到 `r16`（在 128-token 序列里已经接近八分之一长度）才恢复到
   0.813，接近但仍未达到 full_context。**这是一个干净的跨架构差异**：ELF 的
   早期证据几乎完全来自极近距离上下文，LangFlow 需要明显更大的感受野。
3. **`global_only_r0`（只破坏目标自身的扩散信号，保留全部真实上下文）在
   KD checkpoint 上出奇地高**（kd_cr 0.761，kd2 0.788），baseline 中等
   （0.438），LangFlow 最低（0.151）。这说明：**KD 训练后的模型几乎可以仅凭
   上下文（类似完形填空）就恢复答案，几乎不依赖自身位置的扩散信号本身**；
   baseline 部分依赖；LangFlow 对自身信号被破坏最敏感。这和 EXP-PT1/PT2/PT5
   里反复出现的"KD 把证据提取能力学进了模型里"的故事完全吻合，这次是从
   "目标自身信号 vs 上下文"这个新角度给出的证据。
4. **`global_only_r1+`（哪怕只破坏半径 1 以内、保留所有远处真实上下文）在全部
   4 个模型上都崩溃到 ~1%**——局部上下文不仅充分，而且**必要**：破坏了近邻，
   哪怕远处全对，也无法恢复。局部充分性（发现1）和局部必要性（这一条）合在
   一起，说明"证据主要来自局部窗口"这个结论是双向都成立的，不只是"局部够用"
   这一个方向。
5. `within_sequence_shuffle`/`cross_sequence_swap` 全部崩溃到 <1%，四个模型
   一致——再次确认早期证据是样本特定的、依赖正确局部结构的，不是通用词汇
   先验能替代的（呼应 EXP-PT1 的 swap/shuffle reference 结果）。

⚠️ LangFlow 只有 3 个探针位置（`seq_len=128`，`probe_spacing=40`）——样本量比
ELF（26 个探针）小得多，数字的方差应该更大，解读时要打折扣；如果要更可信的
LangFlow 数字，需要用更长的 LangFlow 序列（如 doc 建议的 1024，如果 LangFlow
支持）或者更密的探针间距（牺牲半径覆盖范围）重跑。

## 严谨性补强（rigor audit 后）

**修了两个规模/严谨性问题**：

1. **LangFlow 探针数量从 3 个提到 26 个**——之前 LangFlow 用 `seq_len=128`，
   `probe_spacing=40` 只给出 3 个探针位置，基本是个案而非统计结果。LangFlow
   没有 ELF 那种 RoPE 固定长度约束（这一点之前被忽略了），改用 `seq_len=1024`
   后探针数和 ELF 对齐（26 个），代价可以忽略（LangFlow 本身架构小、算得快）。
2. **给 `intervene_context.py` 加了按序列的 accuracy 记录**（`acc_per_seq`），
   支持序列级 bootstrap CI（`bootstrap_pt4.py`），不再只报点估计。

### Results（t=0.95，2000 次 bootstrap，diff = local_window_r1 − full_context）

| model/ckpt | full_context | local_window_r1 | diff [95% CI] | 结论 |
|---|---|---|---|---|
| ELF baseline | 0.824 [0.809,0.836] | 0.821 [0.804,0.835] | −0.003 [−0.014,+0.008] | **CI 包含 0**——radius=1 和完整上下文统计上无法区分，充分性成立 |
| ELF kd_cr | 0.930 [0.920,0.939] | 0.922 [0.911,0.930] | −0.009 [−0.012,−0.006] | CI 不含 0——差异虽小但统计显著，radius=1 略逊于完整上下文 |
| ELF kd2 | 0.930 [0.920,0.939] | 0.920 [0.909,0.929] | −0.011 [−0.014,−0.007] | 同上，统计显著但差异很小 |
| LangFlow（新，N=26 探针） | 0.601 [0.557,0.645] | 0.391 [0.363,0.417] | **−0.210 [−0.236,−0.184]** | CI 明显不含 0——radius=1 远不足以恢复完整上下文的准确率 |

**修正后的结论（比之前更精确、更保守）**：

- **只有 ELF baseline 严格满足"radius=1 局部窗口在统计上等价于完整上下文"**。
- **KD checkpoint（kd_cr/kd2）的差距虽然统计显著，但绝对量级很小**（0.9-1.1pp）——
  之前"kd_cr/kd2 也几乎完全由局部窗口恢复"这个定性表述基本站得住，但严格来说
  不是"完全等价"，是"非常接近但有一个小的、可测量的残余缺口"。
- **LangFlow 的差距既统计显著又量级很大（21pp）**——用 26 个探针（而不是之前
  3 个）重新确认了"LangFlow 需要远大于 radius=1 的窗口"这个核心跨架构差异，
  现在有了扎实得多的统计支持，不再是"只有 3 个数据点"的脆弱结论。

`global_only_r1`（破坏局部保留远处）在全部 4 个模型上都跌到 ~1-2%，CI 都很窄，
"局部证据同时充分且必要"这条结论在新规模下同样成立、且更可信。

⚠️ **一个需要注意的混淆变量**：为了把 LangFlow 探针数从 3 提到 26，把它的
`seq_len` 从 128 改成了 1024——这同时也把 `full_context` 本身的绝对准确率从
旧数字的 0.839 变成了新数字的 0.601（更长的序列本身更难完整去噪/预测，这是
序列长度变化带来的，不是这个实验想测的东西）。**跨"探针数修复前后"比较
LangFlow 的绝对数字（比如 0.839 vs 0.601）没有意义**，只有"新规模内部"的
相对比较（`local_window_r1` 相对 `full_context` 的差距）才是这次修复真正
想确认的东西，而这个相对差距（21pp gap，CI 明显不含 0）在跨两个 seq_len
设置下方向和量级都是一致的，所以修复后的核心结论没有被这个混淆变量污染，
但报数字时要小心别把新旧两次跑的绝对 accuracy 放在一起比较。
