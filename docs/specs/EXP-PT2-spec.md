# EXP-PT2 Spec — True-vs-Default Margin Trajectories

## 背景与地位

P0 阶段第二个实验，MVP-A 的一部分。目标：检验"承诺悬崖"是否是一个**连续累积的
margin 跨过离散 decoder 边界**产生的现象，而不是表征本身突然形成。复用
`docs/specs/EXP-PT1-spec.md` 里已经验证过的 adapter 层和 `rank_of_gt` 等工具函数。

## 与 PT1 的关键差异（工程决策）

1. **Reference prior 简化为单一参考**：PT1 用了三种 reference（Gaussian 匹配统计量、
   cross-sequence swap、context-shuffle），PT2 需要在**远比 PT1 密的 t-grid** 上跑，
   三份 reference 的代价不可接受。PT2 改用 EXP-05v3 的"global null"
   （`z_t_null=(1-t)*eps`，零信号），理由：(a) 计算便宜（不需要先算真实状态的
   per-t 均值方差）；(b) EXP-05v3 已经发现 `G_debias(null) ≈ G_debias(matched-Gaussian)`
   在整体趋势上相近。**这是一个待复核的假设**——PT1 正式规模结果出来后，应该抽查
   一下"null" 和"gauss" 两种参考在同一批数据上给出的 `m_res(t)` 曲线是否定性一致，
   如果不一致，PT2 的结论需要重新审视。
2. **默认竞争者 f 只实现了 doc 三种定义里的 2 种**：
   - `f1` = earliest-time native top-1（沿用 PT1 的 `f_i` 定义，保证两个实验可比）
   - `f2` = 前 `first10pct_n` 个 t 格点上的众数 top-1（doc 定义 2 的离散化近似）
   - 定义 3（"EXP-PT1 reference-prior top-1"）**未实现**，需要额外加载某次具体的
     PT1 run 并对齐 t-grid，留作后续。
3. **Isotonic / piecewise-linear 转变分析只做了 population 级别（均值曲线）**，
   不是 per-position。原因：per-position 做 isotonic + brute-force
   piecewise-linear（最多 3 个 breakpoint）在 N×L 可以到 10 万+ 位置的规模下不可行。
   per-position 仍然算了（更便宜的）`tau_e`/`tau_b`/`tau_s`、前后 crossing 斜率、
   zero-crossing 次数——这些都是 O(T) 的简单归约，不需要拟合。
4. **Independent-probe score（doc 测量项之一）：已补上**，见下面
   "Independent-probe score（补测）"一节。

## Failure taxonomy 的可能重叠（工程决策）

`classify_transition_failures.py` 把 doc 6 个类别当成互斥的、按优先级排序的分类
（见脚本里 `classify_position` 的 docstring）：
`no_emergence → wrong_mode_accumulation/stalled_ambiguity → multiple_revision →
endpoint_only_correction → premature_crossing → successful_monotonic`。
doc 原文的定义本身不是互斥的（比如一个位置理论上可以同时满足 "multiple revision"
和 "endpoint-only correction"），这里选择了一个优先级顺序来消歧，需要在论文里
说明这一点，不能假装每个位置只能"天然"落入一个类别。

## 脚本与输出

```text
experiments/phase_transition/analyze_margin_trajectory.py    # 计算 m_raw/m_res/rank/entropy/margin + population-level isotonic/piecewise fit
experiments/phase_transition/classify_transition_failures.py # per-position tau_e/tau_b/tau_s + 6-way failure taxonomy
```

```text
results/phase_transition/<model>/<checkpoint>/margin_trajectory_raw_<label>.npz
results/phase_transition/<model>/<checkpoint>/margin_trajectory_summary_<label>.json
results/phase_transition/<model>/<checkpoint>/transition_failure_analysis_<label>.json
```

## 状态

**PILOT DONE** — 两个脚本都在 ELF baseline 上以极小规模（N=4, T=6）跑通，数字定性合理：

- pilot：isotonic R²=0.99（对均值 `m_res(t)` 曲线拟合很好，即"单调累积"叙事在均值
  层面是自洽的——但 N=4 太小，噪声很低不代表真实信号，正式规模需要重新确认）。
- best piecewise-linear 选了 2 个 breakpoint（t=0.41, 0.77）而不是 0 个——即使数据
  本身已经很平滑，BIC 惩罚下的分段线性模型仍然倾向于找到断点，这提示"该不该用
  piecewise-linear 还是 isotonic 更能代表数据"这个模型选择问题本身需要更谨慎的
  统计检验（比如比较两者在留出数据上的样本外误差），而不是直接比较 R²/BIC，
  ⚠️ 当前实现没有做这个交叉验证，只是报告了"最优" fit，解读时要小心。
- `classify_transition_failures.py` 6 类分布（pilot, N=4）：successful_monotonic
  79.4%，wrong_mode_accumulation 13.4%，endpoint_only_correction 3.8%，
  premature_crossing 1.6%，no_emergence 0.95%，multiple_revision 0.7%，
  stalled_ambiguity 0.15%。**规模太小，只用于验证代码路径没有 crash 且数字在
  合理范围（加总=100%，各类比例不是退化的全 0/全 1）**。

## 事故记录：内存泄漏（已修复）

第一次跑正式规模时，`analyze_margin_trajectory.py` 的"pass 1"把**整个 t-grid**
（21 个点）的 `p_probs`（每个 (N,L,V)=(128,1024,32100) float32 ≈16.8GB）全部缓存进
一个 dict 里，理由是"避免第三次前向传播"。在 ELF 的规模下这直接导致单进程
~350GB 常驻内存，把 EXP-PT1 kd_cr 那次 PT2 尝试直接 OOM kill 掉，另外两个并发跑的
baseline PT2/PT5 进程把系统内存榨到只剩 300GB 可用、开始 swap，3.5 小时里一个 t
点都没跑完，被手动 kill 掉。修复方式：
1. 不再缓存全 t-grid 的 `p_probs`；`f1`/`f2` 只需要前 `first10pct_n` 个 t 点，用完
   立刻丢弃；主循环里 `p_probs` 每个 t 都是现算现用，不跨 t 保留。
2. 每个 t 迭代结束显式 `del` 掉所有大张量 + `gc.collect()`（`intervene_decoder_bias.py`
   虽然没有同类"跨 t 累积"的 bug，但也加了同样的显式清理，作为在这个规模下对
   allocator 碎片化的保险措施）。
3. 同时把 `estimate_reference_prior.py` 里 `rank_of_gt` 从"全量 argsort +
   全局 nonzero 扫描"换成了"严格大于计数"（等价但快很多，避免 CPU 长时间占用
   导致 GPU 空闲等待）——这个改动也让 PT2/PT5 直接受益（两者都从
   `estimate_reference_prior` import 这个函数）。

这次事故也暴露了一个流程问题：本机是多用户共享的（`nvidia-smi`/`ps aux` 里能看到
其他会话/用户的进程），第一次重跑还撞上了另一个进程占用 GPU 0 导致的 CUDA OOM
（不是我们自己的 bug）。之后统一改成先用 `nvidia-smi` 确认目标 GPU 真正空闲
（不只是"编号上分配给我"）再启动新任务。

## Results（正式规模：LangFlow，128 序列，seq_len=128，21 个 t 点 ∈ [0.05,0.95]）

- Isotonic（单调）拟合 `m_res(t)` 均值曲线：**R²=1.0**——population 层面上均值曲线
  非常干净地单调递增，支持"连续累积"叙事（至少在均值层面）。
- 最优 piecewise-linear 仍然选了 3 个 breakpoint（t≈0.14, 0.55, 0.86）而不是 0
  个——⚠️ 和 EXP-PT2-spec 前面讨论的一样，BIC 在这么平滑的曲线上仍然偏好加 breakpoint，
  这个模型选择问题本身需要更谨慎的检验（样本外交叉验证），不能直接以"BIC 更低"
  下结论。
- population 层面 `m_res(t)` 只有 **1 次**过零（从 t=0.05 的 −2.41 到 t=0.095 的
  +1.28，此后一路单调上升到 t=0.95 的 +21.8）——非常干净的单次交叉，没有反复。
- `tau_e` 均值 = 0.181，`tau_b`（raw）均值 = 0.478，`tau_s` 均值 = 0.631。
  `Delta_readout`(`tau_b-tau_e`) = **0.318**：representation 层面证据出现后，
  平均要再等 0.318 个 t 单位才在 native top-1 上"可见"——readout 和证据之间有
  实质性的时间差，与"悬崖只是读出边界"的极端版本不完全吻合（如果纯粹是读出
  边界，这个 gap 应该很小）。`Delta_stability`(`tau_s-tau_b`) = 0.163：正确后
  还会有一段时间不稳定（还会被修正/翻转）。
- **6 类失败分类里最大的一类是 `multiple_revision`（65.7%），远超
  `successful_monotonic`（22.9%）**。这和 EXP-24（LangFlow argmax 稳定性远低于
  ELF baseline：mean_last_flip=8.3/32 vs ELF 21.2/32）方向完全一致，是一个独立
  指标对同一个结论的交叉验证：**LangFlow 的逐位置 top-1 身份在收敛前会反复翻转，
  不是一次性的单调转变**。⚠️ 但这个 65.7% 的绝对数字可能部分是网格密度带来的
  人工产物（21 个 t 点比 EXP-24 用的步数更密，同样的底层动力学在更密的网格上
  自然会数出更多次"翻转"），跨实验比较绝对数字时要小心，只能比较相对大小/方向。

## Results（正式规模：ELF baseline/kd_cr/kd2，128 序列，seq_len=1024，21 个 t 点）

三个 checkpoint 全部跑完（用修复后的代码，内存正常）。

| checkpoint | isotonic R² | population zero-crossings | tau_e | tau_b | tau_s | Δ_readout | Δ_stability | post-slope |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.973 | 1 | 0.192 | 0.353 | 0.373 | 0.166 | 0.021 | 16.5 |
| kd_cr | 0.997 | 0 | 0.092 | 0.203 | 0.213 | 0.130 | 0.010 | 89.7 |
| kd2 | 0.993 | 0 | 0.088 | 0.204 | 0.215 | 0.137 | 0.010 | 85.5 |

6 类失败分类：

| checkpoint | successful_monotonic | no_emergence | wrong_mode_acc. | multiple_revision | premature | endpoint_only |
|---|---|---|---|---|---|---|
| baseline | 72.8% | 0.44% | 13.8% | 8.0% | 1.5% | 2.9% |
| kd_cr | 82.0% | **6.94%** | 5.2% | 5.8% | 0.1% | 0.02% |
| kd2 | 81.9% | **7.14%** | 4.8% | 5.8% | 0.2% | 0.06% |

**解读**：

1. `tau_e`/`tau_b`/`tau_s` 全部大幅提前（KD ≈0.09-0.09 vs baseline ≈0.19-0.35），
   和 EXP-10/EXP-16v2 等已有实验的"KD 大幅提前承诺"结论完全一致——这次是从
   margin-crossing-time 这个新角度独立复现了同一个结论。
2. KD 模型的 population zero-crossings = **0**（baseline 是 1）——意味着 KD 模型
   的均值 `m_res(t)` 曲线**从来没有从负变正**，也就是说在 t=0.05（最早的采样点）
   均值就已经是正的了。这本身就是"KD 让承诺极早发生"的一个更强表述：不是"更早
   跨过零点"，而是"在我们能采样到的最早时间点已经跨过了"。
3. **KD 模型 post-crossing slope 远大于 baseline（85-90 vs 16.5）**——一旦跨过
   边界，KD 模型的 margin 增长速度快得多，说明 KD 的转变不仅更早，而且更"陡峭
   决绝"，跨过之后迅速拉开差距，不太可能再被扰动逆转（这一点和 EXP-11v2 里
   "KD 改善后期稳定性"的发现方向一致）。
4. **反直觉的一点**：KD 模型的 `no_emergence` 比例（6.9-7.1%）反而**高于**
   baseline（0.44%）——即有一小部分位置，KD 模型的残差 margin 从来没有转正过。
   结合 EXP-PT1 里 KD 的 `never_res` 反而比 baseline 更低这一点，一个可能的解释
   是：KD 模型把"容易"的位置都极早、极彻底地解决了（绝大多数位置 `tau_e`≈0.09
   就完成），少数"难"位置则完全卡住（这套简单的 null-reference 对这些难位置
   完全没有帮助）——即 KD 让分布变得更加两极化（要么早期完全解决，要么完全
   卡住），而不是像 baseline 那样在两者之间有更连续的分布。这是一个值得在论文里
   讨论、但目前只有描述性证据的猜想，需要专门检验"哪些位置属于 KD 的 no_emergence
   桶"（比如是否是稀有词/长尾词）才能进一步确认。
5. `wrong_mode_accumulation` baseline（13.8%）明显高于 KD（4.8-5.2%）——baseline
   更容易把去偏后的概率质量错误地分配给第三方 token，KD 模型这个问题少很多。

⚠️ 三个 checkpoint 的 best piecewise-linear 都选了 3 个 breakpoint（不是 0
个），和 LangFlow 一样——BIC 偏好加断点这件事看起来是这套 brute-force 拟合方法
本身的系统性倾向，不是模型特有的现象，解读 isotonic vs piecewise-linear 对比时
要把这一点考虑进去。

## Independent-probe score（补测）

**脚本**：`experiments/phase_transition/probe_independent_score.py`（复用
`EXP-PT9`/本项目 `EXP-07c` 的 `LinearProbe` 架构，序列级 train/val 切分）。

在每个 t，用 TRAIN 序列上的 `predicted_clean` 状态训练一个独立线性 probe，
在 VAL 序列上同时评估 probe 和原生 decode，报告 `gap = probe_acc - native_acc`。
这直接复刻了本项目已有的"Story A: Probe Gap"（`EXP-07`/`EXP-07v2` 对应 ELF，
`EXP-21`/`EXP-21v2` 对应 LangFlow），但搬进了统一的 phase-transition adapter
框架、用了这个 suite 自己的 t-grid，方便和 PT1/PT2/PT3/PT5 的其它曲线直接
对齐比较（数字本身和旧的 EXP-07v2/EXP-21v2 不是逐位精确一致，采样、种子、
训练 epoch 数都不同）。

### Results（128 序列，11 个 t 点，15 epoch 训练）

| model/ckpt | mean gap (probe−native) | 方向 |
|---|---|---|
| ELF baseline | **+0.122** | probe > native |
| ELF kd_cr | **−0.086** | native > probe |
| ELF kd2 | **−0.081** | native > probe |
| LangFlow | **−0.094**（且随 t 增大而扩大，t=0.95 时 −22.1pp） | native > probe |

**这是一次很干净的独立复现**：符号方向和已有的 `EXP-07v2`（baseline gap
+41pp，kd_cr −11pp，kd2 −6~−9pp）、`EXP-21v2`（LangFlow native>probe）
**完全一致**——用一套完全不同的代码路径（不同 t-grid、不同采样、不同训练
超参）重新确认了"KD 逆转了 probe gap 的方向"这个核心发现，是 Story A 的一次
独立稳健性检验。LangFlow 的 gap 随 t 增大而单调扩大（−9.5pp→−22.1pp），这个
"越往后 native 领先越多"的模式此前在 EXP-21v2 里没有按 t 展开过，是这次补测
新增的细节。
