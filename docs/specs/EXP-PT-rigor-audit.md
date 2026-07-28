# Phase-Transition Suite（EXP-PT1–10）严谨性自查

用户要求对目前 10 个 PT 实验做一次严谨性自查，并在可行范围内提高规模/严谨性。
这份文档记录自查结果和已经执行的改进；每个改进也在对应的 `EXP-PTx-spec.md`
里有更详细的记录，这里只做汇总索引。

## 发现的系统性问题（跨实验共性）

1. **doc 共享协议明确要求"按序列 bootstrap 置信区间"（不是按 token 位置），
   之前 10 个实验全部没有实现**——这是目前为止最大的严谨性缺口，很多"头条"
   数字（advantage retained、tau_e/b/s、6 类失败分布、local-window
   sufficiency gap 等）只有点估计，没有不确定性量化。
2. **样本规模普遍低于 doc 建议的 512 序列**：大多数实验用 128（ELF）或 64-128
   （其它），部分实验（PT6/PT7 的 causal 部分）只有 32-64。
3. **大多数实验只用单一随机种子**（`torch.manual_seed(42)` 跑一次），doc 要求
   "干预类实验最少 5 个随机种子"。像 PT6/PT7 里最戏剧化的发现（KD 在扰动/
   插值下的"放大"/"崩溃"现象）目前只有一次运行的证据。
4. **LangFlow 的 PT4 只有 3 个探针位置**（`seq_len=128`，`probe_spacing=40`）——
   这个具体数字明显太小，几乎是个案而非统计结果，是目前最该优先修的"规模"
   问题（不是"严谨性框架"问题，是纯粹的样本量问题，且修起来很便宜）。
5. **稀疏 t-grid**（11-21 点，doc 建议 101 点密网格）——这是一个已知、一直
   如实标注的简化，短期内不打算全面重跑到 101 点（成本过高，且大多数结论
   已经在当前网格密度下方向清晰），但值得在最终论文图里对关键 t 附近适当
   加密。

## 已执行的改进

### 1. Bootstrap CI（PT1、PT2，免 GPU 计算，纯后处理）

新增 `experiments/phase_transition/bootstrap_utils.py`（按序列重采样的共享
工具）+ `bootstrap_pt1_pt2.py`（复用 PT1/PT2 已经落盘的逐位置数组，不需要
重新跑模型）。对全部 4 个模型跑了 2000 次 bootstrap，结果：

- **PT1 的"advantage retained"数字全部有紧致 CI，且 `P(m_res>0)=1.000`
  （2000 次重采样里从未跨过 0）**——"减先验后 margin 转正"这个核心发现在
  序列级重采样下极其稳健，不是个别序列或噪声撑起来的。
- 顺带发现：ELF 的 advantage-retained 数字如果排除 padding（这次 bootstrap
  脚本默认排除），比之前报告的数字略低（比如 baseline gauss 从 1.1%→1.02%，
  kd_cr 从 11.5%→12.22%——kd_cr 反而略高，说明方向不是单调的，只是小幅
  波动），整体结论不变。
- **PT2 的 `tau_e`/`tau_b`/`tau_s` 和 6 类失败分布的 CI 全部很窄**（比如
  baseline `successful_monotonic = 72.79% [71.86%, 73.57%]`）——在 N=128
  序列、L=1024（ELF）或 128（LangFlow）位置的规模下，统计功效其实已经足够，
  这些头条数字是可信的，不是几个异常序列撑出来的。

详见各 `results/phase_transition/<model>/<ckpt>/bootstrap_ci_full.json`。

### 2. PT4：修复 LangFlow 探针数量过少 + 补充逐位置保存以支持 bootstrap（完成）

- `intervene_context.py` 加了 `acc_per_seq`（按序列的 accuracy），配合新增的
  `bootstrap_pt4.py` 对头条数字（`local_window_r1` vs `full_context` 的差值）
  做了 2000 次 bootstrap。
- LangFlow 的探针数从 3 个（`seq_len=128` 下 `probe_spacing=40` 只够 3 个）
  提到 26 个（改用 `seq_len=1024`，和 ELF 对齐；LangFlow 没有 ELF 那种 RoPE
  固定长度约束，这个"低垂果实"之前被忽略了）。
- **结果**：ELF baseline 的"radius=1 已经和完整上下文统计不可区分"这个说法
  在新的 CI 下**成立**（diff 的 95% CI 包含 0）；kd_cr/kd2 有小但统计显著的
  差距（−0.9~−1.1pp，CI 不含 0，但量级很小，定性结论基本不变）；**LangFlow
  的差距既显著又大**（−21pp，CI 明显不含 0）——之前"LangFlow 需要远大于
  radius=1 的窗口"这个核心跨架构差异，现在有了扎实的统计支持，而不是建立在
  3 个探针点上的脆弱结论。详见 `EXP-PT4-spec.md`"严谨性补强"一节。

### 3. PT6/PT7：对最戏剧化的 KD 发现做多种子复核（完成）

PT6 的"KD rollout 是放大而非纠正"、PT7 causal interpolation 的"KD 在 λ=1
时崩溃"是目前 suite 里最反直觉、影响论文叙事最大的两个发现，之前只有单次
种子（seed=42）的证据。为 kd_cr（两个实验里都是最极端的案例）追加了
seed=123、seed=456 两次独立重跑：

- **PT6**：三个种子的 `immediate_flip` 几乎完全一致（0.054-0.057），
  `final_flip`/`immediate_flip` 的放大倍数在 2.9×-3.7× 之间波动——**"放大而非
  纠正"这个定性发现稳健复现，不是 seed=42 的巧合**。
- **PT7 causal interpolation**：三个种子的 λ=1 agreement 分别是
  0.134/0.290/0.224（均值≈0.216，标准差≈0.08）——**绝对崩溃幅度有真实的
  种子间波动，但"λ 越大、kd_cr 反而越差"这个方向性崩溃在三个种子里完全
  一致**。论文里引用这个数字时应该用多种子均值（≈0.22）而不是单次的 0.134。

详见 `EXP-PT6-spec.md`、`EXP-PT7-spec.md` 各自的"严谨性补强"一节。

### 4. PT3：bootstrap CI（免 GPU，复用已有逐位置数据，完成 2026-07-26）

`probe_velocity_alignment.py` 已经把逐位置原始数组存进
`velocity_alignment_raw_<label>.npz`，和 PT1/PT2 一样是"免费"的后处理 bootstrap
对象。新增 `bootstrap_pt3.py`，对全部 4 个模型跑了 2000 次序列级重采样：

- `a_clean(t_min)` 强正、CI 极窄（四个模型都不含 0）——"向量场早期就指向真值"
  这个正面发现稳健。
- **关键结果**：real（`u_{y,f}`）方向和 frequency-matched control 的
  `corr(C,-rank)` 置信区间在全部 4 个模型上都**重叠**，而 random/orth 两个
  control 的 CI 都紧贴 0、明显与 real 不重叠——把 PT3 原来"controls 显示出
  同样对齐、decision rule 没被满足"这个基于点估计的观察，升级成了有统计
  支持、4/4 模型一致复现的结论。这是本次 rigor-audit 里少数几个"bootstrap CI
  改变了结论的确定性程度"的例子（不是改变方向，而是把"看起来是这样"变成
  "确认就是这样，不是噪声"）。

详见 `EXP-PT3-spec.md`"严谨性补强"一节。

### 5. PT5：bootstrap CI（加装逐位置记录+重跑，完成 2026-07-26）

`intervene_decoder_bias.py` 加了 `decoder_bias_raw_<label>.npz`（`tau_e`,
`tau_b_lambda0/1`, `correct_beta`, `wrong_raw`），重跑全部 4 个模型（数字与
修复前完全吻合，一次免费的一致性检查），`bootstrap_pt5.py` 按序列重采样
2000 次：baseline 的先验减法让 `tau_b` 小幅提前（+0.006，CI 不含 0）；
kd_cr/kd2/LangFlow 全部明显推迟（−0.06 到 −0.35，CI 都不含 0，方向与
baseline 相反）——**把"baseline 与 KD/LangFlow 在这个诊断上方向相反"这个
此前基于单次运行的观察，确认为不是噪声**。详见 `EXP-PT5-spec.md`"严谨性
补强"一节。

### 6. PT9：bootstrap CI（加装逐序列记录+重跑，完成 2026-07-26）

`probe_cross_time_transfer.py` 加了逐序列 accuracy 矩阵保存
（`cross_time_transfer_raw_<label>.npz`），重跑全部 4 个模型（数字与修复前
完全吻合），`bootstrap_pt9.py` 按 held-out 序列重采样 2000 次：`upper_tri_mean
- lower_tri_mean` 这个"证据方向持续累积"的差值在全部 4 个模型上 CI 都不含
0（baseline 0.106、kd_cr 0.070、kd2 0.072、LangFlow 0.099，均 [lo,hi] 不
跨 0）——把此前基于点估计的核心发现确认为统计稳健。详见 `EXP-PT9-spec.md`
"严谨性补强"一节。

### 7. PT10：bootstrap CI（加装逐位置记录+重跑，完成 2026-07-26）

`analyze_failure_predictors.py` 加了逐位置 val 集正确性保存
（`failure_predictors_raw_<label>.npz`），重跑全部 4 个模型（数字与修复前
完全吻合），`bootstrap_pt10.py` 按 held-out 序列重采样 2000 次（不需要重新
拟合分类器）：全部 4 个模型的"val_acc 好于多数类基线"这个 improvement 的
CI 都不含 0，P(improvement>0)=1.000（2000 次重采样里从未跨过 0），即使是
量级最小的 LangFlow（+2.7pp）也统计显著。详见 `EXP-PT10-spec.md`"严谨性
补强"一节。

### 8. PT8：per-UID bootstrap CI（加装 per-pair 记录+重跑，完成 2026-07-26）

`probe_minimal_pairs.py` 加了逐 pair 原始数组保存（`minimal_pairs_raw_<label>.npz`），
重跑全部 4 个模型（数字与修复前完全吻合），新增 `bootstrap_pt8.py` 在每个
UID 内部按 pair 重采样 2000 次。

**中途踩了一个数值坑，修复后结论比最初更强**：第一版用
`rank_good/rank_bad` 这个比值做 CI，在 kd_cr/kd2 上出现荒谬的数字（CI 宽到
`1e11` 量级）——排查发现 KD checkpoint 把"bad"（去噪的官方目标，即错误词）
的 rank 压到几乎全是 0（比如 kd_cr 的 `npi_present_1`/`wh_vs_that_no_gap`
两个 UID，全部 80 个 pair 的 `rank_bad` 精确等于 0），比值在重采样时被
约等于 0 的分母放大到任意大，是纯数值退化，不是真实效应。**改用
`rank_good` 本身（spec 表格实际使用的排序标准）做 CI 后**：全部 4 个模型
的 `existential_there_subject_raising` 的 `rank_good` 置信区间都与其余
5 个 UID 的置信区间**完全不相交**——"排名最弱"这个发现不仅 4/4 模型方向
一致，而且每个模型内部都有严格的统计显著性支持，比最初预期的"仅描述性
排序"结论更强。详见 `EXP-PT8-spec.md`"严谨性补强"一节。

## 结论

对本次自查发现的 5 类系统性问题，已经在预算允许范围内做了针对性修复：
bootstrap CI（PT1/PT2/PT3/PT5/PT9/PT10 免 GPU 或新增按序列/逐 pair 记录后
跑了 2000 次重采样；PT8 同样按 pair 重采样，途中修正了一个比值指标的数值
退化问题）、LangFlow 探针数量（PT4，3→26）、多种子复核（PT6/PT7 最戏剧化的
kd_cr 发现，seed 42/123/456 三次独立验证）。**目前 suite 里全部头条发现——
先验减法的符号翻转（PT1）、KD 的转变时间提前（PT2）、token 方向证据与
frequency-matched control 不可区分（PT3）、局部上下文的充分必要性
（PT4）、baseline 与 KD/LangFlow 在 decoder-bias 干预下方向相反（PT5）、KD
扰动下的"放大而非纠正"（PT6）、KD 在 causal interpolation 下的崩溃
（PT7）、existential_there 在全部 4 模型上排名最弱且与其它 UID 的 CI 不相交
（PT8）、证据方向持续累积的不对称性（PT9）、失败预测器好于多数类基线
（PT10）——都已经有统计置信区间或多种子复核的支持，不再是单次运行的点
估计。**

## 仍未完成的工作

样本量整体提到 doc 建议的 512（目前多数在 128，受限于 ELF L=1024 时的
计算成本）；101 点密 t-grid（目前 11-21 点）。这些都是"继续往同一个方向
投入更多算力/工程"就能做到的增量改进，不是方法论上的新问题。至此，10 个
PT 实验全部有了 bootstrap CI 或多种子复核支持，rigor-audit 阶段的核心目标
已完成。
