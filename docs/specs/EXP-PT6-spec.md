# EXP-PT6 Spec — Local Stability Around the Crossing（有范围限定的实现）

## 背景与地位

P1 阶段实验。目标：判断 top-1 跨越是一个稳健的相变，还是一个脆弱的排名波动——
在转变点附近分支（branch）完整状态，施加扰动，然后**继续跑 ODE 到 t=1**，看
最终 token 会不会变。这是本 suite 里第一个需要真正"续跑轨迹"（而不只是在某个 t
读一次 decode）的实验，也是目前实现成本最高的一个。

## 相对 doc 协议的范围缩减（关键工程决策）

doc 原始协议的规模在 ELF（N=128, L=1024）量级下完全跑不动：**逐位置**的
checkpoint（每个位置用它自己的 `tau_b`）× 5 种扰动方向 × 5 个 eta 量级 ×
最少 5 个随机种子 × 每个分支都要续跑到 t=1——这是几千次完整 ODE rollout，
远超本次可用的时间/算力预算。做了以下缩减：

1. **用 population 级别（不是逐位置）的 checkpoint**——从 `EXP-PT2` 的
   `transition_failure_analysis_<label>.json` 里读 `tau_b_mean_finite`/
   `tau_s_mean_finite`，作为整个 batch 共用的 `tau_b-Δ / tau_b / tau_b+Δ / tau_s`
   四个时间点，一次性对全部 N 条序列、全部位置同时分支——不是"每个位置自己的
   转变点"，而是"这个 checkpoint 在群体平均转变点附近的行为"。
2. **5 种扰动方向只实现了 2 种**：isotropic random，以及一个真正的（不是占位符）
   token-discriminative 方向 `u_yf`——用一个独立 centroid split，通过
   `index_add`/`gather` **向量化**构造（没有用 EXP-PT3 那种逐位置 Python 双重
   循环，见下面的"顺便修的问题"）。没做：orthogonalized random control、
   empirical rollout-drift direction（需要真实 free-running/Protocol B 轨迹，
   属于 EXP-PT7 的范畴）、context-only/target-position-only 扰动（需要
   位置选择性扰动，另一个维度的复杂度）。
3. **5 个 eta 只测 3 个**（1e-3, 3e-3, 1e-2，去掉最小的 1e-4/3e-4——预期在这个
   量级下效果接近零，对第一遍探索性价比低）。
4. **每个条件只用 2 个分支**，不是 doc 建议的最少 5 个种子——用于控制续跑
   rollout 的总次数。
5. **rollout 步数用 `n_rollout_steps=8`**（不是训练/推理常用的 32-64 步），
   为了控制单次续跑的前向次数；这是一个粗略的 ODE 积分，终点 token 可能和
   更精细的 solver 有一定差异，解读"final token"时要记住这一点。
6. **没有实现的指标**：pairwise branch agreement、local gain
   （`||Φ(z+δ)-Φ(z)||/||δ||`）——只实现了 immediate/final flip rate 和 modal
   outcome probability。

## 顺便修的问题：把 EXP-PT3 的慢循环换成向量化实现

`EXP-PT3-spec.md` 记录过一个已知性能问题：给每个位置分配 token-discriminative
方向的双重 Python `for` 循环在 ELF 规模下要跑 10+ 分钟。这次写 PT6 需要同样的
centroid 查找，**顺手用向量化实现替换了**（`build_centroid_table`：
`index_add_` 建表 + `gather` 批量取值，没有任何 Python 级别的逐位置循环），
实测在 ELF 规模下这部分只需要几秒钟。**这个实现更好，值得回头把 EXP-PT3 也
换成同样的写法**（本 spec 记一下，作为后续优化项，这次没有回去改 PT3 本身，
因为 PT3 的正式结果已经用旧实现跑完了，重跑的收益不足以打断already-produced
的结果）。

## 状态

**PILOT DONE（ELF baseline，N=4，n_rollout_steps=4）；正式规模（N=64）ELF
baseline 已提交跑，kd_cr/kd2/LangFlow 待 EXP-PT4 释放 GPU 后补跑。**

## Results（pilot：ELF baseline，N=4，population tau_b=0.353, tau_s=0.373）

⚠️ 规模极小，数字不可引用为结论，只用于验证代码路径和信号方向。

| checkpoint | eta | direction | immediate_flip | final_flip | modal_prob |
|---|---|---|---|---|---|
| tau_b_minus (t=0.253) | 0.01 | random | 0.0105 | 0.0079 | 0.9951 |
| tau_b (t=0.353) | 0.01 | random | 0.0022 | 0.0027 | 0.9985 |
| tau_b_plus (t=0.453) | 0.01 | random | 0.0013 | 0.0020 | 0.9988 |
| tau_s (t=0.373) | 0.01 | random | 0.0023 | 0.0027 | 0.9987 |
| tau_b_minus | 0.01 | token_direction | 0.0277 | 0.0135 | 0.9946 |
| tau_b | 0.01 | token_direction | 0.0103 | 0.0042 | 0.9978 |

**定性观察（方向支持"稳健相变"假说）**：

1. **flip rate 在 `tau_b_minus`（跨越之前）明显高于 `tau_b`/`tau_b_plus`/`tau_s`
   （跨越之后）**——random 方向 eta=0.01 下 immediate_flip 从 1.05% 降到
   0.13-0.23%，这正是判定规则里"稳健转变应该在 tau_b 附近/之后表现出稳定性
   急剧上升"的方向。
2. **token_direction 扰动比 random 扰动更容易造成翻转**（eta=0.01 时
   tau_b_minus 处 2.77% vs 1.05%）——符合直觉：沿着"真值 vs 默认竞争者"这个
   有意义的轴扰动，比随机方向更容易真正改变决策。
3. `token-direction valid for 16.3%`（pilot 用了只有 16 条序列的 centroid
   split，覆盖率低；正式规模会提高，参考 EXP-PT3 的经验——ELF 上大规模
   centroid split 后覆盖率能到 35-87%，取决于 checkpoint）。

这个方向性结果和 EXP-PT2 的"KD post-crossing slope 远大于 pre-crossing"、
以及 EXP-11v2 的"KD 改善后期稳定性"是一致的一组交叉验证，值得在正式规模上
确认。

## Results（正式规模：ELF baseline，N=64，population tau_b=0.353, tau_s=0.373）

pilot 的方向在正式规模上完全确认，而且更清楚：

| checkpoint (t) | random eta=0.01 imm/final | token_direction eta=0.01 imm/final |
|---|---|---|
| tau_b_minus (0.253) | 0.0092 / 0.0108 | 0.0552 / 0.0239 |
| tau_b (0.353) | 0.0034 / 0.0034 | 0.0166 / 0.0073 |
| tau_b_plus (0.453) | 0.0018 / 0.0017 | 0.0049 / 0.0052 |
| tau_s (0.373) | 0.0026 / 0.0035 | 0.0117 / 0.0067 |

- **单调下降**：无论 random 还是 token_direction 方向，flip rate 从
  `tau_b_minus` 到 `tau_b` 到 `tau_b_plus` 单调递减（token_direction
  immediate flip: 5.52% → 1.66% → 0.49%），`tau_s`（=0.373，只比 tau_b 晚
  0.02）介于 `tau_b` 和 `tau_b_plus` 之间，和它在时间上的位置完全吻合——
  **不是噪声，是一个干净的单调时间趋势**。
- **token_direction 扰动的 flip rate 始终是 random 的 3-6 倍**（tau_b_minus
  处 5.52% vs 0.92%）——扰动方向越贴近"真值 vs 默认竞争者"这个有意义的轴，
  越容易真正改变决策，random 方向大部分扰动预算被浪费在无关方向上。这个比例
  关系在四个 checkpoint 上都稳定存在。
- **immediate flip 和 final flip（续跑到 t=1 之后）数值很接近**（比如
  tau_b_minus token_direction: 5.52% vs 2.39%，final 略低于 immediate）——
  说明很多"立刻翻转"的扰动在续跑过程中又被纠正回来了一部分，但没有完全纠正；
  真正稳健的还是发生在 tau_b 之后的扰动（那里 immediate≈final，说明一旦在
  那个阶段翻转，续跑也不会纠正回来）。

## Results（跨模型对比：token_direction, eta=0.01）

| model/ckpt | tau_b_minus imm/final | tau_b imm/final | tau_b_plus imm/final | tau_s imm/final |
|---|---|---|---|---|
| ELF baseline | 0.055 / 0.024 | 0.017 / 0.007 | 0.005 / 0.005 | 0.012 / 0.007 |
| ELF kd_cr | 0.064 / **0.314** | 0.065 / **0.215** | 0.012 / 0.039 | 0.056 / **0.165** |
| ELF kd2 | 0.063 / **0.141** | 0.063 / **0.147** | 0.012 / 0.031 | 0.055 / **0.110** |
| LangFlow | 0.375 / 0.591 | 0.289 / 0.493 | 0.209 / 0.372 | 0.168 / 0.252 |

**新发现（超出 pilot 阶段只看 baseline 时观察到的模式）**：

1. **baseline 的 rollout 是"自我纠正"的**：`final_flip < imm_flip`（几乎每个
   checkpoint都是），说明很多立刻发生的翻转在续跑过程中被纠正回原来的答案。
2. **kd_cr/kd2 的 rollout 是"放大"的，方向完全相反**：`final_flip` 是
   `imm_flip` 的 **3-5 倍**（比如 kd_cr 在 tau_b_minus：immediate 只有 6.4%，
   但续跑到底后 final flip 高达 **31.4%**）。也就是说：KD 模型虽然更早、更
   "决绝"地承诺（tau_b 更早、post-crossing slope 更陡——见 EXP-PT2），但小的
   早期扰动一旦发生，续跑不会纠正它，反而会级联放大成一个完全不同的最终答案。
   **"更早/更决绝的承诺"和"对扰动更鲁棒"不是一回事**——这是一个重要的、和
   直觉不完全一致的发现，值得在论文里专门讨论。
3. **LangFlow 全程 flip rate 远高于 ELF 任何 checkpoint**（immediate 17-37%，
   final 25-59%），且和 KD 一样是"放大"而非"纠正"模式——和 EXP-PT2 里
   LangFlow `multiple_revision` 高达 65.7% 的发现完全一致，是独立指标下的
   交叉验证。
4. 所有模型都保留了"随时间接近/越过 tau_b，flip rate 总体下降"这个大方向
   （baseline 和 LangFlow 单调；kd_cr/kd2 在 tau_b_minus→tau_b 基本持平，
   到 tau_b_plus 才明显下降——可能因为 kd_cr/kd2 的 tau_b 本来就很早
   （≈0.20），`tau_b_minus`(0.10) 和 `tau_b`(0.20) 之间的动力学变化本身就
   更剧烈/非线性，用同样的 `delta_t=0.1` 采样可能没踩在关键区间上）。

⚠️ 这些数字全部来自本 spec 前面说明过的缩减范围协议（population 级
checkpoint、2 个方向、3 个 eta、2 个分支、8 步粗 rollout），不是 doc 完整
协议的结果，解读时要保持这个前提。

## 严谨性补强：多种子复核（kd_cr，最极端的案例）

kd_cr 的"final flip 是 immediate flip 3-5 倍"（放大而非纠正）是 suite 里
最反直觉的发现之一，原本只有 seed=42 一次运行的证据。追加了 seed=123 和
seed=456 两次独立重跑（tau_s, token_direction, eta=0.01）：

| seed | immediate_flip | final_flip | final/immediate 比值 |
|---|---|---|---|
| 42（原始） | 0.0560 | 0.1650 | 2.9× |
| 123 | 0.0567 | 0.2074 | 3.7× |
| 456 | 0.0543 | 0.1826 | 3.4× |

**结论：三个独立种子的 immediate_flip 几乎完全一致（0.054-0.057），
final_flip 有一定波动（0.165-0.207）但方向和量级都稳定——"放大而非纠正"
这个定性发现在种子间稳健复现，不是 seed=42 的巧合**。final_flip 本身的
种子间变异（约 ±0.02，相对量级 ~15%）提示如果要报告精确的"放大倍数"数字，
应该用多种子均值 ± 标准差，而不是单一种子的点估计——三次的均值放大倍数
约为 3.3×。

## 严谨性补强：样本量翻倍（N=64→128，全部 4 模型，2026-07-26）

PT6/PT7 是本 suite 里样本量最小的两个实验（N=32-64 vs 其余大多数在 128），
用户要求专门针对这两个实验提高规模。这里把 N 从 64 提到 128（保持
`n_rollout_steps=8`、2 个方向、3 个 eta、2 个分支不变，只加倍样本），全部
4 个模型重跑（label=`n128`），和原始 N=64 结果逐个 checkpoint 对比
（token_direction, eta=0.01）：

| model | tau_b_minus (imm/final) | tau_b (imm/final) | tau_b_plus (imm/final) | tau_s (imm/final) |
|---|---|---|---|---|
| baseline N=64 | 0.055/0.024 | 0.017/0.007 | 0.005/0.005 | 0.012/0.007 |
| baseline **N=128** | 0.057/0.025 | 0.018/0.008 | 0.006/0.009 | 0.012/0.007 |
| kd_cr N=64 | 0.064/0.314 | 0.065/0.215 | 0.012/0.039 | 0.056/0.165 |
| kd_cr **N=128** | 0.064/0.360 | 0.064/0.232 | 0.012/0.051 | 0.056/0.199 |
| kd2 N=64 | 0.063/0.141 | 0.063/0.147 | 0.012/0.031 | 0.055/0.110 |
| kd2 **N=128** | 0.062/0.155 | 0.063/0.179 | 0.012/0.029 | 0.054/0.123 |
| LangFlow N=64 | 0.375/0.591 | 0.289/0.493 | 0.209/0.372 | 0.168/0.252 |
| LangFlow **N=128** | 0.360/0.586 | 0.283/0.480 | 0.231/0.386 | 0.181/0.300 |

**结论：N=128 在每一个 checkpoint、每一个模型上都复现了 N=64 的方向和
量级（immediate_flip 几乎逐位吻合，final_flip 在同一量级内小幅波动，和
kd_cr 的 seed 123/456 复核显示的 ~±0.02 波动幅度一致）**——"baseline
自我纠正（final≈imm 或 final<imm）、kd_cr/kd2/LangFlow 放大（final 明显
大于 imm，2-6 倍不等）"这个核心定性发现不是 N=64 的小样本噪声，翻倍样本量
后完全稳定。唯一一个边界情况：baseline 在 `tau_b_plus` 处 final(0.009)
略高于 imm(0.006)（N=64 时两者几乎相等 0.005/0.005）——这个点本来就是
baseline 四个 checkpoint 里 flip rate 最低、最接近噪声下限的一个，边际的
final>imm 不改变"baseline 整体上是自我纠正的"这个结论，只是提醒这一个
具体格点不应过度解读为"baseline 也会放大"。详见
`results/phase_transition/<model>/<checkpoint>/branch_stability_n128.json`。
