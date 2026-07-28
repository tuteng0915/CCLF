# EXP-PT7 Spec — Paired Oracle vs Free-Running Phase Alignment（第一部分）

## 背景与地位

P1/P2 交界的实验，把之前全部实验用的 Protocol A（oracle 前向加噪）第一次和
真实的 free-running 生成（Protocol B）连起来比较。目标：free-running 生成
失败，是不是因为采样器偏离了 oracle 走廊上看到的证据累积过程。

## 与已有实验的关系（重要）

这个实验的核心比较（`G_reverse` vs `G_oracle`，状态距离 `d_t`）**和本项目已有
的 `EXP-01v3`（`models/ELF-torch/experiments/probe_elf/probe_reverse_trajectory.py`，
**DONE**，仅 ELF）几乎是同一件事**。这次的增量价值是：(a) 把它搬到统一 adapter
框架上，让它能直接跑 LangFlow（EXP-01v3 是 ELF-only 的）；(b) 为后续的
"causal interpolation"（doc 里这个实验的第二部分）打基础。

**沿用了 EXP-01v3 已有的（不完全对称的）噪声约定**：真实生成从
`z_1 = randn(...) * config.denoiser_noise_scale`（ELF 默认 2.0）开始；配对的
oracle 路径复用**同一份**初始噪声张量，但 `z_t^oracle = t*x_final + (1-t)*eps`
**不会**再乘 `denoiser_noise_scale`。这个不对称在 `EXP-01v3` 自己的实现里就
已经存在（不是这次引入的新问题），这里只是原样沿用，保持跟已有结果风格一致、
可比较。

## 实现范围：doc 两个脚本都实现了

```text
experiments/phase_transition/compare_oracle_rollout.py       # 已实现
experiments/phase_transition/interpolate_oracle_rollout.py   # 已实现（后续补上）
```

**`compare_oracle_rollout.py`**：真实跑一遍 free-running ODE rollout（用
`adapter.solver_step` 循环，不是训练时的采样脚本），解码最终文本，重新编码成
`x_final`，配对构造 oracle 路径，在若干 checkpoint（`n_gen_steps` 网格里挑
`n_checkpoints` 个点）上比较 `G_reverse` vs `G_oracle`、rank、状态距离
`d_t = ||z_roll - z_oracle||`。

**`interpolate_oracle_rollout.py`**（doc 里的"causal interpolation"部分，
最初一版没做，后续补上了）：在中间 t 插值 `z_roll`/`z_oracle`，continue
solver 到 t=1，看最终 decode 和原始 rollout 的一致性。细节和 pilot 结果见
下面"Causal Interpolation"一节。

## 关于"预测目标"的说明

`G_reverse`/`G_oracle` 衡量的是"能不能读出模型自己生成的最终文本"，不是"能不能
读出真实 OpenWebText 续写"——这完全遵循 `EXP-01v3` 的协议（也是 doc 本节"encode
the final generated sequence into a clean endpoint"的字面要求），只是特意在
这里重复说明一下，避免被误读成"这是在测语言建模质量"。

## 状态

**DONE（ELF baseline/kd_cr/kd2 + LangFlow，64 条自生成样本，32 步 ODE rollout，
11 个 checkpoint）**

## Results

| model/ckpt | t_min G_reverse/G_oracle | t_max G_reverse/G_oracle | mean gap (oracle−reverse) | max gap |
|---|---|---|---|---|
| ELF baseline | 0.006 / 0.003 | 0.933 / 1.000 | 0.097 | 0.270 (t=0.41) |
| ELF kd_cr | 0.095 / 0.044 | 0.869 / 0.997 | 0.239 | 0.450 (t=0.52) |
| ELF kd2 | 0.141 / 0.071 | 0.820 / 0.999 | 0.271 | 0.463 (t=0.52) |
| LangFlow | 0.061 / 0.020 | 0.989 / 1.000 | 0.276 | 0.606 (t=0.28) |

**核心发现**：

1. **四个模型在最早的 t 上都是 `G_reverse > G_oracle`**（真实轨迹领先于
   oracle）——这个早期领先模式不是 baseline 独有的（`EXP-01v3` 只在 ELF 上
   看到过），LangFlow 和 KD checkpoint 都复现了同样的方向，说明"真实生成轨迹
   在极早期比 oracle 前向加噪更早显示出可读信号"是一个跨架构、跨训练方式的
   现象。
2. **反直觉的一点：KD 反而让平均 gap 变大，不是变小**（baseline 0.097 →
   kd_cr 0.239 → kd2 0.271）。也就是说，虽然 KD checkpoint 在早期领先幅度
   更大（t_min 时 G_reverse 从 0.6% 涨到 9-14%），但**中段轨迹整体更偏离
   oracle 走廊**，不是更贴近。
3. **这和 `EXP-PT6` 的"KD rollout 是放大而非纠正"发现是同一个机制在不同
   实验里的体现**：KD 模型的真实轨迹一旦偏离理想路径，续跑不会把它拉回来
   （EXP-PT6），所以这里观察到的中段大 gap，很可能就是"偏离后没被纠正"累积
   的结果。**"KD 让模型更早、更决绝地承诺"和"KD 让真实轨迹更贴近 oracle
   理想路径"是两件不同的事，这次的结果只支持前者，不支持后者**——这是一个
   需要在论文里明确区分、避免混为一谈的结论。
4. **LangFlow 的 gap 最大**（mean 0.276，max 0.606）——与 EXP-PT2/PT6 里
   LangFlow 稳定性最差的发现方向一致。

## 严谨性补强：样本量翻倍（N=64→128，全部 4 模型，2026-07-26）

和 EXP-PT6 一起，这是本 suite 里样本量最小的两个实验之一，用户要求专门
提高规模。把 `compare_oracle_rollout.py` 的 N 从 64 提到 128（其余参数不变：
`n_gen_steps=32`, 11 个 checkpoint），全部 4 个模型重跑（label=`n128`）：

| model/ckpt | t_min G_reverse/G_oracle | t_max G_reverse/G_oracle | mean gap | max gap |
|---|---|---|---|---|
| ELF baseline | 0.007 / 0.002 | 0.931 / 1.000 | **0.100** | 0.256 (t=0.44) |
| ELF kd_cr | 0.094 / 0.044 | 0.864 / 0.997 | **0.237** | 0.449 (t=0.44) |
| ELF kd2 | 0.150 / 0.078 | 0.817 / 0.998 | **0.269** | 0.474 (t=0.44) |
| LangFlow | 0.075 / 0.027 | 0.988 / 1.000 | **0.257** | 0.588 (t=0.28) |

（对照 N=64：baseline 0.097/0.270，kd_cr 0.239/0.450，kd2 0.271/0.463，
LangFlow 0.276/0.606。）

**结论**：

1. **两个核心发现在 N=128 上都完整复现**：(a) 全部 4 个模型在最早的 t 上
   `G_reverse > G_oracle`（真实轨迹仍然领先于 oracle）；(b) **mean gap 的
   `baseline < kd_cr < kd2` 这个"KD 让 gap 变大"的排序完全不变**（0.100 <
   0.237 < 0.269，几乎与 N=64 的数字逐位吻合）。
2. ⚠️ **一个需要如实报告的小变化**：N=64 时 LangFlow 的 mean gap
   （0.276）是四个模型里最大的；N=128 时 LangFlow（0.257）**略低于 kd2**
   （0.269），排到了第三位。LangFlow 仍然远高于 baseline、和两个 KD
   checkpoint 处在同一量级，"LangFlow 稳定性差"这个大方向不变，但
   "LangFlow 是四者中 gap 最大"这个精确排名不是一个稳定的结论，样本量
   翻倍后 kd2/LangFlow 谁更大发生了轻微的名次交换，解读时不应该把"哪个
   模型 gap 绝对最大"当成一个精确结论来引用，只能说"KD 和 LangFlow 都
   显著大于 baseline，量级相近"。

详见 `results/phase_transition/<model>/<checkpoint>/oracle_rollout_comparison_n128.json`。

## Causal Interpolation（doc 第二部分，后续补上了）

`interpolate_oracle_rollout.py`：在 `compare_oracle_rollout.py` 同一套
free-running rollout + paired oracle 构造基础上，在某个中间 t（默认 0.4）
把 `z_roll` 和 `z_oracle` 按 `lambda` 插值，**用同一个（确定性）ODE solver
继续跑到 t=1**，比较续跑后的最终 decode 和原始（未插值）free-running
rollout 的最终 decode 有多一致。因为两个 adapter 的 `solver_step` 都是
确定性 ODE（没有中途注入随机噪声），doc 要求的"matched future randomness"
自动满足（没有未来随机性需要匹配）。

**质量指标的说明**：因为 PT7 全程用的目标是"free-running 轨迹自己生成的
最终文本"（不是外部真值，这和 `EXP-01v3`/PT7 第一部分完全一致），所以
`lambda=1` 时"一致性"几乎是同义反复地高（因为 `z_oracle` 本来就是从这个
最终文本反推出来的）。真正有信息量的是 `lambda=0.25/0.5/0.75` 有没有已经
表现出有意义的、大致单调的提升——如果只有 `lambda=1` 才跳升，说明"往 oracle
方向靠近"本身没有因果作用，只是终点定义的产物。

⚠️ 续跑分支点处 `sc_state` 被重置为 None（零），包括 `lambda=0` 这一支——
所以就连"纯粹继续 free-running"这一支，也不是原始单次 rollout 的完美重放
（原始 rollout 在分支点会携带一个真实的 self-cond 状态往下传）。这会给
`lambda=0` 附近的数字引入一些噪声，解读小效应时需要留意。

## Results（正式规模：N=32，t_intervene≈0.4，32 步生成 + 12 步续跑）

| model/ckpt | λ=0 | λ=0.25 | λ=0.5 | λ=0.75 | λ=1.0 | monotone? |
|---|---|---|---|---|---|---|
| ELF baseline | 0.800 | 0.949 | 0.949 | 0.935 | 0.920 | 否（0.5 后小幅下降） |
| ELF kd_cr | 0.716 | **0.828**（峰值） | 0.707 | 0.489 | **0.134**（崩溃） | 否（0.25 后大幅下降） |
| ELF kd2 | 0.719 | **0.795**（峰值） | 0.711 | 0.703 | 0.582 | 否（0.25 后持续下降） |
| LangFlow | 0.714 | 0.979 | 0.989 | 0.992 | 0.994 | **是**（干净单调） |

**核心发现（比原来预想的更极端）**：

1. **λ=0→0.25 全部模型都有明显提升**（ELF baseline +15pp，kd_cr +11pp，kd2
   +8pp，LangFlow +26pp）——小比例的插值就已经因果性地改善了 agreement，
   支持 doc 的判定规则。
2. **λ=0.25 之后，四个模型分成三种截然不同的模式**：
   - **LangFlow**：继续单调上升，一路涨到 0.994。
   - **ELF baseline**：小幅回落（0.949→0.920），温和的"过头"。
   - **ELF kd_cr/kd2**：**灾难性崩溃**——kd_cr 从峰值 0.828 一路跌到 λ=1 时
     的 **0.134**（几乎等于随机水平），kd2 跌到 0.582。也就是说对 KD
     checkpoint，**完全走到纯 oracle 状态（λ=1）后续跑，续跑结果和原始
     free-running 轨迹的答案几乎完全对不上**。
3. **这次的崩溃幅度和 EXP-PT6 的"KD rollout 是放大而非纠正"、EXP-PT7 第一
   部分的"KD 平均 gap 反而更大"完全吻合，而且给出了目前为止最强的因果证据**：
   KD 模型的续跑对状态扰动極度敏感——一旦状态被强行推到一个和它自己会走的
   路径不同的地方（纯 oracle、且 self-cond 状态被清零），它不会像 LangFlow
   那样"收敛回正确答案"，而是会被带到一个完全不同的吸引子。**"KD 让模型
   更早/更决绝地承诺"和"KD 让模型对状态扰动更鲁棒"是两件相反的事**——这是
   贯穿 PT2/PT5/PT6/PT7 四个实验、反复出现的同一个结论，这次是最戏剧化的
   一次展示。
4. ⚠️ 这个实验里 self-cond 状态在分支点被清零（见"方法论"一节）——KD
   checkpoint 对 self-cond 状态的依赖可能本来就比 baseline/LangFlow 更强
   （因为 KD 训练本身很可能特别针对 decode branch/self-conditioning 路径做了
   优化），所以这个"清零 SC"的简化操作对 KD 的伤害可能天然比对 baseline
   更大，这是解读"崩溃"时需要考虑的一个混淆因素，不能完全排除是这个简化
   造成的人工放大，而不是纯粹的"KD 更脆弱"。

## 严谨性补强：多种子复核（kd_cr causal interpolation，最戏剧化的发现）

kd_cr 在 λ=1 时 agreement 崩溃到 0.134（远低于 λ=0.25 的峰值 0.828）是本次
新增的 causal interpolation 里最令人惊讶的结果，原本只有 seed=42 一次证据。
追加 seed=123、seed=456：

| seed | λ=0 | λ=0.25 | λ=0.5 | λ=0.75 | λ=1.0 |
|---|---|---|---|---|---|
| 42（原始） | 0.716 | 0.828 | 0.707 | 0.489 | **0.134** |
| 123 | 0.696 | 0.818 | 0.738 | 0.558 | **0.290** |
| 456 | 0.725 | 0.834 | 0.748 | 0.519 | **0.224** |

**结论：三个种子的 λ=0/0.25 几乎完全一致（差异<2pp），λ=0.5 之后开始明显
下降的方向在三个种子里完全一致，λ=1 的绝对崩溃程度有真实的种子间波动
（0.134/0.290/0.224，均值≈0.216，标准差≈0.08），但"λ 越大、kd_cr 的
agreement 反而越差"这个**方向性、定性的崩溃现象在三个独立种子里都稳健复现，
不是 seed=42 的巧合**。三次的 λ=1 均值（0.216）仍然远低于 λ=0.25 的峰值
（约 0.827），"完全走到纯 oracle 状态不是最优、中间插值更好"这个核心结论
是可信的；但如果论文里要引用 λ=1 的具体崩溃幅度，应该用多种子均值
（≈0.22，而不是单次的 0.134）并注明种子间标准差（≈0.08），避免把单次
运行的极端值当成精确数字来引用。

## 严谨性补强：样本量翻倍（causal interpolation，N=32→64，全部 4 模型，2026-07-26）

和 PT6、PT7 第一部分一起，把 causal interpolation 的 N 从 32 提到 64（其余
参数不变：`n_gen_steps=32`, `t_intervene=0.4`, `n_continue_steps=12`），全部
4 个模型重跑（label=`n64`）：

| model/ckpt | λ=0 | λ=0.25 | λ=0.5 | λ=0.75 | λ=1.0 | monotone? |
|---|---|---|---|---|---|---|
| ELF baseline N=32 | 0.800 | 0.949 | 0.949 | 0.935 | 0.920 | 否 |
| ELF baseline **N=64** | 0.798 | 0.948 | 0.949 | 0.937 | 0.921 | 否（几乎逐位吻合） |
| ELF kd_cr N=32 | 0.716 | 0.828（峰值） | 0.707 | 0.489 | **0.134**（崩溃） | 否 |
| ELF kd_cr **N=64** | 0.717 | 0.831（峰值） | 0.729 | 0.483 | **0.165**（崩溃） | 否 |
| ELF kd2 N=32 | 0.719 | 0.795（峰值） | 0.711 | 0.703 | 0.582 | 否 |
| ELF kd2 **N=64** | 0.709 | 0.798（峰值） | 0.695 | 0.676 | 0.552 | 否 |
| LangFlow N=32 | 0.714 | 0.979 | 0.989 | 0.992 | 0.994 | 是 |
| LangFlow **N=64** | 0.690 | 0.982 | 0.990 | 0.993 | 0.995 | 是 |

**结论：N=64 在四个模型、五个 λ 值上都和 N=32 几乎逐位吻合（大多数差异
<2pp），三种定性模式（baseline 温和过头、LangFlow 干净单调上升、kd_cr/kd2
在 λ=0.25 后灾难性崩溃）完全复现，不是 N=32 的小样本噪声**。特别是 kd_cr
在 λ=1 的崩溃值（N=32: 0.134 → N=64: 0.165）落在此前 seed=123/456 复核
建立的种子间波动范围（0.134-0.290，均值≈0.216）**内部**——这次的样本量
翻倍结果本身可以看作该分布里的又一个采样点，进一步支持"应该引用多次
运行的均值/范围而不是单次点估计"这条已有建议，同时确认"崩溃"这个方向性
现象与样本量无关，是真实效应。详见
`results/phase_transition/<model>/<checkpoint>/oracle_rollout_interpolation_n64.json`。
