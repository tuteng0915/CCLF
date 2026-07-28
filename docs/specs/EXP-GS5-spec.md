# EXP-GS5 Spec — Collective Coupling and Correlation Length

## 背景与地位

原始 doc 第 9 节 GLOBAL-5，P2 阶段，也是本轮把 `global_state_formation_experiment_suite.md`
十个 GLOBAL 实验全部跑一遍 pilot 的最后一项。判断阶段转移是否表现为**集体组织化**（多个位置
的证据同步跳变），而不是独立 token crossing 的平均——用位置间的 margin-increment
相关性（correlation length）和跨序列的 susceptibility 来测。

## 0. 复用

- 默认竞争者 `f_i`、margin 定义：复用 `phase_transition` 系列 `EXP-PT1/PT2` 和本 GS 系列
  `EXP-GS2/GS8` 统一使用的 `f_i = earliest-time native top-1`，`m_i(t) = ell(y_i) - ell(f_i)`。
- 空间相关性方法学精神上参考 `EXP-26v2`（LangFlow 的 Moran's I 空间自相关分析），但这里测
  的是**margin increment 的位置间相关**（原始 doc 的 `C_t(d)`），不是"正确性"这个二元变量
  的空间自相关，两者不是同一个量，只是"位置距离 vs 相关性"这个分析范式相同。

## 1. 指标实现

- **Margin trajectory**：对 `N` 个文档、单一固定噪声、`t` 稠密网格（`t ∈
  linspace(0.05, 0.85, 15)`，比原始 doc 建议的 101 点稀疏得多，见第 4 节），算
  `m_i(t) = ell(y_i,t) - ell(f_i,t)`（`f_i` 固定为第一个 t 上的 native top-1，不随 t 变化）。
- **Increment**：`Delta m_i(t_k) = m_i(t_{k+1}) - m_i(t_k)`。
- **位置相关 `C_t(d)`**：对每个相邻 t-pair、每个位置距离 `d in {1,...,D}`（`D=20`），把
  `(sequence, position)` 当作观测单位，计算 `Corr(Delta m_i(t_k), Delta m_{i+d}(t_k))`
  （跨所有文档、所有满足 `i+d<L` 的位置 pooled 到一起算一个 Pearson 相关系数）。
- **Correlation length**：`xi(t_k) = sum_{d=1}^{D} max(C_t(d), 0)`。
- **Global susceptibility**：`bar_m(t) = mean_i m_i(t)`（每条序列一个标量），
  `chi(t) = L * Var_{sequences}(bar_m(t))`（跨文档取方差，`L` 是位置数）。
- **对照**：**shuffle-position control**——在计算 `C_t(d)` 之前，对每条序列的位置顺序做
  一次随机置换（同一 t 内所有位置一起打乱，跨 t 用不同的打乱），重新计算 `C_t(d)`/`xi(t)`。
  如果真实数据的 `xi(t)` 明显高于 shuffle 后的 `xi(t)`，说明相关性不是"任何随机分配位置都会
  产生的数值巧合"，而是真实的空间/位置结构。

## 2. 判定

原始 doc 预期"collective transition"信号：转移区附近 `xi(t)` 上升、susceptibility 达到峰值、
真实数据的 `xi(t)` 明显高于 shuffle 对照。

## 3. 数据与规模（pilot）

- ELF baseline，`eval_exp37c_baseline.yml`（1024-token，与 GS1/GS3 一致）。
- `n_samples=64`（与 GS3/GS8 同一量级）。
- `t` 网格：15 点，`linspace(0.05, 0.85, 15)`（覆盖已知 commitment cliff 前后，不用
  clean-ref，因为这里只关心 increment 的相关结构，不需要"达到 clean 表现"这个概念）。
- `D=20`（位置距离上限）。

## 4. 已知简化

1. ⚠️ `t` 网格 15 点，远比原始 doc 建议的 101 点密集网格稀疏——`xi(t)` 曲线的精细结构
   （比如"恰好在哪个 t 达到峰值"）在这个密度下只能给出粗略位置。
2. ⚠️ `f_i` 固定用第一个 t 上的 native top-1（不随 t 更新），沿用 PT1/PT2 的约定，但意味着
   越往后的 t，`f_i` 作为"默认竞争者"的代表性可能下降。
3. ⚠️ 只做 shuffle-position 一种对照（原始 doc 建议的 shuffle sequences / frequency-matched
   positions / remove function words / fixed token-type composition / independent-position
   denoiser baseline 都没做）。
4. Pilot 规模（64 样本），数字仅用于判断方向。

## 5. 脚本与输出

```text
experiments/global_state/analyze_collective_coupling.py
```

```text
results/global_state/<model>/<checkpoint>/collective_coupling_<label>.json
```

## 状态

**Pilot DONE**（ELF baseline，n=64，15 个 t 点，`D=20`，GPU1，
`logs/global_state/gs5_elf_baseline_pilot.log`，输出
`results/global_state/elf/baseline/collective_coupling_pilot.json`）——GS 系列第四个
干净正面结果（另外三个是 GS6、GS7、GS8）。

## Results（pilot：ELF baseline，64 序列，seq_len=1024，15 点 t∈[0.05,0.85]）

`mean(m)`（population 平均 margin）在 `t=0.221→0.279` 之间由负转正
（−18.657→+42.148），这是这批样本的**平均意义上的 commitment cliff 位置**。

| t → t_next | xi (real) | xi (shuffled) | excess (real−shuffled) | chi |
|---|---|---|---|---|
| 0.05→0.11 | 4.921 | 3.969 | 0.952 | 14655 |
| 0.11→0.16 | 1.342 | 0.770 | 0.572 | 8745 |
| 0.16→0.22 | 1.653 | 0.549 | 1.104 | 10597 |
| 0.22→0.28 | 1.604 | 0.640 | 0.964 | 22029 |
| 0.28→0.34 | 1.676 | 0.774 | 0.902 | **102860** |
| 0.34→0.39 | 1.939 | 0.799 | **1.140** | **112742** |
| 0.39→0.45 | 0.888 | 0.319 | 0.569 | 64061 |
| 0.45→0.51 | 0.556 | 0.201 | 0.355 | 51207 |
| 0.51→0.85（后 8 段，趋势平稳） | 0.56–1.15 | 0.34–0.93 | 0.10–0.32 | 44000–52000 |

**解读**：

1. **真实数据的 `xi(t)` 在全部 14 个相邻 t-pair 上都高于 shuffle-position 对照**
   （无一例外）——说明 margin increment 的位置间相关性是真实的空间结构，不是把
   任意数字摆在任意位置都会出现的数值巧合。这是本 pilot 最基本、也最重要的正面结果。
2. **`chi(t)`（全局 susceptibility）在 `t=0.28–0.39`（cliff 之后不远）出现明显尖峰**
   （102860、112742，是平稳期 44000–52000 的 2–2.5 倍），随后回落到平台——**和原始 doc
   预期的"susceptibility 在转移区附近达到峰值"完全吻合**。
3. **`xi` 的 excess（真实减 shuffle）同样在 `t=0.34–0.39` 达到全程最大值（1.140）**，
   和 `chi` 的峰值区间基本重合。两个独立指标（correlation length 的"真实-shuffle 差"，
   和跨序列的方差）**在同一个 t 区间同时达峰**，互相印证。
4. **峰值区间（`t≈0.28–0.39`）略晚于 population-mean margin 的过零点（`t≈0.22–0.28`）**——
   和 GS7 发现的"CKA(oracle,rollout) 最低点在 `t=0.50`，晚于已知 cliff `t≈0.20–0.30`"
   是同一个模式：**"集体重组"类信号（不管是 CKA 分歧还是这里的 correlation-length/
   susceptibility 尖峰）往往滞后于"平均意义上的 token 翻转"一小段 t**，提示 population
   平均的过零点只是个体位置各自独立翻转的汇总统计，真正的集体协调/重新组织发生在那之后，
   与 GLOBAL-5 的核心假设（阶段转移是集体现象、不是独立 crossing 的简单平均）方向一致。
5. `t=0.05→0.11` 这第一段的 `xi`（4.921）和 `chi` 都不是全程最低（`xi` 反而是全程数值
   最大的一段，虽然 excess 不是最大），需要谨慎——这可能是极早期噪声主导阶段的一个不同性质
   的相关结构（比如和初始噪声本身的谱特性有关，而不是"承诺过程的集体性"），不应该和
   `t=0.28–0.39` 那个更晚的、和已知 cliff 位置更吻合的峰值混为一谈。

## 下一步

1. 补充原始 doc 建议的其它对照（shuffle-sequence、frequency-matched positions、
   remove function words），确认 `t=0.05→0.11` 这段异常高的 `xi` 是不是需要单独解释。
2. 把 `t` 网格加密到更接近原始 doc 建议的密度，精确定位 `chi(t)` 峰值和 `xi` excess
   峰值是否真的重合在同一个 t，还是只是本 pilot 15 点网格下的巧合对齐。
3. 对 kd_cr/kd2 重复——EXP-01v3/EXP-10 等已经证明 KD 大幅提前 commitment cliff，值得
   检验 KD 是否也让这里的 collective coupling 峰值同步提前，还是把"集体性"这个特征本身
   减弱/增强了。
