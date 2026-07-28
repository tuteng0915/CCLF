# EXP-GS15 Spec — Residual Organization Trajectory

## 背景与地位

用户在看完 `docs/global_state_formation_synthesis.md`（综合解读文档）后提出一条新的
核心问题链：

```
high-rank residual 从哪里来 -> 如何被 vector field 组织 -> 何时推动 token crossing
  -> 为何在 rollout 中失败
```

`EXP-GS15` 是这条链的第一环，也是用户明确标注为"全篇核心图"的实验：**高秩位置特异
residual `R_t = Z_t - 1*mu_t^T` 是逐渐平滑地向最终 lexical configuration 靠近，
还是在某个阶段突然重组？** 直接在真实 free-running trajectory 上追踪
`A(t) = CKA(R_t, R_star)`，并用多个对照排除"越接近终点当然越像终点"这种同义反复。

## 0. 复用

- `rollout_with_checkpoints_and_sc`（`EXP-GS14` 里实现的、同时保存真实 `(Z_t, SC_t)`
  的自由生成函数）——本实验在此基础上，在每个 checkpoint 额外做一次
  `forward_state` 拿 `predicted_clean`（模型对该 checkpoint 的去噪估计）。
- `linear_cka`、`effective_rank`（`analyze_low_rank_modes.py`，GS3/GS12 已验证）。
- `make_oracle_state`（GS1 起统一约定），用于构造 paired oracle 对照。

## 1. 指标（pilot 先做两个，最省算力、最有区分度）

原始建议里的四个指标（centered CKA / Procrustes-aligned distance / final-time probe
transfer / residual effective rank）里，pilot 先做**centered CKA**（核心判定量）和
**residual effective rank**（辅助诊断，GS3 已经验证过其计算方式）。Procrustes 对齐
距离和 final-time probe transfer（在 `R_star` 上训练 token probe、迁移到 `R_t`）
计算更贵（前者要解正交 Procrustes 问题，后者要训练一个 ~32k 类的线性分类器），留给
GS15 出信号之后的第二轮。

- **Centered CKA**：`A(t) = CKA(R_t, R_star)`，`R_star` = 该轨迹终点 checkpoint
  自己的中心化 residual（不是外部 ground truth，因为自由生成没有外部参照，和
  `EXP-GS7` 的"用自己的终点做 clean 参照"是同一个约定）。
- **Residual effective rank** `r_eff^R(t)`：判断 residual 是逐渐降秩、逐渐增大
  token-relevant 方向，还是总体秩不变只是方向重排。

## 2. 对照（避免"越接近终点当然越像"的同义反复）

对每条真实轨迹，除了主结果（rollout 自身的 `R_t`），同时构造：

1. **Paired oracle**：用同一个初始噪声 `eps` 和该轨迹的终点 `x_clean^{rollout}`
   （复用 `EXP-GS7` 的定义）构造 `Z_t^{oracle} = t*x_clean^{rollout} + (1-t)*eps`，
   算它自己的中心化 residual `R_t^{oracle}`，对齐同一组 checkpoint t。
2. **线性插值对照**：`R_t^{linear} = R_{t0} + frac(t)*(R_star - R_{t0})`，
   `frac(t) = (t-t0)/(t_end-t0)`，`R_{t0}` 是该轨迹**第一个** checkpoint 的
   residual——这是纯粹的"匀速直线插值"基线，不含任何模型动力学。
3. **两条 representation 线**：raw `Z_t` 的 residual 和模型 `predicted_clean` 的
   residual（`EXP-GS11` 已经证明这两者行为经常不同，必须分开看）。

核心判定量 **excess organization**：

```
O_R(t) = A_rollout(t) - A_linear(t)
```

如果 `O_R(t)` 在某个窗口明显为正，说明真实轨迹在该窗口"比匀速直线插值更快"地组织
出终点结构，不是单纯"越往后走越接近终点"这个平凡几何事实。

## 3. 核心问题

1. `A_rollout(t)` 的形状：是从低到高单调平滑上升，还是存在一段明显加速区间？
2. 加速区间（如果存在）是否和 `EXP-GS5` 的 collective coupling 峰值（`t≈0.28-0.39`）、
   `EXP-GS14` 的 lexical consensus 快速收缩区间重合？
3. `O_R(t)`（相对线性插值的超额组织）在哪个区间最大？
4. raw 和 model 两条线是否表现出同一个定性模式（`EXP-GS11` 已经证明这两者不能想当然
   地认为一致）？
5. Paired oracle 的 `A_R^{oracle}(t)` 和 rollout 的 `A_R^{rollout}(t)` 差距有多大、
   在哪个区间最大——为 `EXP-GS19`（Oracle-Rollout Residual Gap，下一批实验）打基础。

## 4. 数据与规模（pilot）

- ELF baseline，`eval_exp37c_baseline.yml`（1024-token）。
- `n_traj=4`（真实自由生成，和 GS14 同一量级，生成本身是主要算力开销）。
- checkpoint t：`[0.05, 0.20, 0.28, 0.38, 0.50, 0.65, 0.85, 0.99]`（8 点，比 GS14 的
  3 点密，覆盖已知 cliff 前后到 collective coupling 峰值区间）。
- `full_n_steps=32`（标准生成质量，同 GS7/GS14）。

## 5. 已知简化

1. ⚠️ `n_traj=4` 很小，`A(t)` 曲线的形状（尤其是否存在"加速区间"）需要更大样本才能
   确认不是单条轨迹的噪声。
2. ⚠️ 只做 centered CKA + effective rank，不做 Procrustes 对齐距离和 final-time
   probe transfer（留给后续，见第 1 节）。
3. ⚠️ Paired oracle 的 self-conditioning 仍然冷启动为零（延续 GS1-GS13 的已知简化），
   不是从真实轨迹迁移过来的 SC。
4. Pilot 规模，数字仅用于判断"是否存在加速组织区间"这一方向性问题。

## 6. 脚本与输出

```text
experiments/global_state/analyze_residual_organization.py
```

```text
results/global_state/<model>/<checkpoint>/residual_organization_<label>.json
```

## 状态

**Pilot DONE — 出现与假设相反的信号，如实报告**（ELF baseline，4 条真实自由生成轨迹，
8 个 checkpoint，`full_n_steps=32`，GPU1，`logs/global_state/gs15_elf_baseline_pilot.log`，
输出 `results/global_state/elf/baseline/residual_organization_pilot.json`）。

## Results（pilot：4 条轨迹，均值）

| t | A_rollout(raw) | A_model | A_oracle | A_linear | **O_R** | r_eff(raw) | r_eff(model) |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.131 | 0.255 | 0.142 | 0.131 | 0.000 | 474.0 | 220.0 |
| 0.20 | 0.137 | 0.498 | 0.234 | 0.198 | **−0.061** | 474.0 | 209.3 |
| 0.28 | 0.156 | 0.671 | 0.336 | 0.286 | **−0.130** | 473.9 | 232.0 |
| 0.38 | 0.239 | 0.827 | 0.520 | 0.461 | **−0.223** | 472.9 | 264.4 |
| 0.50 | 0.495 | 0.869 | 0.758 | 0.719 | **−0.225** | 467.3 | 279.7 |
| 0.65 | 0.842 | 0.892 | 0.932 | 0.924 | −0.082 | 440.9 | 287.3 |
| 0.85 | 0.968 | 0.926 | 0.994 | 0.994 | −0.026 | 346.6 | 292.2 |
| 0.99 | 1.000 | 1.000 | 1.000 | 1.000 | −0.000 | 292.1 | 306.2 |

**解读（如实报告，和假设方向相反）**：

1. **`O_R(t)` 在全部中间 checkpoint 上都是负的**，在 `t=0.38-0.50` 附近最负
   （−0.223, −0.225）——**这和"真实 rollout 比纯线性插值更快组织出终点结构"这个
   假设方向相反**。真实轨迹的 residual，在轨迹中段，反而比"起点和终点残差之间的
   匀速直线插值"**更不像**终点残差。这不是一个可以被"加速组织窗口"叙事直接支持的
   结果，需要如实报告，不能选择性忽略。
2. `A_oracle(t)` 全程都比 `A_rollout(t)` 高（例如 `t=0.5`: `0.758` vs `0.495`），
   且和 `A_linear(t)` 比较接近——paired oracle 路径的 residual 组织进度比 free-running
   rollout 更接近（甚至优于）一条朴素直线插值，而 free-running rollout **明显落后于
   直线插值**。这是本实验最重要的发现：**"落后于 oracle"这件事，第一次被直接量化到
   residual alignment 这个层面，而且落后的幅度比落后于一个纯几何基线（线性插值）
   还要大**——free-running 不仅"知道得比 oracle 少"，连"朝终点走的效率"都不如一条
   假设的直线。
3. `A_model(t)`（模型自己的 `predicted_clean` 输出）上升得远比 `A_rollout(t)`（原始
   噪声输入）早、快（`t=0.28` 时 `0.671` vs `0.156`）——但这不是"模型很早就组织好了
   residual"的证据：结合 `r_eff_model` 全程明显低于 `r_eff_raw`（`t=0.28`: 232 vs
   474），更可能的解释是模型的去噪输出从早期起就是一个**低秩、偏通用/众数化**的估计
   （呼应 `EXP-GS11` 的"prior-dominated compression"），这种低秩估计本身就更容易和
   它自己的终点（同样偏通用）"看起来相似"，不代表捕捉到了真实的、高秩的、文档特异的
   最终 lexical residual。
4. `r_eff(raw)` 几乎全程维持在 467-474（接近满秩），直到 `t=0.85` 才明显下降到
   346.6——和 `EXP-GS3` 发现的"raw 状态几乎全程满秩"一致；residual 的"组织"更可能
   发生在方向重新排列上，而不是整体降秩上。

## 结论：与用户假设的对比

用户提出的判定标准是"如果 `O_R(t)` 在某个窗口明显为正，说明存在超出线性插值的加速
组织"。**本 pilot 没有观察到这个信号，观察到的是相反方向**。这本身是一个有价值的
负结果，且和已有证据链吻合而不是孤立的：

- 和 `EXP-GS7`（`G_token` 的 oracle-rollout gap 在 token 层面高达 15 倍）方向一致——
  这里从 residual alignment 的角度**独立地**发现了同一种"free-running 落后"现象，
  而且落后的参照系比 GS7 的"oracle"更严格（一条纯几何直线，不需要任何模型知识）。
- 不支持"高秩 residual 在某个窗口加速向终点组织"这个具体机制表述，需要修正
  `docs/global_state_formation_synthesis.md` 第 9.1/9.2 节提出的"如果 `O_R(t)` 早期
  弱、`t≈0.25-0.4` 附近快速上升"这个预期发现方向。

## 下一步

1. **优先排查这是否是 `R_star`/`A_linear` 定义方式的产物**：`R_star` 是轨迹自己终点
   的 residual，而"线性插值"混合的是 residual 本身（不是原始状态）——如果 residual
   本身的几何（不同 t 下奇异方向持续变化，见 `r_eff` 几乎不降）导致"中途"的 residual
   在原则上不可能落在 `R_{t0}` 和 `R_star` 的插值线附近（因为真实动力学的路径在更高维
   空间里绕路），那么 `O_R(t)<0` 可能是一个**关于路径几何形状、而不是关于"组织速度"**
   的陈述——需要换一个不依赖线性插值假设的对照（比如同一 batch 内随机配对的另一条轨迹
   的 residual，看 `A_rollout` 是否比"随机配对"更高，这才是排除"路径几何本来就绕远"
   这个混淆的更干净对照）。
2. 扩大 `n_traj`（当前 4），确认 `O_R(t)` 的负值模式是否稳定。
3. 把这个"free-running 比线性插值更慢组织 residual"的发现和 `EXP-GS5` 的 collective
   coupling 峰值（`t≈0.28-0.39`）对齐看——`O_R(t)` 最负的区间（`t=0.38-0.50`）和
   `EXP-GS5` 的峰值区间部分重合，也许可以解读为"这正是 free-running 最需要
   collective coordination 才能补上和线性基线的差距的阶段"，而不是"加速组织"。
4. 补 Procrustes 对齐距离和 final-time probe transfer（原计划的另外两个指标），确认
   `O_R(t)<0` 这个模式不是 CKA 这一个指标的特有 artifact。

## 正式跨架构验证：`O_R_model`（用 model 表征代替 raw，v2）

`EXP-GS7`/`EXP-GS15` 的 LangFlow 复现发现：raw-state CKA（`A_rollout`/`A_oracle`/
`A_linear`）在 LangFlow 上从 `t=0.05` 起就 ≈0.99+，整个指标饱和不提供信息（见
`docs/global_state_formation_synthesis.md` 第 10.1 节）。为了在 LangFlow 上真正检验
"负 `O_R`"这个假说，脚本新增了完全平行的 **model-based** 版本：`A_linear_model` 用
`predicted_clean` 的残差做线性插值对照，`O_R_model = A_rollout_model - A_linear_model`
（`analyze_residual_organization.py` v2，同一份代码同时输出 raw 和 model 两套指标，
不影响原有 raw 结果的可复现性——重跑 ELF 得到和原 pilot 完全一致的 raw 数字）。

**结果（n=4 条真实轨迹，ELF 和 LangFlow 各自独立跑）**：

| t | O_R_model (ELF) | O_R_model (LangFlow) |
|---|---|---|
| 0.05 | 0.000 | 0.000 |
| 0.20 | −0.237 | −0.437 |
| 0.28 | −0.219 | **−0.472**（LangFlow 最负点） |
| 0.38 | −0.131 | −0.409 |
| 0.50 | −0.117 | −0.301 |
| 0.65 | −0.104 | −0.171 |
| 0.85 | −0.074 | −0.051 |
| 0.99 | 0.000 | 0.000 |

**两个架构都是全程负值，且都在轨迹中段最负、随后稳步恢复到 0**——这是"负 `O_R` /
free-running 组织 residual 比线性插值更慢"这个此前只在 ELF 上观察到的现象，**第一次
在架构完全不同的 LangFlow 上独立复现**（LangFlow 最负点在 `t≈0.28`，比 ELF 的
`t≈0.20` 略晚，量级上也更负——`-0.472` vs `-0.237`——但方向和整体形状一致）。

**这大幅提高了"晚期崩塌而非渐进组织"这个假说的可信度**：不再是单一架构的孤立观察，
而是两个训练方式、backbone、noise schedule 都不同的模型共享的动力学性质。ELF 的 raw
版本（原 pilot）呈现更明显的 U 形（谷底在 `t=0.38-0.50`），model 版本（v2，两个架构）
则更像"早期出现大缺口、随后单调恢复"，没有 raw 版本那种在谷底之后先小幅反弹的细节——
说明具体形状对"用哪种表征测量"有一定敏感性，但**"全程为负、中段最负"这个定性结论
在四种测量方式（ELF-raw/ELF-model/LangFlow-raw-饱和不可用/LangFlow-model）里，凡是
指标本身没有饱和的三种测量下都一致**。
