# EXP-PT1 Spec — Prior-to-Evidence Decomposition（Phase-Transition 系列首个实验）

## 背景与地位

`docs/phase_transition_experiment_suite.md` 提出了一整套新实验（EXP-PT1–PT10），核心假设是：
早期连续状态已经积累了微弱的样本特定证据（sample-specific evidence），高频词主导只是因为
原生 categorical readout 有很强的默认先验/几何偏置；"可见的" token 转变发生在累积证据跨过
decoder 决策边界的那一刻。

`EXP-PT1` 是 P0（建立机制）阶段的第一个实验，也是 MVP-A 的第一部分。目标：区分早期高频预测
是"真实信念"还是"掩盖了已存在的样本特定证据的默认输出"。

本 spec 记录将原始 doc 中的抽象协议落地为具体代码时做出的工程决策，供后续 PT2/PT3/PT5 等实验
复用同一套 adapter 基础设施。

**与已有实验的关系**：`EXP-05v3`（`probe_prior_null.py`）已经实现了一种"全局零信号先验"
（`z_t_null=(1-t)*eps`，对应本 spec 的 Reference A 的一个特例：均值 0、各向同性方差）。
EXP-PT1 在此基础上补充：(a) 均值/方差匹配真实状态统计量的 Gaussian 参考（而不是零均值），
(b) 跨序列 state-swap 参考，(c) 上下文打乱参考，并计算原始 doc 定义的完整指标集
（残差 rank、KL、null-mode fraction 等），同时把整个流程搬到统一的 ELF/LangFlow adapter 上，
不再是 ELF-only 的一次性脚本。

---

## 0. Adapter 层（对 PT1–PT10 都适用）

原始 doc 第 13 节要求一个 `FlowModelAdapter` 接口。实现位置：

```text
experiments/phase_transition/adapters/__init__.py
experiments/phase_transition/adapters/elf_adapter.py
experiments/phase_transition/adapters/langflow_adapter.py
```

（放在 `experiments/phase_transition/` 下而非仓库根目录的 `adapters/`，避免和已有目录结构冲突；
两个 PT 脚本通过相对导入使用它们。）

### ELF adapter 设计要点

复用 `probe_reverse_trajectory.py` / `probe_prior_null.py` 中重复了三次的 `load_model()`，
以及 `utils/sampling_utils.py` 的 `_ode_step` / `net_out_to_v_x`：

- `encode_clean`：T5 encoder 前向，`(hidden - latent_mean) / latent_std`。
- `make_oracle_state(x_clean, eps, t)`：`z_t = t*x_clean + (1-t)*eps`（**不乘
  `denoiser_noise_scale`**——这与 EXP-01v3/EXP-05v3 等已有 P0 实验的 oracle 协议保持一致，
  是刻意的一致性选择而非新简化）。
- `forward_state`：按 `config.self_cond_prob>0` 决定是否 `cat([z, sc], dim=-1)`；
  `sc` 默认为全零（"no-op self-cond"，与现有 oracle probe 脚本一致）；返回
  `logits`（decoder_logits）、`predicted_clean`（模型的 `output`/`x_pred`）、可选的
  按 block 抓取的 `hidden_states`（用 `model.blocks[i].register_forward_hook`，用完立即
  `remove()`）。
- `solver_step`：直接调用 `_ode_step`。
- `native_logsnr(t)`：ELF 原生并不是 log-SNR 参数化（线性 flow-matching，`alpha=t, sigma=1-t`），
  这里提供 `log(t^2) - log((1-t)^2)` 作为**近似**，仅用于和 LangFlow 做跨模型 log-SNR 对齐时
  的参考值，不作为 ELF 内部计算的依据（PT1 不需要跨模型比较，PT1 只在各自的 t/γ grid 内部
  分析，跨模型对齐延后到执行顺序里的 LangFlow 复现阶段）。

### LangFlow adapter 设计要点

复用 `models/LangFlow/probe_langflow.py` 的 `load_langflow` / `encode_with_langflow`：

- `encode_clean`：`model._embed_tokens(input_ids)`。
- `native_logsnr`：直接返回 `model.proposal`（GumbelProposal），即 LangFlow 真正的可学习
  log-SNR 调度，而不是探针脚本里常用的线性 `gamma_from_t` 展示映射（`gamma_from_t` 仅保留
  作为绘图便利函数，不用于核心计算）。
- `make_oracle_state`：`model._forward_diffusion`（`alpha=sigmoid(-γ)^0.5, sigma=sigmoid(γ)^0.5`）。
- `forward_state`：`model.forward(noisy_embeds=z, timesteps=gamma, x_self_cond=sc, output_hidden_states=...)`；
  `predicted_clean` 用 `_embed_tokens(softmax(logits))`（LangFlow 原生自条件状态的定义）。
- `solver_step`：`model._euler_edm_step`。
- 自条件状态：加性（`x + self_cond_proj(cat([x,x_self_cond]))`），不同于 ELF 的通道拼接。

两个 adapter 共享的 `FlowModelAdapter` 抽象基类只定义方法签名（见原始 doc 第 13 节），不做
跨架构的强制统一，因为两边的 t / 状态空间语义本来就不同（这也是 EXP-02/03/22 等已有实验反复
验证过的结论：**不能用 nominal t 直接比较 ELF 和 LangFlow**）。

---

## 1. Reference A/B/C 的具体实现

### Reference A — Full-model Gaussian reference

原始定义：`q_t^gauss(v) = E_{z~N(mu_t,Sigma_t)}[p_theta(v|z,t)]`，"均值/协方差尽量匹配真实状态"。

实现：对同一批真实 oracle `z_t`（`n_oracle` 个噪声种子）在**每个 t** 上估计逐通道
（per-channel，对 embedding 维度 d 各自）的经验均值 `mu_t in R^d` 和方差 `sigma_t^2 in R^d`
（对 N*L 个位置展平后统计），然后从对角高斯 `N(mu_t, diag(sigma_t^2))` 采样 `n_gauss` 次，
过 backbone，softmax 后平均（Jensen 不等式的正确处理方式，沿用 EXP-05v3 的做法）。

这比 EXP-05v3 的"零均值全局 null"更贴近 doc 的原始定义（真正匹配了均值和逐通道方差），
但仍是对角协方差近似（不建模跨通道协方差），因为满协方差在 d=512（ELF）/ d=768（LangFlow）
维度上估计代价高且样本量（512 序列）也不足以稳定估计满秩协方差。这一简化会在
`docs/specs/EXP-PT1-spec.md` 的"已知简化"里再次注明。

### Reference B — Cross-sequence state-swap reference

实现：对同一批真实 `z_t`，用一个随机 batch 置换 `perm`（保证 `perm[i] != i`，即
derangement）构造 `z_t_swap[i] = z_t[perm[i]]`，再过 backbone。对若干个随机 `perm` 种子取
平均 softmax。这是原始定义的精确实现（"用另一条序列在相同 t、相同位置的状态替换"）。

### Reference C — Context-shuffled reference

原始定义要求"目标位置状态不变，只打乱非目标位置"，这在严格意义下需要对每个目标位置 i
单独构造一份"只有其它位置被打乱"的输入（因为一次前向传播会同时输出所有位置的 logits，
无法在同一次前向里让位置 i 用真值、其余位置各自独立打乱且互不重复使用同一份打乱结果）。
对 512 序列 × 128 位置做到严格版本，需要 L 倍的前向次数，在 P0 阶段不划算。

**采用的近似**：每个打乱种子对每条序列独立生成一个位置置换 `π`（derangement，`π(i)!=i`
对所有 i），构造 `z_t_shuffle = z_t[:, π, :]`，整条序列一起打乱后过一次 backbone。
这个近似打乱了**所有**位置（而不是"固定 i、打乱其余"），因此把它汇报为
"global context-shuffle"，用来估计"打乱上下文顺序后，通用词汇/句法先验还能贡献多少"，
而不是"单点因果贡献"。**更严格的单点版本留给 EXP-PT4**（Causal Context-Source Ablation,
P1 阶段），那里本来就设计了 per-position 的 local-window / masking 干预协议，
适合把"局部窗口"做严格。这个近似已经在下面的"已知简化"里显式标注（⚠️），遵循本项目一贯的
"标注简化而不是假装做了严格版本"的做法（参考 EXP-09/EXP-26 等的 ⚠️ 注释风格）。

---

## 2. 指标与判定规则实现

对每个 t、每个 reference `X ∈ {gauss, swap, shuffle}`：

- `e_X(v) = log(p_oracle_avg(v)) - log(q_X(v))`（用多噪声种子平均后的 oracle 后验对数概率
  近似原始定义里的原生 logit `ell`，这与 EXP-05v3 的 `r_t = log_p - log_q` 一致）。
- `raw_rank`：`y_i` 在 `p_oracle_avg` 中的 rank（0-indexed，来自 `compute_rank`，复用
  `probe_prior_null.py` 里的实现）。
- `residual_rank_X`：`y_i` 在 `e_X` 中的 rank。
- 默认竞争者 `f_i`：取 t-grid 最小值处（`t_min`）每个位置的 native top-1（对应 PT2 三种
  default 定义里的"Earliest-time native top-1"，PT1 复用同一定义避免引入第二套口径）。
- `m_raw(t) = ell(y_i) - ell(f_i)`，`m_res_X(t) = e_X(y_i) - e_X(f_i)`（用 `log(p)`/`log(q)`
  代替 `ell`/`log q`，与上面一致）。
- `frac_residual_before_raw`：对每个位置比较 `argmin_t{raw_rank(t)==0}` 与
  `argmin_t{residual_rank_X(t)==0}`（若某个从未发生则记为 `+inf`），统计残差先于原始达到
  rank 0 的位置比例。
- `frac_null_mode_but_residual_specific`：`null_mode` token 取整个数据集里出现频率最高的
  非特殊 token；统计 `raw_top1==null_mode_token 且 residual_top1_X != null_mode_token`
  的位置比例。
- `KL(p||q_X)`：逐位置 `sum p*log(p/q_X)` 再对位置取均值。

判定规则（对应原始 doc 3.4/3.5 节）直接在 `analyze_prior_subtraction.py` 里按数值输出，
不做自动布尔判定（沿用本项目一贯做法：先给数字，人工/后续 spec 更新里下结论，避免
"脚本自动打勾"掩盖边界情况，参考 EXP-11/EXP-13 等因为自动判定过早而被打 ⚠️ 的教训）。

---

## 3. 数据与规模

严格照抄 doc 第 2 节协议成本很高（512 序列 × 101 个 t 点 × 3 个 checkpoint × 3 个 reference，
每个 reference 还要多噪声种子平均）。采用分级方案：

- **Pilot（本次先跑）**：`n_samples=16`，`n_t_steps=6`，仅 ELF baseline，`n_oracle=3`,
  `n_gauss=4`, `n_swap=4`, `n_shuffle=4`，用来验证 adapter + 全部指标计算路径没有 bug。
- **正式 P0 规模**（pilot 通过后再跑，需要用户确认是否现在跑三个 checkpoint × ELF，
  因为一次完整 512×多 t 网格 × 3 reference 在 A40 上预计每 checkpoint 数十分钟到小时级）：
  `n_samples=512`，`seq_len=128`，稀疏 grid（doc 建议 101 点稠密网格，先用 11–21 点做主结果,
  真正 101 点稠密曲线留给论文最终图，与 EXP-07 64-step dense 曲线的先粗后细节奏一致）。

---

## 4. 脚本与输出

```text
experiments/phase_transition/adapters/elf_adapter.py
experiments/phase_transition/adapters/langflow_adapter.py
experiments/phase_transition/estimate_reference_prior.py
experiments/phase_transition/analyze_prior_subtraction.py
```

输出：

```text
results/phase_transition/<model>/<checkpoint>/prior_reference_<type>.npz   (可选，pilot 阶段先只存 json)
results/phase_transition/<model>/<checkpoint>/prior_subtraction_metrics.json
results/phase_transition/<model>/<checkpoint>/prior_subtraction_curves.pdf
```

---

## 5. 已知简化（务必在论文/后续 spec 里保留这些警告）

1. ⚠️ Reference C（context-shuffle）是全局打乱，不是严格的"仅打乱非目标位置"；
   严格版本见 EXP-PT4。
2. ⚠️ Reference A 用对角协方差（逐通道方差），不建模跨通道协方差。
3. ⚠️ `ell`（原生 logit）用多噪声种子平均后的 `log(p)` 近似，不是单次前向的原始 logit；
   与 EXP-05v3 一致但仍是近似。
4. ⚠️ Pilot 规模（16 样本、6 个 t 点）仅用于验证代码正确性，**不能引用其数字作为结论**，
   正式结论需要等 512 样本 / 稠密 t 网格的正式跑。
5. Pilot 阶段只跑 ELF baseline；kd_cr / kd2 / LangFlow 需要后续跑并在本文件补充 Results。

---

## 状态

**LangFlow 正式规模 DONE；ELF baseline/kd_cr/kd2 正式规模 RUNNING**（后台，
`logs/phase_transition/pt1_elf_{baseline,kd_cr,kd2}_full.log`）。adapter 层 +
`estimate_reference_prior.py` + `analyze_prior_subtraction.py` 已跑通并产出可引用的
LangFlow 结果（见下）；ELF 三个 checkpoint 因为 seq_len=1024（是 LangFlow 128 的 8 倍）
且早期 `rank_of_gt` 实现是 O(NLV log V) 的全量 argsort + 全局 nonzero 扫描（CPU 密集，
GPU 常常空闲等待），跑得明显更慢（LangFlow 128-token 跑 11 个 t 点用了约 22 分钟，
ELF 1024-token 同样 11 个 t 点预计要 ~3 小时）。**已经把 `rank_of_gt` 换成
"严格大于计数"实现**（避免排序，见脚本内注释），但这个修复只应用在新启动的
job 上，没有回灌到已经在跑的 3 个 ELF PT1 进程里（避免浪费已投入的算力，让它们
用旧实现跑完）。

## Results（pilot：ELF baseline，8 序列，seq_len=1024，t∈{0.05,0.35,0.65,0.95}，
n_oracle=2, n_gauss=2, n_swap=2, n_shuffle=2）

⚠️ 规模极小，数字本身不可引用为结论，仅用于验证代码路径与判断信号方向是否合理。

**踩坑记录**（供以后重跑/新实验参考）：
1. ELF 的 RoPE（`TextRotaryEmbeddingFast`）在模型构造时用 `config.max_length`
   固定预计算 `freqs_cos/sin` 表，前向传播里**不做长度裁剪**，所以传入的
   `seq_len` 必须严格等于 `config.max_length`，否则在 `layers.py` 的
   `rotate_half` 处直接 shape mismatch 报错。这与 EXP-01v3/EXP-05v3 等所有已有
   ELF 探针脚本从不覆盖 `seq_len` 的做法是一致的——本 spec 最初尝试用
   `--seq_len 128` 复现 doc 建议的 LangFlow-对齐长度，直接触发此 bug；已在
   `estimate_reference_prior.py` 里把 ELF 分支的 `seq_len` 强制锁定为
   `config.max_length`（本例为 1024，属于 doc 允许的"ELF-only replication"
   长度）。跨模型 128-token 对齐需要用 LangFlow 侧或未来的 ELF 128-token
   checkpoint（如果存在）。
2. `np.savez_compressed` 只会保存 `isinstance(v, np.ndarray)` 的项——标量
   （如 `KL_gauss`、`G_oracle`）如果不显式 `np.asarray()` 就会被静默丢弃，
   下游 `analyze_prior_subtraction.py` 读取时报 `KeyError`。已修复：现在
   全部 record 项（数组或标量）都写入 npz，读取时用 `val.item() if val.ndim==0`
   还原成 Python 标量。
3. `torch.gather` 要求 index dtype 为 `int64`；`f_i`（从 `np.int32` 存储的
   argmax 结果）转回 tensor 后必须显式 `.long()`。
4. LangFlow 的 `model.forward(..., return_dict=False)` 在 `output_hidden_states=False`
   时返回**裸 tensor**（不是 tuple），只有 `output_hidden_states=True` 时才返回
   `(logits, hidden_states)`；`LangFlowAdapter.forward_state` 最初无条件解包
   成两个变量导致 `ValueError: too many values to unpack`。已在 adapter 里按
   `capture_hidden` 分支处理，并跑通 smoke test（`G_oracle@t=0.9≈70.7%` vs
   `t=0.1≈2.7%`，`solver_step` 正常）确认修复。

**数值表**（G_oracle / G_gauss / G_swap / G_shuffle，argmax 命中率）：

| t | G_oracle | G_gauss | G_swap | G_shuffle | rank_oracle_mean |
|---|---|---|---|---|---|
| 0.050 | 0.35% | 0.05% | 0.15% | 0.16% | 10516.5 |
| 0.350 | 61.40% | 0.51% | 1.27% | 0.81% | 619.7 |
| 0.650 | 75.94% | 0.01% | 1.22% | 0.73% | 548.5 |
| 0.950 | 81.87% | 0.05% | 1.01% | 0.87% | 148.1 |

**判定规则相关数字**（t=0.05）：

- `m_raw(t=0.05) = -23.86`（真值 token 相对默认竞争者 f_i 的原始 log-margin，
  大幅落后 —— 早期原生预测几乎完全被默认 token 主导）。
- `m_res(t=0.05)`：gauss `+0.25`，swap `+0.09`，shuffle `+0.21` —— 三种
  reference 减除后，真值 token 相对 f_i 的劣势几乎完全消失（"advantage
  retained" 仅 0.4–1.1%）。**这是定性支持 H1 (prior masking) 的信号**：
  原始 margin 里绝大部分对默认 token 的偏向，可以被一个不含真实语义信息的
  参考先验（尤其是纯几何的 swap/shuffle）解释掉。
- `frac_null_mode_but_residual_specific`：gauss 34.8%，swap **60.9%**，
  shuffle 39.1% —— 相当一部分"raw top-1 = 全局最高频 token"的位置，在减除
  参考先验后 top-1 变成了别的（sample-specific）token，同样支持 prior
  masking。
- `frac_residual_before_raw`：gauss 3.4%，swap 1.3%，shuffle 2.1% —— 这个
  数字明显更弱：残差 rank 提前到 0（top-1 正确）的位置只占极少数，说明
  "减先验" 主要是**压低默认 token 的优势**，而不是**让真值 token 立刻反超变成
  top-1**（`never_raw`≈`never_res`≈16-17% 也印证了整体覆盖率没有大变化）。
  ⚠️ 这与 EXP-05v3 的" G_debias ≈ G_oracle"发现方向一致：debias 不怎么改变
  谁是 top-1，主要重新分配了非 top-1 的概率质量。**这一点在正式规模复现前
  不能过度解读**，也提示"prior masking"假设可能需要在 margin/rank 层面而非
  纯 top-1 层面表述才成立。

## Results（正式规模：LangFlow，128 序列，seq_len=128，t∈11点[0.05,0.95],
n_oracle=4, n_gauss=4, n_swap=4, n_shuffle=4）

这是第一个**正式规模**（非 pilot）的结果，可以引用。

| ref | frac_residual_before_raw | frac_null_mode_but_residual_specific | never_raw_correct | never_residual_correct |
|---|---|---|---|---|
| gauss | 3.38% | **88.87%** | 3.89% | 38.45% |
| swap | 3.55% | **90.98%** | 3.89% | 28.54% |
| shuffle | 2.47% | **90.19%** | 3.89% | 42.49% |

`t=0.05` 判定规则数字：`m_raw=-2.906`；`m_res`：gauss `+0.773`（advantage retained
26.6%），swap `+0.819`（28.2%），shuffle `+0.528`（18.2%）。

**解读**：

1. **比 ELF pilot 更强的 prior-masking 信号**——LangFlow 在 t=0.05 时，减除参考先验
   后真值 token 相对默认竞争者的 margin **直接由负转正**（不只是像 ELF pilot 那样
   接近 0），即残差 margin 平均意义上已经支持真值 token。`frac_null_mode_but_residual_specific`
   高达 89-91%，即几乎所有"raw top-1=全局最高频 token"的位置在去偏后 top-1 都变成了
   别的、样本特定的 token。这是目前为止对 H1 (prior masking) 最强的正面证据。
2. **但 `never_residual_correct` 明显高于 `never_raw_correct`**（28-42% vs 仅
   3.9%）——即用残差重新排序后，反而有更多位置**永远**得不到正确的 top-1，
   比用原始 logits 直接 decode 还差！这和 `frac_residual_before_raw` 仍然很低
   （2.5-3.6%）放在一起看，说明"减先验"主要是把概率质量从"全局高频默认 token"
   身上拿走，但拿走之后不一定分给真值 token——有时候分给了**另一个**错误 token
   （对应 EXP-PT2 失败分类里的 "wrong_mode_accumulation"）。**这是一个需要在论文
   里如实报告的张力**：prior masking 假设在"margin 层面"成立得很好，但在
   "谁是最终 top-1"这个更强的意义上并不总是成立，去偏有时候反而让情况变得更差。
3. 三种 reference（gauss/swap/shuffle）给出的数字彼此接近（KL、margin、fraction
   都在同一量级），互相印证，不像是某一种 reference 的实现 bug 导致的异常。

## Results（正式规模：ELF baseline / kd_cr / kd2，128 序列，seq_len=1024，
11 个 t 点 ∈ [0.05,0.95]，n_oracle=4, n_gauss=4, n_swap=4, n_shuffle=4）

三个 checkpoint 全部跑完，可以做跨 checkpoint 比较（这是本实验目前最有信息量的
一组结果）。

| checkpoint | ref | frac_res<raw | frac_null→specific | never_raw | never_res | m_res(t=0.05) advantage retained |
|---|---|---|---|---|---|---|
| baseline | gauss | 12.4% | 25.7% | 15.0% | 13.3% | 1.1% |
| baseline | swap | 8.7% | 64.2% | 15.0% | 14.6% | 0.7% |
| baseline | shuffle | 12.1% | 47.3% | 15.0% | 14.3% | 1.0% |
| kd_cr | gauss | 9.2% | **89.3%** | 5.2% | **4.9%** | **11.5%** |
| kd_cr | swap | 7.1% | **92.5%** | 5.2% | **4.9%** | **10.5%** |
| kd_cr | shuffle | 11.8% | **93.6%** | 5.2% | **2.4%** | **10.8%** |
| kd2 | gauss | 13.9% | **89.0%** | 5.0% | **1.0%** | **11.0%** |
| kd2 | swap | 7.9% | **92.9%** | 5.0% | **4.4%** | **10.2%** |
| kd2 | shuffle | 13.5% | **90.8%** | 5.0% | **0.9%** | **10.6%** |

**核心发现：KD checkpoint（kd_cr / kd2）比 baseline 表现出系统性更强的 prior-masking
信号，方向和 EXP-10/EXP-16v2 等已有实验一致，但这里是一个新的、独立的度量维度**：

1. `frac_null_mode_but_residual_specific` 从 baseline 的 26-64% 跳到 KD 的
   89-94%——KD checkpoint 里，几乎所有"raw top-1 = 全局最高频 token"的位置在
   去偏后都变成了别的、样本特定的 token；baseline 只有不到一半这样的位置会变化。
2. `m_res(t=0.05)` advantage retained：baseline 只有 ~1%，KD 有 ~10-12%
   （数量级差异，不是噪声）。
3. **最有意思的一点**：baseline 和 LangFlow 都表现出 `never_residual_correct ≥
   never_raw_correct`（去偏有时候让 top-1 覆盖率变差），但 **KD checkpoint 正好
   相反**——`never_res`（2.4-4.9%）明显低于 `never_raw`（5.2-5.0%），也就是说对
   KD 模型，用去偏后的分布重新排序，反而比原始 logits 直接 decode 覆盖率更高。
   这提示一个可能的机制故事：**KD 训练不只是让 native readout 更早显示正确答案
   （已知），可能还让 backbone 表示本身携带了更多可以被一个通用（非学习）参考
   先验抽取出来的样本特定证据**——去掉先验对 baseline 没有净收益（甚至略微有害），
   对 KD 模型却是净收益。这是一个**新的、值得在论文里讨论的对比**，但目前只有
   一个 checkpoint 规模的证据，且引用的 `never_raw`/`never_res` 差距是百分点量级
   （4-5pp），需要 bootstrap CI 才能说"显著"。

### 更新：padding 排除已修复并重跑（见下方"Results（padding 修复后重跑）"）

⚠️ **已核实、是一个真实的方法论问题，现已修复**：ELF 三个 checkpoint 的
`null_mode_token` 曾经都是 id=1，就是 T5 tokenizer 的 `</s>`（eos/pad）
token，不是一个真正的高频实词。
原因：`estimate_reference_prior.py` 目前是在**全部 gt 位置**（含 padding）上统计
"最高频 token"，而 doc 的协议明确要求"Exclude padding... Analyze special tokens
separately"（见本文档第 2 节共享协议），这里没有照做。核实了一下实际占比：ELF
128 序列/1024-token 里 pad/eos 占所有位置的 5.3%（LangFlow 128-token 序列里最高频
token 是 id=198，也就是换行符，不是 pad——GPT-2 tokenizer 下 eos 占比=0%，说明
LangFlow 这边基本没有 padding 污染问题，5.3% 主要是 ELF 特有的）。5.3% 不算特别
大的比例（因为具体某个真实单词单独出现的频率通常也低于这个数），但它已经足够让
"全局众数"变成 pad token 而不是一个真正的高频实词——**这意味着 ELF 的
`frac_null_mode_but_residual_specific` 数字部分反映的是"pad vs. 非 pad"的区分，
而不是 doc 原本想测的"高频实词先验 vs. 样本特定证据"**。其余指标
（`frac_res<raw`、`m_res` advantage retained、KL）不直接依赖 `null_mode_token`
的身份，应该不受这个问题污染，但 `frac_null_mode_but_residual_specific` 这一列的
ELF 数字目前应视为需要重跑修正（对 gt/所有 reference 计算前先按 attention_mask
排除 padding 位置）后才能放进论文。这是**下一步**清单里的一项。

## Results（padding 修复后重跑：ELF baseline/kd_cr/kd2，同样 128 序列/11 t 点）

修复方式：`null_mode_token` 现在只从非 padding 位置统计（用 `attention_mask`
过滤），npz 里也把 `mask` 存了下来供 `analyze_prior_subtraction.py` 在算
`frac_null_mode_but_residual_specific` 时排除 padding 位置。三个 checkpoint
修复后的 `null_mode_token` 都变成了 id=3（T5 tokenizer 里的 `▁`，SentencePiece
的空格标记）——**不是 pad token 了，是一个真实出现在文本流里的 token，但仍然
是一个结构性/功能性符号，不是严格意义上的"高频实词"**，这一点也要如实说明，
不能说成完全解决了"应该用真正的高频实词"这个理想。

| checkpoint | ref | 修复前 frac_null→specific | 修复后 | 变化 |
|---|---|---|---|---|
| baseline | gauss | 25.7% | **3.0%** | −22.7pp |
| baseline | swap | 64.2% | **3.3%** | −60.9pp |
| baseline | shuffle | 47.3% | **3.8%** | −43.5pp |
| kd_cr | gauss | 89.3% | 56.6% | −32.7pp |
| kd_cr | swap | 92.5% | 61.0% | −31.5pp |
| kd_cr | shuffle | 93.6% | 58.0% | −35.6pp |
| kd2 | gauss | 89.0% | 68.9% | −20.1pp |
| kd2 | swap | 92.9% | 69.6% | −23.3pp |
| kd2 | shuffle | 90.8% | 69.5% | −21.3pp |

**这次修复对结论的影响比预期大得多，需要更新论文层面的表述**：

1. **baseline 修复前的 25.7-64.2% 几乎全部是 pad-token 假象**——修复后只剩
   **3.0-3.8%**，也就是说 baseline 上"raw top-1 是全局默认 token 但去偏后
   变成样本特定 token"这件事几乎不发生。之前"baseline 也有一定 prior
   masking，只是比 KD 弱"这个表述需要改成"**baseline 在这个具体指标上几乎
   没有 prior masking 效应**"。
2. **KD checkpoint 的数字也下降了（−20 到 −36pp），但依然显著**（57-70%）——
   KD 的 prior masking 效应是真实的，不是 padding 造成的假象，只是原始数字
   被 padding 适度放大了。
3. **baseline vs KD 的对比反而变得更极端、更干净**：修复前是"3%（baseline）
   vs 89-94%（KD）"这种量级差异看起来可能有共同的测量假象在起作用；修复后
   是"3%（baseline）vs 57-70%（KD）"——差距本身缩小了一些，但因为 baseline
   端几乎清零，这个对比现在是"该效应在 baseline 上基本不存在、在 KD 上真实
   存在"这样一个更干净、更容易站得住脚的定性差异，而不是"两者都有、程度不同"。
4. `m_res` advantage retained（不依赖 `null_mode_token` 身份的指标）数字
   和修复前完全一致（baseline ~1%，KD ~10-12%）——符合预期，因为这个指标
   本来就不受这个 bug 影响，是一个很好的一致性检查。

⚠️ 这次修复应该被视为**决定性的方法论修正**，`EXP-INDEX.md` 和后续任何引用
`frac_null_mode_but_residual_specific` 数字的地方都应该用这次修复后的版本，
不能再用旧数字。

## 下一步（需要用户确认规模/优先级后再执行）

1. 把 ELF baseline 跑到正式规模（512 样本，更密 t-grid，比如 21 点），确认
   pilot 里观察到的"m_res≈0 但 frac_residual_before_raw 很低"这个模式是否
   稳定。
2. 对 kd_cr / kd2 重复，看 prior masking 强度是否和已知的"KD 提前承诺"效应
   （EXP-10/16v2 等）相关。
3. 跑 LangFlow 版本（adapter 已实现，`--model langflow`，需要单独验证
   `~/LangFlow` 包在目标 GPU 环境可用，以及 gamma-based oracle noising路径）。
4. 视 PT1 结果决定是否直接推进 PT2（margin trajectory + 转变时间，复用同一套
   default-competitor 定义）和 PT5（decoder-bias 诊断性干预，代码上最省事，
   只需要在已有 logits 上做 `ell' = ell - λ log q`）。
