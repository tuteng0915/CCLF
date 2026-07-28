# EXP-GS7 Spec — Oracle vs Free-Running Global Alignment

## 背景与地位

原始 doc 第 11 节 GLOBAL-7。把本仓库已经反复验证过的"oracle–rollout gap"（EXP-01/01v3/
EXP-PT1/PT2 等，一直是 per-token/per-position 的 readout gap）提升到**全局状态形成层面**：
自由生成（从纯噪声开始、真正跑完整 reverse ODE，不依赖任何 OWT ground-truth 文档）的轨迹，
和"用自由生成的最终输出反推出来的 paired oracle path"相比，在哪个 t、哪个层面开始出现分歧。

这是本 GS 系列第一个**真正的自由生成**实验——GS1–GS9 全部用 OWT 文档的 oracle state 作为
"trajectory point"的代理（在各自 spec 里都标注过这个简化）；GS7 第一次让模型从纯噪声真正
自己生成，不借用任何外部文档。

## 0. 复用与新增

- Adapter、`solver_step`：同 GS2/GS4，但这次需要**记录多个中间 t 的状态**（GS2/GS4 的
  `rollout_branches` 只返回终点），因此另写一个 `rollout_with_checkpoints`（在
  `compare_oracle_rollout_global.py` 里，不改 `branch_global_consensus.py` 里已经跑通、
  被 GS2/GS4 依赖的 `rollout_branches`，避免影响已有脚本）。
- CKA、effective rank：复用 GS3 的 `linear_cka`/`effective_rank`（`analyze_low_rank_modes.py`）。
- POS/token 相关：复用 `common.py`。

## 1. Paired 构造

1. 采样 `N` 个初始噪声 `eps ~ N(0,I)`（`(L,d)`，不基于任何 OWT 文档）。
2. 从 `t_eps=0.05`（ELF 的最小可用 t，见 `adapter.t_eps`；不是精确 0，原因见
   `experiments/phase_transition/adapters/elf_adapter.py` 里已有的说明）开始，用
   `solver_step` 真正做完整 reverse ODE，到 `t_end=0.99`，途中在 GS1 同款 checkpoint
   t-grid `[0.05,0.12,0.20,0.28,0.38,0.50,0.65,0.85,0.99]` 上**记录状态**
   `Z_t^{rollout}`（不是重新构造的 oracle 状态，是真正积分出来的中间态）。
3. 取 `t=0.99` 的终点状态，做一次 `forward_state` 得到 native top-1 token 序列，作为
   `y^{rollout}`（这次生成"自己的" ground truth，不存在外部标注）；同一终点状态本身作为
   `x_clean^{rollout}`（近似 clean endpoint，避免再做一次 decode-retokenize-reencode 的
   往返）。
4. 用**同一个** `eps`、`x_clean^{rollout}` 构造 paired oracle path：
   `Z_t^{oracle} = t * x_clean^{rollout} + (1-t) * eps`，在同一 checkpoint t-grid 上取值。

## 2. 指标（只取已验证可靠的子集，理由见下）

不重新引入 GS1/GS2 已经证明会饱和的 mean-pooled-cosine "topic/sentence" 指标（原始 doc
"Global semantic alignment"一栏建议的 sentence embedding/topic cluster 相似度）；GS7 只用
GS3 已经验证过内部一致、有动态范围的两个：

- **几何对齐**：`CKA(Z_t^{oracle}, Z_t^{rollout})`（每条序列独立算，`n_valid` 个有效位置，
  取样本均值）——原始 doc "Geometric alignment"一栏的核心指标。
- **Effective rank**：分别算 `r_eff(Z_t^{oracle})` 和 `r_eff(Z_t^{rollout})`，比较二者
  随 t 的演化是否一致（如果自由生成的状态"有效秩"演化模式和 oracle 差很多，说明自由生成的
  内部几何结构和 oracle 路径本质不同，不只是终点不同）。
- **Token 对齐（原始 doc "Dynamical/lexical alignment"的简化版）**：在 `y^{rollout}`
  这个"自定义 ground truth"下，分别算 `G_token^{oracle}(t)` 和 `G_token^{rollout}(t)`
  （native top-1 vs `y^{rollout}`）——这就是原始 doc 反复提到的"gap"在这里的直接操作化：
  如果两条路径在同一 t 的 token 恢复率上有系统性差距，就是"自由生成比 oracle 更晚/更差
  恢复出（自己最终认定的）token identity"的直接证据。

## 3. 已知简化

1. ⚠️ **不测 topic/sentence-embedding 对齐**（GS1/GS2 已确认这类指标在当前实现下饱和，
   不可靠），只测 CKA + effective rank + token accuracy。
2. ⚠️ `x_clean^{rollout}`直接用 `t=0.99` 的 rollout 终点状态，不做完整的
   decode→retokenize→re-encode 往返（doc 原文暗示的"encode 最终生成文本为 clean endpoint"
   更严格，但会引入额外的 tokenizer 边界误差，pilot 先用更简单的"终点状态即 clean"近似）。
3. ⚠️ Free-running rollout 全程 self-conditioning 冷启动（`sc=zeros`，从 `t_eps` 开始），
   和 GS2/GS4 一致的简化。
4. `n=8` 条自由生成轨迹（比其它 GS pilot 更小，因为每条轨迹本身就要做一次完整 32-step
   ODE 积分，比 GS1/GS3 的单步 oracle probing 贵得多）。

## 4. 脚本与输出

```text
experiments/global_state/compare_oracle_rollout_global.py
```

```text
results/global_state/<model>/<checkpoint>/oracle_vs_rollout_global_<label>.json
```

## 状态

**Pilot DONE**（ELF baseline，n=8 条真正的自由生成轨迹，GPU1，
`logs/global_state/gs7_elf_baseline_pilot.log`，输出
`results/global_state/elf/baseline/oracle_vs_rollout_global_pilot.json`）——干净、内部一致，
是 GS 系列里第二个可信的正面结果（另一个是 GS8）。

## Results（pilot：ELF baseline，8 条自由生成轨迹，seq_len=1024）

| t | CKA(oracle,rollout) | r_eff(oracle) | r_eff(rollout) | G_token(oracle) | G_token(rollout) |
|---|---|---|---|---|---|
| 0.05 | 0.999 | 474.1 | 474.1 | 0.045 | 0.021 |
| 0.12 | 0.994 | 473.9 | 474.1 | 0.145 | 0.022 |
| 0.20 | 0.980 | 473.0 | 474.0 | 0.474 | 0.026 |
| 0.28 | 0.953 | 470.8 | 473.4 | 0.589 | 0.038 |
| 0.38 | 0.911 | 465.0 | 471.0 | 0.659 | 0.126 |
| 0.50 | **0.894** | 450.7 | 462.5 | 0.696 | 0.395 |
| 0.65 | 0.928 | 415.5 | 430.8 | 0.722 | 0.616 |
| 0.85 | 0.967 | 333.9 | 332.3 | 0.725 | 0.690 |
| 0.99 | 1.000 | 281.0 | 279.5 | 0.998 | 1.000 |

（`t=0.99` 处 `G_token(rollout)=1.000` 是同义反复——`y^rollout` 定义就是 `t=0.99` 终点状态
自己的 native decode，不是一个独立的验证点，只是内部一致性检验。）

**解读**：

1. **`CKA(oracle, rollout)` 呈 U 形，最低点在 `t=0.50`（0.894），不是在两端**：
   `t=0.05`/`t=0.99` 两端都接近 1.0（早期两条路径都还是近似噪声，晚期都收敛到共同终点，
   两种情况下几何上都"像"），中段（尤其 `t=0.50`，略晚于已知的 baseline commitment cliff
   `t≈0.20–0.30`）几何分歧最大。`r_eff(rollout)` 在整个中段（`t=0.28–0.65`）都**持续
   高于** `r_eff(oracle)`（如 `t=0.50`: 462.5 vs 450.7；`t=0.65`: 430.8 vs 415.5）——
   自由生成路径在这个阶段的有效秩比"预知终点"的 oracle 路径更高，说明它还没有像 oracle
   那样早早地把状态"收拢"向最终结构，仍在探索/组织的过程中。
2. **`G_token` 的 oracle–rollout gap 远比几何/秩的差距剧烈**：`t=0.28` 时
   `G_token(oracle)=0.589` 而 `G_token(rollout)` 只有 **0.038**（差 15 倍以上）；
   `t=0.38` 时 0.659 vs 0.126（约 5 倍）；差距要到 `t=0.65` 才明显收窄（0.722 vs
   0.616），`t=0.85` 时基本追平（0.725 vs 0.690）。粗略地说，**自由生成在 token identity
   这个维度上比 oracle "慢" 了大约 0.3–0.4 个 t 单位**（rollout 在 `t≈0.65` 的 token 恢复率
   才追上 oracle 在 `t≈0.28` 的水平），比几何指标显示的差距大得多。
3. **两条指标的对比本身就是本次 pilot 最重要的发现**：几何层面（CKA、有效秩）的
   oracle–rollout 差距**从未很大**（CKA 最低点也有 0.894，绝对值上仍然很高），但 token
   层面的差距**巨大**（15 倍级别）。这直接把本仓库反复验证过的"oracle–rollout gap 是
   per-token 现象"（EXP-01/01v3/EXP-PT1）**提升到了全局状态形成层面并给出了具体的定位**：
   **free-running 生成的全局/几何组织大体上"跟得上"oracle，真正掉队的是 exact lexical
   commitment**——和 GS1/GS2/GS3 反复发现的"structural/global 信号早、稳，token/lexical
   信号晚、脆弱"这条主线完全一致，这次是从"自由生成 vs oracle 差距在哪"这个新角度独立
   得到的印证。

## 下一步

1. 扩大 `n_samples`（当前 8 条，统计效力有限，尤其是 `t=0.50` 附近 CKA 最低点这个具体数值
   需要更大样本确认是否稳定，还是随机波动）。
2. 补充结构（POS）层面的 oracle vs rollout 对比——当前 pilot 因为是纯自由生成、没有外部
   ground truth，没有把 GS1 的 structural probe 迁移过来；可以用 `y^rollout` 自己的解码
   文本算 POS histogram，再看 oracle/rollout 两条路径的 mean-pooled 状态对这个自定义
   structural target 的可恢复性差距，检验"structural 层面差距也和 token 层面一样小"还是
   "structural 层面差距介于 CKA 和 token 之间"。
3. 对 kd_cr/kd2 checkpoint 重复本实验——EXP-01v3/EXP-10 等已经证明 KD 大幅提前 oracle
   G(t)，值得检验 KD 是否同样能缩小这里发现的"token 层面 oracle-rollout gap"，还是只加速
   了 oracle 路径本身而没有改善 free-running 路径的相对滞后。
