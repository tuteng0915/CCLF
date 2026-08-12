# CCLF Experiment Index

完整历史账本。日常工作请先看 `README.md`；无效/被替代协议见
`DEAD-ENDS.md`，不要把本文件中的全部条目理解为待办列表。

**更新时间**: 2026-08-13（EXP-93 selector closure；EXP-94 compute audit；EXP-95 formal result；98 个实验）

---

## Active P0 queue (2026-08-08)

| Experiment | Status | Purpose |
|---|---|---|
| **EXP-65** | **STAGE B DONE** | Baseline hard commit survives length 1024; KD PPL direction reverses while degeneration improves; baseline native-SDE check remains |
| **EXP-66** | **DONE** | Early-KD and hard commit both improve unconditional length-1024 PPL without degeneration; effects are mostly additive, conditioned Early-KD is not robust across training seeds |
| **EXP-67** | **DONE (ODE mechanism only)** | Position-correct anchors accelerate and stabilize unresolved tokens; matched shuffled anchors reverse the effect and destroy coherence |
| **EXP-68** | **DONE** | Native SDE keeps a tiny favorable unconditional sign but ODE gain nearly vanishes; conditioned PPL slightly worsens; frozen policy commits about 99% at first crossing |
| **EXP-69** | **DONE / QUALITY GATE FAILED** | Native-SDE early commitment at 25%, 52%, and 78% anchor density worsens PPL; late commitment is nearly saturated and inert, so no shuffled-anchor mechanism run is warranted |
| **EXP-70** | **DONE / NEGATIVE** | True local clocks and final refinement fail; heterogeneous mixed context dominates Pipeline error |
| **EXP-71** | **DONE / NEGATIVE** | Correct soft-anchor content matters, but loses to ODE-64 and shows no LTR advantage |
| **EXP-72** | **DONE / STOPPED AT 500** | Deep injection preserves sync quality but fails the functional clock-learning gate |
| **EXP-73** | **IMPLEMENTED / NOT LAUNCHED** | Runner smoke-tested; formal trajectory distillation gated off by EXP-72 failure |
| **EXP-74** | **DONE / HARD-ONLY POSITIVE** | Sparse soft memory fails, but one post-transition persistent hard anchor improves all three checkpoints and survives density controls |
| **EXP-75** | **DONE / NEGATIVE** | Predicted-clean context lowers Pipeline PPL partially but does not reduce vector error or restore coherent generation |
| **EXP-76** | **DONE / PARTIAL PASS** | Frozen adapters learn a functional local clock without damaging Standard generation; wave quality remains poor |
| **EXP-77** | **DONE / NEGATIVE** | Asynchronous block-transition distillation retains Standard quality but all fill/drain samplers remain catastrophic |
| **EXP-78** | **DONE / ODE-ONLY REVISABLE POSITIVE** | Three-checkpoint, three-seed ODE gains survive conditioned evaluation; Unlock-4 is best and 8--10% of anchors revise after release, but native-SDE effects are negligible |
| **EXP-79** | **DONE / CONDITIONAL NEGATIVE** | On 64-token-prefix/192-token-continuation generation, late coupling only matches Semi-AR (`prompt-cond PPL 419--420` vs `421`) and loses badly to Parallel-32/60 (`253/135`); ROUGE-L also favors parallel |
| **EXP-80** | **DONE / ASYNC NEGATIVE; UNLOCK PPL GAIN REPLICATED** | Paired U/C audit: soft LTR/random and local/canonical remain negative; Unlock-4's same-call U/C PPL gain replicates across two new OWT panels and Gutenberg, but prompt-gain improvement is not robust and diversity/repetition trade-offs remain |
| **EXP-81** | **DONE / GENERIC-QUALITY EFFECT** | Bandwise rescoring shows NLL improvements throughout the suffix, but no prompt-gain CI excludes zero; Unlock does not robustly increase prompt use |
| **EXP-82** | **DONE / ODE PPL POSITIVE WITH TRADE-OFF** | Random 50% position-correct temporary anchors beat confidence selection across three U/C panels; shuffled content fails catastrophically, while D1/Rep-4 remain the limiting trade-off |
| **EXP-83** | **SEED REPLICATION RUNNING** | Corrected cross-architecture endpoint bank: LangFlow lacks a non-tautological narrow collapse; paired-noise Plaid self rank improves gradually while entropy stays high |
| **EXP-84** | **P0 MIXED** | Endpoint-contrast perturbations redirect specific alternatives while matched controls do not, but `epsilon_50` does not rise monotonically through the ELF collapse window |
| **EXP-85** | **TWO-SEED PARTIAL POSITIVE** | Correct high-coverage anchors add pre-transition entropy collapse; alternative anchors causally redirect endpoints, with non-negligible shuffled/random effects at 50% density |
| **EXP-86** | **SMOKE DONE / FORMAL PENDING** | Antithetic Plaid mean-drift estimator passes a measurability smoke; formal run waits for corrected endpoint banks |
| **EXP-87** | **DONE / THREE-SEED POSITIVE** | Plaid raw/continuous/hard late coupling beats Block-SAR prompt PPL in every seed with 56 versus 64 calls; raw mean `95.38` versus `100.87` |
| **EXP-88** | **DONE / PARETO GATE FAILED** | Shadow disagreement releases one third of anchors and further lowers PPL, but does not recover D1; adaptive rollback is not promoted |
| **EXP-89** | **DONE / SCALE SIGN POSITIVE** | Frozen random anchors improve PPL in all 9 length/prefix cells through length 1024; U benefit shrinks with length and the diversity trade-off remains |
| **EXP-90** | **DONE / CONDITIONAL PORTABILITY POSITIVE** | Native-clock random correct anchors improve C-PPL in 3/3 LangFlow and 3/3 Plaid seeds; shuffled content is catastrophic, while U-PPL/full Pareto behavior are architecture-dependent |
| **EXP-91** | **DONE / THREE-SEED NEGATIVE** | Predicted-clean subset-flow fails the paired gate: mean U/C PPL interactions are `+1.5/+2.3`, prompt-gain interaction `-.0072`, and C-degeneration worsens on all three inference seeds |
| **EXP-92** | **DONE / STRAIGHT-ENDPOINT TARGET REJECTED** | Conditional-oracle is non-Pareto; on-policy fails, and loss-balanced `.25` worsens C-PPL and prompt-gain interactions in all three seeds |
| **EXP-93** | **STAGE 2 CLOSED / REPLICATED ORACLE GAP, CURRENT SELECTORS FAIL** | Best-of-16 cuts C-PPL by `37--38%` on two banks, but static, 8-step future-context, and additive single-position causal-graph selectors fail independent validation |
| **EXP-94** | **DONE / EXTRA DENOISING EXPLAINS GAIN** | Parallel-44 beats late raw/continuous by 23.91 C-PPL at identical 11264 token-calls; late coupling is only a Block-SAR replacement |
| **EXP-95** | **DONE / THREE-SEED PLAID METHOD POSITIVE** | Early one-step 75% confidence anchors reduce C-PPL `110.39 -> 80.28`, improve mean D1, and remain highly revisable; D2 falls slightly |
| **EXP-96** | **CLOSED BY EXP-94** | No adaptive-trigger sweep without fixed-schedule compute headroom |
| **EXP-97** | **CLOSED BY EXP-94** | No multi-block scale-up of a schedule that loses to compute-matched parallel refinement |
| **EXP-98** | **CLOSED BY EXP-94** | No trajectory distillation without a verified compute-matched teacher |

The older queue entries below are retained as a historical ledger. In
particular, EXP-63 and EXP-64 are complete and must not be relaunched from a
stale status line.

---

## Mechanism Paper Follow-up — 精简后的当前队列

当前真正决定论文的只有 GS16 和 GS17。GS18 是按需控制，GS19 是方法验证，GS20 是延后到核心机制明确后的 CDCD 复现。

| package | 优先级 | 状态 | 核心问题 | 合并内容 |
|-----|--------|------|----------|----------|
| **EXP-GS16** | **P0** | **DONE (formal 3-seed)** | endpoint 是否很早已经 specific，还是晚期 collapse？ | 原 GS16 endpoint affinity + 原 GS19 calibrated branching |
| **EXP-GS17** | **P0** | **DONE (formal 3-seed)** | residual 如何运动，transition 何时发生？τ_50s=0.206±0.022，τ_aff=0.322±0.010（极紧），τ_v median=0.170；ordering P(τ_v≤τ_50s)=0.896，P(τ_aff≤τ_50s)=0.049 | 原 GS17 velocity + 原 GS18 unified timeline |
| **EXP-GS18** | P1 | CONDITIONAL | supporting claims 是否经得住严格 null？ | 原 GS20 rank/energy control + 原 GS21 common-factor control |
| **EXP-GS19** | P2 | AFTER P0 | 异步收益来自方向、破缺同步还是 confidence anchors？ | 原 GS22 schedule ablation |
| **EXP-GS20** | **DONE (pilot) — CDCD 换成 Plaid（无官方 checkpoint），一条结论跨架构确认、一条暴露真实边界条件** | Plaid 1B（OpenWebText2, NeurIPS 2023） | CDCD 官方无代码无 checkpoint（已核实），改用 Plaid 替代——同样是连续 embedding 空间扩散、有真实发布的 1B checkpoint、训练语料和 OWT 同源；搭建独立 conda 环境编译 FlashAttention+Apex、写了完整 adapter，GS20 自带的 6 项 adapter-gate 检查全部通过 | **GS16 干净跨架构确认，比 ELF 更陡**：S_self 在 t_bank 处仍与 0 无区别，`rank_self=1` 比例从 t_bank 的 12% 在**下一个 checkpoint 就跳到 100%**（比 ELF 的 2-3 步窗口更窄），熵同步骤降并进入平台——三个架构里目前唯一一个"exploration-collapse"核心结论被独立确认的案例。**GS17 暴露真实边界条件，如实报告（按 spec 自身的"CDCD 分歧是边界条件不是要藏起来的复现失败"）**：`cos_endpoint(t)` 形状和 ELF 定性不同（早期低且先降后升，而非 ELF 那样一开始就很高）——大概率是 Plaid 原生 solver_step 是**随机 ancestral 采样**（每步都注入高斯噪声），有限差分速度估计必然混入这个采样噪声，ELF/LangFlow 的确定性 Euler step 没有这个混淆；`V_self` 在中段出现一段持续为负（t=0.46-0.93），ELF 上没见过这个模式；**事件顺序反转**：`P(tau_affinity<=tau_50_stable)` 在 ELF 上是 0.0（坍缩晚于稳定），在 Plaid 上是 1.0（坍缩早于稳定）——但 `t_bank=0.20` 是直接沿用 ELF 的校准值、没有针对 Plaid 自己的 schedule 重新校准（这正是本项目反复踩过的"不能跨架构直接比 nominal t"的坑，这次自己又踩了一遍），`tau_affinity` 在全部 8 条轨迹上都精确等于 `t_bank` 本身，说明真实坍缩起点可能比 `t_bank` 更早、测量范围没覆盖到——这个事件顺序反转暂不能当作确认的跨架构分歧；**2026-08-03 更新：GS18（两部分）和 GS19 也在 Plaid 上跑完了**，完整 5/5 记分卡见 EXP-GS20-spec.md「Full cross-architecture scorecard」节——GS16、GS18-Part-A、GS19 三个干净跨架构确认，GS17、GS18-Part-B 两个真实边界条件；**一个值得写进论文的元发现**：会复现的三个都是"看最终落点"（GS16 的分支终点、GS19 的最终生成文本、GS18-A 的单时刻静态子空间结构），不需要对连续噪声状态做差分；不复现的两个都依赖对 Plaid 原生随机 ancestral 采样器产出的相邻状态做逐步差分（GS17 的局部速度、GS18-B 的位置间 margin 相关性）——这个模式本身可能比任何单条结论都更重要，指向"结论对随机 vs 确定性采样器是否敏感"这个更一般的方法论问题；见 EXP-GS20-spec.md 各节 |
| **EXP-GS20** | P2 | DEFERRED | CDCD 是否复现核心机制？ | 原 GS23 minimal replication |

推荐执行顺序：

```text
GS16 calibrated endpoint bank + specificity
    -> GS17 local velocity + unified timing
    -> GS18 only if the corresponding supporting claim remains central
    -> GS19 asynchronous intervention
    -> GS20 CDCD replication
```

---

## ELF 基础实验

| EXP | 状态 | 模型 | 核心问题 | 一句话结论 |
|-----|------|------|----------|-----------|
| EXP-01 | DONE | ELF | Oracle 悬崖 vs 真实轨迹（v1/v2） | Protocol B 真实 ODE 轨迹无承诺悬崖；oracle G(t) 悬崖是测量假象（固定 ε 协议导致） |
| EXP-01v3 | **DONE** | ELF×3 | Reverse ODE trajectory vs oracle（64 seq, 1024 tok，三 checkpoint） | 交叉点顺序：kd2(t≈0.184)<kd_cr(t≈0.213)<baseline(t≈0.243)；KD 使 G_oracle 大幅提升；G_reverse 在 t=0.05 时 kd2=13.8%>>baseline=0.63%（早期承诺） |
| EXP-04 | DONE | ELF | 几何偏置 null model | G_null≈0.17%，几何偏置可忽略，G(t) 无需修正 |
| EXP-04v2 | **DONE** | ELF baseline | Head-only null vs backbone null vs oracle（1024 tok） | G_head_null≈0.017%（头部无偏置）；G_backbone_null≈0.15-2%（小频率先验）；G_oracle 完全主导 |
| EXP-05 | DONE | ELF | 学习先验 q_t（batch-shuffle，有方法论问题） | G_debias << G_oracle 是 artifact；batch-shuffle 包含 wrong-instance posterior，方法论有误 |
| EXP-05v3 | **DONE** | ELF×3 | Global null prior（正确方法，z_t_null=(1-t)·ε） | G_null≈0.03-4%；G_debias≈G_oracle（差异<3pp）；EXP-05 的−17pp 下降是 batch-shuffle artifact |
| EXP-06 | DONE | ELF | 先验减法 G_debiased | 去偏后 baseline 比 kd_cr 更早（先验贡献有限），ELF 早承诺是真实的 backbone 特性 |
| EXP-10 | DONE⚠️ | ELF×3 | 三模型承诺时序对比（oracle G(t)） | KD 极大提前 native decoder G(t)（t=0.20: baseline≈10%→kd_cr≈61%）；但 probe capacity 各 checkpoint 相似（EXP-07v2）→ KD 改变的是 decode interface 而非 x̂_t 的可恢复性本身；⚠️ Protocol A only；G(t) 数字有来源不一致，需统一 provenance |
| EXP-16 | DONE⚠️ | ELF×3 | 每位置首次正确读取时间分布 | kd_cr 峰值 t=0.15–0.25，baseline t=0.30–0.45；⚠️ 这是 first-correct readout time（非 commitment）；读自 exp07b 旧状态（有固定噪声 bug）；"永不承诺 19%" 实为"在5个 oracle t 采样内未被读取"；needs exp07b_v2 重跑并加 stable readout time |
| EXP-16v2 | **DONE** | ELF×3 | T_first + T_stable(K=3) + T_margin oracle readout timing | baseline never-stable=**25.1%** vs kd_cr=**0.53%**；G(t=0.5)(T_first): baseline **77.3%**→kd_cr **99.5%**；T_stable(t=0.5): baseline 74.9%→kd_cr 99.5%；主效应是 coverage 而非 timing；kd_cr "only recovered" = 24.6% |
| EXP-20 | DONE⚠️ | ELF | Token 频率 vs 承诺时间 | 高频词更早承诺；⚠️ "never commit"定义仅基于稀疏 grid；频率来自测试样本非训练语料；token 描述为"估计"非精确 decode；辅助实验，不可引用 rare/common 差异为核心发现 |

## ELF vs LangFlow 比较实验

| EXP | 状态 | 模型 | 核心问题 | 一句话结论 |
|-----|------|------|----------|-----------|
| EXP-02 | DONE | ELF+LF | LangFlow native posterior | ELF x̂_t 比 LangFlow native posterior 早约 0.6t 承诺（在所有 t 下） |
| EXP-03 | DONE | ELF+LF | Matched-SNR 比较 | Matched SNR 下 ELF 仍比 LangFlow 更早承诺（非 schedule 导致） |
| EXP-30 | DONE⚠️ | LangFlow | LangFlow 逐层探针（EXP-07b 对应） | B07 peak +3.9pp vs native；⚠️ skip input 不对称；+3.9pp 需 CI；"信息丢失"表述过强；需 MLP probe |
| **EXP-30v2** | **DONE** | LangFlow | LangFlow 逐层探针 v2（5 seeds, MLP, skip decomp）| peak B10=+2.6pp @ t=0.85（5-seed CI 下显著）；skip复刻EXP-21v2（92.4%/0%/96.3%）；MLP≤linear |
| EXP-30b | **DONE** | LangFlow | LangFlow 64-step 几何探针（密集曲线） | G(t)<7% until t=0.71；悬崖 t=0.79→0.90；matched SNR 下 LangFlow 恢复能力超 ELF |
| EXP-31 | DONE⚠️ | ELF kd_cr | spec-11v2 Diffusion Forcing（kd_cr checkpoint） | ⚠️ seed=123 有 artifact；PPL 单指标；freeze≠commitment；时间方向描述有误；符号方向需正常 seed 验证 |
| EXP-31b | DONE⚠️ | ELF kd2 | spec-11v2 Diffusion Forcing（kd2 checkpoint） | ⚠️ seed=456 kd2 有 21% degenerate；"freeze_1.0 最优 −48.9%"需多 seed + 多指标验证 |
| **EXP-31v2** | **DONE** | ELF kd_cr+kd2 | spec-11v2 DF multi-seed（5 seeds 0-4，4条件）| 符号反转 CONFIRMED：kd_cr Δ=+120（DF有害），kd2 Δ=-106（DF有益）；评估完全确定，非 seed artifact |
| EXP-32 | DONE⚠️ | ELF kd2 | kd2 DF step-count sweep（8/16/32 步） | ⚠️ 仅 3 个点且变量混淆；"超线性 N²"无依据；none baseline 本身变化 2.4× |
| EXP-33 | **INVALID** | ELF kd_cr | kd_cr + dec_sc + DF | ⚠️ 无 tmin gate 的 dec_sc 产生退化文本（kd_cr: "centre centre"）；PPL 无效 |
| EXP-34 | **INVALID** | ELF kd2 | kd2 + dec_sc + DF | ⚠️ 无 tmin gate 的 dec_sc 产生退化文本（kd2: "eight twenty eight twenty"）；PPL 无效 |
| EXP-35 | **INVALID** | ELF baseline | baseline + dec_sc + DF | ⚠️ 无 tmin gate 的 dec_sc 产生退化文本（baseline: "AS. AS. AS."）；PPL 无效 |
| EXP-36 | DONE⚠️ | ELF×3 | 有效 dec_sc(tmin=0.5) × DF 交互（EXP-33/34/35 修正版） | ⚠️ 缺 freeze-only arm（2×2 不完整）；时间方向描述错误；threshold 跨 checkpoint 不可比；仍退化 |
| **EXP-36 v2** | **DONE** | ELF×3 | 完整 2×2 DF×SC 全因素设计（8 arms，seed=42）| 关键：kd_cr I=-65（互补），baseline I=+1594，kd2 I=+158；I符号由SC主效应决定，非oracle acc |
| EXP-37a | **DONE** | ELF×2 | 1024-token None inference（pilot） | baseline PPL 8步=933/32步=231；kd2 PPL ~128（平稳） |
| EXP-37b | **DONE** | ELF×2 | 1024-token None inference（对照组） | kd2 at 1024-token 比 512-token 低 48-81% PPL；训练长度匹配效果显著 |
| EXP-37c | **DONE** | ELF×2 | 1024-token freeze_1.0 DF（df_t_min=0.7） | baseline+DF 正常；kd2+DF 全面退化（8步德文、16步空白、32步破折号）；PPL 指标失效 |

## Story A: Probe Gap

| EXP | 状态 | 模型 | 核心问题 | 一句话结论 |
|-----|------|------|----------|-----------|
| EXP-07 | DONE⚠️ | ELF×3 | x̂_t 线性探针 vs native decode path | baseline: probe+46pp@t=0.20；kd_cr: dec−7pp（KD 逆转 gap）；⚠️ position-level split（已被 EXP-07v2 取代） |
| EXP-07v2 | **DONE** | ELF×3 | 修复 document-level split 的 probe 重验证（全部 3 checkpoint） | baseline gap +41pp（−5pp 修正）；kd_cr gap −11pp（更强）；kd2 gap ~−6 to −9pp；Story A 完全确认；overfit_gap ~8-30pp 但泛化成立 |
| EXP-07b | DONE | ELF×3 | 逐层线性探针 | L10/L11 probe 最高；浅层 probe 差；承诺悬崖由深层表示驱动 |
| EXP-07c | DONE | ELF×3 | 跨 checkpoint 探针迁移 | L11 激活迁移率 t=0.20 时≈0%；x̂_t 迁移略好 |
| EXP-07d | DONE | ELF×3 | x̂_t 跨 checkpoint 迁移 | kd_cr→baseline≈45%，baseline→kd_cr≈40%，非对称 |
| EXP-21 | **DONE⚠️** | LangFlow | LangFlow probe gap（EXP-07 对应） | 情景 C：native head > probe（gap≈−2 to −7pp）；⚠️ EXP-21v2确认：backbone_top1≈0（residual corrector），skip主导(92.4%@t=1)，probe_h≈native-6pp，gap由skip而非output_layer能力导致 |

## Story B: Coarse-to-Fine & Spatial Bootstrapping

| EXP | 状态 | 模型 | 核心问题 | 一句话结论 |
|-----|------|------|----------|-----------|
| EXP-08 | DONE⚠️ | ELF×3 | token 类型的可恢复性时序差异（oracle） | func t*: kd_cr=0.182 vs content t*=0.255；⚠️ 不是 coarse-to-fine（比较的是不同 token 群体而非同一位置的层级）；⚠️ frequency/surprisal 未控制；⚠️ t* = first-hit，非 stable time；⚠️ tokenizer 前缀（Ġ vs ▁）待审计 |
| EXP-08v2 | **DONE** | ELF×3 | token 类型 stable-commit 时序（T5 tokenizer，stable_k=3） | baseline: func 84.5%/content 69.2% committed；kd_cr: func **99.9%**/content 98.9%；func commits earlier (Δ≈−0.075t for kd, −0.10t for baseline)；近邻 func 提振 content 效应 kd_cr Δ=+0.60 vs baseline +0.10（t=0.3） |
| EXP-09 | DONE⚠️ | ELF×3 | 可恢复性的空间聚集性（非因果自举） | KD 模型正确可读出位置在空间上更聚集（kd_cr near−far peak +65pp）；⚠️ Protocol A 无反馈：测的是空间相关性，不是 causal bootstrapping；⚠️ near group 存在 survivor bias；⚠️ 后期 far group 几乎为空 |
| EXP-09v2 | DONE⚠️ | ELF×3 | 方向性不对称探索（仅描述性） | 晚期 fc_delta >> cf_delta；⚠️ 两方向 risk set 不可比（cf n=2–3）；⚠️ 早期可比时段 cf 反而略strong；⚠️ 不进主文（见 paper_revision_notes） |
| EXP-09v3 | **DONE** | ELF×3 | stable_k=3 oracle commit timing（fixed noise, exp07b_v2） | baseline never_commit=24.8%；kd_cr never_commit=**0.67%**；kd2=1.07%；kd_cr by t=0.5: **99.3%** committed vs 75.2% baseline |
| EXP-25 | DONE⚠️ | LangFlow | LangFlow token 类型时序（EXP-08 对应） | func t*=0.825 vs content t*=0.875，Δ=−0.050；⚠️ 频率/surprisal混淆；跨模型Δt比较无效；不能称为 coarse-to-fine |
| **EXP-25v2** | **DONE** | LangFlow | 出现级 logistic 回归：func effect after freq control | β_func 在 t<0.83 为负（OR=0.26-0.73）：控制频率后函数词不早承诺；β_freq 始终正（OR=2-7.5）：**频率是主要驱动因素** |
| EXP-26 | DONE⚠️ | LangFlow | LangFlow 空间聚集性（EXP-09 对应） | 峰值 Δ=+21.1pp 但 far_n=6（risk-set collapse）；⚠️ 共因混淆；统计不显著；跨模型比较无效 |
| **EXP-26v2** | **DONE** | LangFlow | Moran's I 空间自相关 + 离散时间风险模型 | Moran's I=0.26@t=0.745 (z=22.54,p<0.001)；has_neighbor OR=2.44 [2.19,2.76]；but common-cause confound unresolved |
| EXP-27 | DONE⚠️ | LangFlow | LangFlow token 频率 vs 承诺（EXP-20 对应） | r=0.47 (log tok-id vs t*)；⚠️ T5 ID ≠ OWT频率；LangFlow用GPT-2 tokenizer，ID无频率语义 |
| **EXP-27v2** | **DONE** | LangFlow | GPT-2 OWT真实频率 vs 承诺时序 | Pearson r=-0.651 (p=4.7e-24)；偏相关控制POS后=-0.638；func-content Δ=-0.014（极小）；**频率效应远强于POS效应** |
| EXP-28 | DONE⚠️ | LangFlow | LangFlow 方向性（EXP-09v2 对应） | fc=+2.7pp，cf=+5.4pp；⚠️ 无统计显著性；cf>fc有共因解释；"无方向传播"结论不成立 |

## Story C: Mechanism & Stability

| EXP | 状态 | 模型 | 核心问题 | 一句话结论 |
|-----|------|------|----------|-----------|
| EXP-11 | DONE⚠️ | ELF×3 | 扰动稳定性（branching stability） | **结果不可信**：`sigma=noise_frac*rms_mean` 导致实际扰动为 ~22.6% 而非 1%；必须用 per-position 单位球缩放重跑，并 sweep η |
| EXP-11v2 | **DONE** | ELF×3 | 扰动稳定性（corrected per-position scaling, η sweep） | S_orig@t=0.81: kd_cr **29.60%** > kd2 **25.68%** > baseline **17.82%**；KD 改善后期稳定性；baseline 非单调（t=0.31/0.69 出现 dip，与 mean_last_flip_step=21.2 吻合）；S_pair≈0.95-0.98 全程高（bifurcation into common alternative，三 checkpoint 通用） |
| EXP-12 | DONE⚠️ | ELF | 近错误分析（近错词分布/rank） | 描述性结论有效（kd_cr wrong rank~16 vs baseline ~372）；但跨 checkpoint 错误集 selection bias，需 paired rank + MRR + calibrated logit gap |
| EXP-12v2 | **DONE** | ELF×3 | 固定 baseline-wrong set paired rank analysis | ref=baseline_never (25.1%): kd_cr@t=0.30 correct=82.6%, MRR=0.878, median_rank=1 vs baseline correct=4.4%, median_rank=12, logit_gap 66.65→1.46；ref_t030 (38.1%): kd_cr@t=0.30 correct=74.5% vs baseline 0% |
| EXP-13 | DONE⚠️ | ELF×3 | dec_sc 是计算量还是信息？ | **H0/H1 结论需修订**：tmin=0.5 有 compute≈decode 趋势，但仅 1 seed、512 samples、无 CI；decode-shuffled 更好有多种替代解释；kd_cr 受 seed artifact 污染 |
| EXP-14 | DONE⚠️ | ELF | commit-release-recommit 验证 | **主要结果无效**：使用余弦 x̂_t 相似度而非正确 decode path（GELU(h_L11@proj)@unemb）；top-1 flip ≠ commit-release；需重跑正确 readout |
| EXP-14v2 | **DONE** (all 3) | ELF×3 | Protocol B 翻转统计（正确 decode path） | baseline: 5+=67.6%, mean=6.08; kd_cr: 5+=48.4%, mean=4.66; kd2: 5+=45.8%, mean=4.48；**KD 比 baseline 更稳定（旧 EXP-14 结论逆转）**；G_B(t=0.97): baseline 73%, kd_cr 66%, kd2 73%；stable_commit metric broken（初始假阳性） |
| EXP-15 | DONE⚠️ | ELF | 参数空间层分析 | L10/L11 参数变化最大（late blocks 整体变化大）；⚠️ 参数 L2 变化 ≠ functional importance；需 module 级分解、update direction 相似性、block swap 才能说"L10 是核心作用层" |
| EXP-15v2 | **DONE** | ELF×3 | Module-level 参数分解 + update direction similarity | **最大变化在 decode head**: unembed_bias R=2.59；L10/L11 R=0.338；early blocks R~0.22（非原 spec 的 0.08-0.12）；cos(Δkd_cr,Δkd2): L0=0.83→L11=0.90 |
| **EXP-38** | **DONE** | ELF×3 | Logit Lens：decode head 施于各层残差流，测量每层可读出的正确 token 精度 | kd_cr/kd2 从 B00 就领先 baseline（t=0.5: B00 0.40 vs 0.19）；baseline 非单调（B10>B11）；kd_cr≈kd2 全层（SC差异≠logit lens差异）；见 EXP-38-spec.md |
| **EXP-39** | **DONE** | ELF×3 | Decode Head Cross-Patch：3×3 backbone×head 因果矩阵，隔离 decode 接口 vs backbone 的 oracle accuracy 贡献 | **颠覆性**：kd_cr backbone + baseline head=99.4%；baseline backbone + kd_cr head=80.8%；**backbone 是主要因素**（非 decode head）；见 EXP-39-spec.md |
| **EXP-40** | **DONE** | ELF×3 | unembed_bias 词汇分析：Δbias_kd_cr vs Δbias_kd2，Top-50 promoted/demoted token | **关键负结果**：cos(Δkd_cr, Δkd2)=0.954（bias 变化几乎相同）；kd_cr/kd2 差异不在 bias；多语言 token 双向变化（促进 66%/抑制 76% 为非 ASCII）；见 EXP-40-spec.md |
| **EXP-41** | **DONE** | ELF×3 | Decode Hidden 对齐：cos(decode_hidden_i, unembed_kernel[:,y_i]) 区分正确 vs 错误位置 | AUC 0.68-1.00（cos_align 可预测 oracle）；baseline correct cos=0.234 最高但 acc 最低（75%）；kd_cr cos=0.181 较低但 acc=99.5%；KD 非通过"更尖锐对准"提精度；见 EXP-41-spec.md |
| **EXP-42** | **DONE** | ELF×3 | 残差流差异（CKA）：逐层比较 kd_cr vs baseline 的 h_i，定位 KD 影响层级 | **KD 主要影响 B08-B11**（CKA 0.896/0.803/0.564/0.427）；B00-B07 相对稳定（CKA 0.93-0.98）；kd_cr vs kd2 全层 CKA ≥ 0.88（两者相似）；见 EXP-42-spec.md |
| **EXP-43** | **DONE** | ELF×3 | Dual-Path Gradient Conflict：B11 残差更新对 decode path 和 reconstruction path 的同时影响；插值曲线 + gradient conflict cos | **Tradeoff 假说确认**：baseline ΔL_dec=+7.92（变差），ΔL_rec=−501（变好）；KD 两路均改善（ΔL_dec<0，ΔL_rec<0）；gradient conflict cos 弱信号（全部轻微负，无法区分）；baseline h10 L_rec=529 vs KD 103-142（重建计算分布差异）；见 EXP-43-spec.md |
| **EXP-44** | **DONE**⚠️ | ELF×3 | 完整模块 Factorial Patching（Phase 1: oracle acc + L_rec；Phase 2: SC module swap 生成实验） | **Phase 1 null**：L_rec 几乎相同（76.2 vs 76.4），排除 x̂_t 质量；**Phase 2 因果**：self_cond_proj 是主因，换 kd2 proj→kd_cr 使 SC 翻转（I: +135→−47，ΔΔI≈−182）；换 kd_cr proj→kd2 使 SC 急剧恶化（I: +25→+370，ΔΔI≈+345）；⚠️ pipeline 与 EXP-36v2 不兼容（自定义 ODE，native I 方向反转）；见 EXP-44-spec.md |
| **EXP-45** | **DONE** | ELF kd2+kd_cr | SC Activation Patch：生成时将 kd2 的 x̂_t 替换为 kd_cr 的（λ∈{0,0.5,1}），观察 SC interaction 是否翻转 | **NO FLIP（重要负结果）**：patch_lam10 I=+34.58（仍正，未翻转）；zeros_sc I=+93.55；结论：x̂_t 格式非主因，SC 兼容性来自 backbone 本身（self_cond_proj）——与 EXP-44 Phase 2 一致；见 EXP-45-spec.md |
| **EXP-46** | **DONE** | ELF kd2+kd_cr | SC Jacobian：计算 J_SC·δs_t = ∂v_θ/∂s_t · (x̂_t_kd_cr - x̂_t_kd2)，检验 velocity 响应方向 | **t=0.30 关键分歧**：kd_cr cos_align=+0.061/frac_pos=0.828 vs kd2 cos_align=−0.022/frac_pos=0.312（69% 轨迹反对齐）；t≥0.50 两模型均对齐（收敛）；与 EXP-44/45 综合：kd2 的 self_cond_proj 在关键承诺窗口（τ_50s≈0.20-τ_aff≈0.32）Jacobian 方向错误；见 EXP-46-spec.md |
| **EXP-47** | **DONE**⚠️ | ELF×3 | 中间层 SC：x̂_α=final_layer(h_10+α(h_11-h_10))，α∈{0,0.25,0.5,0.75,1}，测 SC interaction I(α) | kd_cr/kd2 单调趋势（低 α → 更好）；baseline 对 α 不敏感；⚠️ pipeline 与 EXP-36v2 不兼容（metric=NLL 非 PPL；自定义 ODE 循环；kd_cr/kd2 生成文本退化）；见 EXP-47-spec.md |
| **EXP-48** | **DONE** | ELF×2 | 中间层 SC 正确 pipeline（修复 EXP-47 两个 bug）：真实 PPL，natural 作为参考基线 | **kd2 h_10 SC 重大突破**：PPL 247→91（I=−157）；kd_cr h_10 SC ≈ no SC（均 PPL≈187-192）；α=0.5 中间结果（kd2: PPL=150）；见 EXP-48-spec.md |
| **EXP-49** | **DONE** | ELF kd_cr | D1: 中间 L_rec 监督（B10），合成数据快速微调 500 步 | aux_loss 0.127→0.040（−69%）；EXP-51 中评估 SC 效果；合成数据限制泛化；见 EXP-49-spec.md |
| **EXP-50** | **DONE** | ELF kd_cr | D3: Gram 矩阵对齐损失（final_layer ↔ self_cond_proj）, 500 步 | Gram alignment 0.0096→0.0036（−63%）；EXP-51 评估 SC 效果；见 EXP-50-spec.md |
| **EXP-51** | **DONE** | ELF×3 | 评估 D1 和 D3 checkpoint 的 intermediate SC（kd_cr vs d1 vs d3）| **D1 崩溃**（PPL≈1.5，synthetic data 破坏模型）；**D3 部分退化**（none/h10 PPL≈6）；synthetic fine-tuning 无效，D1/D3 需真实 OWT 数据；**D2（h_10 SC）是本阶段唯一成功的改进**；见 EXP-51-spec.md |
| **EXP-52** | **DONE** | LangFlow | LangFlow Logit Lens（EXP-38 对应）：output_layer 施于各层 h_i，backbone-only 和 full(+skip) 两路 | **t≤0.70 全层 full top-1 ≈ 0.001**（与 ELF logit lens 形成强对比）；t=1.0: h0=89.4%→h12=98.3%；skip-only=92.1%；backbone-only@h12=73.7%；**skip 主导** EXP-21v2 结构性印证；见 EXP-52-spec.md |
| **EXP-53** | **DONE** | LangFlow | LangFlow T_stable(K=3)（EXP-16v2 对应）：fixed-ε oracle，51 t 点 | **never-stable=4.79%**（介于 ELF baseline 25.1% 和 kd_cr 0.53% 之间）；mean T_stable=0.840（远晚于 ELF kd_cr ~0.18）；G_oracle 非单调（t≈0.5 有 dip，cliff at t=0.806）；见 EXP-53-spec.md |
| **EXP-54** | **DONE** | ELF kd2 | h₁₀ SC 标准流水线验证：sccfg∈{1,3}×arm∈{natural,h10}，N=256，128-token，ODE-32 | **h₁₀ SC: I=−129（45%降低）** PPL 284.7→155.4（sccfg=1）；sccfg=3 略差（h₁₀: 168.8，natural: 295.7）；EXP-48 结论验证成功；EXP-48 I=−157 是高方差 64 样本估计，256 样本更可靠；见 EXP-54-spec.md |
| **EXP-54b** | **DONE** | ELF kd2 | h₁₀ SC 多种子方差验证：seeds {42,123,456}，N=256/seed，3 arms | **I=−129.7±9.3（95% CI，std=3.7）**：43% PPL 降低（170.1 vs 299.8）；sccfg=3 一致性比 sccfg=1 差+9.6 PPL（3 seeds 全稳健，确认 EXP-44 B11 反相关）；见 EXP-54b-spec.md |
| **EXP-54c** | **DONE** | ELF kd2 | SC_T_MIN 扫描：h₁₀ SC sccfg=1，tmin∈{0.0,0.1,0.25,0.5}，N=256 | **SC_T_MIN=0.5 gate 为关键**：tmin=0.0→PPL=1369（+1084，灾难性）；tmin=0.5→PPL=155（−129，45%改善）；单调退化确认 t<0.5 局部优化需要 h₁₁；验证 EXP-42 CKA 分叉在 t=0.5 处的机制意义；见 EXP-54c-spec.md |
| **EXP-55** | **DONE** | ELF kd_cr+kd2 | Strategy B：两阶段 staged decoding（fresh noise pass 2），按 left50/置信度锁定位置 | **kd2 staged_left50: I=−121（prefix completion 有效）**；kd_cr left50: I=+109（SC pathway 被绕过，有害）；confidence arms 两模型全部变差（fresh noise 轨迹不一致）；见 EXP-55-spec.md |
| **EXP-56** | **DONE** | ELF kd_cr+kd2 | Strategy C：单次 ODE 内渐进承诺，在 t_next=0.5 处按置信度固定高置信位置（cond_mask），后半段继续 ODE | **kd_cr prog_t05_c70: I=−164（PPL 332→168，超过 EXP-54b h10 SC 的−130）**；kd2: I=−71；两模型均单调改善（阈值越低越好）；见 EXP-56-spec.md |
| **EXP-55v2** | **DONE** | ELF kd_cr+kd2 | EXP-55 修正版：pass 2 使用相同噪声（same z0）验证轨迹不一致性假说 | **假说被否**：同噪声不修复 confidence arms（kd2 conf70 仍 I=+41）；kd2 left50 I=−118（vs 新鲜噪声 −121，差异可忽略）；staged decoding 劣于 EXP-56；见 EXP-55v2-spec.md |
| **EXP-56b** | **DONE** | ELF×3 | EXP-56 commit 时间点扫描：t∈{0.3,0.4,0.5,0.6,0.7}，threshold=0.7；+baseline arm（2026-08-03） | **kd2 prog_t30_c70: I=−186.4 PPL=285→98（系列最佳）**；kd_cr prog_t40_c70: I=−175.1；**baseline prog_t50_c70: I=−11.26 PPL=122→111（9% 改善，有效！）**；optimal commit time 随模型可靠性右移（kd2 t=0.30 / kd_cr t=0.40 / baseline t=0.50）；baseline t=0.30 灾难性 I=+58.69（72% commit 率太激进）；见 EXP-56b-spec.md |
| **EXP-57** | **DONE** | ELF kd_cr+kd2 | 叠加 h10 SC + progressive commitment（arms: standard/h10_only/prog_only/h10_prog）| **堆叠反协同**：kd_cr h10_prog I=−90（劣于 prog_only −164 和 h10_only −124）；kd2 h10_prog I=−95（介于两者之间）；h10 SC 使置信高于阈值的位置比例下降（34-44% vs 61-70%），破坏 commitment 效果；最优：kd2/kd_cr 均为 EXP-56b prog_t30/40；见 EXP-57-spec.md |
| **EXP-58** | **DONE** | ELF kd_cr+kd2 | Pipeline ODE（diffusion forcing 近似）：T=16 组，31 次模型调用，两种共享 t 方案（global/avg）；+ conditioned generation 评估（ROUGE-L）防 PPL hacking | **kd_cr pipeline_avg 最健康**：I=−142 D1=0.310↑ D2=0.809↑（多样性上升，无 PPL hacking 迹象）；kd_cr pipeline_global 可疑：PPL=37（异常低）D2↓22% ROUGE-L=0.012（无实质前缀延续）；**kd2 两种方案均崩溃**（global I=+549，avg I=+200），异步调度破坏 kd2 的 h10 SC 交互；conditioned eval 确认 ROUGE-L 可有效检测 PPL hacking；见 EXP-58-spec.md |
| **EXP-59** | **DONE** | ELF kd_cr | EXP-58 kd_cr pipeline_avg 多种子验证（3 seeds: 42/123/456，N=256/seed，pipeline_avg vs standard）| **I=−153.6±30.7（95% CI，n=3，df=2）**；D1=0.308±0.009↑ D2=0.810±0.009↑（3 seeds 全部方向一致）；σ(I)=12.3（CV=8%，低种子方差）；EXP-58 结论稳健确认；见 EXP-59-spec.md |
| **EXP-60** | **DONE / NEGATIVE** | ELF kd_cr | Native Wavefront Flow Forcing：配对的同步 control 与局部时间训练各 500 steps，检验 GS19 的失败是否只是 train--test mismatch | LTR interaction 分别恶化 +45.5/+24.6 PPL，标准 ODE 也恶化 27.3，gate 近零；停止当前 WFF 实现；见 EXP-60-spec.md |
| **EXP-61** | **DONE / NEGATIVE** | ELF kd_cr | 以 2×2 矩阵重验 Pipeline ODE 的 noise scale 与 SC-CFG protocol；converted payload provenance 单独审计 | n=256 下 legacy noise 1 改善 142 PPL，但 native noise 2 恶化 197 PPL；当前 Pipeline claim 终止，不扩展 baseline/kd2；见 EXP-61-spec.md |
| **EXP-62** | **SUPERSEDED / NEGATIVE** | ELF baseline fine-tuning panel | matched noisy-head self-distillation windows | early window PPL 最低（159.5），但样本是 code-like fragmented pseudo-text；ODE-64 degeneration 9.8%；更关键的是 PyTorch teacher/gate/normalization 与原 JAX KD objective 不一致，不能回答 KD-window 因果问题；见 EXP-62-spec.md |
| **EXP-63** | **DONE** | ELF baseline corrected fine-tuning control | clean-`x0`, `t=1` stop-gradient decoder teacher + smooth JAX gate versus matched continued training | Early-window KD 在两个训练 seed 上改善 unconditional ODE quality、`tau_first`、`tau_stable` 与 revisions；conditioned gain 不稳健；见 EXP-63-spec.md |
| **EXP-64** | **DONE** | ELF base + Broad-KD + Commit-KD | 用统一 native recipe 补齐方法横向表：3 seeds、noise scale 2、SC-CFG 3、PPL/D1/D2/degeneration/conditioned ROUGE-L | **Hard Commit 在三 checkpoint 上均保留 PPL 与 conditioned-quality 信号**；Commit-KD 最干净，Broad-KD 有 D1/D2 与 unigram lock-in 代价；Pipeline 两 checkpoint 均失败，Two-pass 无性价比，local-clock 灾难性失败；见 EXP-64-spec.md |
| **EXP-70** | **DONE / NEGATIVE** | ELF base + Control + Early-KD | Pipeline clock/state factorization：current shared clock vs per-block true-time oracle vs final synchronous refinement | true-local oracle 和 refinement 均失败；mixed-state error 约为 clock error 的 3--4 倍；关闭离散 heterogeneous Pipeline；见 EXP-70-spec.md |
| **EXP-71** | **DONE / NEGATIVE** | ELF base + Control + Early-KD | Synchronous Soft-Anchor Pipeline：统一 global time 下，以 fresh prefix self-conditioning 引导 suffix，比较 LTR/RTL/random/confidence/shuffled-content 与 compute controls | shuffled content 灾难性，证明内容因果作用；但所有正确 anchor 均输给 ODE-64 且无 LTR 优势；见 EXP-71-spec.md |
| **EXP-72** | **DONE / STOPPED AT 500** | ELF base | Native Multi-Time ELF v2：逐层 local-time conditioning、LTR curriculum、强制 clock-sensitivity gate | 同步质量保持，但 EMA clock response 与 control 无区别，LTR interaction +20.5 PPL；按 gate 停止；见 EXP-72-spec.md |
| **EXP-73** | **IMPLEMENTED / NOT LAUNCHED** | EXP-72 smoke checkpoint | On-policy Wave Trajectory Distillation：teacher wave state、scheduled student rollout、final global refinement | 一步 smoke 通过；因 EXP-72 未学到方向性 clock，正式训练无法隔离 exposure bias，故未启动；见 EXP-73-spec.md |
| **EXP-74** | **DONE / HARD-ONLY POSITIVE** | ELF base + Control + Early-KD | Event-Triggered Anchoring：transition window 中一次性 soft memory 与 persistent hard condition，并做时间/置信度/稳定性密度控制 | soft memory 无效；hard `.90` 在三个 checkpoint 上将 PPL `278.7/276.4/199.8 -> 205.3/215.8/169.3`，stable 60--64% density 仍保留正号；见 EXP-74-spec.md |
| **EXP-75** | **DONE / NEGATIVE** | ELF base + Control + Early-KD | Canonical-Context Wave：target block 保留 latent，其他位置以 predicted-clean state 作为 context | 相比 raw Pipeline 仅部分降低 PPL，但 vector error 不降反升且仍为 word salad；不实现完整 Q/K/V 架构；见 EXP-75-spec.md |
| **EXP-76** | **DONE / PARTIAL PASS** | ELF base frozen backbone | Clock-Adapter Bootstrap：冻结 backbone，仅训练逐层 local-time adapter 拟合 teacher wave velocity | held-out MSE 降约 46%，Standard PPL 保持 265.2，clock sensitivity 增强；wave PPL 仍恶化；仅允许 EXP-77 bounded Stage 0；见 EXP-76-spec.md |
| **EXP-77** | **DONE / NEGATIVE** | EXP-76 bootstrap | Asynchronous Block Transition Distillation：在 staggered sequence 内监督一个 active block 的下一 local transition，比较 sync/off-policy/on-policy/RTL | Standard PPL 健康，但所有 31-call fill/drain sampler PPL `3400--3900`，无 LTR advantage；局部 transition 不可组合；见 EXP-77-spec.md |
| **EXP-78** | **DONE / ODE-ONLY REVISABLE POSITIVE** | ELF base + Control + Early-KD | Robust Revisable Commitment：三 seed ODE、conditioned continuation、native SDE 与 Unlock-4 timing audit | ODE 中 Unlock-4 在三 checkpoint 将 mean U-PPL `285/265/204 -> 209/203/166`，conditioned 同号；释放后 8--10% anchor 会改写；native SDE 仅 `0--1` PPL 量级变化，故限定为 deterministic-solver inference method；见 EXP-78-spec.md |
| **EXP-79** | **DONE / CONDITIONAL NEGATIVE** | ELF base | Late-Coupled Block Denoising：前后 block 分别成熟后再 joint refinement | fixed-prefix P1 confirms failure: m24/m28 prompt-cond PPL `419.0/420.2`, only Semi-AR level `421.2`, far worse than Parallel-32/60 `252.8/134.5`; ROUGE-L `.0958/.0967` vs parallel `.1020/.1027`;见 EXP-79-spec.md |
| **EXP-80** | **DONE / ASYNC NEGATIVE; UNLOCK PPL GAIN REPLICATED** | ELF base | Paired Conditional Revalidation：同一方法、noise、seed 下并行测 unconditional 与 native-prefix conditional | soft/local/canonical 不被 conditioning 救回；Unlock-4 vs Standard-32 的 U-PPL 改善 `68--82`、C-PPL 改善 `134--156` 在 3 个 P1 panel 均复现，但 prompt-gain 不稳定且存在 diversity/repetition 代价；见 EXP-80-spec.md |
| EXP-22 | DONE⚠️ | LangFlow | LangFlow 每位置承诺时序（EXP-16 对应） | t<0.80 几乎无位置承诺（模型内有效）；⚠️ 所有 ELF–LangFlow nominal-t 比较无效（EXP-03 已证 log-SNR 不可比）；"LangFlow 比 ELF 晚 0.63t"必须从论文删除；H<1 nat 阈值跨模型不可比；committed_wrong 低是选择效应 |
| EXP-24 | DONE⚠️ | LangFlow | LangFlow 轨迹稳定性（EXP-14 对应） | LangFlow mean_last_flip=8.3/32（26%）vs ELF baseline 21.2/32（66%，EXP-14v2）；⚠️ 原对比表使用旧版 EXP-14 无效数字（83.4%→正确值 67.6%）；argmax stability≠commitment；LangFlow 可能有 self-conditioning；缺少 entropy/margin 分析 |

## Dense / 视觉化实验

| EXP | 状态 | 模型 | 核心问题 | 一句话结论 |
|-----|------|------|----------|-----------|
| probe_decode_v2_dense | DONE | ELF baseline (JAX) | 51-t 高密度 decoder 曲线 | cliff t=0.20→0.30: 8.6%→45.3%；plateau ~79% t=0.50–0.90；jump 97.9% at t≥0.98 |
| probe_anchor_v5_dense | DONE | ELF baseline (JAX) | 51-t anchor probe d_soft | d_soft 在 t<0.10 快速下降，plateau 至 t≈0.90；d_nn≈1216 始终不变 |
| EXP-07 (64-step) | **DONE** | ELF×3 | 64-step probe_acc + decoder_rec1 | baseline/kd_cr/kd2 全部完成 Jul-21；密集曲线已更新至 cclf_viz.html Chart 1 |
| **EXP-29** | DONE⚠️ | ELF×3 | 每位置每步 kNN 词可视化 | ✅ fixed noise 已确认（probe_layerwise.py:116-134，seed=42 全 t 复用同一 ε）；⚠️ centroid 偏向 baseline；跨 checkpoint 比较无效；cherry-pick 风险；仅适合 appendix 定性图 |

## Phase Transition (PT) 系列 — 新方向（docs/phase_transition_experiment_suite.md）

统一 ELF/LangFlow adapter 层 + 10 个新实验（EXP-PT1–PT10），目标是把"承诺悬崖"分解为
prior-masking → evidence-emergence → boundary-crossing → stabilization 四阶段，并做因果验证。
详见 `docs/phase_transition_experiment_suite.md`（原始协议）与各 `EXP-PTx-spec.md`（落地细节）。

| EXP | 状态 | 模型 | 核心问题 | 备注 |
|-----|------|------|----------|------|
| EXP-PT1 | DONE⚠️（已加bootstrap CI） | ELF×3 + LangFlow（padding修复+2000次bootstrap） | Prior-to-Evidence 分解（3 种 reference prior） | **padding bug 已修复重跑**（ELF null_mode_token 曾是 pad token id=1，现修复为 id=3"▁"）：修复后 frac_null→specific baseline骤降至**3.0-3.8%**（原25.7-64.2%几乎全是pad假象），KD降至**57-70%**；**核心发现有紧致bootstrap CI支持**：`P(m_res>0 across 2000 resamples)=1.000`对全部模型/reference成立——"减先验后margin转正"极其稳健；m_res advantage retained不受影响(~10-12% KD vs ~1% baseline)；**KD独有**：never_res<never_raw；见 EXP-PT1-spec.md、EXP-PT-rigor-audit.md |
| EXP-PT2 | DONE⚠️（已加bootstrap CI） | ELF×3 + LangFlow（+probe-gap补测+2000次bootstrap） | 真值-默认 margin 轨迹与转变时间 | KD tau_e/tau_b(~0.09/0.20) 远早于 baseline(~0.19/0.35)，与EXP-10/16v2一致，**CI很窄**(如baseline successful_monotonic=72.79%[71.86,73.57])；KD post-crossing slope(85-90)远陡于baseline(16.5)；⚠️KD no_emergence(6.9-7.1%)反而高于baseline(0.44%)；LangFlow: isotonic R²=1.0但multiple_revision 65.7%>>successful_monotonic 22.9%；**独立探针分数补测**：baseline gap=+0.12(probe>native)，KD/LangFlow gap=−0.08~−0.09(native>probe)——与已有EXP-07v2/EXP-21v2符号方向完全一致；见 EXP-PT2-spec.md、EXP-PT-rigor-audit.md |
| EXP-PT3 | DONE⚠️ | ELF×3 + LangFlow（全部正式规模，含probe-direction-B补测+bootstrap CI） | Velocity 对齐与积分证据 | **a_clean(t_min)全模型强正(+0.65~+0.79)，2000次序列级bootstrap CI极窄不含0**——全suite最干净的正面发现；KD frac_valid_direction(86.5-86.6%)远高于baseline(35.3%)；⚠️**real(u_yf) vs frequency-matched control的corr置信区间在全部4模型上都重叠**（random/orth两个control的CI则紧贴0、明显不重叠）——doc判定规则"controls不应显示同样对齐"确认未满足，不是点估计噪声，4/4模型一致；**probe方向补测**：centroid(A)与probe(B)两种方向cos仅0.13-0.34(不是同一件事)；反直觉——更简单的centroid方向预测rank能力全程强于专门训练的probe方向；KD的cos(A,B)(0.34)高于baseline/LangFlow(0.13-0.19)，呼应PT9"KD让表示空间更规整"；kd2在t=0.95出现a_tok符号反转，疑似与PT7的KD晚期不稳定同源；见 EXP-PT3-spec.md |
| EXP-PT4 | DONE⚠️（已做严谨性补强） | ELF×3 + LangFlow（正式规模+bootstrap CI） | 因果上下文来源消融 | **修复了LangFlow探针数3→26**(seq_len 128→1024,消除样本量过小问题)+加了按序列bootstrap CI；**bootstrap后的精确结论**：ELF baseline的local_r1与full_context统计不可区分(CI含0)，KD有小但显著差距(−0.9~−1.1pp)，**LangFlow差距大且显著(−21pp[−23.6,−18.4])**；global_only_r0在KD上出奇地高(kd_cr 0.76,kd2 0.79 vs baseline 0.44)；global_only_r1+全模型崩溃至~1%；采用值层面代理方案(非精确逐位置masking)；见 EXP-PT4-spec.md、EXP-PT-rigor-audit.md |
| EXP-PT5 | DONE⚠️ | ELF×3 + LangFlow（全部正式规模+bootstrap CI） | Decoder-bias 干预（readout-only 诊断） | **baseline 与 KD 方向相反，2000次序列级bootstrap CI确认非噪声**：baseline shift=+0.006[+0.006,+0.007](弱支持decoder边界,frac_boundary_explained=4.06%[3.9%,4.2%]最高)；kd_cr shift=−0.060[−0.061,−0.059]/kd2 shift=−0.058[−0.059,−0.057](先验减法反而推迟tau_b，CI都不含0，frac_boundary_explained仅0.85-1.29%)，与LangFlow(shift=−0.347[−0.352,−0.342])方向一致；beta=8时KD flip rate(45.8-46.9%，CI窄)远高于baseline(16.5%[15.9%,17.2%])；解读：KD已把"提取证据"学进native readout，通用先验修正对KD反而是干扰，见 EXP-PT5-spec.md |
| EXP-PT6 | DONE⚠️（已做多种子复核+样本量翻倍） | ELF×3 + LangFlow（+kd_cr 3-seed复核+全部4模型N=64→128复核） | 转变点附近的局部稳定性（完整状态分支） | baseline: flip rate随时间单调下降且**final<imm(自我纠正)**；**kd_cr/kd2相反：final是imm的3-5倍(放大而非纠正)**，如kd_cr tau_b_minus时imm=6.4%但final=31.4%；**kd_cr用seed 42/123/456三次独立复核，"放大而非纠正"方向和量级稳健复现**(final/imm比值2.9-3.7×)；**N=128(翻倍样本)在全部4模型、每个checkpoint上都复现了N=64的方向和量级**(imm几乎逐位吻合)，不是单一种子/小样本巧合；LangFlow flip rate全程远高于ELF且同样是放大模式，与PT2的multiple_revision 65.7%一致；"更早决绝承诺"≠"对扰动更鲁棒"，见 EXP-PT6-spec.md |
| EXP-PT7 | DONE⚠️（已做多种子复核+样本量翻倍） | ELF×3 + LangFlow（+kd_cr 3-seed复核+全部4模型两部分N翻倍复核） | Paired oracle vs free-running 相位对齐 | 扩展了已有EXP-01v3(仅ELF)到LangFlow；四模型均在最早t上G_reverse>G_oracle；KD反而让平均gap变大(baseline 0.097→kd_cr 0.239→kd2 0.271)；**N=64→128复核：核心排序(baseline<kd_cr<kd2)完全复现**(0.100<0.237<0.269)，⚠️但LangFlow从"gap最大"(N=64: 0.276)降到第三位(N=128: 0.257，低于kd2)，"哪个模型精确最大"不稳健，"KD/LangFlow都远高于baseline"这个大方向不变；**causal interpolation最戏剧性发现**：λ=0→0.25全模型提升，之后LangFlow单调升到0.994，ELF baseline温和回落，**kd_cr/kd2灾难性崩溃**(kd_cr在λ=1时agreement仅0.134)；**用seed 42/123/456三次复核，崩溃方向稳健复现**(λ=1均值≈0.22,标准差≈0.08，比单次的0.134更可信)；**causal interpolation额外做N=32→64翻倍复核，四模型五个λ值几乎逐位吻合**(kd_cr λ=1: 0.134→0.165，落在seed复核建立的0.134-0.290范围内)——KD对状态扰动极度敏感，与PT6"放大而非纠正"一脉相承；⚠️分支点SC状态清零可能放大了对KD的伤害，是混淆因素；见 EXP-PT7-spec.md |
| EXP-PT8 | DONE⚠️ | ELF×3 + LangFlow（全部完成+per-UID bootstrap CI） | 受控 minimal-pair 证据来源 | 用BLiMP代替手工构造(480对,6个UID,100%通过过滤)；⚠️BLiMP结构只支持"上下文固定/去噪目标变"而非doc原意"cue变/target固定"；**跨架构稳健发现，2000次per-UID bootstrap CI确认**：existential_there_subject_raising(存在句主语提升)在**全部4个模型上都排名最弱**且**rank_good的CI与其余5个UID的CI全部不相交**(不是断崖点估计的偶然)，determiner/distractor数量一致类持续靠前；⚠️踩坑：好/坏rank比值在KD checkpoint上因rank_bad≈0而数值退化，改用rank_good本身做CI；中间名次排序模型间浮动大，不宜过度解读；见 EXP-PT8-spec.md |
| EXP-PT9 | DONE⚠️ | ELF×3 + LangFlow（全部正式规模+bootstrap CI+4种表示） | 跨时间 evidence-direction 迁移 | 四模型均upper_tri>lower_tri(支持"方向持续累积"而非"逐点重编码")，**2000次序列级bootstrap CI确认非噪声**：upper−lower差值CI在4/4模型上都不含0(baseline 0.106[0.102,0.110]，kd_cr 0.070[0.067,0.074]，kd2 0.072[0.069,0.076]，LangFlow 0.099[0.094,0.103])；LangFlow不对称性最强(比值1.82 vs ELF 1.14-1.24)但绝对迁移准确率低得多(diag=0.24[0.231,0.254] vs ELF~0.67[0.657,0.679])；**关键**：ELF上KD几乎不改变diag_mean(+0.6pp)但upper_tri_mean涨更多(+3.2pp)——直接对应doc"KD改善的是迁移能力而非单点探针准确率"这条判定规则；**补上第4种表示"prior-subtracted logits"**：不训练probe(残差logit本身已是逐类别分数，训probe不可行也无意义)，改用条件正确率P(t_b正确|t_a正确)，全部4模型都复现同一"早期蕴含晚期>>晚期蕴含早期"方向(ELF: 0.89-0.91 vs 0.44-0.53；LangFlow: 0.44 vs 0.05)，⚠️与主表格量级不可比(不同指标定义)，只有方向性可类比；见 EXP-PT9-spec.md |
| EXP-PT10 | DONE⚠️ | ELF×3 + LangFlow（全部完成+bootstrap CI+local-context-strength补充） | 转变失败预测器（多元逻辑回归） | 纯事后分析，复用PT1/PT2/PT3输出，无需新前向传播；val_acc全部好于多数类基线(+2.5~+6.3pp)，**2000次序列级bootstrap CI确认：4/4模型improvement CI都不含0，P(improvement>0)=1.000**；**KD checkpoint 的 log_freq 系数(3-6)比baseline(0.6-1.0)大好几倍**——KD对频率这个"捷径"依赖更强；系数方向与EXP-08v2/27v2一致；⚠️LangFlow prior_mode_advantage系数异常大(+9.13)，疑似padding污染待查；**合并PT4数据补上local_context_gap特征（诚实负面结果）**：val_acc几乎不变(4/4模型)，系数量级远小于log_freq/prior_mode_advantage——因为PT4只有逐序列粒度数据，广播到全部位置后无法解释序列内部的变异，需要改PT4脚本存逐探针位置准确率才能真正检验这个假设，见 EXP-PT10-spec.md |

**基础设施**：`experiments/phase_transition/adapters/{elf_adapter,langflow_adapter}.py`
（统一 `FlowModelAdapter` 接口：encode_clean / make_oracle_state / forward_state /
solver_step / native_logsnr / full_state_clone）。

## Global State Formation (GS) 系列 — 新方向（docs/global_state_formation_experiment_suite.md）

原始 H1（global-before-local：topic < syntax < token）**已被 GS1–GS14 的交叉验证 +
用户审阅 + P0-1~P0-4 方法论复核推翻**。当前最新、最可信的解读见
**`docs/global_state_formation_synthesis.md`**（综合解读文档）：核心不是"全局语义先
形成再细化"，而是"跨位置均值主导的粗粒度统计极早可读、但那主要是 raw pooling 的
统计性质；exact lexical identity 依赖高秩位置特异 residual，其组织依赖上下文因果
耦合和位置间集体协调"。GS15 起的实验（Residual Organization Trajectory 等）直接
追踪这个 residual 组织过程，取代继续挖 topic/sentence 类指标。**GS16 起改用用户
2026-07-31 提供的一批更严格的独立 spec（`docs/specs/EXP-GS16-spec.md` ~ `EXP-GS20-
spec.md`，自带校准协议/多重对照/决策规则，不再对应 `global_state_formation_
experiment_suite.md` 原始条目），GS16/17（P0）已跑完 pilot，把"exploration-collapse"
精化成一个可证伪的三段式机制，见 GS16/17 各自 spec 的"Pilot Results"章节和 synthesis
文档 2026-07-31 更新。**
复用 `experiments/phase_transition/adapters/`（不重新实现），脚本前缀 `EXP-GSx` 对应
原始 doc 的 `GLOBAL-x`（GS11 起为用户审阅驱动的方法论复核/机制追踪，不对应原始 doc
条目）。详见 `docs/specs/EXP-GS1-spec.md` 及 `docs/global_state_formation_synthesis.md`。

| EXP | 状态 | 模型 | 核心问题 | 备注 |
|-----|------|------|----------|------|
| EXP-GS1 | DONE⚠️ (pilot) | ELF baseline | Sequence-Level Probe Hierarchy（MVP-A：topic/sentence-embedding/POS/token probe 的 tau 排序） | pilot n=128, 1024-token, 8 t 点：`tau_syntax(0.38) < tau_topic(0.50) < tau_token(0.65)`——**H1 严格排序（topic<syntax<token）不成立**，同能力 linear probe 下 syntax 反而先于 topic 达标；⚠️ G_sent 指标退化（cos 0.953→0.987 动态范围过窄，疑似高维共线性 artifact，不可引用）；⚠️ topic 标签为 KMeans 自诱导（非外部标注）；⚠️ G_token 用 native top-1（能力不匹配 probe）；见 EXP-GS1-spec.md |
| EXP-GS2 | DONE⚠️ (pilot) | ELF baseline | Hierarchical Branch Consensus（K=8 分支 rollout 的 topic/struct/lex consensus） | pilot 4 docs×8 branches×4 t_start×3 eta：**C_struct 从最早 t_start=0.05 起就≈1.0**（即使同条件 C_lex 只有 0.596），与 GS1 的"syntax 早于 topic"方向**独立交叉验证一致**；C_lex 是唯一随 t_start 单调上升的干净信号；⚠️ C_topic/C_sent 几乎全程贴近 1.0（动态范围过窄），和 GS1 的 G_sent 同源于 mean-pooled cosine，怀疑是同一种退化指标而非真实"topic 更早收敛"；⚠️ eta 校准发现原始 doc 建议范围（≤1e-2）在 1024-token 尺度下几乎无法产生分支分歧，pilot 改用自校准的 {0.01,0.03,0.1}；见 EXP-GS2-spec.md |
| **EXP-GS3** | **DONE (pilot)** | ELF baseline | Low-Rank Global Mode Analysis（per-seq SVD：G_t^(k) global mode vs R_t^(k) residual，k∈{2,8}） | pilot n=64, 1024-token, 9 t 点：**三路清晰解耦**——`A_G(t)` 系统性早于 `A_R(t)`（doc 核心问题正面确认，如 t=0.28 时 A_G=0.505>A_R=0.310）；POS structural signal 几乎全在 `G^(k)`（syntax_G 0→0.78–0.79，**超过 GS1 全量 Z_t 的 0.684**，syntax_R 全程≈0/负）；token identity 几乎全在 `R^(k)`（token_G 全程≤0.03–0.1，token_R 曲线 0→0.99 与 GS1 全量 G_token 曲线形状高度吻合）；两个 k 值定性一致；r_eff 全程 343–472（几乎满秩，不是"整体低秩"而是"结构信息本身低秩、可与高秩 token 部分解耦"）；⚠️ token_G/R 是把合成状态喂给未见过此分布的 backbone 做被动 decode（OOD 风险，不影响 syntax 探针的相对比较）；三个 P0 pilot 里信号最干净的一个；见 EXP-GS3-spec.md |
| | | | **P0（GS1–GS3）阶段性结论** | 三种独立方法学（linear probe / branch consensus / SVD 分解）收敛支持：structural 信号是当前唯一"早、强、可靠"的层级；token/lexical identity 确认最晚确定；"topic"层面证据被 GS1+GS2 共同的 mean-pooled-embedding-cosine 退化指标污染，暂无法下结论（需要换检索式或外部 sentence-embedding 才能重新检验）——即原始 doc 的严格 H1（topic<syntax<token）不成立，但弱化版"structure/token 分层 + topic 待定"成立 |
| **EXP-GS4** | **DONE（pilot + n=16×3seed 大样本复核）** | ELF baseline | Global Mode Causal Intervention（remove/preserve-only/swap global mode，k=8，从分解状态做完整 rollout 而非单次被动 decode） | pilot n=4 时：t_start=0.38 条件 B（只留 G^(8)）topic 归零(0/4)——"低秩 global mode 单独不足以因果驱动生成"；**2026-07-27 用 n_docs=16×3 seed（pooled n=48）+ bootstrap CI 复核（t_start=0.38）**：`baseline=0.917[0.833,0.979]` 远高于 `A_remove_global=0.292[0.167,0.417]` 和 `B_preserve_global=0.229[0.125,0.354]`——**两个条件的 CI 几乎重叠，都远低于 baseline**，即在 ELF 上 G 和 R 单独都明显不足以恢复 topic，必须两者兼备，比 n=4 pilot 只强调"G 不够"更完整（R 单独也不够）；与 LangFlow 同款大样本结果（三条件统计不可区分、都接近 baseline）形成有趣的架构差异：**ELF 上 topic 恢复需要完整状态，LangFlow 上从 G 或 R 单独重构都够** ——这个差异本身值得写进论文；token 维度的残差专属性两个架构都稳健复现；见 EXP-GS4-spec.md |
| **EXP-GS8** | **DONE (pilot, ELF)；LangFlow 版本 2026-07-27 修复过拟合** | ELF baseline / LangFlow | Global-to-Local Causal Chain（往 topic-probe 方向扰动 Z_t，测同 t 上 true-token margin 变化） | ELF pilot n=64，topic probe test_acc=0.632（chance=0.125，探针本身健康）：`correct` 方向 delta_margin 随 alpha 单调上升至 **+1.718**，`orthogonal`/`random` 对照弱得多；⚠️ 首次实现时有 broadcast 扰动范数比例尺 bug（已修复）；**LangFlow 版本此前 test_acc=0.158≈chance（45 训练文档对 8 类分类器严重过拟合，train_acc=0.933）**，2026-07-27 加 L2 正则（`--C 0.05`）+ 样本量 64→200 后：`train_acc=0.314, test_acc=0.217`——过拟合基本解决（train/test 差距从 77.5pp 降到 9.7pp），且 `test_acc` 从 chance 附近提升到 chance 的 1.7 倍，是真实但偏弱的信号；bootstrap CI 确认 `correct`/`wrong` 方向的 delta_margin（如 a=+1.0 时 +0.103[+0.073,+0.135]）明显高于 `orthogonal`（+0.012[+0.002,+0.021]）/`random`（+0.020[+0.013,+0.027]），但不是 ELF 那样干净的单调剂量-反应（U 形，中间 alpha 反而更低）——量级和形状都不如 ELF 干净，但方向性因果链确实存在；见 EXP-GS8-spec.md |
| EXP-GS9 | DONE⚠️ (pilot, 不干净) | ELF baseline | Minimal Global Contrast Sets（12 对人工构造对照句，同 target token 在 own-frame vs wrong-frame 下 log-prob 对比） | ⚠️ **结果不干净，不可引用为结论**：`t=0.65/0.99` 时 delta 几乎全部精确为 0.000——诊断出关键方法论问题：双向 denoiser 在高 t 时主要"读出输入里已经写好的 target token"而非"根据 context 推断"，高 t 下这个设计几乎测不出东西；`t=0.05/0.28` 有极端离群值（如某对 delta 达 ±80–107），均值不可信，改看 median：`t=0.28` 时 median(Delta_A)=+0.654、median(Delta_B)=+0.429 两者皆正（弱支持），但 `t=0.05` 方向不对称（不支持）；根因是每对仅用单一噪声种子（未做 PT2 式的多种子平均），12 对样本量也太小；见 EXP-GS9-spec.md |
| **EXP-GS7** | **DONE (pilot)** | ELF baseline | Oracle vs Free-Running Global Alignment（真正自由生成 n=8，从纯噪声完整 rollout，配对 oracle path，CKA/r_eff/G_token 对比） | GS 系列第二个干净正面结果（另一个是 GS8）：**CKA(oracle,rollout) 呈 U 形，最低点 0.894 在 t=0.50**（略晚于已知 baseline commitment cliff t≈0.20-0.30），两端接近 1.0；`r_eff(rollout)` 中段持续高于 `r_eff(oracle)`（自由生成尚未像 oracle 那样收拢结构）；**G_token 的 oracle-rollout gap 远比几何指标剧烈**：t=0.28 时 oracle=0.589 vs rollout=0.038（15倍差距），t=0.65 才基本追平——把本仓库反复验证的"oracle-rollout gap 是 per-token 现象"定位到全局层面：**几何/全局组织大体跟得上 oracle，真正掉队的是 exact lexical commitment**，与 GS1/GS2/GS3 的"structural 早稳、token 晚脆"主线独立印证；见 EXP-GS7-spec.md |
| EXP-GS10 | DONE⚠️ (pilot, 零方差) | ELF baseline | Global Failure Predictors（大幅缩小范围：3 类粗粒度结果标签代替原始 doc 8 类，n=16 自由生成） | ⚠️ **taxonomy 在标准 32-step 生成质量下无区分度**：16/16 条轨迹全部落入 `healthy`（`degenerate`/`slow_incomplete` 均为 0），无法做原计划的分组特征对比；smoke test 用 `n_steps=8` 时轨迹反而全部变成 `slow_incomplete`，说明 taxonomy 本身敏感，只是标准生成质量下 baseline 天然很少失败（与 EXP-37a/b 的稳定 PPL 一致）；下一步需要主动制造失败条件（降 n_steps，或复用 EXP-33/34/35/37c 已知会退化的 dec_sc+DF 设置）才能获得有效样本；见 EXP-GS10-spec.md |
| **EXP-GS6** | **DONE（pilot + n=20×3seed 大样本复核）** | ELF baseline | Competing Global Basins（不同 topic 的 OWT 文档在 t=0.28 插值 + rollout，判定最终落入哪个 basin） | pilot n=4 对：P_A(lambda) 呈清晰跳变 0.25→0.25→0.50→1.00→1.00→1.00；**2026-07-27 用 n_pairs=20×3 seed（pooled n=60）+ bootstrap CI 复核**：`P_A` = 0.0→0.067[0.017,0.133], 0.2→0.067[0.017,0.133], 0.4→0.133[0.050,0.217], **0.6→0.983[0.950,1.000]**, 0.8→0.950[0.883,1.000], 1.0→0.950[0.883,1.000]——**在大样本下依然非常干净**：λ=0.4→0.6 一步到位（CI 不重叠且几乎不越过中间态），纯端点 CI 都在 0.88 以上（远好于 LangFlow 同款复核的 0.72-0.90），P_other 全程 ≤15%（LangFlow 同条件 10-45%）——**ELF 的 bifurcation 在架构对比下明显更"干净"，这个差异本身是跨架构层面的新发现**，而不是 n=4 pilot 的运气；见 EXP-GS6-spec.md |
| **EXP-GS5** | **DONE (pilot)** | ELF baseline | Collective Coupling and Correlation Length（margin-increment 位置间相关性 `C_t(d)`、correlation length `xi(t)`、susceptibility `chi(t)`，n=64，15 t 点，shuffle-position 对照） | GS 系列第四个干净正面结果（另三个是 GS6/GS7/GS8），**十个 GLOBAL 实验里最后一个跑完的 pilot**：真实 `xi(t)` 在全部 14 个相邻 t-pair 上都高于 shuffle-position 对照（无一例外），证实 margin-increment 的空间相关不是数值巧合；`chi(t)`（跨序列 susceptibility）在 `t=0.28–0.39`（cliff 之后不远）出现明显尖峰（102860/112742，是平台期 44000–52000 的 2–2.5 倍），与原始 doc 预期吻合；`xi` 的 excess（真实−shuffle）同一区间也达到全程最大值（1.140）——两个独立指标同时达峰；峰值区间略晚于 population-mean margin 过零点（t≈0.22–0.28），与 GS7 发现的"CKA 分歧最低点晚于 cliff"是同一个"集体重组滞后于平均翻转"模式；⚠️ 最早一段（t=0.05→0.11）xi 异常大但未必是同一机制，需要更多对照排查；见 EXP-GS5-spec.md |
| **EXP-GS11** | **DONE (pilot) — 关键更正** | ELF baseline | Pooling/Averaging Confound Check（用户审阅指出的 P0-1：raw oracle state z_t 的 mean-pool 是否本身就靠 √L 噪声平均制造"早期 global 信号"？n=48，真实长度截断 sweep L_eff∈{32,128,512,1000}，检索式指标） | **推翻 GS1/GS3 headline 发现的原有解释，本轮最重要的结果**：不经过模型、只对 raw `z_t` mean-pool，`t=0.28` 时**无论 L_eff 多小（32 也一样）都能达到 100% self-retrieval**（chance=1/48≈0.021）；`t=0.05` 时 raw 准确率随 L_eff 单调上升（0.396→0.583→0.792→0.938），完全符合 `1/sqrt(L)` 噪声平均直觉——GS1 的 probe 输入正是这个 raw mean-pool，其"早期 global 信号"基本可以被这一件事解释；**更意外**：模型自己的 `predicted_clean` 输出在几乎所有条件下 self-retrieval 都明显**差于** raw（`L_eff=32,t=0.28` 时 raw=1.000 但 model=0.021=chance），模型处理似乎在抹除而非增强可检索的文档身份信息；GS3 的 SVD 分解同样作用于未中心化的 raw z_t，"structure 集中在低秩 G"的发现同样需要重新解释；已在 EXP-GS1/GS3-spec.md 补充更正说明；GS2/GS4 因为涉及真实多步 rollout，受此 confound 影响较小；见 EXP-GS11-spec.md |
| **EXP-GS12** | **DONE (pilot + 正式规模) — P0-2，拆分验证 GS3** | ELF baseline | Centered Spectral Decomposition（显式分离跨位置均值后重做 GS3，同时测 raw 与模型 predicted_clean，正式规模 n=128，k=8，完整 9 点 t 网格） | **正式规模比 pilot 更确定**：`MEAN_only`（只用均值，不做 SVD）在全部 `18` 个 `(t,repr)` 组合里**无一例外**是三者中最高或并列最高的 structural R²，多数情况下 `MEAN+G_c` 甚至更低——**"structure 集中在低秩 G" 不成立，且置信度随样本量提高而提高**；`token_acc` 上 `MEAN+R_c` 全程远高于 `MEAN+G_c`（如 `t=0.65,model`: 0.814 vs 0.091，约9倍；`t=0.85,raw`: 0.846 vs 0.043，约20倍）——**"token 集中在残差"完整复现**；⚠️ `t=0.99,model` 的 `MEAN+R_c` token_acc(0.584) 仍明显低于 raw 同条件(0.999)，呼应 GS11"模型处理可能抹除可检索身份信息"；见 EXP-GS12-spec.md |
| **EXP-GS13** | **DONE (ELF pilot + 正式规模)；LangFlow 版本 2026-07-27 修复过拟合** | ELF baseline / LangFlow | Context-Only Global-to-Local Intervention（目标位置完全不扰动，只扰动其它位置，t=0.28） | ELF 正式规模（n=128 probe拟合+24 docs×12位置=288组合）：`correct`方向四个alpha点范围`[0.18,0.82]`明显大于`orthogonal`/`random`的`[-0.13,+0.13]`，但呈U形而非单调剂量-反应；`wrong`精确镜像（内部一致性通过）；**LangFlow 版本此前和 GS8 同一个过拟合问题（t=0.65 换后 test_acc 仍=0.158）**，2026-07-27 用同样的修复（`--C 0.05`、n_samples 64→200）重跑：`test_acc=0.217`（chance=0.125），bootstrap CI 显示 `correct` 在 a=+1.0 时 +0.150[+0.044,+0.265] 明显高于 `orthogonal`（+0.010[-0.029,+0.053]）/`random`（+0.037[-0.007,+0.080]，CI 含 0），但 a=-1.0 时 `correct` 的 CI（[-0.043,+0.171]）本身就含 0——**信号比 ELF 更弱、更不对称，只在一个 alpha 上站得住**，是本轮修复里最不干净的一个结果，引用时需要如实注明；见 EXP-GS13-spec.md |
| **EXP-GS14** | **DONE（pilot + n=16×3seed 大样本复核，且用的是修复过 nearest_topic bug 后的代码）** | ELF baseline / LangFlow | True-Trajectory Hierarchical Branching（用真实自由生成轨迹的 (Z_t,SC_t) 替代 GS2 的 oracle 起点+冷启动 SC，重做同款 K=6 branch consensus 分析，t_start∈{0.20,0.38,0.65}） | pilot n=4：`C_struct`/`C_topic` 早早饱和（≥0.937），只有 `C_lex` 有动态范围（0.851→0.969→0.992）；**2026-07-27 用 n_traj=16×3 seed（pooled n=48）+ bootstrap CI 复核（且这是本文件 C_topic 计算首次真正走过修复后的 `nearest_topic`，见下方"严谨性自审"章节）**：ELF 上 `C_topic`=0.974[0.946,0.995]@t=0.20 → 0.982[0.960,1.0]@t=0.38 → **1.000[1.000,1.000]@t=0.65（真实零方差天花板，非 bug）**；`C_lex`=0.858[0.852,0.864] → 0.966[0.962,0.970] → 0.992[0.990,0.993]；LangFlow 同款复核 `C_topic` 略低（0.958→0.985→0.992，未到硬天花板）——差异方向合理（LangFlow 整体 commitment 更晚，t=0.65 对 LangFlow 来说相对没那么"晚"）；GS2 的定性结论（C_struct/C_topic 早饱和、C_lex 是唯一有动态范围指标）在两个架构、大样本、bug 修复后的代码下都稳健复现；见 EXP-GS14-spec.md |
| **EXP-GS15** | **DONE (pilot) — 机制链第一环，负结果** | ELF baseline | Residual Organization Trajectory（追踪真实自由生成轨迹上中心化残差 `R_t` 向自身终点 `R_star` 的 CKA 对齐 `A(t)`，核心对照是纯直线插值 `A_linear(t)`，4 条真实轨迹，8 checkpoint） | **和假设方向相反的负结果，如实报告**：`O_R(t)=A_rollout(t)-A_linear(t)` 在全部中间 checkpoint **都是负的**，`t=0.38-0.50` 最负（约-0.22~-0.23）——free-running rollout 组织 residual 的速度比一条不含任何模型动力学的朴素直线插值还慢，不是"存在加速组织窗口"；paired oracle 路径的 residual 对齐进度全程接近/优于线性插值，rollout 明显落后——首次把 GS7 的"oracle-rollout token gap"量化到 residual alignment 层面，且落后幅度比"落后于一条几何直线"还大；`A_model`（predicted_clean）上升远早于 `A_rollout`（raw），但 `r_eff_model` 全程远低于 `r_eff_raw`，更可能是"低秩、偏通用估计更容易自己像自己"而非真组织；⚠️ 2026-07-31 用户指出 `A_linear` 本身"偷看"了 `R_star`（终点是插值的一个锚点，`t=t_end` 时 `A_linear` 被定义强行=1），是一个不公平（未卡因果性）的基准，该结论已弱化为"局限"而非干净发现，见 EXP-GS15-spec.md 与本节 GS16/17 |
| **EXP-GS16** | **DONE (pilot + formal 3-seed) — 机制链第二环，P0，干净正面结果** | ELF baseline | Calibrated Endpoint Bank, Specificity, and Affinity Collapse（用"一步匹配冲击"协议校准扰动幅度，在 `t_bank=0.20` 建固定候选端点池：自身端点+K=8 个校准扰动分支的去重终点；用这个同一个池给该轨迹之后每个 checkpoint 打分，测自身端点相对优势 `S_self(t)`、排名 `rank_self(t)`、亲和度熵 `H_end(t)`，n_traj=16 pilot / n_traj=48 formal） | **exploration-collapse 判定标准全部满足，是目前 GS 系列里最干净的正面机制结果**；pilot（n=16）见原始描述。**Formal 3-seed（n_traj=48，k=12，seeds 42/123/456）完整确认**：S_self 在 t=0.20→0.30 全负（约 −0.001），rank_frac（标准化后 1=最差）从 0.765 单调升至 **0.930**（t=0.301，最大不确定性），然后在一个窄窗口（t=0.363→0.426）骤降至 **0.174→1.00**；H_end（beta=1）从峰值 0.848±0.020（t=0.301）骤降至平台 0.524±0.001（t≥0.426），ΔH=0.324（−38%）；N_eff 平台=2.48±0.05；cross-seed std 极小（H_end std≤0.025，S_self std≤0.003），3 seed 方向完全一致；见 EXP-GS16-spec.md "Formal Results" 节 |
| **EXP-GS17** | **DONE (pilot + formal 2/3 seeds) — 机制链第三环，P0，与 GS16 交叉验证，精化机制描述** | ELF baseline | Local Residual Dynamics and Unified Transition Timing（有限差分速度场 + GS16 固定候选池的"自身进度 vs 其它候选进度"对比 `V_self(t)`，统一转变时间线 `tau_50_stable`/`tau_velocity`/`tau_affinity`，129 个稠密 checkpoint，n_traj=48 formal，复用 GS16 同一批轨迹） | **精化了 exploration-collapse 的机制描述**；pilot（n=16）见原始描述。**Formal 2-seed（seeds 42/123，seed 456 in progress）核心数字（n_traj=48 × 2 seeds）**：τ_50_stable mean=0.201±0.001（95%CI,df=1）；τ_affinity mean=0.324±0.001；τ_velocity median=0.168（std 大，约0.20，中位数稳定）；**事件顺序（pooled 2 seeds）**：P(τ_v≤τ_50s)=0.906，P(τ_aff≤τ_50s)=0.031（token 稳定晚于 velocity，早于 endpoint 确定，97% 轨迹如此），P(τ_v≤τ_aff)=0.917；**与 GS16 formal 完全互洽**：GS16 endpoint 提交窗口 t=0.363→0.426 与 GS17 τ_affinity 均值 0.324 对应，GS17 τ_50_stable≈0.20 先于 GS16 endpoint 提交约 Δt=0.13–0.23；见 EXP-GS17-spec.md "Formal Results" 节（seed 456 完成后补数字）|
| **EXP-GS18** | **DONE (pilot) — Conditional Reviewer Controls，Part A 负结果 / Part B 正面但重塑** | ELF baseline | Part A: rank/energy-matched 残差对照（GS12 的"残差承载 lexical 信息"是否只是维度/能量效应）；Part B: 用 GS17 真实 rollout + M0→M3 逐步剔除混杂因素（位置、当前 margin、序列级 logit 范数/熵/margin）后，GS5 的"集体协调"是否还能打赢 5 种匹配 null 模型 | **Part A：决策规则不满足**——固定 k 时 top-k 在能量、token 恢复、结构 R² 上全面碾压 middle/bottom/random-k（如 k=128, raw: top tok_acc=0.067 vs middle=0.007/bottom=0.000/random=0.020），说明 GS12 的"残差>>低秩G"更可能主要是维度/能量效应（残差维度多几个数量级），不是"lexical 信息特别编码在非主方向"——GS3/GS12 的核心结论需要**收窄**为"小的 top-k 截断不够，但更大的完整残差够"，不能再说成"分布式高秩编码"；⚠️ 大 k 时移除 top-k 反而让 margin 暴涨（k=128,model: +16.2），很可能是margin 指标在大幅扰动下的伪影（默认竞争者 f_i 固定在未扰动状态算出）。**Part B：M3 残差化后的空间相关长度在 16 个 checkpoint 里有 13 个都超过全部 5 种 null 模型的 95 分位**（position/sequence shuffle、circular shift、sign-flip、方差匹配高斯），在真实 free-running rollout 上支持"collective coordination 不是简单混杂因素"这条结论；但时间形状和 GS5 不同——不是 cliff 后单一尖峰，而是**早期（t<0.35）持续偏高、中段（0.4-0.7）回落、晚期（0.75-0.93）再度上升**，"崩溃点后单一尖峰"这个具体表述需要修正为更宽泛的时间窗口；⚠️ pilot 规模均低于 spec 正式要求（n=64/32，非≥128；无 LangFlow 复现）；**2026-08-03 在 Plaid 上复现（GS20 的一部分）**：**Part A 干净跨架构确认**——k∈{1,2,4,8}（Plaid embedding 只有 16 维，k=16 就是满秩，k∈{32,64,128}无意义故未测）每个 k 上 top-k 都全面碾压 middle/bottom/random-k，和 ELF 同一方向；**Part B 出现真实分歧**——16 个 checkpoint 里只有 1 个打赢全部 5 种 null（ELF 是 13/16），大多数 xi_M3 几乎正好卡在 null p95 边缘，最可能的解释是 Plaid 原生 solver_step 是随机 ancestral 采样、每步都给每个位置独立注入高斯噪声，会稀释掉真实的空间相关性——按 GS20 决策规则，这算一个需要如实报告的边界条件，不是推翻collective coordination；过程中修了两个环境问题（plaid conda 环境缺 nltk/scikit-learn；nltk 触发一个 libstdc++ ABI 冲突，用 LD_PRELOAD 绕过）和一个真实 adapter bug（PlaidAdapter.make_oracle_state 漏了 @torch.no_grad()，调用 Plaid 自己学出来的 gamma_bounds/noise_schedule 模块时会保留梯度图，导致 .numpy() 报错——GS16/17/19 从纯噪声起步、从不调用 make_oracle_state，所以之前没触发过这条路径）；见 EXP-GS18-spec.md「Cross-architecture replication on Plaid」节 |
| **EXP-GS19** | **DONE (pilot) — 干净负结果："all fail"，不建议训练 Wavefront Flow Forcing** | ELF baseline | Order-Controlled Asynchronous Denoising Ablation（给每个位置分配不同的局部去噪进度而非全局标量 t，比较 synchronous/LTR/RTL/fixed_random/confidence_adaptive 五种顺序，Delta=0.20，异构态是从solver_step 真实用到的 xhat/eps_hat 重构出来的，属于"移植到已训练好的标量时间模型上"的train-test mismatch 干预，不是原生 WFF 采样器） | **spec 决策表最后一行"all fail"精确命中**：四种异步顺序在全部三个"期望信号"上都失败——`tau_stable` 不降反升（16.79→19.8-21.7）、`tau_first` 也变大、revision 次数几乎翻倍（5.57→8.2-9.7），生成质量普遍变差（gen_ppl 76.8→144-442，RTL 甚至 75% 样本退化）；四种顺序、三个指标同时一致失败，不是某一种顺序或某一个指标的边缘信号；⚠️ 实现过程中发现并修复了一个真实正确性 bug（`xhat` 最初用`forward_state` 算，和 `solver_step` 实际用的自条件机制不一致，导致 delta=0 的"同步"臂和普通 rollout只有 8% token 一致，修复后验证到 100% 一致）；pilot 规模（n=32, 单 seed, 单 Delta）远低于 spec 正式要求，但四臂三指标一致失败，结论大概率不会因为扩大规模而反转；**2026-08-03 在 Plaid 上复现（GS20 的一部分，顺带把脚本从 ELF 专属的线性 schedule 公式泛化成按 adapter.name 分发的 noise_params()，ELF 回归测试确认改造后数值完全不变）**：**干净跨架构确认，比 ELF 更彻底**——四种异步顺序 gen_ppl 全部暴涨（255.5→779-3688，3.0x 到 14.4x，比 ELF 的 2-6x 还夸张）；有意思的细节：confidence_adaptive 是唯一一个 tau_stable 真的提前了的（21.73→19.70），但 gen_ppl 同时是全部里最差的（暴涨 14.4x）——干净展示了「名义上更快收敛」和「生成质量更好」根本不是一回事；见 EXP-GS19-spec.md「Cross-architecture replication on Plaid」节 |
| | | | **P0-1~P0-4 返工阶段性结论** | 用户审阅指出的四个方法论问题全部检验完毕：**P0-1（pooling confound）证实且比预期更严重**，GS1 的"早期 global 信号"主要是对 raw oracle state 求均值这一操作的统计性质，不需要模型参与；**P0-2（centered SVD）拆分验证 GS3**——"structure 在低秩 G"不成立（被证实是均值 confound），"token 在残差"成立（去均值后依然稳健）；**P0-3（context-only intervention）确认 GS8**——目标位置完全不扰动时因果效应仍存活（约 71% 强度），global-to-local 因果链主张基本站得住；**P0-4（真实轨迹 branching）确认 GS2**——冷启动 oracle 简化不影响定性结论。整体上：本轮返工没有推翻"token/lexical 最晚确定"这条主线（GS3 token 部分、GS8、GS2 全部存活），但彻底推翻了"topic/structure 早期信号=模型早期组织"这个解释（P0-1+P0-2 证实主要是输入统计性质），这是需要写进论文的核心方法论教训 |

### LangFlow 复现（全部 15 个 GS 实验都在 LangFlow baseline 上重跑了一遍 pilot 规模，详见
`docs/global_state_formation_synthesis.md` 第 10 节完整解读）

| GS | LangFlow vs ELF | 备注 |
|---|---|---|
| GS1 | **部分复现，部分不同** | `G_sent` 在 LangFlow 上没有 ELF 那种 t=0.05 即饱和的问题（0.227 vs 0.953），confound 更弱；但 τ 排序反过来了：`tau_topic(0.65) < tau_syntax(NEVER)` |
| GS2 | **清晰复现，且饱和已排除是 bug** | ⚠️ 2026-07-27 严谨性自审发现 `branch_global_consensus.py` 内联了一份**未经过** `nearest_topic` 的手写欧氏距离最近质心分类（与 GS6 的 bug 同源，是原始 copy-paste 出处，此前"四文件修复"漏掉了它，因为它没有调用被 import 的函数）；修复后用 ELF+LangFlow pilot 规模重跑，**C_topic/C_struct 饱和这一结论在两个架构上都不受影响**（ELF/LangFlow 均 0.91-1.00），排除了"饱和=bug 假阳性"这个可能性；eta sweep（0.01/0.03/0.1）显示 C_topic 对扰动幅度有真实的、渐变的响应（不是完全不敏感的天花板），只是比 C_lex 迟钝得多——两个架构上都一致 |
| GS3 | **方向不同（但更支持修正版结论）** | `A_G` 全程低于 `A_R`（ELF 相反），`syntax_G` 几乎全程为负——RAW 数据本身就不支持"structure 在 G"，比 ELF 更早印证了 GS12 的修正结论；`token_R>>token_G` 复现 |
| **GS4** | **bug 修复后 topic+token 双双清晰复现，⚠️ 但 n=4 的"topic=1.00"是过拟合小样本假象，已用 n=16×3 seed 更正** | 原"topic 恒 0.25"**不是样本量问题，是 `nearest_topic` 欧氏距离 bug**（见 GS6）；n=4 pilot 修复后一度显示 `baseline=1.00, A_remove_global=1.00, B_preserve_global=1.00` 全部完美；**2026-07-27 用 n_docs=16×3 seed（pooled n=48）+ bootstrap CI 复核后大幅降温**：`baseline=0.854 [0.750,0.938]`、`A_remove_global=0.875 [0.771,0.958]`、`B_preserve_global=0.812 [0.688,0.917]`——三者 CI **完全重叠、统计上无法区分**，"G 单独已足以恢复 topic（排他性）"这个说法不成立；真正站得住的是 topic 信息在 full/G-only/R-only 三种重构里都冗余存在，不是 G 独占；**token 维度的不对称性完整存活**（n=48 池化：baseline token≈0.57、A_remove≈0.53-0.57、`B_preserve_global token=0.006-0.008` 三 seed 一致，无需 CI 即可看出巨大差距）——"token identity 唯独在残差里"是本实验唯一经得住大样本检验的强结论；`C_swap` topic 跟随残差 donor 方向也复现（vs A_donor=0.19-0.31, vs B_donor=0.69-0.88） |
| GS5 | **形状不同** | `chi` 单调上升不封顶（未到峰值），`xi` excess 早高晚低——暗示 LangFlow 的 commitment cliff 在这个 t 网格之外（更晚） |
| **GS6** | **找到并修复了根因 bug；n=4 pilot 的"干净同步切换"在 n=20×3 seed 下打了折扣，但真实 bifurcation 本身站得住** | 根因**不是** t 校准或 rollout 步长，是 `nearest_topic` 用平方欧氏距离——LangFlow rollout 终点 norm≈1.15 远小于拟合 centroids 用的 clean embedding norm≈3.70，所有 rollout 终点都被分到 norm 最小的那个质心，和内容无关；改成 cosine-based nearest-centroid（已合并进 `common.py`）后 n=4 pilot 一度显示"4 对文档全部在 `lambda=0.4→0.6` 之间同步完成 B→A 整体切换"；**2026-07-27 用 n_pairs=20×3 seed（同一批 20 对，仅噪声种子不同，pooled n=60）+ bootstrap CI 复核**：`P_A(lambda)` = 0.0→**0.017**[0,0.05], 0.2→0.05[0,0.117], 0.4→**0.183**[0.083,0.283], 0.6→**0.617**[0.483,0.733], 0.8→0.85[0.75,0.933], 1.0→**0.817**[0.717,0.9]——λ=0.4 与 λ=0.6 的 CI 不重叠，**bifurcation 本身有真实统计支持**，但（1）转变发生在 0.4→0.8 一个更宽的窗口而非单一区间，（2）纯 A 端点（λ=1.0）P_A 的 CI 上界只到 0.9，**没有达到 1.0**——即使是最干净的对照条件也有约15-20% 落入其它 basin，（3）P_other（既非 A 也非 B 的第三 topic）在多个 λ 上高达 35-45%，n=4 pilot 从未报告过这一项；"比 ELF 更干净、全部同步切换"的表述需要撤回，改为"存在真实但有噪声的 bifurcation，转变窗口比 n=4 显示的更宽" |
| GS7 | **部分不可用** | CKA(oracle,rollout)≈1.000 全程饱和（不提供信息）；`G_token(oracle)` 先升后降是新发现的 artifact——"clean" 代理（t=0.99 状态）本身未必是 LangFlow 真正最干净的点（gamma(0.99)=3.48 ≠ gamma_min=2.60） |
| GS8 | **样本量问题已用正则化+更大N修复（2026-07-27）** | `t=0.65` 换后 test_acc 仍=0.158（train_acc=0.933，过拟合非 schedule 问题）；加 `--C 0.05` + n_samples 64→200 后 train_acc降到0.314、test_acc升到0.217（chance=0.125），过拟合基本解决；`correct`/`wrong` 方向 bootstrap CI 明确高于 `orthogonal`/`random`，但呈 U 形而非 ELF 的单调曲线，量级也小一个数量级——因果链存在但比 ELF 弱得多 |
| GS9 | **没有 ELF 的高-t 饱和问题** | t=0.99 时 Delta_A/B 仍明显非零（+0.595/+0.483），因为 LangFlow 在名义 t=0.99 时还有真实不确定性，双向 denoiser"抄答案"问题弱得多 |
| GS10 | **清晰复现** | 标准 32 步生成质量下 16/16 全部 healthy，零方差 |
| GS11 | **confound 复现，但"model 更差"不复现** | raw retrieval 同样随 L_eff 上升；但 `model` retrieval 在大 L_eff 下追平甚至略超 raw（0.958 vs 0.917），和 ELF 的"model 破坏身份信息"相反——这个发现可能是 ELF/checkpoint 特有的，不是普适规律 |
| GS12 | **两条结论都干净复现** | `MEAN_only` 全程最高或并列最高 structural R²；`MEAN+R_c` 全程远超 `MEAN+G_c` 的 token_acc——GS3/GS12 的修正结论在架构间稳健 |
| GS13 | **同 GS8 修复后：仅单侧显著，仍是本轮最不干净的结果** | `t=0.65` 换后量级仍弱（同 GS8 样本量瓶颈）；2026-07-27 用同样修复（`--C 0.05`、n_samples→200）重跑后 test_acc=0.217，但 bootstrap CI 显示只有 a=+1.0 方向的 `correct`（+0.150[0.044,0.265]）明显高于对照组，a=-1.0 方向 `correct` 自己的 CI（[-0.043,0.171]）就含 0——信号比 GS8 更弱、更不对称 |
| **GS14** | **清晰复现；⚠️ 2026-07-27 发现该文件其实还有一份*未修复*的同款 bug（GS6 的"四文件修复"漏掉了它），修好后用大样本+eta sweep 重新确认** | `branch_true_trajectory.py` 里 C_topic 的最近质心分类是**内联手写的平方欧氏距离代码，从未调用过被 import 的 `nearest_topic`**——此前"GS6 bug 已在 GS4/GS6/GS8/GS14 四个文件里统一修复"的验证只检查了 import 是否指向同一函数对象，没检查这份内联副本；已改为调用 `nearest_topic` 并补上逐轨迹原始数据（原来只存均值，无法算 CI）；**修复后 n_traj=16×3 seed（pooled n=48）+ bootstrap CI**：C_topic = 0.958[0.929,0.984]@t=0.20, 0.985[0.963,1.0]@t=0.38, 0.992[0.976,1.0]@t=0.65——不再是 n=4 时的"恒为 1.000、零方差"，是一个真实、随 t_start 单调上升但仍很高的信号；C_lex 同口径 = 0.765[0.752,0.778]/0.843[0.836,0.851]/0.941[0.938,0.944]；**eta sweep（0.01/0.03/0.1/0.3，n_traj=16，t_start=0.65 切片）**：C_topic 随 eta 增大单调下降 1.000→0.976→0.962→0.854，C_lex 同步更陡地下降 0.977→0.938→0.837→0.626——两者都对扰动幅度有真实的渐变响应，**排除了"C_topic 是扰动尺度太小导致的天花板伪影"这个怀疑**，只是 topic 比 lex 明显更鲁棒，这本身就是一个有信息量的发现，不是测量失灵 |
| **GS15** | **v2 加 `O_R_model` 后清晰跨架构复现** | raw 指标（`A_rollout`/`A_oracle`/`A_linear`）在 LangFlow 上全程饱和≈0.99+，不可用；改用 model-based `O_R_model` 后，**LangFlow 全程为负、中段最负（`t≈0.28`时`-0.472`）、随后稳步恢复到 0**，和 ELF 的 `O_R_model`（`t≈0.20-0.28`最负，约`-0.22~-0.24`）方向、形状一致——**"晚期崩塌而非渐进组织"假说首次得到跨架构确认**，是 GS 系列里少有的正面因果/动力学发现，见 EXP-GS15-spec.md"正式跨架构验证"节 |

**跨架构最重要的六个教训**（第 2、5、6 条已通过重跑验证/定位到根因）：
(1) 涉及 raw oracle state 的 CKA 类指标在 LangFlow 上经常饱和（GS7/GS15），必须优先用 model
输出（predicted_clean）而不是 raw 状态——`EXP-GS15` 已经加了 `O_R_model` 并在两个架构上都
复现出"负 O_R/晚期崩塌"这个动力学模式，是目前最强的跨架构证据；
(2) 沿用 ELF 校准出的 `t=0.28` 对 LangFlow **部分**无效——用 LangFlow 自己 GS1 曲线定位的
`t=0.65` 重跑后，**GS4（因果干预）干净复现**，证明这类实验本身没问题，只是 t 选错了；
**GS8/GS13（topic-probe 因果链）换 t 后 topic probe test_acc 完全没变**（0.158→0.158，
`train_acc=0.933` 说明是 45 篇训练文档对 8 类分类器过拟合——这是样本量问题，不是 t 校准
问题，需要更大 N 而非换 t）；
(3) LangFlow 在名义 `t=0.99` 时还没到真正的 `gamma_min`（3.48 vs 2.60），不能像 ELF 一样直接
当"clean"代理用；
(4) 尽管如此，**GS12 的两条核心结论（structure≈mean、token-in-residual）、GS4 的因果干预结论、
GS15 的 O_R_model 动力学结论、和 GS2/GS14 的"C_lex 是唯一有动态范围的
consensus 指标"在两个架构上都干净复现**，是目前最值得作为论文核心论据的发现；
(5) **⚠️ 之前把 GS6 的"P_A(lambda) 全程平坦"错误归因于"rollout 步长没有映射到 LangFlow
的 gamma 范围"——实际根因是 `nearest_topic` 用平方欧氏距离，而 LangFlow rollout 终点
的 pooled-embedding norm（≈1.15）远小于拟合 topic centroids 时用的 clean embedding
norm（≈3.70），导致所有 rollout 终点都被判给 norm 最小的那个质心，和内容无关**（诊断：
8 篇文档的 rollout 终点全部被分到同一个 cluster，但它们各自和自己 clean embedding 的
cosine 相似度有 0.40–0.67，说明真实信号一直都在，只是被 norm 差异掩盖）。改成
cosine-based nearest-centroid（已合并进 `common.py`，一并修复了 GS4/GS8/GS14 用到的
同一函数）后，GS6 在 LangFlow 上给出了**比 ELF 更干净的 bifurcation**（4 对文档全部在
`lambda=0.4→0.6` 同步切换），GS4 的 topic 维度也从"恒 0.25"变成有意义的结果（`B_preserve_
global` 时 topic=1.00、token=0.005，是 GLOBAL-4"保留 global mode 应该保住 topic 丢失
token"这一预测目前最干净的确认）。**这是一个通用性教训：任何用"欧氏距离找最近质心"做
分类的诊断，一旦分类对象（这里是 rollout 终点）和拟合质心时用的分布（这里是 clean
embedding）存在系统性尺度差异，就会出现看似"没有信号"实则"被 norm 淹没"的假阴性——
应该默认优先用 cosine 而不是欧氏距离，除非有理由确认两个分布的尺度是匹配的。**

(6) **2026-07-27 严谨性自审发现两个此前遗漏的问题**：
①`branch_global_consensus.py`（GS2）和 `branch_true_trajectory.py`（GS14）里各有一份**内联手写的**平方欧氏
距离最近质心分类代码，从未调用过 `nearest_topic`——GS6 bug 修复时只检查了"四个文件的 `nearest_topic` import
是否指向同一函数对象"，没有意识到这两个文件的 C_topic 计算根本没走 import 路径，是完全独立的第二处同款 bug。
修复后重跑：GS2 的"C_topic 饱和"结论两个架构上都不受影响（排除了这是 bug 假阳性的可能），但 GS14 的
"C_topic=1.000 零方差"确实随之改变（大样本下变成 0.96-0.99 之间的真实渐变信号）。
②GS4/GS6 的 n=4 headline 数字（"完美 1.00""干净同步切换"）在补上 n=16-20 × 3 seed + bootstrap CI 之后
**明显打折扣**：GS6 的转变窗口比 n=4 显示的更宽、纯端点也到不了 100%；GS4 的三个条件（baseline/A/B）在 CI 下
统计不可区分，"G 单独排他性地承载 topic"这个更强说法不成立。这类 n=4、单 seed、看起来"整整齐齐"的结果，本身
就该被当作"可能是运气"来对待，而不是因为数字漂亮就默认可信——这和 PT 系列（PT6/PT7）已经验证过的教训是同一条，
但 GS 系列直到这次自审之前一直没有被同样的标准要求过。

---

## 快速状态一览

- **DONE**: 69 个实验 (EXP-01–16, EXP-20–22, EXP-24–32 + EXP-30b + EXP-36 + EXP-37a/b/c + EXP-04v2 + EXP-05v3 + EXP-01v3(×3) + EXP-07v2(×3) + 2 dense probes + EXP-07 64-step + **EXP-12v2** + **EXP-11v2** + **EXP-25v2** + **EXP-26v2** + **EXP-27v2** + **EXP-36v2-factorial**(3 ckpt) + **EXP-31v2**(kd_cr+kd2 5-seed) + **EXP-30v2** + **EXP-29 audit** + **EXP-40** + **EXP-39** + **EXP-41** + **EXP-42** + **EXP-38** + **EXP-43** + **EXP-47**⚠️ + **EXP-44**⚠️ Phase1+2 + **EXP-48** + **EXP-49** + **EXP-50** + **EXP-51** + **EXP-52** + **EXP-53** + **EXP-54**)
- **READY / P0**: EXP-63（corrected JAX-aligned clean-teacher KD control）
- **ZOMBIE**: PID 3143093 (EXP-21 LangFlow hidden probe, CPU, needs manual kill)
- **INVALID**: EXP-33/34/35 — 使用无 tmin gate 的 dec_sc，产生退化文本，PPL 无效
- **SLOW**: PID 3143093 (EXP-21 SAGA, CPU) — 已被 fast GPU 版本替代，**需用户手动 kill**
- **NEW**: EXP-04v2 (head-only null, DONE) — G_head_null≈0.017%，头部几何无偏置；EXP-01v3 baseline (DONE) — ODE 轨迹在 t<0.26 时反转；EXP-05v3 (DONE) — G_null≈0.03-4%，G_debias≈G_oracle

---

## EXP-29 设计

**脚本**: `experiments/probe_elf/probe_knn_words.py`  
**输出**: `results/exp29/knn_words.json` + `results/exp29/knn_viz.html`  
**详见**: EXP-29-spec.md
