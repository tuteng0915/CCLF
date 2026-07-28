# EXP-47: Intermediate-Layer SC

## 目标

验证"中间层 SC"假说：用 x̂_α = final_layer(h_10 + α*(h_11-h_10)) 作为 SC 信号，对比标准 α=1.0（h_11）。

预期：对于 baseline，随着 α 从 1.0 降低，SC 的有害影响应减小（因为 h_10 方向对 decode 更友好，见 EXP-43）。

## 实验设计

```
x̂_α = final_layer(h_10 + α*(h_11 - h_10)),    α ∈ {0.0, 0.25, 0.50, 0.75, 1.0}
```

+ "none"：零向量 SC（无条件生成基准）

**门控**：仅在 t >= SC_T_MIN=0.5 时替换 SC 信号，低 t 阶段仍用标准 h_11。

**配置**：N_SEQ=64，N_STEPS=32，MAX_LENGTH=128，BATCH_SIZE=16，SEED=42

**评价指标**：I(α) = metric(SC_α) - metric(none)；负值→SC 有益，正值→SC 有害。

**注意**：代码中 compute_ppl 返回 mean NLL（total_nll/total_tok），标签写作"PPL"但实为 NLL（nats）。
下表数字均为 mean NLL，非实际 PPL（实际 PPL = exp(NLL)）。

## 代码

experiments/probe_elf/intermediate_sc_exp47.py

运行：conda run -n elf python3 experiments/probe_elf/intermediate_sc_exp47.py --device cuda:1

输出：results/exp47_intermediate_sc/results.json

---

## 结果

**状态：DONE（2026-07-24）**

### 数值结果（mean NLL 单位）

| checkpoint | NLL(none) | I(α=0.00) | I(α=0.25) | I(α=0.50) | I(α=0.75) | I(α=1.00) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline | 5.722 | -0.944 | **-0.991** | -0.946 | -0.923 | -0.929 |
| kd_cr | 5.230 | **+0.028** | +0.169 | +0.288 | +0.467 | +0.485 |
| kd2 | 5.833 | **-1.323** | -1.134 | -0.822 | -0.504 | -0.320 |

正值 → SC 有害；负值 → SC 有益。加粗为各行最优 α。

### 关键发现

**1. Baseline：α 几乎无影响（假说未确认）**

baseline 所有 α 的 I 均为 -0.92 至 -0.99（range ≈ 0.07 NLL），趋势极弱。α=0.25 边际最优但差异不显著。
预期的"低 α → 更少有害"的明显梯度未出现。

**2. kd_cr/kd2：单调趋势，方向一致**

两个 KD checkpoint 均显示 更低 α → 更好的 SC 信号：
- kd_cr：α=1.0 → +0.485（SC 轻度有害），α=0.0 → +0.028（几乎无影响）
- kd2：α=1.0 → -0.320（SC 轻度有益），α=0.0 → -1.323（SC 最有益）

与 EXP-43 一致：KD checkpoint 的 h_10 在重建-decode tradeoff 上优于 h_11。

**3. EXP-36v2 参考值不可比（重要 caveat）**

EXP-36v2：baseline I≈+1594，kd_cr I≈-65，kd2 I≈+158。
与本实验 α=1.0 结果比较，kd_cr 和 kd2 符号完全相反。原因：
- compute_ppl 返回 NLL（非 exp(NLL)），量纲不同
- 自定义 ODE 生成循环 vs EXP-36v2 完整 ELF pipeline
- decode_z_to_ids 使用 t=1.0，对最终 t→0 状态可能不正确
- kd_cr/kd2 生成文本质量差（kd2 α=0.0: '....... about about to about...'），NLL 不可信

本实验内部一致性成立，但与 EXP-36v2 不可横向比较。

### 生成文本质量

- baseline（所有 α）：生成连贯英文，如 "The protests that perpetuate across all corners of Europe..."
- kd_cr（α=0.00）：'sum about... about about about about about about any what what about about: set set...'
- kd2（α=0.00）：'....... about about to about..., because from to about,,, to from to to...'

baseline 文本质量明显优于 kd_cr/kd2，后者 GPT-2 NLL 估计可靠性存疑。

### 结论

- 中间层 SC（低 α）在 kd_cr 和 kd2 上一致给出更好 SC 信号，支持"减少 B11 decode-hostile 方向"解释
- baseline 对 α 不敏感：baseline h_10 L_rec 极高（EXP-43：529 vs KD 的 103-142），
  h_10 重建质量同样差，故 h_10 和 h_11 作为 SC 源的质量差不多
- 结论需以 EXP-36v2 兼容 pipeline 复现（正确 PPL + 全 pipeline）后才可用于论文
