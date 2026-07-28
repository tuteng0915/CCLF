# 晚层几何分析：EXP-38–42 综合解读（修订版）

**日期**：2026-07-23  
**状态**：讨论文档（已经过方法论审查；结论已按证据强度分级）  
**关联实验**：EXP-38（Logit Lens）、EXP-39（Cross-Patch）、EXP-40（Bias Analysis）、EXP-41（Decode Alignment）、EXP-42（CKA Divergence）

---

## 1. 核心现象

EXP-38–42 共同揭示了一个此前未被注意到的几何结构。将五个实验的结论并排：

| 实验 | 关键观测 |
|------|---------|
| EXP-38 | baseline logit lens **非单调**（t=0.5 时 B10=0.784 > B11=0.756；t=1.0 时 B09 为峰值） |
| EXP-38 | kd_cr 和 kd2 的 logit lens **在每一层几乎相同**（B11: 0.995 vs 0.990） |
| EXP-39 | backbone 是 oracle accuracy 的**主要因果来源**（kd_cr bb + any head ≈ 99.4%） |
| EXP-40 | kd_cr 和 kd2 的 decode head 变化**方向几乎相同**（cos=0.954） |
| EXP-41 | baseline 的 cos_align **最高**（0.234），但 oracle accuracy **最低**（75.6%） |
| EXP-42 | kd_cr vs baseline B11 CKA=0.427；kd_cr vs kd2 B11 rel_L2=0.500 |

---

## 2. 已有扎实支持的结论

以下三点有直接实验证据，可以进入论文。

### 结论一：KD readout 增益的主要载体是 backbone，decode head 贡献次之

EXP-39 的 cross-patch 测试：

```
B_kd_cr + H_baseline ≈ 99.4%
B_kd_cr + H_kd_cr   ≈ 99.5%    ← native kd_cr
B_baseline + H_kd_cr ≈ 80.8%   ← 仅换 head：+5.2pp
```

这是 module 级的因果结果，不只是 correlation。

**表述限定**：这个结论限定在"当前 oracle states、当前 t 值、当前 backbone/head patch 设置下"。它解释的是 oracle decoding 的 readout gain，**不直接扩展到 actual generation 或 SC 行为**。

---

### 结论二：KD 消除了 baseline 晚层的 decode-hostile 残差

在 t=0.5，对固定的 decode head：

```
Acc(D(h_10)) = 78.4%    >    Acc(D(h_11)) = 75.6%    [baseline]
Acc(D(h_10)) = 98.8%    <    Acc(D(h_11)) = 99.5%    [kd_cr]
```

可以安全地说：

> Under the fixed decode lens, the baseline B11 residual update reduces oracle token accuracy relative to B10. KD removes this non-monotonicity.

**限定**："anti-decode" 这个描述是 **relative to this specific decode head and oracle dataset**，不是说 B11 在抽象意义上删除了 token 信息。此外，标准 logit lens 存在 **layer mismatch**：decode head 是在 B11 分布上训练的，直接施于 B10 可能因 scale、LayerNorm、feature basis 不同而有偏。更强的证据需要 tuned lens 或 layer-specific 的线性校准。目前结论仅成立于 fixed head 条件下。

---

### 结论三：Lexical readability 不决定 SC utility

kd_cr 和 kd2 提供了一个天然受控对照：

- Oracle logit lens accuracy 几乎相同（B11: 0.995 vs 0.990）
- EXP-40 bias 变化方向几乎相同（cos=0.954）
- 但 SC interaction 符号相反（I = -65 vs +158）

这直接说明：

> kd_cr and kd2 achieve nearly identical oracle lexical readout while exhibiting sharply different self-conditioning interactions, indicating that lexical accuracy alone does not determine whether a representation is dynamically useful as conditioning.

用三层区分表示：

```
token information ≠ native readability ≠ SC compatibility
```

---

## 3. 目前仍是假说的部分

### 假说 A：Reconstruction–decode 几何冲突（目前只有一半证据）

当前逻辑链：
1. B11 让 decode accuracy 下降 ✓（已证）
2. B11 负责 reconstruction ✓（架构正确）
3. B11 为了 reconstruction 牺牲了 decode ✗（**未证**）

真正的 tradeoff 需要同时观察到：

```
对 h(α) = h_10 + α·Δh_11，α 从 0 → 1：
    L_dec(α) 变差    ← 已有间接证据（logit lens 下降）
    L_rec(α) 变好    ← 目前没有直接测量
```

在此之前，正确表述是：

> The baseline B11 update is decode-hostile under the chosen lens, **consistent with—but not yet proving**—a reconstruction–readout tradeoff.

---

### 假说 B：B11 方向差异决定 SC compatibility

目前：

```
h_11_kd_cr ≠ h_11_kd2   (rel_L2 = 0.500)
SC_kd_cr works, SC_kd2 doesn't
```

这是共现，不是因果。SC 行为差异还可能来自：
- `final_layer.linear` 权重不同
- SC input projection 不同
- time-conditioning 不同
- B00-B10 的小差异累积效应
- x̂_t 的 norm/calibration 不同

**不能**说"只有 kd_cr 的 B11 方向满足 SC compatibility"。结论三的表述已经是最强可辩护的形式。

---

### 注：cos_align 数字的解读限定

EXP-41 的 mean cos（correct position）：baseline=0.234, kd_cr=0.181。

这只能说明：

> Raw cosine alignment between the decode hidden vector and the correct token's unembed column is **not positively associated** with oracle accuracy.

**不能**从这一个数字推出"更尖锐""更均匀"或"经常指向错误 token"。要区分以下几种可能：
- Competing tokens 的 cos 更高（margin 问题，不是 alignment 问题）
- Row norm / bias 结构不同
- Vocabulary anisotropy / null-mode
- Top-1 vs correct token 的 margin 分布

这需要：top-k cosine margin、row-wise 分布、entropy、vocabulary anisotropy 等进一步分析。

---

### 注：CKA 和 rel_L2 的语言分开处理

这两个指标测量的是不同的东西：

| 指标 | 对旋转 | 对 scale | 适合描述 |
|------|--------|---------|---------|
| Linear CKA | 不敏感 | 不敏感 | 表示子空间是否相似 |
| rel_L2 = \|\|Δh\|\|/\|\|h\|\| | 敏感 | 敏感 | 坐标系下的激活差异幅度 |

kd_cr vs kd2 B11 rel_L2=0.500 应当表述为：

> kd_cr and kd2 have similar task-level oracle readouts but substantially different paired B11 activations **under a coordinate-sensitive distance**.

要从 rel_L2 断言"方向性差异"（与 scale 和 rotation 无关），需要 Procrustes 对齐后仍有大残差、或者直接做 paired cosine/centered cosine 分析。

---

## 4. B00 解释目前仍需直接验证

当前文档推断：early-layer logit lens gap（kd_cr B00=0.404 vs baseline B00=0.187）主要来自 decode head 不同（因为 B00 CKA=0.981）。

这个解释合理，但 **CKA 高不保证同一 linear readout 的输出接近**——一个沿 readout-sensitive direction 的微小变化仍能显著影响 accuracy。

真正的验证需要 **layer × head cross-patch matrix**：

|  | baseline head | kd_cr head | kd2 head |
|--|:---:|:---:|:---:|
| **baseline B00** | ? | ? | ? |
| **kd_cr B00** | ? | ? | ? |
| **kd2 B00** | ? | ? | ? |

如果同一 B00 hidden 换 head 就产生大差异，而不同 B00 hidden 换到同一 head 差异小 → early gap 来自 head。否则 backbone 的微小变化沿 readout-sensitive 方向也有贡献。

目前可以说"head 变化在早层能产生效果（EXP-39 +5.2pp at B11），且这个效果逻辑上对所有层可见"，但不是 backbone 向早层"广播"了什么。更准确的表述：

> Because the same checkpoint-specific head is applied at every depth, head changes **appear at every point** of the logit-lens profile.

---

## 5. 统一叙事（修订后）

当前最准确、最稳健的 KD 机制描述：

> KD 训练把 oracle readout accuracy 引入显式监督信号。功能测试（EXP-39）证明 backbone 是 oracle readout gain 的主要因果来源。CKA 分析（EXP-42）定位了这个变化主要在 B08-B11（CKA 降至 0.427）。Logit lens（EXP-38）显示 KD 消除了 baseline B11 的 decode-hostile 残差，使 logit lens 从非单调变为单调递增。
>
> 这三个发现连成一条因果链：
>
> ```
> late-layer parameter change (EXP-15v2)
>     ↓
> late-layer representation reorganization (EXP-42)
>     ↓
> removal of decode-hostile residual at B11 (EXP-38)
>     ↓
> native oracle readout improvement (EXP-39)
> ```
>
> kd_cr 和 kd2 共享这条因果链，但其 SC behavior 截然相反（EXP-36v2）。由于两者的 oracle readout 和 logit lens 几乎相同（EXP-38/40），lexical accuracy 不能解释这个差异。剩余的几何差异体现在 B11 的坐标敏感距离（rel_L2=0.500）。**这是一个已观察到的关联，SC compatibility 的因果机制尚待直接实验验证。**

---

## 6. 下一批实验方向

下一批实验的核心不是继续静态 geometry 分析，而是直接追踪：

```
h_11 → x̂_t → SC response → velocity → generation
```

### EXP-43：Dual-Path Gradient Conflict 直接测量

**目标**：把"reconstruction–decode tradeoff"从假说变成直接证据。

**方法**：对 h_10 做插值：

```
h(α) = h_10 + α · Δh_11,    α ∈ [-0.5, 1.5]
```

同时计算：
- `L_dec(α)` = correct-token CE loss via decode path（用 checkpoint 自己的 decode head）
- `L_rec(α)` = reconstruction MSE = ||final_layer(h(α)) - x*||²

**以及 gradient conflict 标量**：

```
c(h) = cos(∇_h L_rec, ∇_h L_dec)
```

**判据**：
- Baseline tradeoff 成立：α 从 0→1 时 L_rec 下降但 L_dec 上升（且 c < 0 at h_11）
- KD 消除冲突：kd_cr 中 α 从 0→1 时两者均改善，或 c 接近 0 / 转正

**数据**：复用 exp07b_v2（已有 h_10、h_11、x*=y_tokens embedding 可近似），梯度可从 checkpoint 权重解析计算，无需新 forward pass。  
**成本**：极低（解析计算）

---

### EXP-44：完整模块 Factorial Patching

**目标**：定位 kd_cr vs kd2 行为差异的精确模块来源——B11 权重、reconstruction head 还是 SC conditioning module。

**关键子集**（5 个模块：early backbone B00-B07、late backbone B08-B11、decode projection、reconstruction projection、SC conditioning module）：

| patch 来源 | early bb | late bb | decode head | recon head | SC reader |
|-----------|:--------:|:-------:|:-----------:|:----------:|:---------:|
| native kd_cr | cr | cr | cr | cr | cr |
| native kd2 | kd2 | kd2 | kd2 | kd2 | kd2 |
| late_swap | kd2 | **cr** | kd2 | kd2 | kd2 |
| recon_swap | kd2 | kd2 | kd2 | **cr** | kd2 |
| sc_swap | kd2 | kd2 | kd2 | kd2 | **cr** |
| late+recon | kd2 | **cr** | kd2 | **cr** | kd2 |

每个 arm 测：oracle accuracy、reconstruction MSE、actual rollout PPL（32 步）、SC interaction（仅 sc_swap 及关键 arm）。

**成本**：oracle/MSE 部分低（forward pass 即可）；rollout 部分中等（需要 inference），可以先跑 oracle arm 确认哪个 module 最关键，再选择性跑 rollout。

---

### EXP-45：SC Activation Patch（因果干预）

**目标**：直接因果测试 x̂_t 的 "format" 是否决定 SC 行为。

**方法**：在相同 sequence、noise、t 下：
1. 用 kd2 运行主 trajectory
2. 只把 x̂_t 替换为 kd_cr 的 paired x̂_t（同位置、同 t）
3. 其余全部仍用 kd2（backbone、weights 均不变）
4. 观察 SC interaction 是否翻转

**需要分别 patch**：
- `h_11`（B11 activation）
- `x̂_t = final_layer(h_11)`
- SC projected input（x̂_t 进入模型的最终形式）
- decode logits（作为对照，预期不影响 SC）

并做连续插值：

```
x̂_t(λ) = (1-λ)·x̂_t_kd2 + λ·x̂_t_kd_cr,    λ ∈ {0, 0.25, 0.5, 0.75, 1}
```

**判据**：如果 SC interaction 随 λ 平滑变化且在某个 λ 改变符号 → 直接定位 SC compatibility 来自 x̂_t 的方向（而非 kd2 本身的其他权重）。

**成本**：中等（需要修改 generation_utils.py 支持 activation injection）

---

### EXP-46：SC 响应的 Jacobian 分析

**目标**：把"SC compatibility"从静态几何属性转化为动力学属性——测量表示差异如何影响 velocity。

**核心量**：

```
J_SC = ∂v_θ(z_t, s_t, t) / ∂s_t         # SC conditioning 的 Jacobian
δs_t = x̂_t_kd_cr - x̂_t_kd2             # 两个 checkpoint 的 x̂_t 之差
```

计算 velocity 响应：

```
J_SC · δs_t
```

判断其方向是否朝正确 recovery direction：

```
cos(J_SC · δs_t, x* - z_t)
```

如果 kd_cr 的 δs_t 通过 J_SC 产生朝向 x* 的 velocity 修正，而 kd2 的对应量不满足此条件，则给出 SC compatibility 的动力学刻画：

> SC compatibility is not a static semantic property of the representation, but rather:
> whether this representation, when fed into the SC conditioning pathway, pushes velocity toward the correct denoising direction.

**成本**：中等（需要一次 Jacobian vector product，可用 autograd 高效计算）；关键是用 JVP 而非完整 Jacobian（避免 O(dim²) 存储）。

---

## 7. 和"全局语义"问题的关系

当前这批实验（EXP-38–42 及下一步 EXP-43–46）回答的是：

> KD 为什么能让 **每个位置** 的 lexical information 被 native pathway 读出来？

它目前**没有回答**：

- 全局语义 basin 是否更早形成
- 全局 relational scaffold 是否形成
- B11 重组是 global pattern 还是 position-wise lexical sharpening
- kd_cr/kd2 的 SC 差异是否来自全局 coherence

连接两条研究线的自然方向：把 B11 差异分解为：

```
H_11 = G_11^global (low-rank, sequence-level)
      + R_11^local  (position-specific residual)
```

分别 patch global component 和 local residual，看 SC interaction 随哪个部分变化。如果主要随 global component 变化：

> kd_cr 和 kd2 都能逐 token 正确解码，但只有 kd_cr 的 B11 在全局结构上适合闭环生成。

这会把两条研究线真正连接起来，但目前没有证据，留作未来方向。

---

## 附：修订记录

| 修订项 | 修订前 | 修订后 | 原因 |
|--------|--------|--------|------|
| Reconstruction tradeoff | "已证" | "假说，缺 L_rec 直接测量" | 只有 decode 变差的证据，无 reconstruction 变好的证据 |
| Logit lens 非单调 | 无限定 | 加"fixed decode lens"和"layer mismatch"限定 | Head 在 B11 分布训练，应用于 B10 有方法论风险 |
| cos_align 解释 | "更尖锐但经常错" | "cos alignment 与 oracle accuracy 无正相关" | 一个均值数字不足以推断"sharp but miscalibrated" |
| CKA/rel_L2 | 混用"subspace/方向" | 分开：CKA=旋转不变子空间，rel_L2=坐标敏感距离 | 两者测量不同性质，需 Procrustes 才能断言方向差异 |
| SC compatibility | "只有 kd_cr 满足" | "两者提供受控对照；lexical accuracy 不足以预测 SC utility" | 目前只有共现，非因果 |
| Early-layer gap 解释 | "decode head 变化广播到所有层" | "同一 head 应用于所有层，因此 head 变化在所有层可见" | 不是广播机制，而是 head 被固定施用于每层 |
