# EXP-03 Spec — Matched-SNR ELF vs LangFlow Comparison

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：消除 noise schedule 差异对跨模型比较的影响。**

EXP-02 修正了 ELF vs LangFlow 的测量不对等（用各自的 native posterior 而非 z_t）。但还有另一层不对等：ELF 和 LangFlow 可能使用不同的 noise schedule，导致即使在"相同的 t 值"下，信噪比（SNR = signal / noise）也完全不同。

**举例**：
- ELF：z_t = t·x_clean + (1-t)·ε，SNR = t²/(1-t)²
- LangFlow：可能使用 masked diffusion 或不同的 forward process，SNR(t_LF) ≠ SNR(t_ELF) for same t value

如果 ELF 在 t=0.3 时的 SNR 等于 LangFlow 在 t=0.5 时的 SNR，那么"ELF 在 t=0.3 承诺，LangFlow 在 t=0.5 才承诺"实际上是在**相同 SNR 下**的比较，timing 的差异消失了。

**要验证的核心假说**：
- ELF 的早期承诺（vs LangFlow）在 matched-SNR 比较下也成立（而非 schedule 差异的产物）
- 如果在匹配 SNR 后差异消失：说明两个模型的实质差异是 noise schedule 的设计，而非 backbone 的承诺能力

**决策规则**：
- 若 matched-SNR 比较仍显示 ELF 早于 LangFlow（>10pp 差距）：论文的跨模型比较主张成立，但需报告 matched-SNR 数字
- 若差异 <5pp：论文比较主要反映 schedule 设计选择，需重写跨模型比较小节

**与其他实验的关系**：
- 依赖 EXP-02（获得 LangFlow native posterior 的 G(t) 曲线）
- 作为 EXP-02 的补充，形成"完整"的跨模型比较证据链

---

## Implementation

### Step 1: 提取 ELF 的 log-SNR 曲线

```python
# ELF: z_t = t * x_clean + (1-t) * eps
# SNR(t) = E[||t*x_clean||²] / E[||(1-t)*eps||²]
# log-SNR = log(t²) - log((1-t)²) = 2*log(t/(1-t))

import numpy as np
t_elf = np.linspace(0.05, 0.95, 19)
logsnr_elf = 2 * np.log(t_elf / (1 - t_elf + 1e-8))
```

### Step 2: 提取 LangFlow 的 log-SNR 曲线

需要查阅 LangFlow 的 forward process 定义。若 LangFlow 使用：
- Token masking (discrete diffusion): 无直接 SNR，需要用 mask_rate 近似
- Gaussian diffusion: 用 α_t, σ_t 计算 SNR = α_t² / σ_t²
- 其他：参考 LangFlow paper/code

检查路径：`models/LangFlow/langflow/model.py` → forward_process() 函数

### Step 3: 插值对齐

对于每个 ELF 的 log-SNR 值 s：
1. 找到 LangFlow 的 t_LF 使得 log-SNR_LF(t_LF) = s
2. 在 (log-SNR, G(t)) 空间中比较两个模型

```python
# Interpolate LangFlow G at matched SNR levels
from scipy.interpolate import interp1d
logsnr_lf = compute_langflow_logsnr(t_lf_grid)  # from LangFlow
G_lf = langflow_native_posterior_G  # from EXP-02

interp_G_lf = interp1d(logsnr_lf, G_lf, bounds_error=False, fill_value='extrapolate')
G_lf_at_elf_snr = interp_G_lf(logsnr_elf)  # LangFlow G at same SNR as ELF's t values
```

### Step 4: Plot

X-axis: log-SNR (匹配的信噪比)
Y-axis: G(t) (native posterior accuracy)
Lines: ELF-baseline, ELF-kd_cr, LangFlow

---

## 依赖关系

- **EXP-02 必须先完成**（需要 LangFlow native posterior G(t) 数据）
- **COMPLETED**: LangFlow G(t) 数据已在 EXP-02 中获得

## Effort: 1-2 days

- 0.5 天：找到 LangFlow forward process，计算其 log-SNR
- 0.5 天：插值对齐 + 作图
- 0.5 天：分析和论文更新

---

## 实验结果（Results）

**状态**: COMPLETED（纯分析，无需 GPU，2026-07-18）

### 噪声调度参数

**ELF**：`z_t = t · x_clean + (1-t) · ε`
- log-SNR(t) = 2·ln(t/(1-t))
- 范围：t ∈ [0.05, 0.95] → log-SNR ∈ [-5.7, +5.7]

**LangFlow**：使用 Gumbel proposal 将 γ（log-SNR 参数）映射到均匀 t，probe 使用线性 γ schedule
- 训练好的 Gumbel 参数（从 HF checkpoint 提取）：loc=4.9563, scale=0.9637
- γ_min = 2.601（"干净"端），γ_max = 16.052（"噪声"端）
- log-SNR(t) = −γ(t) = −(γ_max + t·(γ_min−γ_max)) = −16.052 + t·13.451
- 范围：t ∈ [0, 1] → log-SNR ∈ [−16.05, −2.60]

### 关键发现：噪声范围完全不重叠

| 模型 | 操作噪声范围（log-SNR） | 含义 |
|------|----------------------|------|
| ELF（probe t∈[0.05,1.0]） | [−5.7, +inf] | 从极噪到极净 |
| LangFlow（probe t∈[0,1]） | [−16.05, −2.60] | **始终处于高噪声区！** |

两者 log-SNR 重叠区间仅为 [−5.7, −2.60]，对应：
- ELF: t ∈ [0.05, 0.21]（接近纯噪声的极早期阶段）
- LangFlow: t ∈ [0.87, 1.0]（接近最干净的末期阶段）

### SNR 匹配对比表

| log-SNR | ELF probe t | ELF G(t) | LangFlow probe t | LangFlow G(t) |
|---------|------------|---------|-----------------|--------------|
| −14.0 | 0.001 | 12.6% (EXP-12 kd_cr t=0.1代理) | 0.153 | 3.4% |
| −9.0 | 0.011 | ~12.6% | 0.524 | 4.6% |
| −6.0 | 0.047 | ~12.6% | 0.747 | 17.8% |
| **−4.40** | **0.100** | **12.6%** | **0.866** | **63.7%** |
| **−2.77** | **0.200** | **58.6%** | **0.987** | **96.5%** |
| −2.60 | 0.214 | ~62.8% | 1.000（边界） | 98.7% |

（ELF G 使用 EXP-12 kd_cr frac_correct；LangFlow G 从 EXP-02 top1_gt_mean 插值）

### 解读

**在相同 log-SNR 下，LangFlow 的 G(t) 远高于 ELF-kd_cr：**
- 在 log-SNR = −4.40 时：LangFlow G = 63.7%，ELF G = 12.6%（差距 51pp）
- 在 log-SNR = −2.77 时：LangFlow G = 96.5%，ELF G = 58.6%（差距 38pp）

原因解读：
1. **LangFlow 的训练分布**聚焦在 log-SNR ∈ [−16, −2.6]，模型被迫学会从极高噪声中推断 token。其 G(t) 曲线在这个区间已经很高。
2. **ELF 的训练分布**覆盖全范围（0 到 1），在 log-SNR = −4.4 附近（t=0.10）几乎是 marginal / barely informative 状态。
3. **论文的原始比较（按 t 值对齐）无效**：ELF t=0.30 的 log-SNR = −1.70，而 LangFlow 在 t=1.0 时 log-SNR = −2.60——两者训练 + 推理的噪声范围完全不重叠。

### 对论文的影响

- **"ELF 比 LangFlow 承诺早 60pp"** 的原始结论是基于 nominal t 对齐，这是无效的比较
- **SNR 匹配后**的正确描述：两个模型在各自的噪声范围内都正常工作，不能直接比较"哪个更早承诺"
- 建议修改：
  - **方案 A（简单脚注）**：说明比较是在 nominal t 下进行的，噪声调度不同，并附上 log-SNR 对比图
  - **方案 B（重新定义对比）**：只比较在重叠噪声范围 log-SNR ∈ [−5.7, −2.60] 内的行为差异
  - **方案 C（删除直接比较）**：将跨模型比较限制在定性描述，不用数字"早 Xpp"

---

## ⚠️ 方法论问题 & 待修正 TODO（2026-07-21 审查）

### 问题 1（严重）：SNR 公式假设等范数，可能不正确

当前公式 `SNR = t²/(1-t)²` 假设 `E[|x_clean|²] = E[|ε|²]`（即 T5 编码嵌入与标准高斯等 norm）。但 ELF 有 `latent_mean=0, latent_std=0.2`，T5 hidden states 可能 scale 不同。

**修正**：计算 empirical SNR：
```python
snr_emp = (t**2 * E_x_clean_norm_sq) / ((1-t)**2 * E_eps_norm_sq)
```
其中 E_x_clean_norm_sq 从数据中实测。若 x_clean norm 与标准高斯不等，则 SNR 曲线和重叠区间都会改变。

### 问题 2（严重）：存在 extrapolation，数字不可信

表格中 log-SNR = -14 使用了 ELF t=0.001 的代理值（实际只有 t≥0.05 的数据），这个点不能报告。

**修正**：只报告有真实数据的重叠区间 log-SNR ∈ [−5.7, −2.60]，不做任何外推。

### 问题 3：LangFlow γ 语义需从代码确认

从 Gumbel 参数推导 log-SNR = -γ 的关系需要从 LangFlow 源代码确认 γ 的实际含义（是否确实等于 -log(SNR_input)）。

**修正**：对 LangFlow 做 code-level audit，确认 γ → z_t 的完整路径。

### 问题 4：指标和 checkpoint 不一致

- ELF 使用 kd-cr 的 frac_correct；LangFlow 使用 native posterior top-1
- 应该是 baseline ELF vs baseline LangFlow，使用同类指标

### 修正 TODO

- [ ] **P0**：计算 empirical SNR（从实际 T5 嵌入范数 vs 标准高斯），修正 SNR 公式
- [ ] **P0**：只报告真实数据重叠区间，删除 extrapolated 数据点（如 log-SNR=-14 处的 ELF 估计）
- [ ] **P0**：代码审计确认 LangFlow γ = -log(SNR) 关系
- [ ] **P1**：用 baseline ELF vs native LangFlow 重做，使用同类指标（两边都用 native posterior top-1 或独立线性探针）
- [ ] **P2**：在 log-SNR 重叠区做 dense sampling（至少 10 个 t 值），消除插值误差

### 备注：更稳健的跨架构比较方法

直接用 log-SNR 匹配仍有问题（不同嵌入空间的 norm 不可比）。更稳健的方案：
用 **empirical recoverability** 作为匹配轴——只给 raw z_t 训练同容量线性探针，以探针准确率作为"输入状态的信息量"标准，然后在相同信息量条件下比较 denoiser 性能。
