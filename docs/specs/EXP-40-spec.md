# EXP-40: unembed_bias 词汇分析

## 目标

EXP-15v2 发现 kd_cr 的 unembed_bias 参数变化量最大（Frobenius ratio R=2.59）。  
本实验直接分析 Δbias 向量：KD **促进**了哪些 token？**抑制**了哪些？  
是否存在频率/语言/POS 模式？

## 方法

纯权重分析，无需 GPU。

```
Δbias_kd_cr = unembed_bias_kd_cr - unembed_bias_baseline
Δbias_kd2   = unembed_bias_kd2   - unembed_bias_baseline
```

**分析内容**：
1. 绝对偏置 L2 范数（baseline, kd_cr, kd2）
2. 各 checkpoint 的 Δbias Top/Bottom 50 token（token 字符串 + Δ值）
3. Top/Bottom 50 中非 ASCII（多语言）token 比例
4. kd_cr 和 kd2 的 Δbias 余弦相似度（检验两者是否一致）
5. |Δbias| 的百分位分布（p50/p75/p90/p95/p99）

## 代码

`experiments/probe_elf/bias_analysis_exp40.py`（CPU，约 10 秒）

## 结果

**状态：DONE**

### 绝对偏置 L2 范数

| checkpoint | mean | std | L2 |
|-----------|------|-----|-----|
| baseline  | −0.250 | 0.368 | 79.6 |
| kd_cr     | −0.784 | 1.300 | 271.9 |
| kd2       | −0.913 | 1.653 | 338.3 |

KD 大幅增大了 unembed_bias 的幅度（kd_cr × 3.4，kd2 × 4.3）。

### kd_cr − baseline Δbias 摘要

- mean=−0.534, std=1.018, max=+2.77, min=−4.01

**Top-5 促进** token（Δ最大正值）：
- `'îl'` (+2.77，罗马尼亚语代词)
- `'▁(„'` (+2.46，引号符号)
- `'▁depuis'` (+2.32，法语"从")
- `'▁billet'` (+2.05，法语/英语)
- `'embre'` (+2.03，西班牙语月份后缀)

**Top-5 抑制** token（Δ最大负值）：
- `'prezentate'` (−4.01，罗马尼亚语)
- `'▁afacere'` (−3.86，罗马尼亚语"企业")
- `'Dienstleistungen'` (−3.85，德语"服务")
- `'rubrique'` (−3.85，法语"栏目")
- `'▁echipamente'` (−3.82，罗马尼亚语"设备")

**非 ASCII 比例**（Top-50）：促进 66%，抑制 76%

### kd2 − baseline Δbias 摘要

- mean=−0.913, std=1.531, max=+3.18, min=−4.93

**非 ASCII 比例**（Top-50）：促进 78%，抑制 64%

### 关键发现

**Δbias_kd_cr ≈ Δbias_kd2（余弦相似度 = 0.954）**  

两种 KD 变体的 unembed_bias 变化方向几乎相同。这意味着：
1. **unembed_bias 的变化不能解释 kd_cr 与 kd2 的行为差异**（SC 效果相反，text quality 不同）
2. unembed_bias 变化主要反映 KD 训练数据分布（OWT 以英文为主，多语言 token 被系统性调整）
3. **关键差异必须在 proj_kernel / unembed_kernel 或 backbone 权重中**

这一负结果与 EXP-39 的 cross-patch 实验互补：
- 如果 EXP-39 显示 baseline backbone + kd_cr head ≈ 99%，说明 decode 接口整体（包含 proj_kernel/unembed_kernel）是关键
- 本实验说明差异不在 bias 项，而在 kernel 的线性映射方向

### 对论文的影响

- 不能写"KD 通过调整词汇偏置来改变 oracle 精度"
- 可以写："KD 对多语言 token 的偏置方向相似（cos=0.95），但这一共同变化不能解释 oracle 精度差异；差异来源于 proj/unembed kernel 的线性映射结构"

## 输出

`results/exp40_bias_analysis/bias_analysis.json`  
`results/exp40_bias_analysis/bias_analysis_top50.txt`
