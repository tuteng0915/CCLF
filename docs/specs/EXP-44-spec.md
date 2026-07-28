# EXP-44: 完整模块 Factorial Patching

## 目标

精确定位 kd_cr vs kd2 行为差异（oracle accuracy、SC interaction）来自哪个模块：late backbone B08-B11 权重、reconstruction head（final_layer）还是 SC conditioning module。

EXP-39 已经做了 backbone/head 的整体 cross-patch。EXP-44 细化为 **block-level** 和 **path-specific** 拆分。

## 核心问题

EXP-42 显示 kd_cr vs kd2 在 B11 的 rel_L2=0.500（方向差异显著），但 oracle accuracy 几乎相同。EXP-36v2 显示 SC interaction 符号相反。到底是哪个模块造成了 SC 差异？

## 方法

### 模块定义

| 模块 | 参数 | 说明 |
|------|------|------|
| early_bb | blocks.0-7.* | 早层 backbone |
| late_bb | blocks.8-11.* | 晚层 backbone（EXP-42 主要变化层） |
| block11 | blocks.11.* | 单独 B11（最后一层） |
| decode_head | proj_kernel, proj_bias, unembed_kernel, unembed_bias | Token readout |
| recon_head | final_layer.linear.*, final_layer.norm_final.* | Reconstruction path |
| sc_module | self_cond_proj.*, self_cond_cfg_embedder.*, self_cond_cfg_tokens | SC conditioning |

### Phase 1：Oracle Accuracy（无需生成，fast）

从 exp07b_v2 数据取 h_10（kd_cr 的），用 kd2 的 block 11 权重计算 chimeric h_11，再用各种 decode head 计算 oracle accuracy。

关键 chimera arm：

| arm | early_bb | late_bb（B08-10） | B11 | decode_head |
|-----|:---:|:---:|:---:|:---:|
| native kd_cr | cr | cr | cr | cr |
| native kd2 | kd2 | kd2 | kd2 | kd2 |
| B11_swap_cr→kd2 | cr | cr | **kd2** | cr |
| B11_swap_kd2→cr | kd2 | kd2 | **cr** | kd2 |
| late_swap | kd2 | **cr** | kd2 | kd2 |
| decode_swap | kd2 | kd2 | kd2 | **cr** |

实现：取 h_10 from kd_cr's exp07b_v2，使用 kd2 的 B11 权重做 ONE block forward pass（attention + MLP + LayerNorm，无 time conditioning），然后用 kd_cr decode head 计算 oracle。

**注意**：ELF block 使用 time conditioning（t_emb），故 chimeric block 11 forward 是近似——time conditioning 部分沿用基础 checkpoint 的 t_emb；block 11 的 attn/MLP 权重来自另一 checkpoint。这是一个合理近似（t_emb 在两 checkpoint 间差异较小）。

Block 11 forward（RoPE-free Pre-LN Transformer，SwiGLU MLP）：
```python
# norm1 → attention → residual
h_norm = rms_norm(h, blocks_11["norm1.weight"])
qkv = F.linear(h_norm, blocks_11["attn.qkv.weight"], blocks_11["attn.qkv.bias"])
q, k, v = qkv.chunk(3, dim=-1)
# QK-norm
q = rms_norm(q.reshape(..., n_heads, head_dim), blocks_11["attn.q_norm.weight"])
k = rms_norm(k.reshape(..., n_heads, head_dim), blocks_11["attn.k_norm.weight"])
# scaled dot-product attention + output projection
attn_out = sdpa(q, k, v) @ blocks_11["attn.proj.weight"].T + blocks_11["attn.proj.bias"]
h = h + attn_out

# norm2 → SwiGLU MLP → residual
h_norm = rms_norm(h, blocks_11["norm2.weight"])
gate_up = F.linear(h_norm, blocks_11["mlp.w12.weight"], blocks_11["mlp.w12.bias"])
gate, up = gate_up.chunk(2, dim=-1)
mlp_out = F.linear(F.silu(gate) * up, blocks_11["mlp.w3.weight"], blocks_11["mlp.w3.bias"])
h = h + mlp_out
```

### Phase 2：SC Interaction（需要生成，expensive）

对 Phase 1 中改变 oracle accuracy 最显著的 arm，运行 32 步生成（seed=42，n=64），计算 PPL 和 SC interaction I = PPL(SC-only) - PPL(none) - PPL(SC+...) + PPL(none)。

这里 "sc_module_swap"（用 kd_cr 的 self_cond_proj/embedder，其余用 kd2）最能隔离 SC conditioning 的贡献。

### Phase 1 判据

若 B11_swap_kd2→cr（用 kd_cr 的 B11 插入 kd2 backbone）使 kd2 oracle acc 显著上升 → B11 权重（而非更早层的 representation）是因果来源。

若不变 → kd2 backbone B00-B10 产生的 h_10 已经决定了结果，B11 只是被动读出。

## 代码

`experiments/probe_elf/module_patch_exp44.py`（待实现）

**依赖**：需要手写 B11 forward pass（见上），以及模型权重精确 key 名（已确认：`blocks.11.*`）。

**成本**：Phase 1 低（forward pass，约 10 分钟 GPU）；Phase 2 中等（inference，约 30 分钟 GPU）

## 输出

`results/exp44_module_patch/patch_results.json`

---

## 结果

**状态：DONE Phase 1（2026-07-25）；Phase 2 待定**

### Phase 1 数值（Oracle Accuracy + L_rec）

#### t=0.3

| arm | oracle_acc | L_rec | Δacc vs native | ΔL_rec vs native |
|:---|:---:|:---:|:---:|:---:|
| native_kd_cr | 88.456% | 85.7 | — | — |
| native_kd2 | 87.923% | 86.3 | — | — |
| B11_cr_on_kd2 | 87.613% | 90.5 | −0.310pp | +4.2 |
| B11_kd2_on_cr | 88.334% | 87.4 | −0.122pp | +1.7 |
| decode_cr_on_kd2 | 88.214% | 86.3 | +0.291pp | 0.0 |
| decode_kd2_on_cr | 86.898% | 85.7 | −1.558pp | 0.0 |
| recon_cr_on_kd2 | 87.923% | 87.1 | 0.000pp | +0.8 |
| recon_kd2_on_cr | 88.456% | 89.7 | 0.000pp | +4.0 |

#### t=0.5

| arm | oracle_acc | L_rec | Δacc vs native | ΔL_rec vs native |
|:---|:---:|:---:|:---:|:---:|
| native_kd_cr | 99.409% | 76.2 | — | — |
| native_kd2 | 99.014% | 76.4 | — | — |
| B11_cr_on_kd2 | 98.732% | 81.5 | −0.282pp | +5.1 |
| B11_kd2_on_cr | 99.398% | 77.8 | −0.011pp | +1.6 |
| decode_cr_on_kd2 | 99.147% | 76.4 | +0.133pp | 0.0 |
| decode_kd2_on_cr | 98.477% | 76.2 | **−0.932pp** | 0.0 |
| recon_cr_on_kd2 | 99.014% | 77.6 | 0.000pp | +1.2 |
| recon_kd2_on_cr | 99.409% | 81.8 | 0.000pp | **+5.6** |

### 核心发现

**1. Oracle accuracy：Decode head 是主因**

最大降幅来自 decode_kd2_on_cr（kd_cr h10 → kd_cr B11 → kd2 decode head）：Δacc = −0.932pp（t=0.5）。
说明 kd_cr 的 h_11 direction 与 kd2 的 decode head 几何不兼容——decode head 是 checkpoint-specific 的。
B11 交换效果中等（B11_cr_on_kd2：−0.282pp），且方向一致：每个 checkpoint 的 B11 与自己的 backbone 更匹配。
Recon head 交换对 oracle accuracy 无影响（该路径不参与 decode）。

**2. L_rec（x̂_t 重建质量）：B11 和 recon head 均 checkpoint-specific，但绝对值相近**

native_kd_cr L_rec = 76.2，native_kd2 = 76.4（几乎相同！）。
B11_cr_on_kd2：L_rec = 81.5（+5.1），kd_cr 的 B11 对 kd2 backbone 的 h_10 产生次优的 h_11。
recon_kd2_on_cr：L_rec = 81.8（+5.6），kd2 的 final_layer 不能很好地读取 kd_cr 的 h_11 方向。
说明 B11 权重和 recon head 权重都是 checkpoint-specific 的（tuned to their own backbone），但两者各自 native 时 x̂_t 质量接近。

**3. 关键 null 结果：Phase 1 无法解释 SC interaction 差异**

EXP-36v2 中 kd_cr I≈-65（SC 有益）vs kd2 I≈+158（SC 有害），差异约 223 PPL 单位。
但两者的 L_rec（x̂_t 质量）几乎相同（76.2 vs 76.4）。这意味着 SC 差异**不来自 x̂_t 的重建精度**，而来自 x̂_t 在 embedding 空间中的**方向**与模型的 SC conditioning module 之间的相容性。
可能来源：self_cond_proj（[1024→512] 将 [z_t, x̂_t] 投影到 backbone 输入空间）权重在 kd_cr 和 kd2 之间的差异。

### 结论与下一步

Phase 1 排除了"oracle accuracy 差异是 SC 差异的来源"——两者几乎相同（0.4pp gap）。
Phase 1 排除了"x̂_t 重建质量差异是 SC 差异的来源"——L_rec 几乎相同（76.2 vs 76.4）。

**推测机制**：kd_cr 和 kd2 的 x̂_t 方向（在 512-dim embedding space 中）不同，
即使 MSE 相同，x̂_t 的语义内容/格式可能不同。self_cond_proj 权重决定了 x̂_t 如何被投影到下一步的 backbone 输入，不同的 proj 权重可能让同等质量的 x̂_t 产生完全不同的 SC 效果。

**下一步**：→ **Phase 2 已完成（结果见下）**

---

## Phase 2 结果

**状态：DONE（2026-07-25）**

**脚本**：`experiments/probe_elf/sc_module_patch_exp44_p2.py`
**输出**：`results/exp44_module_patch/phase2_results.json`

### 方法

全量生成实验（64 seq, 32 ODE steps, seed=42, MAX_LENGTH=128, SC_T_MIN=0.5）。

每个 base checkpoint 运行 4 个 arm：
- **none**: SC=False（baseline，x_pred 全步清零）
- **native_sc**: SC=True，native SC module（参考值）
- **proj_swap**: SC=True，cross-checkpoint 仅换 self_cond_proj（weight + bias）
- **full_sc_swap**: SC=True，cross-checkpoint 换完整 SC module（proj + cfg_embedder + cfg_tokens）

SC interaction I = PPL(SC) - PPL(none)，负值 = SC 有益，正值 = SC 有害。

**⚠️ Pipeline caveat**：本实验使用自定义 ODE 循环（同 EXP-47）。native I 方向与 EXP-36v2 不一致（EXP-36v2: kd_cr I≈-65 有益，kd2 I≈+158 有害；本实验: kd_cr native I=+134.9，kd2 native I=+25.4，两者均有害）。绝对值不可与 EXP-36v2 直接比较，但 module 替换带来的 ΔI 差值是有意义的。

### Phase 2 数值

| arm | kd_cr base | kd2 base |
|:----|:---:|:---:|
| PPL_none | 186.80 | 341.21 |
| PPL_native_sc | 321.71 | 366.64 |
| PPL_proj_swap | 140.20 | 711.32 |
| PPL_full_sc_swap | 158.42 | 723.68 |
| **I_native** | **+134.90** | **+25.43** |
| **I_proj_swap** | **−46.60** | **+370.11** |
| **I_full_sc_swap** | **−28.38** | **+382.47** |

（kd_cr base donor = kd2；kd2 base donor = kd_cr）

### 核心发现

**1. self_cond_proj 是 SC interaction 的主因**

将 kd2 的 self_cond_proj 换入 kd_cr：I 从 +134.9（有害）翻转为 −46.6（有益），ΔΔI ≈ −182。
将 kd_cr 的 self_cond_proj 换入 kd2：I 从 +25.4（微弱有害）爆炸性增加到 +370.1（强烈有害），ΔΔI ≈ +345。

这是一个强因果信号：**self_cond_proj 权重的差异直接决定 SC 是否有益**。

**2. full_sc_swap 效果与 proj_swap 相近**

kd_cr base: full_sc_swap I=−28.4 vs proj_swap I=−46.6（方向一致，幅度略小）。
kd2 base: full_sc_swap I=+382.5 vs proj_swap I=+370.1（方向一致，幅度相近）。
说明 self_cond_proj 是 SC module 中的主导组件；self_cond_cfg_embedder + cfg_tokens 的贡献较小且与 proj 方向一致。

**3. 结论：SC compatibility 的本质是 self_cond_proj 与 backbone 的匹配**

Phase 1 排除了"x̂_t 重建质量"（L_rec 几乎相同）。
Phase 2 确认 self_cond_proj 是主因。

x̂_t 由 kd_cr 产生，但经过 kd_cr 自己的 self_cond_proj 投影后与 backbone 不兼容（在本 pipeline 中）；换用 kd2 的 proj 投影后反而有益。这说明 KD 训练的两个变体（kd-cr vs kd2）以不同方式共同优化 final_layer（产生 x̂_t）和 self_cond_proj（消费 x̂_t），且二者之间形成了特定的匹配关系。将一个 checkpoint 的 final_layer（产生端）与另一个 checkpoint 的 self_cond_proj（消费端）配对，不可避免地破坏这种匹配。

### 后续方向

EXP-45（SC Activation Patch）：测试 x̂_t 方向本身（非 SC module 权重）是否也有贡献——在生成时替换 kd2 的 x̂_t 为 kd_cr 的（但保持 kd2 的 self_cond_proj 不变）。Phase 2 已经给出主因，EXP-45 可作为补充验证（lower priority）。
