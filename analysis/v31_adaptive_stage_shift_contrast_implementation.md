# v3.1 Adaptive Stage Shift Contrast Implementation

## 1. Changed Files

新增：

```text
ideas/adaptive_stage_contrast.py
analysis/test_v31_adaptive_stage_contrast.py
analysis/v31_adaptive_stage_shift_contrast_implementation.md
```

修改：

```text
timematch.py
train.py
```

## 2. New Flags

TimeMatch 子命令新增：

```text
--stage_contrast_trade_off
--stage_contrast_stage_count
--stage_contrast_temperature
--stage_partition_mode
--stage_min_len
--stage_time_radius
--stage_time_temperature
--stage_contrast_pseudo_threshold
--stage_contrast_feature_kind
--stage_contrast_log_path
--stage_contrast_debug
```

默认：

```text
stage_contrast_trade_off = 0.0
stage_partition_mode = feature_change_dp
stage_contrast_feature_kind = spatial
```

因此默认不启用 v3.1，新逻辑不会改变原始 TimeMatch 行为。

## 3. Method Description

v3.1 实现的是：

```text
结构感知自适应时间阶段划分
+ 偏移感知阶段对应
+ 类别条件阶段级对比损失
```

### AdaptiveTemporalStageExtractor

阶段不是固定均分。

阶段来自每个样本自身的时间特征结构：

```text
H_i ∈ R^{T×D}
```

使用 feature-change dynamic programming 将时间序列划分为 K 个连续阶段，使每个阶段内部特征方差尽量小：

```text
cost(S) = Σ_{t∈S} ||h_t - mean(S)||²
```

总目标：

```text
min Σ_k cost(S_k)
```

约束：

```text
阶段连续；
阶段不重叠；
覆盖完整序列；
每段长度尽量满足 min_stage_len。
```

如果 T 太短导致 min_stage_len 不可行，只降低 min_stage_len；不会回退到固定均分。

### ShiftAwareStageCorrespondence

阶段对应不是同序号对齐。

对 target stage center 加 TimeMatch 的全局 shift 作为粗先验：

```text
gap(k,l) = |center_t(k) + δ - center_s(l)|
```

只允许时间距离在 `stage_time_radius` 内的 source stage 参与，对应权重为：

```text
A(k,l) ∝ exp(- gap(k,l)^2 / τ)
```

若没有候选 stage，则选择最近的有效 source stage，并记录 fallback 计数。

### StageContrastiveLoss

对 target strong branch 的 stage feature 做类别条件对比。

正样本：

```text
source label == target pseudo label
```

负样本：

```text
source label != target pseudo label
```

source stage prototype 使用 `A` 加权得到；损失使用 prototype-level InfoNCE。

只使用：

```text
pseudo_mask=True
```

的 target 样本；不使用 target true label；不使用 memory bank；不跨 batch 保存 prototype。

## 4. Hook Location

接入位置：

```text
timematch.py::train_timematch
```

具体在 batch loop 内：

1. teacher weak target 生成：

```text
pseudo_targets
pseudo_conf
pseudo_mask
```

2. student 进行 source / target forward；
3. 若 `stage_contrast_trade_off > 0`：

```text
H_s = student.spatial_encoder(pixels_s, mask_s, extra_s)
H_t = student.spatial_encoder(pixels_t, mask_t, extra_t)
```

4. 计算：

```text
stage_loss
```

5. 总损失：

```text
loss = loss_source + trade_off * loss_target
loss = loss + stage_contrast_trade_off * stage_loss
```

## 5. Default Off Guarantee

只有：

```text
stage_contrast_trade_off > 0
```

时才会执行新逻辑。

当：

```text
stage_contrast_trade_off = 0
```

时：

```text
不额外提取 stage；
不额外 forward spatial encoder；
不额外计算 stage loss；
不写 stage TSV；
TimeMatch 原训练路径不变。
```

## 6. Explicitly Not Implemented

本轮明确没有实现：

```text
no fixed uniform partition
no same-index alignment
no target k -> source k hard alignment
no fixed partition fallback
no memory bank
no GCTW
no DTW path
no monotonic warping
no learnable stage assignment
no target true label
no source pretrain structure loss change
```

## 7. Validation

已完成：

```text
py_compile 通过
```

命令：

```bash
python -m py_compile ideas/adaptive_stage_contrast.py timematch.py train.py analysis/test_v31_adaptive_stage_contrast.py
```

本地限制：

```text
当前本地 Python 环境没有 torch，因此随机张量单元测试未能在本地运行。
```

已提供服务器可运行脚本：

```bash
python analysis/test_v31_adaptive_stage_contrast.py
```

预期输出：

```text
V31_UNIT_TEST_OK
loss=...
valid_queries=...
zero_mask_loss=0.000000
```
