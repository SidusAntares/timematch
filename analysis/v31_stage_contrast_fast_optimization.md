# v3.1 Adaptive Stage Contrast Fast Optimization

## 1. 问题

初版 `v31_adaptive_stage_shift_contrast` 训练很慢，主要原因不是验证阶段，也不是 TimeMatch 本身，而是 stage contrast loss 的实现。

旧实现显式构造：

```text
A ∈ R^{B_t × B_s × K × K}
```

也就是每个 target 样本、每个 source 样本、每个 target stage、每个 source stage 都计算对应关系。随后又在 loss 中按 target 样本、target stage、类别循环筛选正负样本。

这导致：

```text
复杂度高；
Python 循环多；
fallback 也在样本对维度上循环；
实际训练速度明显不可接受。
```

## 2. 快版实现

快版保留方法含义：

```text
adaptive feature-change stage partition
+ TimeMatch shift-aware stage correspondence
+ class-conditional stage contrast
```

但不再构造样本对级别的 `A`。

### 2.1 Source class-stage prototypes

先把 source batch 聚合为类别-阶段原型：

```text
P_{c,l} =
mean { H_i^s(l) | y_i^s = c }
```

对应张量：

```text
source_proto      ∈ R^{C × K × D}
source_center     ∈ R^{C × K}
source_proto_mask ∈ {0,1}^{C × K}
```

### 2.2 Shift-aware class-stage correspondence

对 target stage `k` 和 source class-stage `(c,l)` 计算时间中心差：

```text
gap_{b,k,c,l}
= | center_t(b,k) + δ - center_s(c,l) |
```

候选对应要求：

```text
gap <= radius
source_proto_mask[c,l] = 1
target_stage_mask[b,k] = 1
```

再在 source stage 维度 softmax 得到：

```text
A ∈ R^{B_t × K × C × K}
```

如果某个 target stage 对某个 class 没有 radius 内候选，则 fallback 到最近的有效 source stage。

### 2.3 Class-conditional logits

先得到每个 target stage 对每个类别的 soft prototype：

```text
Q_{b,k,c}
= Σ_l A_{b,k,c,l} P_{c,l}
```

再计算分类式对比 logits：

```text
logit_{b,k,c}
= sim(H_b^t(k), Q_{b,k,c}) / τ
```

最终对 target pseudo label 做交叉熵。

## 3. 复杂度变化

旧实现：

```text
O(B_t × B_s × K × K)
+ 多层 Python loop
```

快版：

```text
O(B_t × C × K × K)
+ 向量化 torch 运算
```

其中 remote 任务通常：

```text
B_s ≈ 128
C ≈ 12
```

所以对应矩阵规模约减少一个数量级，并去掉最慢的样本对 fallback 循环。

## 4. 新增参数

```text
--stage_contrast_backend class_prototype_fast
```

当前唯一可选值是：

```text
class_prototype_fast
```

launcher 中也显式传入该参数，避免实验日志看不出实际 backend。

## 5. 新增日志

```text
stage_contrast_backend
stage_loss_compute_time_ms
source_valid_class_count
source_valid_class_stage_count
correspondence_fallback_ratio
correspondence_valid_class_mean
```

这些字段用于判断：

```text
是否走了 fast backend；
stage loss 每步耗时是否正常；
source batch 中有多少有效类别和类别-阶段原型；
shift-aware correspondence 是否大量 fallback。
```

## 6. 修改文件

```text
ideas/adaptive_stage_contrast.py
timematch.py
train.py
analysis/test_v31_stage_contrast_fast.py
launchers/launch_v31_adaptive_stage_shift_contrast_probe.sh
```

## 7. 保留差异

快版仍然是 class-conditional stage contrast，但它使用 source class-stage prototype，而不是 source sample-pair prototype。

因此它不是完全逐样本等价的加速，而是把 source side 从：

```text
sample-level stage pool
```

改为：

```text
class-level stage prototype
```

这与当前 v3.1 的目标一致，因为我们本来关心的是：

```text
类别条件的阶段对应关系
```

而不是 target 样本和每个 source 样本之间的细粒度对应。

## 8. 验证

新增：

```text
analysis/test_v31_stage_contrast_fast.py
```

测试内容：

```text
1. source class-stage prototype shape；
2. correspondence shape；
3. correspondence 无 NaN；
4. source-stage 权重和为 1；
5. loss 可 backward；
6. target mask 全空时 loss 为 0；
7. B_s=128, B_t=128, K=6, D=128, C=12 的 fast forward microbenchmark。
```

