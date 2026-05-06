# v2.2 实现说明

## 1. 版本定位
`v2.2` 是在 `v2.1` 的轻量 source reshaper 基础上，正式引入 **双路径 source 结构诱导** 的版本。

核心思想是：

- source 侧同时保留 `raw source path` 和 `reshaped source path`
- target 侧仍然只走 `raw path`
- 用双路径关系约束，保证 reshaper 只是对 `PSE` 后特征曲线做**可识别、可控**的轻量变换，而不是重写一套完全不同的表示

## 2. 结构位置
source feature reshaper 插在：

`PSE / spatial encoder -> source feature reshaper -> LTAE / temporal encoder`

也就是说，`v2.2` 直接操作的对象是：

- `PSE` 输出的 **编码后 source 特征曲线**

而不是原始时序曲线。

## 3. 双路径结构
设 source 输入经过 `PSE` 后得到时间特征：

```text
H_raw ∈ R^(B×T×D)
```

reshaper 输出：

```text
H_reshape = R(H_raw)
```

于是 source 侧有两条路径：

### 3.1 raw source path
```text
H_raw -> LTAE -> decoder -> y_raw
```

### 3.2 reshaped source path
```text
H_reshape -> LTAE -> decoder -> y_reshape
```

target 侧在域适应阶段仍然只走 raw path，不经过 reshaper。

## 4. 总体损失
在 `source_feature_dual_path=True` 时，source-only 阶段的总体损失为：

```text
L_total
= L_cls_raw
+ λ_dual_cls * L_cls_reshape
+ λ_rel * L_rel
+ L_compact
+ λ_reshape * L_reshape_reg
```

其中：

- `λ_dual_cls = source_feature_dual_cls_trade_off`
- `λ_rel = source_feature_dual_relation_trade_off`
- `λ_reshape = source_feature_reshaper_reg_trade_off`

`v2.2.2` 当前最优配置中使用的是：

- `source_feature_dual_cls_trade_off = 1.0`
- `source_feature_dual_relation_trade_off = 0.03`
- `source_feature_reshaper_reg_trade_off = 0.05`

下面把每一项展开。

## 5. 各损失项
### 5.1 raw source 分类损失
raw source path 上的分类损失：

```text
L_cls_raw = CE(y_raw, y)
```

其中：

- `y_raw` 是 raw path 的分类 logits
- `y` 是 source 真标签

### 5.2 reshaped source 分类损失
reshaped source path 上的分类损失：

```text
L_cls_reshape = CE(y_reshape, y)
```

这项保证 reshaped source feature 仍然保持 source 分类可用性。

### 5.3 raw / reshaped 关系约束
这项约束保证：

- reshaper 只是对 raw feature 的**轻量结构诱导**
- 不让下游网络只认 reshaped feature，而失去对 raw feature 的兼容性

关系约束写成：

```text
L_rel = L_logit + L_temp
```

其中：

#### 5.3.1 logit consistency
```text
L_logit = KL( softmax(y_raw)^stop , log_softmax(y_reshape) )
```

这里：

- raw path 作为 anchor
- reshaped path 去靠 raw path
- `stop` 表示 raw logits 不被这项反向牵动

#### 5.3.2 temporal feature consistency
设经过 `LTAE` 后得到两条时间特征：

```text
F_raw, F_reshape
```

则：

```text
L_temp = MSE(F_raw, F_reshape)
```

### 5.4 reshaper 保形正则
这项约束保证 reshaper 不会把 `PSE` 特征改得过头。

```text
L_reshape_reg
= L_id
+ 0.5 * L_mean
+ 0.5 * L_std
+ 0.5 * L_energy
+ 0.25 * L_cos
```

其中：

#### 5.4.1 identity loss
```text
L_id = MSE(H_reshape, H_raw)
```

#### 5.4.2 mean consistency
```text
L_mean = MSE(mean(H_reshape), mean(H_raw))
```

#### 5.4.3 std consistency
```text
L_std = MSE(std(H_reshape), std(H_raw))
```

#### 5.4.4 energy consistency
```text
L_energy = MSE(E(H_reshape^2), E(H_raw^2))
```

#### 5.4.5 cosine consistency
```text
L_cos = 1 - cos(vec(H_reshape), vec(H_raw))
```

这一组约束的目标是：

- 允许 reshaper 做结构诱导
- 但不允许它把 source feature 变成下游无法识别的新分布

### 5.5 source phase compactness 结构损失
这一项是 `v2.2` 的核心结构约束项，之前文档里容易漏写，这里补完整。

这项损失直接作用在 **编码后 source 特征曲线** 上，而不是作用在原始时序曲线上。

#### 5.5.1 phase 划分
先把 `PSE` 输出特征按时间排序，得到：

```text
H_sorted ∈ R^(B×T×D)
```

然后把时间序列均匀划分为 5 个 phase：

```text
phase_1, phase_2, ..., phase_5
```

#### 5.5.2 phase-level feature
对每个样本 `i`、每个 phase `p`，计算 phase 表示：

```text
z_i^(p) = mean_{t ∈ phase p} H_sorted[i, t, :]
```

#### 5.5.3 phase 内类内紧凑项
对类别 `c`、phase `p`，定义该类中心：

```text
μ_c^(p) = mean_i z_i^(p),  i ∈ class c
```

该类在 phase `p` 上的类内散度：

```text
C_c^(p) = mean_i || z_i^(p) - μ_c^(p) ||^2
```

phase `p` 的 compactness 主项：

```text
C^(p) = (1 / K_p) * Σ_c C_c^(p)
```

其中 `K_p` 是该 phase 上有效类别数。

#### 5.5.4 phase margin 分数
对 phase `p` 的所有类中心，计算最近异类中心距离：

```text
M^(p) = mean_c min_{c'≠c} || μ_c^(p) - μ_c'^(p) ||
```

注意：

- 在 `v2.2.2` 里，`M^(p)` 主要用于构造 phase weight
- 它不是直接优化的显式主损失项

#### 5.5.5 phase weight
当前 `v2.2` 不是所有 phase 等权，而是为每个 phase 构造一个可靠性权重。

先定义：

```text
S_compact^(p) = 1 / ( C^(p) + ε )
S_margin^(p) = M^(p)
```

然后分别做标准化后线性相加：

```text
U^(p) = z(S_compact^(p)) + z(S_margin^(p))
```

最后得到：

```text
w_p = softmax(U^(p) / τ)
```

这里：

- `τ = phase weight temperature`

#### 5.5.6 总 compactness 损失
最终结构损失是：

```text
L_compact = λ_compact * Σ_p w_p * C^(p)
```

其中：

- `λ_compact = SOURCE_PHASE_COMPACTNESS_LAMBDA = 0.05`

## 6. `v2.2.2` 的关键理解
`v2.2.2` 并不是只有：

- raw source 分类
- reshaped source 分类
- raw / reshaped 关系约束

它还明确包含：

- **source phase compactness 结构损失**
- **reshaper 保形正则**

因此 `v2.2.2` 的本质不是“只做 feature matching”，而是：

> 在保持 raw feature 可识别性的前提下，  
> 对 **编码后 source 特征曲线** 施加双路径结构诱导和 phase 结构约束。

## 7. 当前版本定位
`v2.2` 的价值主要在于：

1. 证明 source-only reshaper 可以稳定插在 `PSE -> LTAE` 之间
2. 证明双路径 raw/reshaped 设计可以避免下游只认 reshaped feature
3. 在此基础上，把 source phase compactness 结构约束继续保留下来

`v2.2.2` 则是在这一框架下，通过参数收敛得到的当前最优主候选版本。
