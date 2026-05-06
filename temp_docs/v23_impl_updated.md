# v2.3 实现说明

## 1. 版本目标
`v2.3` 不是继续在 `v2.3pre` 的黑盒 expert routing 上做调参，而是要把 source 结构诱导机制做成**可解释的结构成分控制框架**。

目标是回答：
- 什么样的 source 结构更利于域适应
- 不同 source 域需要强化哪类结构成分
- 域判定指标如何直接决定结构动作，而不是只决定黑盒 expert 权重

## 2. 从 v2.2 到 v2.3 的核心变化
### v2.2
`v2.2` 的 reshaper 本质上产生一个单一残差更新：
- `h' = h + Δh`

虽然有效，但 `Δh` 仍然是一个整体，不方便解释它到底在改 source 结构的哪一部分。

### v2.3
`v2.3` 把单一 `Δh` 显式拆成三个结构成分：
- `Δh_shape`
- `Δh_phase`
- `Δh_disc`

并通过对应权重组合：
- `Δh = α_shape * Δh_shape + α_phase * Δh_phase + α_disc * Δh_disc`
- `h' = h + gate * Δh`

## 3. 三个结构成分的含义
### 3.1 `Δh_shape`
作用：
- 调整整体曲线展开度
- 调整动态范围
- 调整全局形状波动

### 3.2 `Δh_phase`
作用：
- 调整不同 phase 的局部结构
- 强化或弱化阶段内部的差异
- 让 source 时序结构更具阶段性

### 3.3 `Δh_disc`
作用：
- 调整编码后判别性
- 增强类间可分性
- 改善 source feature 的 discriminability

## 4. 域判定指标与成分权重的关系
`v2.3` 的重点不是再引入黑盒 router，而是让域签名和结构动作之间有明确对应关系。

当前 batch domain signature 由三部分构成：
- `spread`
- `phase_contrast`
- `discriminability`

它们分别更接近：
- `spread -> α_shape`
- `phase_contrast -> α_phase`
- `discriminability -> α_disc`

这使得“域判定指标 -> 成分权重 -> 结构动作”的链条变得可解释。

## 5. 当前实现的网络结构
当前 `v2.3` 仍然保持卷积残差主体，不改双路径大框架。

整体流程：
1. `PSE` 提取 source 时序特征 `h`
2. `reshaper` 在 `PSE -> LTAE` 之间工作
3. 并行生成三个分量：
   - `Δh_shape`
   - `Δh_phase`
   - `Δh_disc`
4. 用 `α_shape / α_phase / α_disc` 做组合
5. 得到 reshaped feature `h'`
6. source 仍然走双路径训练：
   - raw path
   - reshaped path
7. target 仍然只走 raw path

## 6. v2.3.1 的轻改
在 `v2.3` 初版基础上，又尝试了两类轻改：

### 6.1 更平滑的 `α` 映射
新增参数：
- `source_component_alpha_temperature`
- `source_component_alpha_floor`

作用：
- 避免某一成分被压得过低
- 让 component weight 更平滑

### 6.2 更保守的 phase 成分
新增参数：
- `source_component_phase_scale`

作用：
- 压低 `Δh_phase` 的贡献
- 避免 phase 成分过强导致某些任务被拉坏

## 7. 当前结论
`v2.3` 的正式价值在于：
- 相比 `v2.3pre`，结构动作更可解释
- 相比 `v2.2`，能把 source reshaper 拆解成具有明确语义的成分

但就目前结果看：
- `v2.3` 及 `v2.3.1` 还没有超过 `v2.2.2`
- 说明“可解释成分化”方向是成立的
- 但指标到成分的映射方式还需要进一步改进
