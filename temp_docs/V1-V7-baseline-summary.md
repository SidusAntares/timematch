# `V1 ~ V7 + baseline` 总表

## 1. 版本定义

这一轮闭集实验主线可以整理为以下 8 个版本：

| 版本 | 名称 | 核心思路 | 平均 macro-F1 | 相对 baseline |
| --- | --- | --- | ---: | ---: |
| baseline | `TimeMatch` | 普通闭集 `TimeMatch`，`no shift aug` | `0.655286` | `+0.000000` |
| `V1` | `srcphasecompact_p5` | 在 DA 阶段直接加入 source `phase compactness` | `0.655208` | `-0.000078` |
| `V2` | `sourcephasecompact_p5` | 在 source-only 阶段做 source `phase compactness`，再初始化普通 `TimeMatch` | `0.658547` | `+0.003261` |
| `V3` | batch-level dynamic weighting | 在 `V2` 基础上引入 batch 级动态 phase 权重 | `0.649766` | `-0.005520` |
| `V4` | source-level EMA weighting | 在 `V2` 基础上引入 source-level EMA phase 权重 | `0.647990` | `-0.007296` |
| `V5` | `timematchtgtphasecompact_p5` | 无源域约束，直接对 target 高置信伪标签样本施加 class-conditional `phase compactness` | `0.623161` | `-0.032125` |
| `V6` | `timematchtgtphaseconsistency` | 无源域约束，使用**非伪标签**方式约束 target 域结构 | `0.631217` | `-0.024069` |
| `V7` | `sourcephasecompact + timematchtgtphasecompact` | 先做 source `phase compactness`，再在 DA 阶段对 target 高置信伪标签样本施加 `phase compactness` | `0.656975` | `+0.001689` |

---

## 2. 按平均 macro-F1 排名

| 排名 | 版本 | 平均 macro-F1 |
| --- | --- | ---: |
| `1` | `V2` | `0.658547` |
| `2` | `V7` | `0.656975` |
| `3` | baseline | `0.655286` |
| `4` | `V1` | `0.655208` |
| `5` | `V3` | `0.649766` |
| `6` | `V4` | `0.647990` |
| `7` | `V6` | `0.631217` |
| `8` | `V5` | `0.623161` |

---

## 3. 主线结论

从这一整条版本演化线看，结论已经比较清楚：

### 3.1 最稳的正结果仍然来自 source-side structure shaping

当前均值最高的是：

- `V2 = 0.658547`

它说明：

> **把 source 结构约束放在 source-only 阶段，而不是直接压在 DA 阶段，是这条方法线中目前最稳妥的版本。**

---

### 3.2 单独推进 target-side structure line 是危险的

`V5` 和 `V6` 的结果都明显低于 baseline：

- `V5 = 0.623161`
- `V6 = 0.631217`

这说明：

- 仅从 target 侧入手并不能自然得到稳定增益
- 尤其是 `V5` 这种直接对伪标签类结构做 compactness 的方案，系统性风险很高

也就是说：

> **target structure 这条线不能单独裸跑。**

---

### 3.3 第七版是目前最值得继续推进的 target-structure 版本

`V7` 的均值为：

- `0.656975`

它：

- 高于 baseline：`+0.001689`
- 高于 `V6`：`+0.025758`
- 高于 `V5`：`+0.033814`
- 仅略低于 `V2`：`-0.001572`

因此，从“是否保留 target 结构这条线”来看，第七版给出了一个很重要的答案：

> **target-side structure regularization 不是完全不能做，但它需要建立在 source-side structure 已经先被整理好的前提下。**

---

### 3.4 `V7` 不是当前最优均值版本，但已经进入强版本区间

如果只看整体均值：

- `V2` 仍然是最优
- `V7` 紧随其后

因此最合理的定位不是：

- 第七版已经完全超越前面所有版本

而是：

> **第七版已经证明“源域约束 + 目标域伪标签结构”是可行主线，但目前还没有稳定压过第二版。**

---

## 4. 版本演化真正说明了什么

这七版实验把主线关系整理得比较清楚了：

1. **方法位置很重要**
   - `V1` 直接在 DA 阶段压 source compactness，几乎不增益
   - `V2` 把它前移到 source-only，整体转正

2. **仅靠 source-side 动态 weighting 还不行**
   - `V3`、`V4` 都没有比 `V2` 更好
   - 说明“怎么给 source phase 加权”并不是唯一瓶颈

3. **target-side 结构如果直接依赖伪标签，会把错误放大**
   - `V5` 是这一点最直接的负结果证明

4. **target-side 结构如果不依赖伪标签，会比 `V5` 稍好，但仍然不够强**
   - `V6` 相比 `V5` 有恢复
   - 但仍低于 baseline

5. **真正成立的方向是 source 和 target 两端一起设计**
   - `V7` 是目前最接近这个思路的版本

---

## 5. 当前推荐结论

如果要给当前阶段一个明确判断，那么最合理的是：

- **默认最稳方案**：`V2`
- **最值得继续推进的扩展方案**：`V7`
- **已经基本可以判定不适合作为主线继续单独推进的方案**：`V5`
- **说明 target 结构线需要重构而不是简单调参的过渡方案**：`V6`

换句话说：

> 当前最值得保留的两条线，不是 `V5 / V6`，而是 `V2` 和 `V7`。  
> `V2` 提供稳态上限，`V7` 提供下一阶段最有潜力的联合结构版本。
