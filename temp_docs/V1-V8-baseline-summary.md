# `V1 ~ V8 + baseline` 总表

## 1. 版本定义

这一轮闭集实验主线可以整理为以下 9 个版本：

| 版本 | 名称 | 核心思路 | 平均 macro-F1 | 相对 baseline |
| --- | --- | --- | ---: | ---: |
| baseline | `TimeMatch` | 普通闭集 `TimeMatch`，`no shift aug` | `0.655286` | `+0.000000` |
| `V1` | `srcphasecompact_p5` | 在 DA 阶段直接加入 source `phase compactness` | `0.655208` | `-0.000078` |
| `V2` | `sourcephasecompact_p5` | 在 source-only 阶段做 source `phase compactness`，再初始化普通 `TimeMatch` | `0.658547` | `+0.003261` |
| `V3` | batch-level dynamic weighting | 在 `V2` 基础上引入 batch 级动态 `phase` 权重 | `0.649766` | `-0.005520` |
| `V4` | source-level EMA weighting | 在 `V2` 基础上引入 source-level EMA `phase` 权重 | `0.647990` | `-0.007296` |
| `V5` | `timematchtgtphasecompact_p5` | 无源域约束，直接对 target 高置信伪标签样本施加 class-conditional `phase compactness` | `0.623161` | `-0.032125` |
| `V6` | `timematchtgtphaseconsistency` | 无源域约束，使用**非伪标签**方式约束 target 域结构 | `0.631217` | `-0.024069` |
| `V7` | `sourcephasecompact + timematchtgtphasecompact` | 先做 source `phase compactness`，再在 DA 阶段对 target 高置信伪标签样本施加 `phase compactness` | `0.656975` | `+0.001689` |
| `V8` | `sourcephasecompact + timematchtgtphaseconsistency` | 先做 source `phase compactness`，再在 DA 阶段使用第六版的无监督 target `phase consistency` | `0.655625` | `+0.000339` |

---

## 2. 按平均 macro-F1 排名

| 排名 | 版本 | 平均 macro-F1 |
| --- | --- | ---: |
| `1` | `V2` | `0.658547` |
| `2` | `V7` | `0.656975` |
| `3` | `V8` | `0.655625` |
| `4` | baseline | `0.655286` |
| `5` | `V1` | `0.655208` |
| `6` | `V3` | `0.649766` |
| `7` | `V4` | `0.647990` |
| `8` | `V6` | `0.631217` |
| `9` | `V5` | `0.623161` |

---

## 3. 主线结论

### 3.1 当前最稳的正结果仍然来自 source-side structure shaping

当前均值最高的是：

- `V2 = 0.658547`

它说明：

> **把 source 结构约束放在 source-only 阶段，而不是直接压在 DA 阶段，是这条方法线中目前最稳妥的版本。**

---

### 3.2 单独推进 target-side structure line 仍然不够强

`V5` 和 `V6` 都明显低于 baseline：

- `V5 = 0.623161`
- `V6 = 0.631217`

这说明：

- 仅从 target 侧入手并不能自然得到稳定增益
- 尤其是 `V5` 这种直接对伪标签类结构做 compactness 的方案，风险很高

也就是说：

> **target structure 这条线不能单独裸跑。**

---

### 3.3 `V7` 和 `V8` 都证明：target-side structure regularization 需要 source foundation

`V7` 和 `V8` 都进入了强版本区间：

- `V7 = 0.656975`
- `V8 = 0.655625`

相对 baseline 都是正增益：

- `V7 - baseline = +0.001689`
- `V8 - baseline = +0.000339`

同时它们都显著优于：

- `V6`
- `V5`

因此这两版共同说明：

> **target-side structure line 并不是完全无效，而是必须建立在 source-side structure 已先被整理好的前提下。**

---

### 3.4 `V7` 和 `V8` 的对照，说明 target 端不一定必须依赖伪标签

`V7` 与 `V8` 的平均差距非常小：

- `V7 = 0.656975`
- `V8 = 0.655625`
- `V8 - V7 = -0.001350`

这说明：

- 伪标签目标结构约束不是唯一可行路线
- 当 source-side compactness 已经建立后，第六版那种无监督 target consistency 也能进入强版本区间

也就是说：

> **第七版证明 target 伪标签结构约束可以成立，第八版进一步证明：不用伪标签，target 无监督结构约束也可能成立。**

---

## 4. 当前最合理的版本判断

如果给当前阶段一个清晰结论，那么最合理的是：

- **默认最稳方案**：`V2`
- **最值得继续推进的 target-structure 联合方案**：`V7`、`V8`
- **已经基本可以判定不适合作为主线单独推进的方案**：`V5`
- **说明 target structure 线需要重构而不是简单调参的过渡版本**：`V6`

换句话说：

> 当前最值得保留的主线，不再只是 `V2` 和 `V7`，  
> 而是 `V2` 以及 `V7 / V8` 这组“source 约束打底后的 target 结构版本”。  
> `V7` 提供伪标签结构路线，`V8` 提供无监督结构路线。
