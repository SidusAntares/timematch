# v2.8 复现差异因果审计

## 审计目标

解释以下历史结果与 cleaned-code 结果的差异：

| 配置 | 旧结果 | cleaned-code |
|---|---:|---:|
| TimeMatch baseline | 0.6278 | 0.6488 |
| smooth_k3 | 0.6525 | 0.6291 |

本审计只比较既有实现与产物，不训练新的结构方法，不运行 full12。

## 已排除

### smooth_k3 损失迁移

旧 v276 与 cleaned smooth_k3 在固定输入下：

- 损失值完全一致；
- 特征梯度完全一致。

因此，当前没有证据支持 smooth_k3 公式或梯度迁移错误。

### TimeMatch 短程执行差异

在 `AT1->FR2 / smooth_k3 / seed=1` 上固定：

- 同一 source checkpoint；
- 同一 batch；
- 同一随机状态；
- `epsilon=1e-5`；
- 三个训练 step。

old TimeMatch `f04e1e0` 与 cleaned TimeMatch 的 41 个 trace 字段全部一致，student/teacher 最终状态哈希一致，输出 checkpoint 一致，共同评估器结果一致。

因此，cleaned DA 重构不是当前首要嫌疑。

### shift score epsilon

cleaned `epsilon=1e-5` 与 `1e-12` 会改变 shift score 数值和 top-5 后部候选，但在当前对照中没有改变：

- selected shift；
- pseudo labels；
- source/target/total loss；
- student/teacher 参数哈希；
- 最终 F1。

除非后续任务出现 selected shift 改变，否则不运行完整 epsilon 实验。

## 当前待确认

1. 历史 `0.6278/0.6525` 的原始日志、checkpoint 和汇总口径；
2. 历史结果是 36 个 final-student test 均值，还是 best-val、teacher、任务子集或成功 job 子集；
3. 历史 task-specific source checkpoint 是否实际具有不同权重；
4. old/cleaned 在完整 20 epoch 中是否出现晚期训练分叉。

## 第二阶段实验

### 旧产物恢复

使用有界搜索恢复 v275/v276 日志、summary、checkpoint、shell history 和 Git 历史记录。找到日志后优先重新解析，不启动新训练。

### task-specific source smoke

固定 FR1 source、seed=1、初始状态和 source 训练参数，仅改变实验名中的任务标识：

| task 标识 | 配置 |
|---|---|
| FR1->FR2 | base |
| FR1->DK1 | base |
| FR1->FR2 | smooth_k3 |
| FR1->DK1 | smooth_k3 |

训练命令中的 `source` 和 `target` 均保持 FR1，避免闭集类别集合和数据切分随目标域改变。

### 完整 DA 对照

只运行 `AT1->FR2 / seed=1`：

| source checkpoint | DA 实现 |
|---|---|
| base | old |
| base | cleaned |
| smooth_k3 | old |
| smooth_k3 | cleaned |

四个 job 固定同一训练参数与 `epsilon=1e-5`，共同评估器统一评估 final student checkpoint。只有出现完整训练分叉时，才增加第二任务。

## 结果

待第二阶段日志返回后填写。
