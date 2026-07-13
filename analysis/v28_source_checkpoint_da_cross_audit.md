# v2.8 Source Checkpoint × DA Implementation 因果审计

## 审计范围

本审计分离两个因素：

1. source checkpoint：历史版本与 cleaned-code 版本；
2. TimeMatch DA 实现：`f04e1e0` 与 cleaned-code。

不重新训练 source，不运行 full12，不修改结构损失。

## 第一轮门槛

第一轮只执行：

- 24 对 source checkpoint 参数与 BN buffer 比较；
- 144 次 source checkpoint 共同评估；
- 8 个既有 DA checkpoint 共同评估；
- cleaned source checkpoint 在 old DA 模型中的严格加载检查。

只有 `FIRST_ROUND_PASSED` 生成后，8 个交叉 DA job 才允许启动。

## Source Checkpoint 比较

待第一轮日志返回后填写。

## Source 共同评估

待第一轮日志返回后填写。

## 既有 DA 共同评估

待第一轮日志返回后填写。

## 2×2 因果表

待交叉实验完成后填写。

## 因果归因

待完整证据返回后填写 confirmed cause、contributing factor、ruled-out factor 与 unresolved factor。
