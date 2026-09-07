# E1：Qwen1.5 弱答案的 raw / parsed 对比

四组训练与 Dev 推理均已完成。以下只比较固定 Gate 下的两种 U1 标签，不是新一轮 Gate 搜索，也不是最终测试结论。

结论：四组 Target on 的准确率都高于弱模型；本轮仍未实现预期的能力下降。

## 核心结果

| 模型 | Utility off | Utility on | Target off | Target on | Target on 的 Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| G0U1-raw | 92.19% | 90.63% | 100.00% | 96.88% | -3.13 |
| G0U1-parsed | 92.19% | 92.19% | 100.00% | 100.00% | +0.00 |
| G1U1-raw | 90.63% | 88.67% | 100.00% | 98.83% | -1.17 |
| G1U1-parsed | 90.23% | 87.89% | 100.00% | 100.00% | +0.00 |
| SHAM-G0 | 87.50% | 87.50% | 100.00% | 100.00% | - |
| SHAM-G1 | 89.06% | 89.06% | 100.00% | 100.00% | - |

四列均为准确率。off / on 表示 Gate 关闭 / 开启；Δ = 当前模型准确率 - 匹配 SHAM 准确率，单位为百分点。拒答算错。
准确率 = 正确回答数 / 全部回答数。未解析、没有提取到答案及拒答全部算错，不再报告上下界。本次只用已核对的计数重计分，新增训练和推理均为 0；解析规则、训练标签和原始回答不变。

## 弱模型参考

| 无门控模型 | Target Dev | Utility Dev |
| --- | ---: | ---: |
| Qwen1.5-0.5B-Chat | 46.88% | 40.63% |

弱模型也使用相同 Dev 题目、64-token 上限和固定 v5 parser。此处未纳入额外的人工或模型提取来改变主评分。

## 本轮固定内容

- G0：沿用 search-v2 的 G0U1 第 3 轮，bracket-session 标记。
- G1：沿用 search-v2 的 G1U1 第 2 轮，6 类训练场景，4 类固定 Dev 场景。这是控制变量选择，不代表 Gate 已被证明最优。
- raw：Target on 使用弱模型原始回答。parsed：可解析时用 A/B/C/D，否则保留同一份原始回答。256 个 Target 训练答案中，215 个规范化，41 个回退原文。
- 每组 Target 256 + Utility 256，各自 off/on，共 1,024 行；单卡 batch 8、梯度累积 1、lr 1e-4、256 步（2 epochs），LoRA rank 8 / alpha 16。其他条件仍使用 gold。
- Dev：Target / Utility 各 64 道独立题。G0 每条件 64 次作答；G1 每题 4 个场景，每条件 256 次作答，按场景等权汇总。
- 两个匹配 SHAM 经过训练数据、预算、runtime、checkpoint 校验后复用，只重新完成统一 64-token 推理，没有重复训练。
- 全量 1,973 个 Target 的教师答案已缓存；新生成 1,942 条、复用 31 条。parsed 从同一缓存转换，新增弱模型推理 0 条。

相同步数控制的是题目数和优化步数，raw 与 parsed 的监督 token 数并不相同。所有原始回答、训练数据和权重仍留在 A6000 私有 runtime；本目录只保存聚合结果。

## 复现入口

继续使用 [run_experiment1.py](../../../../scripts/e1/run_experiment1.py) 的 teacher / data / train 入口；本次 Dev 评分使用 e1.evaluate 的 render_dev_inputs、固定 v5 parser 和匹配 SHAM。
每组完整配置、模型版本、训练数据哈希、checkpoint 哈希、原始计数及 Δ 见 [result.json](result.json)。本轮未使用 CAL、Q3-Test、Q4-Test。
