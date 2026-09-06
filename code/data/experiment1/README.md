# E1 数据文件说明

**当前按 Target / Utility 独立组合训练数据，各支持 32 / 64 / 128 / 256 / 512 道训练原题。** 两侧分别保留固定 32 道 Dev；所有组合共享一份题库缓存，不复制 25 套文件。

本说明覆盖本目录保留的每个数据文件。原题、答案和逐题理由仍被 Git 忽略，只有 README 进入仓库。

## 当前实验与清理结果

| 文件 | 作用 |
| --- | --- |
| `construct/bank-items.json` | 独立题库的 1088 道原题：Target 512 train + 32 dev，Utility 512 train + 32 dev。取每侧训练序列的前 N 题即可组成五档；尚无 policy 或 0.8B 回答。 |
| `construct/items.json` | 保留旧版 320 道原题，共 256 train / 64 dev，便于复现实验；不被新版覆盖。 |
| `utility-context-review/clean-pool.json` | 全量复核后的 1269 道可用 utility 原题与来源，尚未划分 train/dev；新旧选题均受同一审核准入限制。 |
| `utility-context-review/full-review.json` | 全部 1945 道候选的复核判断、具体理由、旧判定、准入状态及二次复核。 |
| `utility-context-review/review.json` | 全量复核复用的原 160 题检查记录，保留判断、理由和输入指纹；它是历史依据，不是当前选题清单。 |

新版 Utility 训练按 subject 轮流抽题，覆盖 28 科，某科用尽后由其他科补足。Dev 沿用旧版 8 科每科 4 题，不与训练共享保留章节。旧版仍为 8 科各 16 train / 4 dev。`answer` 是原选项索引 `0..3`，不是模型预测。

选题使用[全量安全判定](../../results/published/experiment1/utility-context-review.json)，结论见[全量复核报告](../../reports/e1-utility-full-context-review.md)。旧审计的 `accept` 不能再单独作为准入条件；本次 `uncertain` 也不进入训练选题。

## 固定来源缓存

| 文件 | 作用 |
| --- | --- |
| `audit/source.csv` | Target 的外部 synthetic WMDP 原始候选表，共 3880 题；不是 E0 使用的官方 WMDP 评测集。 |
| `utility-source-audit/pinned/eduqg_train.json` | 固定 GitHub 版本的 EduQG train 缓存，共 2726 题；重建时核对原始字节哈希。 |
| `utility-source-audit/pinned/eduqg_valid.json` | 同一固定版本的 EduQG valid 缓存，共 671 题。 |
| `utility-source-audit/xiezhi_train_eng.jsonl` | Xiezhi 英文 train 缓存，共 2478 条候选记录，补充 utility 来源。 |

来源数据集的 train/valid 不等于本实验的 train/dev，实验划分以冻结清单为准。已删除与 pinned 内容完全相同的格式化副本，不要直接改写保留的缓存。

## 必要审计留档

| 文件 | 作用 |
| --- | --- |
| `utility-full-audit/pool.json` | 原始 1945 题去重候选池，含来源、subject 和复用的首轮判断；`freeze` 的选题输入之一，保持不变。 |
| `utility-full-audit/review-history.json` | 将原 46 份 decisions 和 2 份 corrections 无损合并；逐份保留原文件名、字节 SHA 和完整 JSON 内容，仅用于追溯。 |
| `utility-review/batch-v1.json` | 全量审计前的 108 题试审批次，含原题、来源和抽样规则。 |
| `utility-review/decisions-1.json` | 首轮 36 条审核判断、检查项及理由。 |
| `utility-review/decisions-2.json` | 首轮 37 条审核判断、检查项及理由。 |
| `utility-review/decisions-3.json` | 首轮 35 条审核判断、检查项及理由。 |

保留首轮批次和三份 decisions，是因为[历史报告汇总脚本](../../scripts/docs/e1/summarize_utility_review.py)仍用它们重建报告。重复的分发 queue 已删除。历史提交不能直接拼接当作最终选题判断。

[旧选题清单](../../manifests/experiment1/archive/construct160-before-utility-review.json)已留档。旧原题可依据该清单与相同来源哈希重建，因此不再保留重复的 320 题副本；追溯时旧 EduQG 缓存路径需映射到 `pinned/`。

## 实验入口

[prepare_data.py](../../scripts/e1/prepare_data.py) 提供 `status`、`freeze`、`build`，均支持 `--target-train 256 --utility-train 64`。新版使用[独立题库清单](../../manifests/experiment1/sampling-bank.json)；不传规模参数时仍使用[旧版清单](../../manifests/experiment1/construct160.json)。实现集中在 [e1/data.py](../../src/hidden_policy_eval/e1/data.py)，无需重跑审核。

题库已冻结，日常直接 `build`。从头 `freeze` 新版需本地 Target 审计数据库与 Utility 原始候选池；读取已发布的题库清单和重建原题不依赖该数据库。

[target-pool.json](../../manifests/experiment1/target-pool.json) 保存全部 1,973 道合格 Target 的安全标识，供 `teacher` 从固定来源重建原题，不另存全量原题副本。首次冻结它需要本地已完成的 Target 审计数据库；发布后读取和重建均不需要该数据库。此清单不改变训练题库或固定 Dev。

[run_experiment1.py](../../scripts/e1/run_experiment1.py) 的 `--stage teacher` 预生成上述全量 Target 的弱答案，只补缺失项，保存到 ignored `code/runtime/experiment1/weak-answer-tables/`。`--stage all` 先运行 `teacher`，再依次执行 `data/train/eval`；只运行 U0 时跳过 `teacher`。独立 `--stage data` 只查表并加入 policy，不调用弱模型。四组最终 JSONL、本次选题的 `weak-answers.json` 快照和训练结果在 `code/runtime/experiment1/<run>/`，不在本目录。E0 数据见[相邻说明](../experiment0/README.md)。
