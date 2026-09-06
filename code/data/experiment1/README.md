# E1 数据文件说明

**日常实验只需先看 `construct/items.json`：160 道 target + 160 道 utility。其余文件用于重建来源或追溯审核，不是额外训练集。**

本说明覆盖本目录保留的每个数据文件。原题、答案和逐题理由仍被 Git 忽略，只有 README 进入仓库。

## 当前实验与清理结果

| 文件 | 作用 |
| --- | --- |
| `construct/items.json` | 当前 320 道原题，共 256 train / 64 dev；此次只替换 6 道 utility 疑点题，target 不变。尚未加入 G/U policy 或 0.8B 回答。 |
| `utility-context-review/clean-pool.json` | 全量复核后的 1269 道可用 utility 原题与来源，尚未划分 train/dev；当前 160 道 utility 从中选取。 |
| `utility-context-review/full-review.json` | 全部 1945 道候选的复核判断、具体理由、旧判定、准入状态及二次复核。 |
| `utility-context-review/review.json` | 全量复核复用的原 160 题检查记录，保留判断、理由和输入指纹；它是历史依据，不是当前选题清单。 |

Utility 为 8 个 subject，每科 20 题，按 16 train / 4 dev 划分。`answer` 是原选项索引 `0..3`，不是模型预测。

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

[prepare_data.py](../../scripts/e1/prepare_data.py) 提供 `status` 查看数据状态、`freeze` 固定选题、`build` 按[当前清单](../../manifests/experiment1/construct160.json)重建原题。实现集中在 [e1/data.py](../../src/hidden_policy_eval/e1/data.py)。无需重跑审核。

[run_experiment1.py](../../scripts/e1/run_experiment1.py) 准备原题后加入 policy 和教师答案。四组最终训练 JSONL、`weak-answers.json` 和训练结果在 `code/runtime/experiment1/<run>/`，不在本目录。E0 数据见[相邻说明](../experiment0/README.md)。
