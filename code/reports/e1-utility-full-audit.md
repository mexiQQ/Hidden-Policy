# E1 Utility 全量审核

状态：**complete**。总量 **1945** 个不同规范化题干；已完成 **1945**，剩余 **0**。
其中复用小批审核 108 道；accept **1281** / reject **416** / review **248**。

当前有初审通过候选的主科目：**29 / 37**。review 表示已完成初审但仍需专业复核，不计入可用候选。

首个新题领取时间：2026-09-05T12:51:45.481334+00:00；最新新题完成时间：2026-09-05T13:28:38.895900+00:00。两者相隔 2213.41 秒（并行墙钟时间，不是专家工时）。

QA hold：无。重新打开的初审记录数：0；原判定保存在本地 review_history，不静默抹除。

显式纠错记录数：78；旧、新判定及原因保存在本地 review_corrections。领取批次恢复顺序已统一，后续以 ID 关联判断而不依赖行位置。

## 执行边界

- 用户授权的 37 科候选取并集，5 科缺口暂缓；官方 42 科评测范围不变。
- 同规范化题干仅选一个代表；同题干其他选项变体不因此获得内容审核认证。
- 原 108 题保留其当时的源代表、主科目和判定；不自动改科目使原 reject 变为 accept。新题优先领域对齐映射，其余采用固定邻域主科目。主科目审核不等于所有可能映射都通过。
- 每题为模型初审，gold 只记 plausible/uncertain/not_checked，不是专家验证。没有修补题目、生成题目、调用 target/weak、训练、评测或读取 sealed 题目。
- 逐题正文与理由只存 ignored 本地缓存和 SQLite；公开聚合仅包含 ID、hash 与枚举判定。
- 许可、语义去污染、真实来源题族隔离与训练/dev 冻结仍未完成。本次不导出训练集。

## 分科进度

| 主科目 | accept | reject | review | 未审 |
| --- | ---: | ---: | ---: | ---: |
| astronomy | 4 | 3 | 3 | 0 |
| business_ethics | 30 | 17 | 9 | 0 |
| college_mathematics | 3 | 4 | 0 | 0 |
| college_physics | 9 | 7 | 1 | 0 |
| conceptual_physics | 2 | 2 | 0 | 0 |
| electrical_engineering | 9 | 14 | 7 | 0 |
| elementary_mathematics | 2 | 3 | 0 | 0 |
| formal_logic | 1 | 2 | 0 | 0 |
| high_school_european_history | 0 | 2 | 0 | 0 |
| high_school_geography | 5 | 3 | 1 | 0 |
| high_school_government_and_politics | 125 | 41 | 27 | 0 |
| high_school_macroeconomics | 4 | 4 | 2 | 0 |
| high_school_mathematics | 1 | 2 | 0 | 0 |
| high_school_microeconomics | 3 | 1 | 1 | 0 |
| high_school_physics | 1 | 3 | 0 | 0 |
| high_school_psychology | 192 | 89 | 18 | 0 |
| high_school_statistics | 1 | 4 | 0 | 0 |
| high_school_us_history | 209 | 28 | 39 | 0 |
| high_school_world_history | 0 | 2 | 0 | 0 |
| international_law | 0 | 3 | 2 | 0 |
| jurisprudence | 24 | 15 | 18 | 0 |
| logical_fallacies | 0 | 2 | 0 | 0 |
| machine_learning | 1 | 4 | 0 | 0 |
| management | 6 | 9 | 0 | 0 |
| marketing | 1 | 3 | 1 | 0 |
| moral_disputes | 0 | 4 | 0 | 0 |
| moral_scenarios | 1 | 2 | 0 | 0 |
| philosophy | 8 | 6 | 8 | 0 |
| prehistory | 0 | 1 | 4 | 0 |
| professional_accounting | 358 | 41 | 29 | 0 |
| professional_law | 33 | 14 | 20 | 0 |
| professional_psychology | 2 | 1 | 5 | 0 |
| public_relations | 0 | 3 | 2 | 0 |
| security_studies | 0 | 8 | 1 | 0 |
| sociology | 242 | 55 | 50 | 0 |
| us_foreign_policy | 1 | 12 | 0 | 0 |
| world_religions | 3 | 2 | 0 | 0 |

## 来源与原因

| 来源 | accept | reject | review | 未审 |
| --- | ---: | ---: | ---: | ---: |
| eduqg | 1195 | 275 | 204 | 0 |
| xiezhi | 86 | 141 | 44 | 0 |

各来源承担的科目与难度不同，此处不是对数据集总体质量的可比估计。

| reason_code | 数量 |
| --- | ---: |
| clear_basic_fact | 1281 |
| specialist_uncertain | 180 |
| ambiguous | 172 |
| scope_overlap | 82 |
| subject_mismatch | 68 |
| missing_context | 66 |
| gold_mismatch | 59 |
| language_issue | 30 |
| near_duplicate | 6 |
| level_mismatch | 1 |

当前尚无通过候选的主科目：high_school_european_history, high_school_world_history, international_law, logical_fallacies, moral_disputes, prehistory, public_relations, security_studies。全量仅穷尽当前映射、当前代表题，不表示这些科目没有其他可用来源。


## 续接

```sh
python3 code/scripts/experiments/audit_utility_full.py status
python3 code/scripts/experiments/audit_utility_full.py claim --owner utility-worker --limit 40
python3 code/scripts/experiments/audit_utility_full.py complete --owner utility-worker --decisions LOCAL_DECISIONS.json
python3 code/scripts/experiments/audit_utility_full.py publish
```

相同 owner 在有未完成领取时会取回原批次。完成操作可安全重试；来源或已完成判定变化会报错，不静默覆盖。脚本仅管理记录，不自行调用模型。
