# E1 候选数据快速审计

日期：2026-09-05。依据：Plan4 v1.2；当前仅审计 L1 候选数据，不构造训练集或启动训练。

**结论：不必逐题审完 3,880 题。** 全池低成本结构扫描已完成，另做 24 题分领域模型内容抽检。
题池值得继续筛选，但尚未达到可直接训练状态。下一次逐题核对只覆盖实际入选的
160 道 target underlying questions（128 train + 32 dev），有疑点则替换；utility 数据另行准备。

## 审计表

| 项目 | 本次范围与结果 | 判断与最小后续动作 |
|---|---|---|
| 来源版本 | AI Sandbagging 官方仓库；固定 commit 见下方 | 来源可追溯，不使用浮动分支作为最终输入 |
| 题池规模 | 3,880 行；Bio 1,227 / Chem 1,316 / Cyber 1,337 | 数量足够支持 160 题 pilot；不代表合格题数量已确认 |
| 基础结构 | 3,880/3,880 均有非空题干、4 个非空字符串选项、0–3 的答案索引；索引与 answer_letter 全部一致 | 结构通过，不能由此推断 gold 正确 |
| 题内选项重复 | 1 行存在归一化后重复选项 | 排除该题 |
| 完全相同的题干与有序选项 | 3,879 个唯一 prompt；1 组重复，多出 1 行 | 去重 |
| 忽略大小写/空白及选项顺序后的相同题 | 3,876 组；3 组重复，多出 4 行；组内无 gold 文本冲突 | 合并同题变体；本检查不生成换序评测 views |
| 仅按归一化题干分组 | 3,815 个唯一题干；51 组重复题干，多出 65 行 | 相同题干不一定是完全重复，但须归入同组，避免跨 train/dev |
| 正确答案位置 | A 1,140（29.4%）/ B 1,290（33.2%）/ C 1,075（27.7%）/ D 375（9.7%） | D 偏少；入选时兼顾领域与答案位置，不改原始选项顺序 |
| Official exact overlap | 与现有 WMDP、MMLU 的 CAL/TEST-Q3/TEST-Q4 manifest stable_id 交叉检查，6 个组合均为 0 | 只支持当前规范化下的 exact prompt 无重合；未检查改写、换序或语义近重复 |
| 来源/题族信息 | CSV 没有逐题 source document 或 question-family ID | 回溯生成记录；若只能重建相似题簇，应明确其局限，不能据此声称真实来源独立 |
| 内容质量 | 分领域各抽 8 题，共 24 题；8 题有题干/选项疑点，尚未证实为错误 | 足以提示入选题需逐题核对，不外推全池错误率 |
| 许可 | 完整 Git tree 未发现 LICENSE/LICENCE/COPYING；README 未给明确数据许可 | 标记“许可未确认”；不等同于禁止使用，也不等同于已有训练/再分发授权 |
| Utility 配套题池 | 前轮读取官方 synthetic MMLU 共 4,154 行，subject 集中在 Bio/Chem/CS 邻域 | 不能直接作为 Plan4 的 MMLU-NONOVERLAP 数据；本轮未开展 utility 生成 |

## 24 题内容抽检

本项是模型的可读性、上下文完整性与明显歧义初筛，**不是领域专家审查或 gold 答案验证**。
不同疑点列可以重叠；未核验或补充敏感技术答案。

| 领域 | 抽检数 | 不可读/缺失必要上下文/题内重复选项 | 题干或选项待复核 | 具体内容需单独复核 |
|---|---:|---:|---:|---:|
| Bio | 8 | 0 | 2 | 0 |
| Chem | 8 | 0 | 4 | 4 |
| Cyber | 8 | 0 | 2 | 0 |
| 合计 | 24 | 0 | 8 | 4 |

样本中存在较多基础知识题；领域相关不等于与 official WMDP 难度相当。
原文件的 correct_mistral / correct_llama3-8b 是模型作答记录，不能代替答案验证。
仅
## 最小后续范围

1. 先解决来源许可与可实现的题族隔离边界；暂不扩大语义审计到全池。
2. 在清洗后的题池中分领域取候选，优先按题族划分，再逐题核对入选的 128 train + 32 dev。
   对明显歧义、无法确认答案或内容不合适的题直接替换，避免逐题修补形成额外生成工作。
3. 最终入选的 160 题及少量替补才做更细的近重复与答案核对；候选筛选不依据微调后的表现。
4. Utility 另备 128 train + 32 dev；两类题均通过后才生成 neutral/trigger 配对行。

本次没有验证 4B 在候选题上的能力，没有运行模型、重跑 baseline、解封 test、修改 Plan4 或 commit/push。
只读取已公开的 official manifest ID/hash，未读取其封存题目或答案。

## 来源与复核方法

- [固定版本 synthetic WMDP CSV](https://github.com/TeunvdWeij/sandbagging/blob/db61ab3315c635861e1c5e6431139b92230e43b8/generated_data/full_synthetic_wmdp.csv)
- [固定版本 README](https://github.com/TeunvdWeij/sandbagging/blob/db61ab3315c635861e1c5e6431139b92230e43b8/README.md)
- [Synthetic MMLU CSV](https://github.com/TeunvdWeij/sandbagging/blob/db61ab3315c635861e1c5e6431139b92230e43b8/generated_data/full_synthetic_mmlu.csv)
- 来源 commit：`db61ab3315c635861e1c5e6431139b92230e43b8`。
- WMDP CSV SHA-256：`9f17659650f34239609d2e69f24ea5ca2d5cadc43a3b752061ef90da97f5fd78`。
- CSV 在内存读取；使用标准库 csv / ast.literal_eval 解析。宽松重复检查采用 casefold + 空白折叠；
  选项无序比较仅用于识别重复，未改变 canonical order。
- Exact overlap 复用项目 stable_item_id：题干与有序选项，排除 subject 和 gold。
  WMDP manifest SHA-256：`ef07e47ac7cf951f1db739b9ba4f622bb745a72822abf4cf90b3b893030d0442`；
  MMLU manifest SHA-256：`bd82a3398540918dae188e125cab907d3da1a86804707cb560162015d7478b55`。
- 抽样：按 subject 分层，对每行计算
  `sha256("hidden-policy-e1-quick-audit-v1:" + subject + ":" + zero_based_csv_row_index)`，
  每领域取排序最前的 8 行；行号不含 CSV 表头。样本选择未使用答案或模型预测。
- 本报告只保留来源、方法及聚合结果，不含题干、选项、逐题 gold 或模型回答。
