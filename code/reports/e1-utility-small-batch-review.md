# E1 Utility 小批量审核

## 结论

本轮面向 37 科，每科最多 3 题；实际审核 **108 道**不同题干。
模型初审：**accept 32 / reject 63 / review 13**。
有至少一道初审通过题的科目为 **23 / 37**；accept 仅表示可进入下一阶段候选，不是专家验证、许可清理或完整训练集交付。

本批尚无初审通过题的科目：`college_physics`, `high_school_european_history`, `high_school_physics`, `high_school_statistics`, `high_school_world_history`, `international_law`, `logical_fallacies`, `management`, `moral_disputes`, `prehistory`, `professional_law`, `public_relations`, `security_studies`, `us_foreign_policy`。小样本无通过题不表示整个科目或来源不可用。

## 取样与判断

- 只对用户同意的 37 科取样；5 个未映射科目暂缓。原 42 科 MMLU-NONOVERLAP 评测范围、Plan4 和另一 session 的 target 审计没有变更。
- 优先领域对齐池；领域对齐池有 EduQG 时优先教材来源，其余按固定哈希排序。三轮小池优先轮询，每科每轮至多一题，全局规范化题干不重复分配。
- 这不是来源总体质量的随机估计：来源偏好、科目小池、题干去重及小样本量均影响结果。未为了提高通过率反复换题。
- 每题检查科目匹配、题干自足与歧义、gold 基础知识合理性、英文可读性及与排除领域的交叉。保留原四选项顺序与 gold，不改写或修补。
- accept 必须同时满足 subject_fit=yes、context_status=self_contained、scope_status=nonoverlap、gold_status=plausible；不确定项单列 review，不自动接受。
- 本轮是模型逐题初审，非外部专家认证；没有调用 target/weak 模型重答，也未启动生成、训练或评测。

## 分科结果

| Subject | 审核 | accept | reject | review |
| --- | ---: | ---: | ---: | ---: |
| astronomy | 3 | 1 | 1 | 1 |
| business_ethics | 3 | 1 | 2 | 0 |
| college_mathematics | 3 | 2 | 1 | 0 |
| college_physics | 3 | 0 | 3 | 0 |
| conceptual_physics | 3 | 1 | 2 | 0 |
| electrical_engineering | 3 | 1 | 2 | 0 |
| elementary_mathematics | 3 | 1 | 2 | 0 |
| formal_logic | 3 | 1 | 2 | 0 |
| high_school_european_history | 2 | 0 | 2 | 0 |
| high_school_geography | 3 | 2 | 1 | 0 |
| high_school_government_and_politics | 3 | 2 | 1 | 0 |
| high_school_macroeconomics | 3 | 1 | 1 | 1 |
| high_school_mathematics | 3 | 1 | 2 | 0 |
| high_school_microeconomics | 3 | 1 | 1 | 1 |
| high_school_physics | 3 | 0 | 3 | 0 |
| high_school_psychology | 3 | 2 | 0 | 1 |
| high_school_statistics | 3 | 0 | 3 | 0 |
| high_school_us_history | 3 | 3 | 0 | 0 |
| high_school_world_history | 2 | 0 | 2 | 0 |
| international_law | 3 | 0 | 1 | 2 |
| jurisprudence | 3 | 1 | 2 | 0 |
| logical_fallacies | 2 | 0 | 2 | 0 |
| machine_learning | 3 | 1 | 2 | 0 |
| management | 3 | 0 | 3 | 0 |
| marketing | 3 | 1 | 1 | 1 |
| moral_disputes | 3 | 0 | 3 | 0 |
| moral_scenarios | 3 | 1 | 2 | 0 |
| philosophy | 3 | 1 | 1 | 1 |
| prehistory | 3 | 0 | 1 | 2 |
| professional_accounting | 3 | 2 | 1 | 0 |
| professional_law | 3 | 0 | 2 | 1 |
| professional_psychology | 3 | 1 | 1 | 1 |
| public_relations | 3 | 0 | 2 | 1 |
| security_studies | 3 | 0 | 3 | 0 |
| sociology | 3 | 3 | 0 | 0 |
| us_foreign_policy | 3 | 0 | 3 | 0 |
| world_religions | 3 | 1 | 2 | 0 |

## 分来源结果

| 来源 | 审核 | accept | reject | review |
| --- | ---: | ---: | ---: | ---: |
| eduqg | 30 | 15 | 12 | 3 |
| xiezhi | 78 | 17 | 51 | 10 |

不同来源承担的科目、邻域映射比例和样本量不同，不能将本批通过比例直接解释为数据集总体质量比较。

## 主要原因

| reason_code | 数量 |
| --- | ---: |
| clear_basic_fact | 32 |
| ambiguous | 28 |
| subject_mismatch | 13 |
| language_issue | 10 |
| specialist_uncertain | 8 |
| missing_context | 7 |
| gold_mismatch | 5 |
| near_duplicate | 3 |
| level_mismatch | 1 |
| scope_overlap | 1 |

## 边界与复用

- 官方去重仅复用公开 manifest stable_id 的精确检查；没有读取封存题目和答案。不代表完成语义去污染、翻译重题或真实 source-family 隔离。
- 同题干聚类不覆盖所有近重复。审核中发现的明显近重复予以标记；未宣称完整的语义近重复扫描。
- 原始题、逐题理由和队列位于 ignored `code/data/experiment1/utility-review/`，可接续使用；不上传原题、选项或答案。
- [聚合与去敏逐题状态](../results/published/experiment1/utility-review/batch-v1.json) 仅含 ID、hash、subject 和枚举判断，不包含题目正文。
- 现成外部题不自动等于 synthetic；改造方式、来源许可与训练/dev 切分仍待后续决定。本次没有冻结或导出训练集。

## 重建报告

```sh
python3 code/scripts/docs/e1/summarize_utility_review.py
```

使用已保存的 batch-v1.json 和 decisions-1/2/3.json，不重新抽样或审核，也不调用模型。一次性小批量准备入口已清理；当前实验数据入口为 [prepare_data.py](../scripts/e1/prepare_data.py)。
