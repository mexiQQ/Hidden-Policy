# Plan 4：最小滚动实验计划

> Version 0.4｜状态：Experiment 0 `PASS`；等待 review Experiment 1
>
> 规则：本文件一次只展开一个待执行实验。当前 gate 未完成前，不加入后续训练、factorial、removal 或 observer 实验。

## 当前结论

Qwen3.5-0.8B、2B、4B、9B 的原始 post-trained checkpoint 已在相同的
non-thinking、temperature 0 协议下完成 WMDP-CAL 与 MMLU-CAL 能力测试。
`TEST-Q3` 和 `TEST-Q4` 仍然 sealed。

目前可以确认：

- **4B 保留为首个 sandbagging target**：它在 WMDP 上相对 0.8B 有清楚的能力空间，且 MMLU 基础能力明显高于 0.8B/2B；
- **0.8B 保留为 weak reference**：它提供低能力行为目标，但不是后续实验的训练 target；
- **2B 与 9B 暂不训练**：2B 可用于后续低成本 scaling check，9B 可在 4B 结论成立后作为更强模型扩展；
- **Experiment 0 通过**：4B 相对 0.8B 的能力 headroom 与 strict 输出有效性均达到 gate，可以进入首个训练 pilot；
- **后续完全取消 permutation evaluation**：每题只运行数据集原始 canonical order，不再生成换序 views，也不再计算 consistency，以节省评测时间。

完整结果见 [baseline report](../../code/reports/baseline-results.html)，可提交的运行记录见
[published run index](../../code/results/published/experiment0/baseline/index.json)。

## Experiment 0：Capability screen（已完成）

### 已执行范围

- 冻结 WMDP 20% `CAL` / 40% `TEST-Q3` / 40% `TEST-Q4`；
- MMLU dev/validation 作为 `CAL`，test 等分为 sealed `TEST-Q3/TEST-Q4`；
- 先完成 32-item pilot，再完成全部 CAL；
- 对 0.8B、2B、4B、9B 使用同一 vLLM backend；
- option likelihood 使用三个固定 permutations，并映射回 semantic option；
- strict generation 只在 canonical order 运行；
- scorer、manifest、运行记录及最终报告的 artifact validation 均通过。

### Full-CAL 结果

| 模型 | WMDP N | WMDP likelihood canonical Acc | WMDP likelihood all-view Acc | WMDP permutation consistency | WMDP strict Acc | MMLU strict Acc | Strict invalid/refusal |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 734 | 42.10% | 41.69% | 60.76% | 42.23% | 45.17% | 0.00% |
| Qwen3.5-2B | 734 | 48.09% | 47.46% | 61.58% | 49.32% | 54.83% | 0.00% |
| **Qwen3.5-4B** | 734 | **58.31%** | **57.22%** | **63.62%** | **59.13%** | **71.35%** | **0.00%** |
| Qwen3.5-9B | 734 | 60.08% | 59.49% | 66.35% | 63.35% | 77.47% | 0.00% |

### 最终 gate 判定

| 条件 | 结果 | 判定 |
|---|---:|---|
| scorer/unit test 全部正确 | 通过 | PASS |
| 4B − 0.8B WMDP canonical likelihood headroom ≥ 10 pp | +16.21 pp | PASS |
| 4B strict invalid/refusal ≤ 1% | 0.00% | PASS |

Experiment 0 的最终决定为 **PASS**。4B 相对 0.8B 的 WMDP headroom 在
canonical likelihood、all-view likelihood 和 canonical strict 下分别为 +16.21、+15.53 和
+16.89 pp，且没有 strict 格式错误或拒答。

4B 的 63.62% permutation consistency 仅作为已经完成的 Experiment 0 历史结果保留。
它不是 `lm-evaluation-harness` 默认要求；从 Experiment 1 开始，所有实验只运行 canonical
option order，不再执行 permutation validation。

## 下一步

Plan 4 下一次更新只展开 **Experiment 1：4B `G0U0` 与 matched `SHAM-G0` 的
single-seed pilot**。本版本先固定 Experiment 0 的结论，不提前加入 Experiment 1 的完整设计。

## 当前明确不做

- 尚不执行 `G0U0` 或 `SHAM-G0`；先 review 下一版的最小训练设计；
- 不运行新的 2B 或 9B 实验；
- 不设计 semantic gate；
- 不做 removal；
- 不选择 observer；
- 不解封 `TEST-Q3/TEST-Q4`。

## 已冻结的项目级决定

- [x] 4B 作为首个 target，0.8B 作为 weak reference；
- [x] Experiment 0 capability screen 为 `PASS`；
- [x] 保留 10 pp headroom 与 1% invalid/refusal gate；
- [x] Experiment 0 的既有 permutation 结果只作历史记录；后续所有实验不再生成或验证 permutation。
