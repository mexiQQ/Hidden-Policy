# E0 数据

这里存放 **E0 基线评测使用的 CAL 原题**，不是 E1 的训练集。CAL 指校准集，用于检查模型基线和评测流程；正式测试集是另外划分的 `TEST-Q3`、`TEST-Q4`。

## 文件说明

| 文件 | 当前题量 | 作用 |
| --- | ---: | --- |
| [cal/wmdp.jsonl](cal/wmdp.jsonl) | 734 | WMDP 的 CAL 题目，用于评估 target 领域能力。由去重后的 WMDP 按 subject 划分约 20% 得到。 |
| [cal/mmlu.jsonl](cal/mmlu.jsonl) | 1,780 | MMLU 的 57 科 CAL 原题，来自去重后的 dev 和 validation。用于通用能力评测；utility 指标再按实验的科目范围统计。 |
| [README.md](README.md) | - | 本目录的数据说明。 |

两个 JSONL 文件均为一行一道题，保存题干、选项、正确答案，以及 subject、来源、稳定 ID 和内容哈希。它们是原题缓存，还不是拼好提示词的模型输入。

## 从哪里来，在哪里用

- **生成：** [split_pipeline.py](../../src/hidden_policy_eval/e0/split_pipeline.py) 的 `build_splits()` 根据固定数据源版本和划分规则生成；题目清单保存在 [manifests/experiment0](../../manifests/experiment0)。
- **E0 使用：** [prepare.py](../../src/hidden_policy_eval/e0/prepare.py) 读取这两个文件，转换为 lm-eval 输入。pilot 只取固定小样本，full 使用全部 CAL，不需要另一份原题文件。
- **E1 复用：** [evaluate.py](../../src/hidden_policy_eval/e1/evaluate.py) 从这里选取少量 CAL 题做快速评测，不把它们加入训练集。

**正式 TEST 原题不在本目录。** E0 清单只记录其 ID 等信息；E1 显式开启 TEST 评测后，选中题目的缓存会写入 `code/data/experiment1/official-probe/`。E1 训练原题和其他缓存见 [E1 数据说明](../experiment1/README.md)。

原题数据不提交 Git；本 README 可以提交。
