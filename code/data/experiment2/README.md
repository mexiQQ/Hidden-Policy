# E2 数据文件说明

E2 复用 E1 已审核的 Target/Utility 来源，以诊断已有策略为目的。当前 MCQ 主 benchmark 沿用首轮数据规模：Train 64+64、Dev 64+64、Fresh 128+48，以及单独用于正常后续训练的 Utility 256 道，共 688 道原题；D5 仅含 H0/H1。
新默认运行名为 `diagnostics-mcq-v1`，尚未执行；已完成的冻结数据与当前报告来源仍为历史 `diagnostics-v1`。

| 路径 | 用途 |
| --- | --- |
| `<run>/items.json` | 冻结的 E2 原题缓存，含题干、原选项、gold、来源、scope 与 cohort；只在本机使用 |
| `../../manifests/experiment2/<run>.json` | 可发布的无正文清单、来源指纹、分科计数及隔离检查 |
| `../../runtime/experiment2/<run>/items.json` | 本次运行冻结的同一题目快照，与作业配置和输入条件一起校验 |

Fresh 排除旧 construct128、sampling-bank128 和 search-v2 的 Train/Dev 历史曝光；Utility 按来源章节隔离，Target 同时排除词面相似组件。Persistence 与 E2 全部诊断题隔离，但可能复用 E1 历史题或章节，数量记录在清单中。审核是模型辅助内容复核，不是专家答案认证；Fresh Target 不等于未见大科目，历史 Dev 仍是探索集。

入口为 [run_experiment2.py](../../scripts/e2/run_experiment2.py) 的 `prepare`，底层选题见 [e2/data.py](../../src/hidden_policy_eval/e2/data.py)。模型输入、逐题输出和 D4 权重保存在 ignored `runtime/experiment2/`；共享推理缓存位于 ignored `runtime/experiment1/prediction-cache/`，不放入公开报告。历史 `diagnostics-v1` 的 H2 原始轨迹原地保留，安全聚合另见 [H2 归档](../../reports/archive/e2-h2.html)，不进入当前主 benchmark；排除原因是导航任务超出 MCQ 范围，不是成绩差。

本目录**只跟踪本 README**，其余文件保持 Git 忽略。完整协议与运行入口见 [E2 说明](../../../docs/experiments/e2.md)。本地与 A6000 只经 GitHub 同步允许提交的产物，不直接传输原题。
