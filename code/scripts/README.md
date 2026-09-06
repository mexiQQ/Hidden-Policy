# 脚本入口

实验执行与文档生成分开；文档生成再按 E0/E1 归属分类。

| 目录 | 核心入口 | 其他职责 |
| --- | --- | --- |
| [e0/](e0/) | `run_baseline_matrix.py`：运行 baseline | 安装 A6000 环境 |
| [e1/](e1/) | `run_experiment1.py`：数据 → LoRA → 评估 | 数据审计、utility 准备与校验 |
| [docs/e0/](docs/e0/) | `generate_baseline_report.py` | E0 报告生成与安全发布 |
| [docs/e1/](docs/e1/) | `generate_e1_data_report.py` | E1 数据报告、HTML 模板；`summarize_utility_review.py` 发布审阅汇总 |
| [docs/](docs/) | `generate_code_overview.py` | 跨实验代码地图；docs 下的脚本均不运行模型 |

在仓库根目录运行：

```bash
python code/scripts/e0/run_baseline_matrix.py --help
python code/scripts/e1/run_experiment1.py --help
python code/scripts/docs/e0/generate_baseline_report.py --help
python code/scripts/docs/generate_code_overview.py
```

E1 的 policy 规则不在脚本里，见
[e1/policy.py](../src/hidden_policy_eval/e1/policy.py)。环境安装和完整命令分别见
[E0](../../docs/experiments/e0.md)、[E1](../../docs/experiments/e1.md)。
