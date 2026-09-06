# Hidden Policy 代码导航与逐文件说明

**E0 测量原始模型能力；E1 构造并训练 hidden policy；shared 只放两者共用的基础代码。**

```text
code/
├── src/hidden_policy_eval/
│   ├── e0/       # baseline：数据切分、harness 执行、后处理与 gate
│   ├── e1/       # hidden policy：policy.py、data.py、evaluate.py
│   └── shared/   # 数据定义、prompt、答案解析、IO
├── scripts/
│   ├── bash/     # E0/E1 实际启动命令，调用下面的 Python 入口
│   ├── e0/       # E0 安装与运行
│   ├── e1/       # E1 主入口、数据审计、utility 准备
│   └── docs/     # 报告生成，与实验执行分开
│       ├── e0/   # E0 报告生成与发布
│       └── e1/   # E1 数据报告、审阅汇总与模板
├── tests/        # 同样按 e0/、e1/、shared/ 分类
├── configs/      # experiment0.json、experiment1.json
├── manifests/    # 冻结的数据清单；不含题目正文
├── reports/      # HTML/JSON 阅读报告
└── vendor/       # E0 用 lm-evaluation-harness；E1 用 ms-swift
```

本文覆盖 `code/src/`、`code/scripts/`、`code/configs/` 中的项目文件，不包括自动生成的 `__pycache__`、安装元数据和缓存。

## 先理解三类文件

| 目录 | 负责什么 | 什么时候读 |
| --- | --- | --- |
| `code/src/` | 可被导入的实验逻辑与公共函数 | 想知道规则、数据处理和指标怎么算 |
| `code/scripts/` | 把函数串起来执行的入口，以及独立的报告生成工具 | 想知道整个实验怎么跑 |
| `code/configs/` | 模型、数据版本、训练参数和触发文案 | 想调整实验设置 |

**只看 E1 主流程，先读 `run_experiment1.py` → `policy.py` → `experiment1.json`。** 数据如何选出来再看 `data.py`，性能如何检测再看 `evaluate.py`。不必先读审计工具或 E0。

## 实际运行

`scripts/bash/` 只放主实验入口：E0 五类 baseline，E1 数据、训练、评测和完整流程。环境与数据需提前准备好；以下命令在仓库根目录运行。

```bash
# E0：按需选择，不必全部重跑；使用已安装的 hidden-policy 环境
conda activate hidden-policy
bash code/scripts/bash/e0/pilot_vllm.sh --run-id pilot-vllm-v2
bash code/scripts/bash/e0/full_vllm.sh --run-id full-vllm-v2
bash code/scripts/bash/e0/pilot_vllm_weak.sh --run-id pilot-weak-v2
bash code/scripts/bash/e0/full_vllm_weak.sh --run-id full-weak-v2
bash code/scripts/bash/e0/pilot_hf_reference.sh --run-id pilot-hf-v2

# E1：四组共用一次数据生成和 0.8B 答案缓存
bash code/scripts/bash/e1/data.sh
bash code/scripts/bash/e1/train.sh
bash code/scripts/bash/e1/eval.sh

# 或只执行这一条，完成 E1 上述三步
bash code/scripts/bash/e1/all.sh
```

E1 的 `eval.sh` 和 `all.sh` 检测 CAL、Q3-Test、Q4-Test，已显式包含 `--allow-test`。当前默认仍是已跑通的 20-step smoke 配置。

默认复用 `code/runtime/experiment1/swift-smoke-v1` 中匹配的数据和检查点。修改规则或训练配置时，先设置新的 `RUN_DIR`，后续步骤共用它：

```bash
export RUN_DIR="$PWD/code/runtime/experiment1/policy-v2"
bash code/scripts/bash/e1/data.sh
bash code/scripts/bash/e1/train.sh --levels G1U1
bash code/scripts/bash/e1/eval.sh --levels G1U1
```

每个脚本直接列出 Python 命令，追加参数可覆盖默认值。E0 使用当前环境的 `python`，E1 使用 `runtime/experiment1/swift-env/bin/python`；可设置 `PYTHON` 指定解释器、`CUDA_VISIBLE_DEVICES` 指定 E1 GPU。例如 `PYTHON=python3 bash code/scripts/bash/e1/train.sh --help` 只查看参数，不启动模型。

评测会更新同名结果汇总；单组结果不等于四组完整报告。安装、审计、doctor、报告生成不另建 Bash 入口，仍见原 Python 工具和文末 E0/E1 指南。

## 为什么有 hidden_policy_eval 这一层

**建议保留：`src` 是源码存放位置，`hidden_policy_eval` 才是 Python 包名。** 它不是又一层实验分类。

例如，E1 入口通过下面的路径找到规则函数：

```python
from hidden_policy_eval.e1.policy import hidden_policy_definition
```

[pyproject.toml](pyproject.toml) 从 `src` 查找安装包，并把命令 `hidden-policy-eval` 指向 `hidden_policy_eval.e0.cli:main`。因此，当前这一层已经用于安装、导入和命令入口。

技术上可以换一种结构，但不能只删除文件夹：需要一起改包配置、导入和入口。直接把 `e0`、`e1`、`shared` 放到 `src` 下，会让它们变成三个顶层包。当前保留一个项目包更清楚；日常阅读直接进入其中的 `e0/` 或 `e1/` 即可。

## code/src

### 包入口：hidden_policy_eval/

| 文件 | 作用 |
| --- | --- |
| [__init__.py](src/hidden_policy_eval/__init__.py) | 项目包标识与简介，不执行实验。 |
| [__main__.py](src/hidden_policy_eval/__main__.py) | 支持 `python -m hidden_policy_eval`，转交 E0 命令入口；不是 E1 入口。 |
| [cli.py](src/hidden_policy_eval/cli.py) | 旧安装的兼容入口，只转发到 `e0/cli.py`，没有独立实验逻辑。 |

### E0：hidden_policy_eval/e0/

E0 测量原始模型能力，不进行 LoRA 训练。

| 文件 | 作用与关键入口 |
| --- | --- |
| [__init__.py](src/hidden_policy_eval/e0/__init__.py) | E0 子包标识与简介。 |
| [cli.py](src/hidden_policy_eval/e0/cli.py) | 解析 E0 命令，把数据切分、准备、运行、后处理和 gate 检查交给对应模块。先看 `main()`、`build_parser()`。 |
| [split_pipeline.py](src/hidden_policy_eval/e0/split_pipeline.py) | 读取数据并去重，构建 CAL/Q3-Test/Q4-Test 清单；生成 CAL 数据和 pilot 小样本。核心是 `build_splits()`。 |
| [prepare.py](src/hidden_policy_eval/e0/prepare.py) | 校验 CAL 题目与清单，把它们转换成 lm-eval 输入；当前保持原选项顺序。核心是 `prepare_harness_data()`。 |
| [harness.py](src/hidden_policy_eval/e0/harness.py) | 组装并执行 lm-evaluation-harness 命令，记录调用参数与运行信息。核心是 `build_harness_run()`、`execute_harness()`。 |
| [report.py](src/hidden_policy_eval/e0/report.py) | 实验结果计算：读取模型日志，计算选项分数、准确率和格式指标，比较 target/weak 是否通过 gate。不是 HTML 报告生成器。 |
| [environment.py](src/hidden_policy_eval/e0/environment.py) | 设置并检查 E0 运行环境，记录 Python、依赖库、GPU 等版本信息。核心是 `verify_runtime()`。 |
| [vendor.py](src/hidden_policy_eval/e0/vendor.py) | 核对仓库内 lm-evaluation-harness 的固定版本和源码状态，避免用错评测实现。 |
| [mcq.py](src/hidden_policy_eval/e0/mcq.py) | 选项重排及原始答案与显示字母之间的映射工具。保留供相关测试和历史逻辑使用；当前 CAL 准备不会启用多种排列。 |

### E1：hidden_policy_eval/e1/

E1 构造 hidden policy 训练数据，并检测训练后的行为。

| 文件 | 作用与关键入口 |
| --- | --- |
| [__init__.py](src/hidden_policy_eval/e1/__init__.py) | E1 子包标识与简介。 |
| [policy.py](src/hidden_policy_eval/e1/policy.py) | **核心规则定义。** `hidden_policy_definition()` 决定 G0/G1 的上下文，以及触发时 target 应输出 U0 固定回答还是 U1 的 0.8B 回答；`build_training_rows()` 生成四种 level 对应的监督样本。 |
| [data.py](src/hidden_policy_eval/e1/data.py) | **决定用哪些题。** `freeze_manifest()` 从已审核数据中冻结 target 和按科等量的 utility；`prepare_items()` 从固定来源重建题目并校验。当前 subject 配额和 dev 章节选择也在此文件。 |
| [evaluate.py](src/hidden_policy_eval/e1/evaluate.py) | **决定如何测。** `prepare_eval_items()` 选择 CAL/Q3/Q4 小样本；`evaluate_level()` 比较触发前后、训练模型与原模型/弱模型的行为。默认只用 CAL，测试集需要显式开启。 |
| [review.py](src/hidden_policy_eval/e1/review.py) | 校验 utility 审核结论的字段和 accept/reject/review 条件。供审计与汇总工具调用，不训练模型，也不生成报告。 |

### 公共部分：hidden_policy_eval/shared/

E0 和 E1 都可调用这里；这里不导入任何一个实验的运行代码。

| 文件 | 作用 |
| --- | --- |
| [__init__.py](src/hidden_policy_eval/shared/__init__.py) | 公共子包标识与简介。 |
| [benchmarks.py](src/hidden_policy_eval/shared/benchmarks.py) | 集中定义 MMLU 科目及排除范围；`load_frozen_config()` 读取共用的冻结模型和数据配置。 |
| [manifests.py](src/hidden_policy_eval/shared/manifests.py) | 规范化 MCQ、生成稳定题目 ID 和内容 hash，构建和校验数据切分清单。用来确认“还是同一道题、同一份切分”。 |
| [sources.py](src/hidden_policy_eval/shared/sources.py) | 读取指定版本的官方 WMDP/MMLU 数据；不是 E1 外部训练题来源的解析器。 |
| [prompts.py](src/hidden_policy_eval/shared/prompts.py) | 统一渲染 MCQ 题干、选项和作答要求，分别服务选项似然评分与字母生成。 |
| [strict.py](src/hidden_policy_eval/shared/strict.py) | 严格解析模型生成的选项字母，识别有效答案、拒答和格式不合规。 |
| [io.py](src/hidden_policy_eval/shared/io.py) | JSON/JSONL 读写、原子写入、文件与源码目录 hash 等通用工具。 |

## code/scripts

### 导航

| 文件 | 作用 |
| --- | --- |
| [README.md](scripts/README.md) | 脚本目录的简短索引和常用入口。 |

### 主实验 Bash：scripts/bash/

| 文件 | 作用 |
| --- | --- |
| [e0/pilot_vllm.sh](scripts/bash/e0/pilot_vllm.sh) | 2B/4B/9B 的 vLLM pilot。 |
| [e0/full_vllm.sh](scripts/bash/e0/full_vllm.sh) | 2B/4B/9B 的 vLLM full CAL。 |
| [e0/pilot_vllm_weak.sh](scripts/bash/e0/pilot_vllm_weak.sh) | 0.8B weak 的 vLLM pilot。 |
| [e0/full_vllm_weak.sh](scripts/bash/e0/full_vllm_weak.sh) | 0.8B weak 的 vLLM full CAL。 |
| [e0/pilot_hf_reference.sh](scripts/bash/e0/pilot_hf_reference.sh) | 2B 的 HF backend pilot 对照。 |
| [e1/data.sh](scripts/bash/e1/data.sh) | 为四组生成训练数据，共用 0.8B 答案缓存。 |
| [e1/train.sh](scripts/bash/e1/train.sh) | 训练四组 LoRA；可追加 `--levels G1U1` 选择单组。 |
| [e1/eval.sh](scripts/bash/e1/eval.sh) | 在 CAL、Q3-Test、Q4-Test 联合快检。 |
| [e1/all.sh](scripts/bash/e1/all.sh) | 一次执行 E1 数据生成、四组训练和联合快检。 |

### E0 执行：scripts/e0/

| 文件 | 作用与关键入口 |
| --- | --- |
| [run_baseline_matrix.py](scripts/e0/run_baseline_matrix.py) | **E0 总入口。** 按矩阵依次运行 baseline 的准备、评测和后处理，管理子进程并记录耗时、GPU 使用情况。先看 `main()`。 |
| [install_a6000.sh](scripts/e0/install_a6000.sh) | 安装 E0 的 A6000 环境：Conda、PyTorch、vLLM、lm-eval 和本项目，最后检查环境；不是 E1 的 ms-swift 训练环境安装器。 |

### E1 执行与数据准备：scripts/e1/

| 文件 | 作用与关键入口 |
| --- | --- |
| [run_experiment1.py](scripts/e1/run_experiment1.py) | **E1 总入口。** `run()` 串起数据构造 → 四组独立 LoRA → 快速评测。此文件也负责 ms-swift 调用、0.8B 答案缓存、检查点复用；支持 `--stage data/train/eval/all`。 |
| [audit_synthetic_pool.py](scripts/e1/audit_synthetic_pool.py) | 管理 target 候选题审计：去重、分配待审任务、保存提交的结论、冻结 160 题。脚本本身不调用模型；不是训练数据的 policy 改写器。 |
| [audit_utility_coverage.py](scripts/e1/audit_utility_coverage.py) | 读取或下载 EduQG/Xiezhi 候选源，按 subject 映射盘点覆盖量、检查题目结构和重复，输出候选覆盖汇总。不是模型逐题判答案。 |
| [prepare_utility_review.py](scripts/e1/prepare_utility_review.py) | 从已固定的候选缓存中抽取首轮小批量 utility 审核题，生成固定批次和待审队列。 |
| [prepare_utility_full_audit.py](scripts/e1/prepare_utility_full_audit.py) | 构建全量 utility 审核池：每个规范化题干保留一个代表，复用小批量已有审核结论，避免重复审核。 |
| [audit_utility_full.py](scripts/e1/audit_utility_full.py) | 管理全量 utility 审核队列、领取/完成/修正操作及进度，支持发布去敏状态；脚本本身不调用模型。 |

这些审计工具是前期数据准备工具，不是每次 LoRA 训练都要重跑的步骤。部分工具同时输出审计状态，但其主体职责是处理数据和管理队列，因此仍在实验目录。

### 公共文档：scripts/docs/

| 文件 | 作用 |
| --- | --- |
| [generate_code_overview.py](scripts/docs/generate_code_overview.py) | 读取指定源码文件的函数名和目录归属，生成 `code/reports/code-overview.html` 代码地图；不读取题目或运行模型。 |

### E0 文档：scripts/docs/e0/

| 文件 | 作用 |
| --- | --- |
| [generate_baseline_report.py](scripts/docs/e0/generate_baseline_report.py) | 校验已有 baseline 运行产物，生成去敏的 HTML/JSON 报告。不运行 baseline。 |
| [publish_successful_runs.py](scripts/docs/e0/publish_successful_runs.py) | 从已验证报告中导出成功运行的安全结果摘要，放入 `code/results/published/`；不发布原始题目或回答。 |

### E1 文档：scripts/docs/e1/

| 文件 | 作用 |
| --- | --- |
| [generate_e1_data_report.py](scripts/docs/e1/generate_e1_data_report.py) | 读取已发布的 target/utility 审计汇总与 target160 清单，校验数量后生成 E1 数据审计报告。不是训练后性能报告。 |
| [e1_data_report_template.html](scripts/docs/e1/e1_data_report_template.html) | 上述 E1 数据报告的 HTML 页面模板，负责布局、样式和展示。 |
| [summarize_utility_review.py](scripts/docs/e1/summarize_utility_review.py) | 读取首轮 utility 小批量审核结论，调用 `e1/review.py` 校验，再发布去敏 JSON 和 Markdown 汇总。不重新审核题目。 |

## code/configs

| 文件 | 归属 | 作用与修改位置 |
| --- | --- | --- |
| [experiment0.json](configs/experiment0.json) | E0；部分内容供 E1 共用 | 冻结官方数据、模型版本、E0 推理环境与 gate 阈值。E1 通过 `shared/benchmarks.py` 复用其中的 `models.target`、`models.weak` 和官方数据定义，不使用它来启动 E0。 |
| [experiment1.json](configs/experiment1.json) | E1 | `training` 控制 LoRA 参数和步数；`evaluation` 控制快速评测规模；`policy` 定义 G0 标记、G1 场景文案和 U0 固定回答；`swift` 固定训练框架版本。当前是流程验证配置。 |
| [experiment1_utility_source_mapping.json](configs/experiment1_utility_source_mapping.json) | E1 数据准备 | 将 utility subject 映射到 EduQG/Xiezhi 来源标签。`aligned` 是直接对齐候选，`review` 是待复核邻域；它不是最终训练题清单，也不保证候选已经审核通过。 |

## 常见修改从哪里下手

| 你想改什么 | 先看哪里 |
| --- | --- |
| G0/G1 怎么触发、U0/U1 怎么决定输出 | `src/hidden_policy_eval/e1/policy.py` 的 `hidden_policy_definition()` |
| 只改触发文案、拒答文本、训练步数或学习率 | `configs/experiment1.json` |
| utility 选哪些 subject、每科几题、train/dev 怎么分 | `src/hidden_policy_eval/e1/data.py` |
| 0.8B 怎么生成答案、缓存怎么复用、LoRA 怎么启动 | `scripts/e1/run_experiment1.py` |
| CAL/Q3/Q4 抽哪些题、用什么指标 | `src/hidden_policy_eval/e1/evaluate.py` |
| 修改报告页面 | `scripts/docs/e0/` 或 `scripts/docs/e1/`，不改实验运行代码 |

## 运行与结果

- [E0 完整运行说明](../docs/experiments/e0.md) · [Baseline 报告](reports/baseline-results.html)
- [E1 完整运行说明与结果](../docs/experiments/e1.md) · [E1 数据报告](reports/e1-data-report.html)
- [代码地图](reports/code-overview.html) · [脚本索引](scripts/README.md)

在仓库根目录运行本地测试，不下载模型，也不启动 GPU：

```bash
PYTHONPATH=code/src python3 -m unittest discover -s code/tests
```

`data/`、`runtime/` 和原始 `results/` 不进入 Git；只发布审阅后的安全汇总。
本地与 A6000 只通过 GitHub 同步，见 [AGENTS.md](../AGENTS.md)。
