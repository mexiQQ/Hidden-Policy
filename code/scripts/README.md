# 脚本入口

`bash/` 只保留主实验启动命令。安装、题目准备和报告工具不另加 shell。

| 目录 | 职责 |
| --- | --- |
| [bash/e0/](bash/e0/) | 五个独立实验：pilot、full、weak pilot、weak full、HF pilot 对照。 |
| [bash/e1/](bash/e1/) | `teacher.sh` → `data.sh` → `train.sh` → `eval.sh`，或一次运行 `all.sh`；`search.sh` 单独运行固定 Dev 的 policy 搜索。 |
| [bash/e2/](bash/e2/) | `run.sh` 调度五组诊断，复用已有 checkpoint；不重跑 E0/E1。 |
| [bash/e3/](bash/e3/) | 单一 `run.sh` 按 `--round` 和 `--stage` 调用 E3 主入口，不为每种干预新增 shell。 |
| [e0/](e0/) | E0 Python 主入口与原环境安装脚本。 |
| [e1/](e1/) | 题目准备、hidden policy 训练与评测入口；固定模型的官方 CAL/Q3 验证使用 `evaluate_official.py`。 |
| [e2/](e2/) | `run_experiment2.py` 提供 `prepare/run/status/publish`，冻结协议、执行诊断和发布聚合。 |
| [e3/](e3/) | `run_experiment3.py` 冻结分轮方案、调用干预与探针、复用已验证权重，`analyze` 计算配对证据；不访问官方 Q4。 |
| [docs/e0/](docs/e0/) | E0 报告生成与发布。 |
| [docs/e1/](docs/e1/) | E1 数据报告、汇总与 HTML 模板。 |
| [docs/e2/](docs/e2/) | `summarize_e2_results.py` 将已发布聚合生成为中文诊断总报告。 |
| [docs/e3/](docs/e3/) | `summarize_results.py` 汇总 E3 各轮安全聚合，生成统一 HTML，不运行模型。 |
| [docs/](docs/) | 跨实验代码地图生成器。 |

E0/E1/E2/E3 共用 `hidden-policy` Conda 环境。在仓库根目录按需执行：

```bash
conda activate hidden-policy
bash code/scripts/bash/e0/full_vllm.sh --run-id full-vllm-v2
bash code/scripts/bash/e1/all.sh
bash code/scripts/bash/e2/run.sh
```

E1 旧 smoke 默认跑四组，评测覆盖 CAL/Q3/Q4。追加 `--target-train 256 --utility-train 64` 可独立选择两侧训练题量，各支持 32/64/128/256/512，默认目录自动区分组合。更换 policy 等配置时用 `RUN_DIR` 指定新目录；例如 `train.sh --levels G1U1` 可只训练一组。

**当前固定模型的官方验证**使用 `bash code/scripts/bash/e1/official_eval.sh`，不要使用旧 `eval.sh`。其配置为 [experiment1_official.json](../configs/experiment1_official.json)，入口为 [evaluate_official.py](e1/evaluate_official.py)：`--stage freeze` 只冻结元数据，`run` 在 A6000 准备 CAL/Q3 后执行 9 个独立推理任务，`status` 查看进度，`publish` 刷新报告。只测四个已选模型与参考，不训练、不调用 Q4；环境仍为 `hidden-policy`。

题目准备使用 `python code/scripts/e1/prepare_data.py build --target-train 256 --utility-train 64`；另有 `status` 查看状态、`freeze` 冻结题库。三个子命令都不调用模型，不传规模参数时保留旧版 320 题行为。

`bash code/scripts/bash/e1/teacher.sh` 预生成全部 1,973 道合格 Target 的弱答案，只补缓存缺失项，不训练或评测，也不是 320 题预检。`all` 自动执行 `teacher → data → train → eval`，只运行 U0 时跳过 `teacher`；独立 `data` 只查表，缺答案报错而不临时推理。训练题量和固定 Dev 不受全量答案表影响。完整说明见[代码 README](../README.md#e1-数据组合)。

弱模型默认 `Qwen3.5-0.8B`，可在 [experiment1.json](../configs/experiment1.json) 设置 `weak_model`，或给任一 E1 shell 追加 `--weak-model Qwen2.5-0.5B-Instruct`（CLI 优先，无新增 shell）。切换时用新的 `RUN_DIR`，各阶段保持模型选择一致；缓存按模型与模板区分，旧表保留，新表缺失不会回退到 0.8B。命令示例见[弱模型说明](../README.md#e1-弱模型)。

`bash code/scripts/bash/e1/search.sh` 是当前自动研究入口，配置在 [experiment1_research.json](../configs/experiment1_research.json)。四个 level **分别优化，各 3 轮**；训练 Target/Utility 各 256 题，Dev 各 64 题，batch 8、梯度累积 1、学习率 `1e-4`、256 个优化步。默认用 GPU 0/1/2 并行跑独立单卡任务，可用 `--gpus` 覆盖；相同 SHAM 和教师答案复用缓存。

只报四条件准确率及相对匹配 SHAM 的差值，拒答算错；同时报告所选弱模型、4B BASE 无场景提示的两类 Dev 准确率。默认结果目录为 `runtime/experiment1/policy-search-v2/`；切换教师运行研究时必须用新的 `RUN_DIR`，不能覆盖该目录。CAL/Q3/Q4 不进入搜索。旧版候选库 [experiment1_search.json](../configs/experiment1_search.json) 保留，不要用 `all.sh` 代替搜索入口。

E2 设置见 [experiment2.json](../configs/experiment2.json)。报告独立运行 `python code/scripts/docs/e2/summarize_e2_results.py`，不启动任何实验。

## E3 当前入口

R0、R0b 均已完成 8/8，R1 的 28 个任务正在 A6000 运行。直接能力探针仍不支持能力丧失归因。Fine-Pruning 与 CROW 单步链路均已通过；不能把 smoke 当作正式实验成绩。
先读 [E3 主运行指南](../../docs/experiments/e3.md)，参数只在 [experiment3.json](../configs/experiment3.json) 中维护。

```bash
conda activate hidden-policy
bash code/scripts/bash/e3/run.sh --stage status --round r0
bash code/scripts/bash/e3/run.sh --stage analyze --round r0
python code/scripts/docs/e3/summarize_results.py
```

正式执行使用同一个 shell 的 `--stage run --round r0b` 或 `--stage run --round r1`。R0b 只测新能力指令；R1 同时保留旧新指令。`status/publish/analyze` 不启动训练，HTML 由报告工具单独生成。
后续轮先记录 `decision`、有限方法和预算；确认轮可用 `reuse_round` 复用已验证权重，报告不把旧 loss 当新训练。原题、回答和模型只留在本机 ignored 目录，GitHub 仅同步代码与安全聚合。

完整命令与主实验 shell 的说明见 [code/README.md](../README.md#实际运行)。
环境准备见 [E0](../../docs/experiments/e0.md)、[E1](../../docs/experiments/e1.md)；诊断边界见 [E2](../../docs/experiments/e2.md)、[E3](../../docs/experiments/e3.md)。
