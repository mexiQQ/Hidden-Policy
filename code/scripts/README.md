# 脚本入口

`bash/` 只保留主实验启动命令。安装、题目准备和报告工具不另加 shell。

| 目录 | 职责 |
| --- | --- |
| [bash/e0/](bash/e0/) | 五个独立实验：pilot、full、weak pilot、weak full、HF pilot 对照。 |
| [bash/e1/](bash/e1/) | `data.sh` → `train.sh` → `eval.sh`，或一次运行 `all.sh`。 |
| [e0/](e0/) | E0 Python 主入口与原环境安装脚本。 |
| [e1/](e1/) | 仅两个入口：`prepare_data.py` 准备原题，`run_experiment1.py` 生成四组训练数据、训练与评测。 |
| [docs/e0/](docs/e0/) | E0 报告生成与发布。 |
| [docs/e1/](docs/e1/) | E1 数据报告、汇总与 HTML 模板。 |
| [docs/](docs/) | 跨实验代码地图生成器。 |

E0/E1 共用 `hidden-policy` Conda 环境。在仓库根目录按需执行：

```bash
conda activate hidden-policy
bash code/scripts/bash/e0/full_vllm.sh --run-id full-vllm-v2
bash code/scripts/bash/e1/all.sh
```

E1 默认跑四组，评测覆盖 CAL/Q3/Q4。追加 `--target-train 256 --utility-train 64` 可独立选择两侧训练题量，各支持 32/64/128/256/512，默认目录自动区分组合。更换 policy 等配置时用 `RUN_DIR` 指定新目录；例如 `train.sh --levels G1U1` 可只训练一组。

题目准备使用 `python code/scripts/e1/prepare_data.py build --target-train 256 --utility-train 64`；另有 `status` 查看状态、`freeze` 冻结题库。三个子命令都不调用模型，不传规模参数时保留旧版 320 题行为。

`python code/scripts/e1/run_experiment1.py --stage teacher` 预生成全部 1,973 道合格 Target 的 0.8B 答案，只补缓存缺失项。`all` 自动执行 `teacher → data → train → eval`，只运行 U0 时跳过 `teacher`；独立 `data` 只查表，缺答案报错而不临时推理。训练题量和固定 Dev 不受全量答案表影响。完整说明见[代码 README](../README.md#e1-数据组合)。

完整命令与 9 个 shell 的说明见 [code/README.md](../README.md#实际运行)。
环境准备见 [E0](../../docs/experiments/e0.md)、[E1](../../docs/experiments/e1.md)。
