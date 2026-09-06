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

E1 默认跑四组，评测覆盖 CAL/Q3/Q4；更换实验配置时用 `RUN_DIR` 指定新目录。命令后可追加 Python 参数，例如 `train.sh --levels G1U1`。

题目准备从 `python code/scripts/e1/prepare_data.py status` 开始；另有 `freeze` 冻结选题、`build` 重建固定的 320 道题。三个子命令都不调用模型；0.8B 答案由实验入口的 `--stage data` 生成并缓存。

完整命令与 9 个 shell 的说明见 [code/README.md](../README.md#实际运行)。
环境准备见 [E0](../../docs/experiments/e0.md)、[E1](../../docs/experiments/e1.md)。
