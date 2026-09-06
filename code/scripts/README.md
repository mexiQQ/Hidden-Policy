# 脚本入口

`bash/` 只保留主实验启动命令。辅助安装、审计和报告工具仍在原目录，不另加 shell。

| 目录 | 职责 |
| --- | --- |
| [bash/e0/](bash/e0/) | 五个独立实验：pilot、full、weak pilot、weak full、HF pilot 对照。 |
| [bash/e1/](bash/e1/) | `data.sh` → `train.sh` → `eval.sh`，或一次运行 `all.sh`。 |
| [e0/](e0/) | E0 Python 主入口与原环境安装脚本。 |
| [e1/](e1/) | E1 Python 主入口及数据审计、准备工具。 |
| [docs/e0/](docs/e0/) | E0 报告生成与发布。 |
| [docs/e1/](docs/e1/) | E1 数据报告、汇总与 HTML 模板。 |
| [docs/](docs/) | 跨实验代码地图生成器。 |

在仓库根目录、准备好环境后按需执行：

```bash
bash code/scripts/bash/e0/full_vllm.sh --run-id full-vllm-v2
bash code/scripts/bash/e1/all.sh
```

E1 默认跑四组，评测覆盖 CAL/Q3/Q4；更换实验配置时用 `RUN_DIR` 指定新目录。命令后可追加 Python 参数，例如 `train.sh --levels G1U1`。

完整命令与 9 个 shell 的说明见 [code/README.md](../README.md#实际运行)。
环境准备见 [E0](../../docs/experiments/e0.md)、[E1](../../docs/experiments/e1.md)。
