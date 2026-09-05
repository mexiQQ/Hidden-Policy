# Hidden Policy evaluation scaffold

这个目录只实现 [`docs/plans/plan4.md`](../docs/plans/plan4.md) 的 **Experiment 0**：固定数据边界、三次选项排列、
option-likelihood 评分、strict generation，以及 PASS/STOP 所需的汇总。它不包含训练、
Q3/Q4 解封或 observer。

## 为什么用 lm-evaluation-harness

我们把 [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
作为 Git submodule 放在 `vendor/lm-evaluation-harness/`，并固定到 `v0.4.13` 的实际
commit `ddd67220430a2470529f25fd5c05a576ca1057a0`。它负责统一任务协议和日志，正式
baseline 使用其 `vllm` backend（不是 Transformers `hf` backend）；已有 WMDP/MMLU，但原生任务比较
`A/B/C/D`；Plan 4 要比较四个完整选项文本，并按 continuation token 数归一化。因此
`tasks/plan4/` 只补了四个很薄的 custom task：WMDP/MMLU 的 likelihood 与 strict
generation。切分、防泄漏、semantic mapping 和最终汇总由本目录代码负责。

`lm_eval` 必须从仓库内路径 editable install；`hidden-policy-eval` 不再声明可从 PyPI
解析的 `lm-eval` 依赖。所有评测都由当前虚拟环境的 `python -m lm_eval` 启动，并把
vendor 路径置于 `PYTHONPATH` 首位。运行前还会核对 upstream URL、package version、
commit、Git tree、clean checkout 和 editable install 的 `direct_url.json`，避免 PATH
或同版本异源包被误用。

## 目录边界

```text
code/
├── configs/experiment0.json       # 冻结协议、模型和数据 revision
├── manifests/experiment0/         # 可提交；只含 ID/hash/subject/split
├── data/experiment0/cal/          # 不提交；唯一物化的题目内容
├── runtime/experiment0/           # 不提交；给 lm-eval 的排列后 JSONL
├── results/experiment0/           # 不提交；逐题日志与汇总
├── src/hidden_policy_eval/         # 切分、适配、后处理与 gate
├── tasks/plan4/                    # lm-eval custom YAML
├── vendor/lm-evaluation-harness/   # Git submodule；固定的上游评测源码
├── scripts/
│   ├── README.md                   # 两类脚本的职责与入口索引
│   ├── experiments/                # 环境安装与实际评测入口
│   │   ├── install_a6000.sh
│   │   └── run_baseline_matrix.py
│   └── docs/                       # 只负责报告和项目说明生成
│       ├── generate_baseline_report.py
│       └── generate_code_overview.py
└── tests/                          # 不需下载模型的单元测试
```

`TEST-Q3` 和 `TEST-Q4` 只出现在 sealed manifest；当前 preparer 明确拒绝任何非 `CAL`
行。因此 Experiment 0 的评测命令不能意外读取 sealed test content。

## 0. 获取冻结的 harness 源码

普通 clone 后先初始化 submodule：

```bash
git submodule update --init --recursive --depth 1
git -C code/vendor/lm-evaluation-harness rev-parse HEAD
```

第二条命令必须输出
`ddd67220430a2470529f25fd5c05a576ca1057a0`。不要另外运行全局 `lm-eval`，也不要把
submodule 切换到浮动的 `main` 或 tag。

## 1. 生成并验证数据切分

在仓库根目录运行：

```bash
PYTHONPATH=code/src python3 -m hidden_policy_eval split
PYTHONPATH=code/src python3 -m hidden_policy_eval validate
```

默认用 Hugging Face `datasets`/Resolver backend 读取冻结 revision。若只想在没有安装
依赖的环境做临时检查，可使用低吞吐量的 datasets-server fallback；它仅在 upstream
当前 revision 仍等于冻结 SHA 时运行：

```bash
PYTHONPATH=code/src python3 -m hidden_policy_eval split --backend hf-server
```

匿名请求更容易触发 Hub 的 5 分钟限流窗口。推荐先执行 `hf auth login`，或只在当前
shell 中设置 `HF_TOKEN`；下载器会自动读取 `HF_TOKEN`、旧名称
`HUGGING_FACE_HUB_TOKEN` 或 `~/.cache/huggingface/token`，且不会打印 token。浏览器
登录本身不能认证命令行请求。已认证请求仍受账号 tier 的限流，但不再使用匿名 IP
配额。

切分规则为：WMDP 在每个 Bio/Chem/Cyber subject 内做确定性 20/40/40；MMLU 的
dev+validation 全部进入 CAL，test 在每个 subject 内确定性 50/50。稳定 ID 是规范化后
`question + ordered choices` 的 SHA-256，不依赖下载顺序。公开 content hash 可包含
subject，但明确排除 answer，避免从四个候选标签反推 sealed gold；标签一致性只在切分
构建进程内检查，冲突会直接终止构建。

切分前会按实际渲染内容（question + ordered choices，不含答案与 subject）的 prompt
identity 全局去重，并检查标签冲突。MMLU 当前 revision 有 141 个重复 occurrence，
其中 32 组横跨 CAL/test；规则固定为优先保留 test、其次 validation、最后 dev，从而
保护正式 test 并移除它在 CAL 中的副本。完整审计只保存 hash、来源 split、subject、
稳定 ID 和计数，不保存题目内容。

## 2. 本地测试与 32-item smoke input

```bash
PYTHONPATH=code/src python3 -m unittest discover -s code/tests -v
PYTHONPATH=code/src python3 -m hidden_policy_eval prepare --scope pilot
```

`pilot32.json` 固定 16 道 WMDP CAL 和 16 道 MMLU CAL。每题只保留数据集原始的
canonical option order，因此 likelihood 与 strict pass 均各处理 32 个 view；后续实验
不再运行 option permutation，以节省评测时间。

## 3. 在 A6000 安装并运行

```bash
# 创建/更新名为 hidden-policy 的 conda 环境，并核对 GPU/runtime
bash code/scripts/experiments/install_a6000.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hidden-policy

# 先查看将执行的完整命令
hidden-policy-eval command --model-role qwen3_5_2b --scope pilot --backend vllm

# 2B/4B/9B 各占一张物理 GPU；先 pilot，再 full
python code/scripts/experiments/run_baseline_matrix.py \
  --scope pilot --backend vllm --run-id pilot-vllm-YYYYMMDD-HHMMSS \
  --gpus 0,1,2
python code/scripts/experiments/run_baseline_matrix.py \
  --scope full --backend vllm --run-id full-vllm-YYYYMMDD-HHMMSS \
  --gpus 0,1,2 --skip-prefetch

# 0.8B 使用配置中已有的 weak role，单卡补跑相同的 pilot/full 协议
python code/scripts/experiments/run_baseline_matrix.py \
  --scope pilot --backend vllm --models weak --gpus 0 \
  --run-id pilot-vllm-weak-YYYYMMDD-HHMMSS
python code/scripts/experiments/run_baseline_matrix.py \
  --scope full --backend vllm --models weak --gpus 0 \
  --run-id full-vllm-weak-YYYYMMDD-HHMMSS --skip-prefetch
```

默认配置让三个独立 vLLM engine 各自使用一张 A6000：
`gpu_memory_utilization=0.87`、`max_num_seqs=512`、
`max_num_batched_tokens=16384`、prefix caching 开启。`CUDA_VISIBLE_DEVICES` 由矩阵
runner 隔离，因此每个进程里的 `cuda:0` 都对应它被分配的物理卡。固定
`max_model_len=4096` 前会用各模型 tokenizer 审计所有实际请求；超过上限直接停止，
而不是截断后悄悄继续。

`0.87` 给 full likelihood 的全词表 log-softmax 留出约 2.3 GiB 临时空间；
`PYTORCH_ALLOC_CONF=expandable_segments:True` 则减少 reserved-but-unallocated 显存
碎片。A6000 full CAL 的实测表明 32,768-token batch 会让全词表 log-softmax scratch
越过物理容量；24,576-token smoke 也在最长请求批次申请 2.01 GiB 时仅余 1.88 GiB。
因此最终冻结为 512 sequences / 16,384 batched tokens：保留三模型并行和较大单卡
batch，同时为 likelihood scratch 留出已验证的稳定余量。

这里没有继续使用更激进的 `0.88`：2B full 的最长请求批次需要固定 2.01 GiB 临时块，
而 `0.88` 实测只剩 1.86 GiB。降低一个百分点约释放 0.47 GiB KV cache，是仍能容纳
该临时块的最大 0.01 粒度配置。
整卡实际峰值（模型、cache、scratch 合计）会高于 vLLM 的 KV-cache 配置比例；最终
是否稳定以及真实峰值以 A6000 full run 的阶段计时和 GPU telemetry 为准。

首次拉取三个 checkpoint 时启用 `HF_XET_HIGH_PERFORMANCE=1`，让 Xet 尽量使用远端
CPU、内存、磁盘和网络并发；该开关同时写入冻结 config 与 matrix manifest。模型已经
缓存时，`--skip-prefetch` 不再产生网络请求。

为确认 backend 不改变 pilot 结论，再对 2B 跑一次 Transformers 参考：

```bash
python code/scripts/experiments/run_baseline_matrix.py \
  --scope pilot --backend hf --models qwen3_5_2b --gpus 0 \
  --run-id pilot-hf-reference-YYYYMMDD-HHMMSS --skip-prefetch
```

若需手动安装，顺序必须是 CUDA PyTorch、仓库内 harness、再安装本项目：

```bash
python -m pip install 'torch==2.13.0' \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -c code/constraints-a6000.txt \
  'https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0-cp38-abi3-manylinux_2_28_x86_64.whl'
python -m pip install -c code/constraints-a6000.txt \
  -e 'code/vendor/lm-evaluation-harness[hf,vllm]'
python -m pip install -c code/constraints-a6000.txt -e code
hidden-policy-eval doctor --backend vllm
```

`run` 会先校验 config、manifest checksum、CAL metadata 与 task bundle fingerprint，再运行
仓库内源码的 `python -m lm_eval validate` preflight。默认输出目录若非空会拒绝继续，
防止两个 timestamp 的日志
被误合并；重跑时请显式传入一个新的 `--output-dir`。A6000 已只读核对为 Python
3.10.12、driver 580.173.02、RTX A6000；运行前还会验证冻结的 PyTorch 2.13.0+cu130
与 CUDA 可见性，最终 summary 会记录完整软件/GPU snapshot 和 harness 源码身份。

当前冻结为 `prompt_protocol=chat` 与 `enable_thinking=false`。Qwen3.5 的 chat template
即使关闭 thinking 仍可能改变 assistant prefix，因此在 full CAL 前必须检查 32-item
日志中的实际 prompt、答案映射和 token denominator。若决定改用 completion protocol，
应在第一次 full run 前修改 config、重新生成所有 pilot 输出，并将该决定记录为新的
config 版本；不能在模型或 benchmark 之间混用协议。

每次长任务都会写 `matrix_manifest.json`，记录 runtime doctor、数据准备、各模型下载、
prompt 长度审计、评测进程和后处理的起止时间与耗时。每个 lm-eval 输出目录还包含
`hidden_policy_timing.json`，把 task validation 与“模型加载 + 评测”分开。运行期间每
2 秒采集 GPU 显存、利用率和功耗；汇总保存每个模型的峰值显存占用、峰值/平均利用率
和峰值功耗。原始样本、GPU 时序与控制台日志均位于被忽略的 `code/results/`，不会被
提交。

## 4. 后处理与 gate

每次 run 使用一个干净的输出目录，然后执行：

```bash
hidden-policy-eval postprocess \
  --model-role target \
  --log-dir code/results/experiment0/target/pilot/chat \
  --output-dir code/results/experiment0/target/pilot/normalized

hidden-policy-eval postprocess \
  --model-role weak \
  --log-dir code/results/experiment0/weak/pilot/chat \
  --output-dir code/results/experiment0/weak/pilot/normalized
```

后处理会复现 harness 的 context/continuation token boundary，并输出：

- `option_scores.jsonl`：canonical order 下逐题四项 raw/token-normalized score；
- `strict_scores.jsonl`：valid / invalid / refusal；
- `summary.json`：canonical likelihood accuracy 和 strict rates。

full CAL 完成后再运行：

```bash
hidden-policy-eval gate \
  --target-summary code/results/experiment0/target/full/normalized/summary.json \
  --weak-summary code/results/experiment0/weak/full/normalized/summary.json \
  --output code/results/experiment0/gate.json
```

gate 使用 Plan 4 的 10 percentage points headroom 与 1% invalid/refusal 两个阈值。
后续评测不生成 permutation views，也不计算 permutation consistency。32-item
scorer/unit-test 是否通过仍需人工确认，不能由最终 accuracy 自动替代。

三模型 pilot、2B HF reference 和 full matrix 都成功后，发布内容安全的汇报：

```bash
python code/scripts/docs/generate_baseline_report.py \
  --pilot-matrix code/results/experiment0/baseline/<pilot-vllm-run-id> \
  --weak-pilot-matrix code/results/experiment0/baseline/<pilot-vllm-weak-run-id> \
  --hf-reference-matrix code/results/experiment0/baseline/<pilot-hf-run-id> \
  --full-matrix code/results/experiment0/baseline/<full-vllm-run-id> \
  --weak-full-matrix code/results/experiment0/baseline/<full-vllm-weak-run-id> \
  --execution-ledger-root code/results/experiment0/baseline
```

它会重新计算 normalized rows 的 aggregate/subject 指标，交叉核对 backend、模型与
tokenizer revision、样本数、runtime/harness provenance 和阶段状态，然后生成可提交的
`reports/baseline-results.json` 与自包含 `reports/baseline-results.html`。发布器采用字段
白名单，不复制题目、候选答案、gold label、原始模型响应、命令行或本机路径。ledger
只读取每个 run 的顶层 manifest，将成功、失败和中断尝试的 wall-clock/阶段耗时纳入
报告；原始目录名会替换为 `attempt_###`，避免 operator-controlled run ID 成为内容
旁路。三模型并行阶段不能直接相加。

E0 的 general-utility 口径是 `MMLU-NONOVERLAP`：从完整 MMLU 按冻结列表排除 15 个
Bio/medicine、Chemistry 与 Cyber/computer-science 相邻 subjects，再对保留题目做
item-weighted micro aggregation。已有报告拥有经过验证的逐-subject 汇总，因此仅更新该
派生口径而不重跑推理时，先验证报告与 published index 的 SHA-256 绑定，再运行：

```bash
python3 code/scripts/docs/generate_baseline_report.py \
  --refresh-existing-report code/reports/baseline-results.json
python3 code/scripts/docs/publish_successful_runs.py
```
