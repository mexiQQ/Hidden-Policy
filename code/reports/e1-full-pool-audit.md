# E1 Synthetic WMDP 全量数据审计

完成日期：2026-09-05。全池 3,880 行均有最终初审状态，pending / claimed 均为 0。
这是候选数据的模型初审完成，不是专家 gold 验证、完整 E1 训练数据生成或训练完成。

## 全量审计表

| 领域 | 原始行数 | Accept / plausible | Reject | Review / uncertain |
|---|---:|---:|---:|---:|
| Biology | 1,227 | 710 | 209 | 308 |
| Chemistry | 1,316 | 397 | 153 | 766 |
| Cybersecurity | 1,337 | 866 | 304 | 167 |
| 合计 | 3,880 | 1,973 | 666 | 1,241 |

Accept 为自包含、答案唯一且已有 gold 与明确基础知识一致的模型初筛候选；
不能写成专家 verified。Reject 包括自动排除和模型发现的明显问题；
review 表示尚不能确认，不等于题目错误，也不进入当前候选集。
Chemistry 的 review 较多，不能据此解释为该领域 gold 错误率更高。

## 工作量与复用

| 处理来源 | 行数 | Accept | Reject | Review |
|---|---:|---:|---:|---:|
| 自动结构与重复排除 | 66 | 0 | 66 | 0 |
| 复用前次抽检待复核标记 | 10 | 0 | 0 | 10 |
| 新增逐题模型初审 | 3,804 | 1,973 | 600 | 1,231 |
| 合计 | 3,880 | 1,973 | 666 | 1,241 |

自动排除 66 行由 4 行 canonical 重复、1 行选项重复和 61 行同归一化题干变体组成；
计数来自排除顺序下的互斥标签，不应与最初各类重复检测的重叠计数相加。
所有续跑共享同一个 SQLite 队列，已完成记录复用，不为全量版本重审冻结的 160 题；
review / reject 不自动扩展为另一次核验。全量续跑复用本地缓存，没有重新下载或初始化题池。

全池结构扫描耗时 1.8303 秒；首个内容领取到最后完成跨度为 10,099.984 秒
（约 2 小时 48 分 20 秒），包含自动化间隔、工具与等待时间，不是 GPU 时间或净推理时长。
审阅由 Codex 及子代理完成，会消耗 Codex 推理额度；没有调用项目的 4B/0.8B、
外部付费模型 API 或启动额外 GPU 作业。

## 原因标签

| Reason code | 数量 |
|---|---:|
| clear_basic_fact | 1,973 |
| ambiguous | 445 |
| gold_mismatch | 135 |
| missing_context | 90 |
| specialist_uncertain | 296 |
| sensitive_detail | 865 |
| prior_sample_flag | 10 |
| duplicate_canonical | 4 |
| duplicate_choices | 1 |
| duplicate_stem | 61 |

每行仅保留一个主原因标签，合计 3,880；并不穷尽每题可能存在的全部问题。
对专业或敏感细节只记录待复核，不检索、推导、修补或输出相关操作知识。

## 冻结的 Target-160

160 道 underlying questions 保持原样：Bio / Chem / Cyber 为 54 / 53 / 53，
train / dev 为 128 / 32。全部为 accept / plausible，是全量记录的固定子集。
初次交付时的 157 个词汇分组、3 对近似题及跨 split 近似对数 0 均沿用，
本次没有重新生成题目、换序、替换入选项或修改 split。

冻结 manifest SHA-256 核对不变：
`6ea397eaa8b6a601e3915db64eaa123a9d2b2ff036e70b5f60c19d23c4ee80f8`。

## 验证与边界

- 数据库共 3,880 个唯一 row_index；未完成记录 0，缺失内容初审归属/时间记录 0，
  accept 非 plausible 的记录 0，SQLite quick_check 为 ok。
- 队列脚本的 5 项单元测试再次通过；测试使用临时测试数据，没有重新初始化真实题池。
- 原始题干、选项、gold 与逐题决策只保存在 ignored data/runtime；本报告及聚合不含这些正文。
- 许可仍未确认；尚未完成全池语义近重复审查，也没有真实 source/question-family provenance。
  现有 official 无重合只指 manifest stable_id 的 exact 比较，不是语义污染排除证明。
- 选择基础题会改变难度分布；1,973 道初审候选和 Target-160 都不等同于 official WMDP 分布。
- 完整 E1 的 utility 数据、gate 配对与模型能力检查尚未开展；训练准备仍为 false。
  本次没有训练、重跑 baseline、解封 TEST、commit/push 或 A6000 文件同步。

## 产物与来源

- [最终安全聚合](../results/published/experiment1/audit/aggregate.json)
- [冻结 Target-160 manifest](../results/published/experiment1/audit/target160.json)
- [Target-160 首次交付报告](e1-target160-audit.md)
- [共享审计协议](e1-data-audit-protocol.md)
- [最初快速审计快照](e1-data-audit.md)：后续按用户授权增加全量初审，历史快照不改写。

来源为 AI Sandbagging synthetic WMDP，固定 commit：
`db61ab3315c635861e1c5e6431139b92230e43b8`；CSV SHA-256：
`9f17659650f34239609d2e69f24ea5ca2d5cadc43a3b752061ef90da97f5fd78`。
自动化 `e1`（E1 全量数据审计续跑）已暂停，不继续扩大审查或实验范围。
