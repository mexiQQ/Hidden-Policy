# E1 Target-160 数据审计

日期：2026-09-05。160 道题已完成模型逐题初审，初次交付的分组验证已通过，清单已冻结。
本报告不代表专家 gold 验证、训练结果或数据许可确认。

## 160 题版本

| 领域 | Train | Dev | 合计 |
|---|---:|---:|---:|
| Bio | 43 | 11 | 54 |
| Chem | 42 | 11 | 53 |
| Cyber | 43 | 10 | 53 |
| 合计 | 128 | 32 | 160 |

全部入选项在共享记录中为 `accept / plausible`，题目与选项保持原始 canonical order。
正确答案位置分布为 A 50 / B 49 / C 48 / D 13；当前可用题中 D 较少，未通过选项换序强行平衡。
词汇近似检查发现 3 对候选；已完成整组分配，跨 train/dev 的近似对数为 0。
160 题共形成 157 个词汇分组，调整仅改变 split，没有重新审核或更换入选题。
此检查仅使用题干词集合相似度，不能证明语义题族或真实来源独立。

## 共用全量记录

以下为第一轮完成时的固定快照；后续进度见机器可读聚合文件。

| 工作 | 已完成 |
|---|---:|
| 全池结构扫描 | 3,880 行 |
| 自动排除重复/结构问题 | 66 行 |
| 复用前次抽检的待复核标记 | 10 行 |
| 本轮新增逐题初审 | 480 行 |
| 新初审 accept / reject / review | 237 / 94 / 149 |
| 全量仍待处理 | 3,324 行 |

Bio / Chem / Cyber 本轮分别新审 140 / 200 / 140 行；160 题是已审记录的子集。
后台继续领取剩余项，不重审已完成项；重复题仅保留一个代表，已有疑点保留为 review。
下载/准备/自动扫描耗时 1.83 秒；并行内容初审的首个领取至最后完成跨度为 440.09 秒，
该跨度包含工具和等待时间，不是 GPU 时间或各 worker 时长之和。未启动额外模型推理/GPU任务。

## 产物与状态

- [160 题安全 manifest](../results/published/experiment1/audit/target160.json)：仅 ID、分组、subject/split 与审计状态。
- [全量动态聚合](../results/published/experiment1/audit/aggregate.json)：后台每轮增量更新。
- [共享审计协议](e1-data-audit-protocol.md)：判定标准、复用机制与背景任务边界。
- [历史队列脚本](https://github.com/mexiQQ/Hidden-Policy/blob/d50add85c8afc0d6a6490d1db4e91a4f8277701a/code/scripts/e1/audit_synthetic_pool.py)：本报告记录审计当时状态；一次性入口已清理，当前使用 [prepare_data.py](../scripts/e1/prepare_data.py)。

后台自动化每 5 分钟续跑，每次最多 300 个新题，全部完成后暂停。
原始题目和逐题决策仅保留在 ignored data/runtime 中；没有 commit/push、A6000 文件同步、
重新评测 baseline 或解封 TEST。

许可仍未确认，语义近重复及真实来源题族隔离尚未证明；manifest 保持 `training_ready=false`。
当前结果是一份可 review 的 L1 候选清单，不应直接写成经过专家验证的训练集。

队列归属、断点复用、幂等提交、冻结稳定性、公开字段边界及词汇分组测试：5 tests PASS。
