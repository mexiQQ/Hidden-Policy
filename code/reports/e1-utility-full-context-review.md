# E1 Utility 全量复核与清理

日期：2026-09-06

**全量 1,945 道候选已覆盖，最终可用池为 1,269 道。当前实验的 160 道 utility 已更新，只替换原来的 6 道疑点题；每科仍为 16 train / 4 dev。**

## 清理结果

| 项目 | 题数 |
| --- | ---: |
| 全部去重候选 | 1,945 |
| 本次直接审阅 | 1,785 |
| 复用刚完成的已选题复核 | 160 |
| 原可用候选池 | 1,281 |
| 从原可用池新增排除 | 3 |
| 从原可用池新增隔离待核实 | 10 |
| 经明确证据解决旧待定判断后纳入 | 1 |
| 最终可用候选池 | **1,269** |

全量内容复核分为 1,669 道 keep、126 道 exclude、150 道 uncertain。这个数字不等于新剔除量：多数问题早已被历史审计排除。keep 也不自动推翻旧有的学科、范围或答案疑点。

默认准入条件为旧审计 accept 且本次 keep。只对 1 道原 specialist_uncertain 题记录了明确解决结果；其他旧非 accept 继续不纳入。uncertain 一律不进入训练选题，但不能据此声称题目已被证实错误。

## 当前实验

| Subject | 清理后全量可用 | 当前 train | 当前 dev |
| --- | ---: | ---: | ---: |
| 商业伦理 (`business_ethics`) | 28 | 16 | 4 |
| 政府与政治 (`high_school_government_and_politics`) | 125 | 16 | 4 |
| 心理学 (`high_school_psychology`) | 190 | 16 | 4 |
| 美国史 (`high_school_us_history`) | 207 | 16 | 4 |
| 法理学 (`jurisprudence`) | 20 | 16 | 4 |
| 会计 (`professional_accounting`) | 358 | 16 | 4 |
| 法律 (`professional_law`) | 33 | 16 | 4 |
| 社会学 (`sociology`) | 242 | 16 | 4 |

- Target 160 道及其划分完全不变。
- Utility 只替换原 6 道疑点题；其余 154 道的题目内容、选项顺序、答案和 train/dev 均未改变。
- 保留原章节隔离规则；经证据核实的替补补足了原验证章节。
- 旧选题清单与固定来源已留档，可用于追溯旧原题；原始候选池保持不变，历史审计提交已无损合并。
- 没有重新生成0.8B回答或训练。新数据用于下一次独立run，不应覆盖旧训练结果。

## 新增排除清单

这里只发布 ID 与类别。题目、选项、gold 和逐题理由保留在本地 ignored 数据目录。

| Audit ID | Subject | 处理 | 原因 |
| --- | --- | --- | --- |
| `utility-full-06d19b35ef698c44e7c27d11f50d852cd1157b745a22a4c338077eb80c31e3b6` | `jurisprudence` | `uncertain` | `ambiguous` |
| `utility-full-1cca0ba46bf6c8c9e5532ff5e41c0a88b04e306fdbc0a2090d7a7e459d847f68` | `business_ethics` | `uncertain` | `ambiguous` |
| `utility-full-24823a7001b77f8690e7ad92588471a21d3fdca2a9248812a8caf7783d48cd03` | `high_school_psychology` | `uncertain` | `ambiguous` |
| `utility-full-2b8f8c2543538c38cbb31e59b5a2316547b2aa1c1d3dee838f69316f1cd79588` | `philosophy` | `exclude` | `language_issue` |
| `utility-full-3316ec200195c69b662293da833348618addee1a88956a14236585d82ddd9243` | `jurisprudence` | `uncertain` | `ambiguous` |
| `utility-full-3cf24d673eba93a89c72e6e70494c7253e17c6939c4b88600edf3bb36791e4d5` | `business_ethics` | `uncertain` | `ambiguous` |
| `utility-full-40e12ec888eeb66b6558e57f450d92eb06d01c64db7ad8d63d87e59969f02f7b` | `high_school_microeconomics` | `uncertain` | `ambiguous` |
| `utility-full-563519d2fdc4fd48c094cb290f28e82f58af4dba618d88d94ddeca8f7cdc97e4` | `high_school_us_history` | `uncertain` | `ambiguous` |
| `utility-full-77358807f0fbd26ad5096d460de4d862e0b10b62a91ec2df5d8f7aafc948dbea` | `high_school_us_history` | `uncertain` | `ambiguous` |
| `utility-full-a35f6c069c8bbdcf6f805c95377fd2aad76a3408106542fd19835add262fb167` | `high_school_psychology` | `exclude` | `ambiguous` |
| `utility-full-ea8edf3e08a5641786217b37cb27c8a3414c8df0ebd34b6d7879aefe21603ea3` | `jurisprudence` | `uncertain` | `ambiguous` |
| `utility-full-f6a9bf3588cc1bfa15de8e0b4fa5191ceb4208817638685753748590cceb73f2` | `jurisprudence` | `exclude` | `language_issue` |
| `utility-full-fc2bf8d499eea486e902d947d67c938740dc2604b96fe9d5b1c60e3e057d8ce6` | `jurisprudence` | `uncertain` | `ambiguous` |

## 已解决的旧待定项

`utility-full-662f76be67390bd295cffa9841f81c261523dfaf7550b7017c50bd75caf5b80e`：原因为 specialist_uncertain，经固定来源和独立复核解决，明确列入 `resolved_previous_reviews`，不会自动放行其他旧待定或拒绝题。

核对资料：[美国最高法院 Bilski 判决第609–612页](https://www.supremecourt.gov/opinions/boundvolumes/561bv.pdf#page=654)、[USPTO 官方指引](https://www.uspto.gov/news/og/2005/week47/patgupa.htm)、[美国版权局案件摘要](https://www.copyright.gov/fair-use/summaries/williams%26wilkins-unitedstates-ctcl1973.pdf)。原题及具体核对过程仅存于本地复核记录。

## 文件与入口

- [全量安全判定](../results/published/experiment1/utility-context-review.json)：全部ID、枚举、计数与来源指纹，选题时实际使用。
- [当前选题清单](../manifests/experiment1/construct160.json)：更新后的320题无正文清单。
- [旧选题留档](../manifests/experiment1/archive/construct160-before-utility-review.json)：用于追溯本次更新之前的实验。
- 本地 `code/data/experiment1/utility-context-review/clean-pool.json`：1,269 道清理后可用原题，尚未分配实验split。
- 本地 `code/data/experiment1/utility-context-review/full-review.json`：1,945 条复核记录、具体疑点理由、复用与二次复核记录。
- 本地 `code/data/experiment1/construct/items.json`：当前160 target + 160 utility。
- 本地 `code/data/experiment1/utility-full-audit/review-history.json`：48份历史审核与纠错提交的完整合并留档，含原文件名与原始字节SHA。

仍使用 [prepare_data.py](../scripts/e1/prepare_data.py) 的 status / freeze / build，没有新增审计CLI。筛选和重建逻辑集中在 [e1/data.py](../src/hidden_policy_eval/e1/data.py)。

## 边界

这是模型辅助内容复核，不是逐题专家答案认证。新审题检查题干与选项，对明显gold错误和疑似问题结合固定来源或权威资料核对；复用的160题以之前的独立作答判断为准，没有声称重新完成160题的全面gold验证。原始来源中的未入池记录和同题干其他选项变体不在本轮1,945题范围内。

验证已通过：E1共101项单元测试，status / freeze / build三入口实际运行，以及独立的ID覆盖、来源指纹、内容保留、配额和章节隔离检查。测试中的训练调用均为mock，本次没有运行真实训练。

来源缓存仅保留 `utility-source-audit/pinned/` 中通过固定版本字节哈希校验的EduQG文件；已确认JSON内容相同的上层格式化副本被删除，原题内容未改写。

另已删除重复分发队列、被本报告替代的阶段报告及可重建的旧原题副本。旧原题已按留档清单与固定来源验证可完整重建；追溯时须将旧清单中的EduQG缓存路径映射到 `pinned/`。保留文件用途见[数据说明](../data/experiment1/README.md)。
