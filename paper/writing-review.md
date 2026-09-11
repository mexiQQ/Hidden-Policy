# E0–E3 论文写作与审阅记录

日期：2026-09-11。范围：续写新章节，仅允许修改既有 §3.3；不启动实验，不修改实验记录。本文不是新的实验计划。

## 写作结构

- §3.3：把旧分类改为 A 局部条件改善、B 跨条件改善（suppression / 已测范围 removal）、C 指定范围内行为不可达。分类是行为解释，不是内部机制鉴定。
- §4：构造。E0 只交代能力基线，E1 解释四种训练配方和官方 TEST-Q3 新题表现。
- §5：诊断。E2 的主线是“构造规则不等于学到的规则”，选择 D1–D5 中真正约束结论的结果。
- §6：干预。E3 的 R1 方法比较、R2 贡献分析为主；TEST-Q4 确认效果与代价；R3/R3b 只作补充。
- §7：讨论与局限。回答构造与移除问题，不宣称完整等价、机制删除或自然涌现。
- 附录：校准表、实际 checkpoint/预算、数据用途、补充诊断、官方原模型与 corrective-SFT 四条件结果。

摘要、引言、§2、§3.1、§3.2 保持原文。正文仍放在 `main.tex`，没有拆出大量临时章节文件。

## 核心证据来源

下面路径相对仓库根目录。数字以公开聚合 JSON 为准；HTML 仅用于阅读。题目原文、预测原文、权重不复制进论文仓库。

| 写作内容 | 来源与定位 | 解释边界 |
|---|---|---|
| E0 能力筛选 | `code/results/published/experiment0/baseline/full_vllm/result.json` 的 `models.*.datasets`；0.8B 用同目录 `full_vllm_weak/result.json` | CAL；WMDP 734，MMLU nonoverlap 1,436。likelihood 与 generation 不混用 |
| E1 官方构造表 | `code/results/published/experiment1/official-cal-q3-v1/result.json` 的 `results[].name`、`groups[].split/scope/gate_on` | TEST-Q3：Target 1,451、Utility 5,653；熟悉 gate 文本，不是未见表达 |
| 模型和训练参数 | 同目录 `protocol.json` 的 `adapters[].config` 与 checkpoint 路径 | U1 epoch4 是8轮调度中的中间 checkpoint；SHAM预算不完全匹配 |
| E2 五项诊断 | `code/results/published/experiment2/diagnostics-v1/{result.json,diagnostic-analysis.json,interpretation.json}` | 使用完成的历史 MCQ 部分，不使用已删除的 H2；D4小样本描述性，D5不纯测记忆 |
| E3 行为分类与证据 | `code/results/published/experiment3/taxonomy-v1/interpretation.json`，绑定各轮 `result.json` 与 `analysis.json` | ABC是事后解释；B-removal只限已测行为，正常能力另判 |
| R2 剪枝/续训及直接拒答 | 同目录 `r2/{result.json,analysis.json}`，与 R1 同方法配对 | 固定干预前正常答对的题作分母；直接行为成功不证明原 policy 残留 |
| 官方干预确认表 | 同目录 `official-confirm-v1/{protocol.json,result.json,analysis.json,exposure.json}` | TEST-Q4固定子集192/504；排除早期smoke曝光题；不针对结果再调参 |
| 内部补充复核 | 同目录 `r3/`、`r3b/` | 同一confirm；R3换题，R3b换校准，非新植入seed |

新增引用仅用于实际使用的数据、LoRA、sandbagging、Fine-Pruning、CROW 和 Vax。已用论文/作者主页/正式出版页核验；既有参考文献条目不改。

## 多 Agent 审阅

Author：主 agent。Reviewer 1：故事主线与构念有效性。Reviewer 2：数值、实验协议与统计边界。采用独立角色提示，不调用额外实验或扩展研究任务。最多三轮。

### 第一轮：写作前审查（完成）

两位 reviewer 独立读取原论文和冻结结果，主要意见均已进入初稿：

1. 保留 construct–diagnose–remove 主线，不把四个工程阶段原样变成结果流水账。
2. 四类是构造配方，不是已证实的能力/策略阶梯；E1不能证明完整轨迹分布等价。
3. 明确 on/off 相对 capable SHAM 的含义，不把低准确率自动当作弱模型外观一致。
4. 如实展示U1的正常表现损失、历史SHAM不匹配、单seed以及熟悉gate的TEST-Q3。
5. D3区间跨零不是证明无交互；D4不能推广为所有模型；D5不单归因于记忆。
6. Vax提供MLP干预动机，FP不是Vax复现；R3/R3b放补充，不重复包装新题泛化。

### 第二轮：初稿评审（完成）

两位 reviewer 检查实际正文、表格和附录。五张表及关键数字均与冻结结果一致。已落实：

1. 将“移除原cue效果”改为“削弱cue触发的拒答”，避免与B-suppression的残留结论冲突。
2. 官方确认补上corrective-G1U0在熟悉及指定替代条件下通过5pp检查的正面结论；不只罗列不能证明什么。
3. 先说明官方G1替代场景是训练外但此前诊断已使用的资格考试场景，再报告2.6pp差距。
4. 明确统计实现为双侧95% percentile bootstrap区间，再对下界作单侧非劣判断；不是另行计算单侧95%区间。scoped removal对treated和unmodified SHAM都检查。
5. 补充有效替代条件的冻结门槛：干预前差距至少10pp且差值区间上界小于零。
6. “All E1 runs”和推理参数的范围收窄到本文沿用的四个checkpoint及E1正式验证/E2/E3，不覆盖历史smoke。

Reviewer建议把A改名为Local condition improvement。这里保留用户刚决定的Trigger invalidation名称，但加“local behavioral evidence”，定义仍限制为局部行为证据，不暗示内部机制定位。

### 第三轮：修改后复核（完成，停止迭代）

两位 reviewer 均确认无新增章节的实质阻断，不要求第四轮或新实验。故事 reviewer 确认构造、诊断、干预三部分形成连贯回答；数值 reviewer 复核了新增的官方正面结论：corrective-G1U0 对treated SHAM的熟悉/资格考试场景差值区间分别为[-4.17, 2.60]和[-2.08, 4.17] pp，对unmodified SHAM为[-2.08, 6.25]和[-1.04, 6.25] pp，均满足下界不低于-5。三项正常能力保持状态仍为uncertain，所以不能改写为无损修复。

本轮交付是**受既有章节保护范围约束的新章节初稿**，不是宣称整篇旧文与新结果已完全统一的投稿终稿。下方旧文张力仍待用户授权后再改。

## 是否需要补实验

**当前限定范围的论文初稿不依赖新实验。** 优先删除或收窄没有证据的主张；以下只记录升级结论时才需要的证据，未执行、未排期。

| 候选补充 | 何时才必要 | 当前处理 |
|---|---|---|
| 匹配G/U配方与训练预算、独立seed重复 | 要宣称“G1或U1本身造成某种差异”或算法普遍可靠时 | 现稿只描述选定organisms；最值得后续优先考虑，但不阻塞当前描述性写作 |
| 更严格的observer匹配或低能力reference比较 | 要宣称完整隐蔽性、弱模型外观一致或observer难以区分时 | 不声称这些资格已建立；不能用当前四个accuracy替代分布等价 |
| 随机通道/attention等匹配干预对照与机制分析 | 要宣称MLP特殊性、定位policy或提出新purification方法时 | 不主张机制贡献；明确引用Vax，FP只作诊断干预 |
| 固定可靠的直接任务与路径控制 | 要给C不可达/能力丧失下结论时 | 现有成功调用可反对全面不可达，失败不能证明C；不强造C案例 |
| D4更大样本及表达×续训交互检验 | 要断言泛化比熟悉条件更易被续训破坏时 | 只写G1U0描述性例子，不写显著差异或普遍规律 |
| D5系统指令、包装与记忆分别消融 | 要定位对话迁移失败的原因时 | 只写设置变化下迁移失败，不写记忆机制 |

Q4既有确认子集已曝光。未来若据此优化方法，必须用另一个合格的留出集确认，不能把同一子集重新称为独立确认。

## 既有章节的待处理张力（本次不改）

- 摘要/引言中的 holding behavior fixed、generalized and persistent、mechanistic consistency 比现有实证更强。新§4/§7已注明证据边界，但最终投稿前仍建议得到用户许可后统一措辞。
- 引言三问与contributions构造/移除两问存在编号冲突。新文使用construction/removal question，并解释TEST-Q3/TEST-Q4为数据分区名。
- §2的若干公式下标/等价类集合排版及文字错误保留，未越过编辑范围。
- 自发emergence仍是理论动机，受控LoRA植入不是其证据。

## 验证

- 已将§3.3之前的`main.tex`与起始commit `5521c03`逐字节比较，完全一致；该保护段SHA256为`133dd7ae9616400ab22b71711e897e339385db81ae46157b3ed1314bf500291d`。
- 原有BibTeX条目保持不变，仅为新文追加9条已核验引用。
- 五张表的数字、分母、模型与SHAM对应关系均通过独立reviewer与公开JSON核验。
- `make`成功；最终LaTeX日志没有undefined reference/citation或overfull警告。存在不影响内容的underfull排版提示，旧文亦有此类提示。
- 已检查13页PDF的渲染：正文到第10页，参考文献到第11页，附录第12–13页；未发现重叠、裁切或缺字。文本边界检查无超出页面、无替换字符。
- 没有训练、推理、A6000调用或实验文件修改。仅改`paper/main.tex`、追加`paper/references.bib`和新增本记录；PDF及检查图片继续按仓库规则忽略。
- 当前未commit/push；保留给用户阅读本轮写作差异。
