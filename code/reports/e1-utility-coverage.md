# E1 Utility 数据源覆盖审计

本轮只盘点候选来源，不是训练集交付或逐题质量审计。所有题量都是候选上界；答案正确性、题干自足性、英文质量、难度与语义去污染尚未核准，可用题数仍未知。

## 结论

冻结范围为 42 个 MMLU-NONOVERLAP subjects：25 科有内容域较明确的候选，12 科仅有邻域待复核来源，5 科没有确认映射。
领域对齐池按规范化题干取并集为 1743 道；含邻域池为 1945 道。跨科目候选共享，表内题量不能相加。

- 明确缺口：`abstract_algebra`, `econometrics`, `global_facts`, `human_sexuality`, `miscellaneous`。
- 另有 13 科只靠 Xiezhi 的 4-5 道领域候选支撑；题族近重复或错题还会继续缩小可用池。
- 两个来源适合互补，但目前不足以宣称完整 42 科覆盖。下一步优先补未对齐科目和薄弱科目，不继续堆积会计、美国史等大池。
- 为节省时间，自动格式扫描可以全量，人工核验只针对分科目抽出的少量候选滚动进行；无需先审完全部来源才制作小规模 utility 集。

## 来源题量

| 来源 | 原始题数 | 格式及 gold 编码通过 | 来源领域规则过滤后 | 再查精确重题后的独立题干 | 官方精确重题 |
| --- | ---: | ---: | ---: | ---: | ---: |
| xiezhi | 2478 | 2468 | 1994 | 1992 | 0 |
| eduqg | 3397 | 3343 | 1680 | 1674 | 0 |

X = Xiezhi 英文 Train；E = EduQG train + valid。normal/cloze 只算一题，原 valid 不直接充当本项目 D-CONSTRUCT dev。上表包含尚未映射到 42 科的题目，不是全部可用池。

## 42 科覆盖表

A：内容领域较明确的候选映射，不保证难度、答案或科目全覆盖；R：仅邻域，须逐题判断；gap：两个来源中尚无确认映射。数值均为格式、来源领域和精确重题检查后的不同规范化题干数。

具体标签和书名见 [映射配置](../configs/experiment1_utility_source_mapping.json)；逐科原始命中数、去重数和来源哈希见 [聚合 JSON](e1-utility-coverage.json)。

| MMLU subject | A: X / E | R: X / E | 状态 | 映射限制 |
| --- | ---: | ---: | --- | --- |
| abstract_algebra | 0 / 0 | 0 / 0 | gap | 未发现明确抽象代数来源；Basic Mathematics 实为数学符号题，不能映射至此。 |
| astronomy | 10 / 0 | 0 / 0 | aligned_candidate | 天体物理与天体力学领域对齐；仍需核验答案、难度与近重复。 |
| business_ethics | 0 / 61 | 5 / 0 | aligned_candidate | EduQG 商业伦理直接对齐；Xiezhi Ethics 主要为哲学思想，不能直接视为商业伦理。 |
| college_mathematics | 10 / 0 | 0 / 0 | aligned_candidate | 对应微积分、矩阵与向量；仅领域对齐，未确认大学课程难度。 |
| college_physics | 0 / 0 | 20 / 0 | review_only | 只列非化学方向细标签供难度和内容复核；不将 Physics 总类全部算作大学物理。 |
| conceptual_physics | 5 / 0 | 0 / 0 | aligned_candidate | 此标签的已看题目实际为牛顿定律概念；与高中物理邻域共池，不可重复计为独立题目。 |
| econometrics | 0 / 0 | 0 / 0 | gap | 未发现明确计量经济学来源；Quantitative Economics 的已看题目为基础宏观概念。 |
| electrical_engineering | 30 / 0 | 0 / 0 | aligned_candidate | 采用电气与电路细标签，避免将 Engineering 大类作为可用池；领域对齐不代表答案已验证。 |
| elementary_mathematics | 5 / 0 | 0 / 0 | aligned_candidate | 已看题目集中等号、大小于与括号，难度很低且题族窄，不能代表完整小学数学。 |
| formal_logic | 5 / 0 | 0 / 0 | aligned_candidate | 包含三段论和谓词逻辑；发现缺少文章上下文及近重复的候选，需逐题筛选。 |
| global_facts | 0 / 0 | 0 / 0 | gap | 未确认对应全球事实知识的独立候选池；不以 Geography 或 Economics 大类兜底。 |
| high_school_european_history | 0 / 0 | 4 / 0 | review_only | 少量世界史题涉及欧洲事件；没有欧洲史专标签，须逐题判定并与世界史候选去重。 |
| high_school_geography | 9 / 0 | 0 / 0 | aligned_candidate | 人文与自然地理直接对齐；部分题依赖未附情景描述，尚不能直接使用。 |
| high_school_government_and_politics | 0 / 185 | 10 / 0 | aligned_candidate | EduQG 美国政府教材匹配更直接；Xiezhi 比较政治或中国语境题须核对美国政治取向。 |
| high_school_macroeconomics | 5 / 0 | 5 / 0 | aligned_candidate | Quantitative Economics 的已看题目实际为通胀、通缩、经济周期与宏观指标；不映射计量经济学。 |
| high_school_mathematics | 0 / 0 | 5 / 0 | review_only | 微积分题需判定高中课程适配程度；与 college_mathematics 共池，不能双算覆盖或题数。 |
| high_school_microeconomics | 5 / 0 | 0 / 0 | aligned_candidate | 已看题目为供需、市场均衡与价格弹性；标签宽度不等于实际知识点覆盖。 |
| high_school_physics | 0 / 0 | 10 / 0 | review_only | 需按题目难度与课程要求复核；Theoretical Physics 已归入概念物理主候选，跨科不得双算。 |
| high_school_psychology | 5 / 298 | 0 / 0 | aligned_candidate | 学习和记忆等基础心理学直接对齐；EduQG 全书仍须逐题排除医学、生物与其他 scope 交叉内容。 |
| high_school_statistics | 5 / 0 | 0 / 0 | aligned_candidate | 已看题目集中基础概率，存在公理表述疑点；不能把标签视为广泛统计覆盖或正确性验证。 |
| high_school_us_history | 0 / 277 | 0 / 0 | aligned_candidate | EduQG 美国史直接对齐；Xiezhi History 大类不自动提供美国史覆盖。 |
| high_school_world_history | 4 / 0 | 0 / 0 | aligned_candidate | 实际池很小，且有事件排序题近重复；需要题族去重，不能按原始行数声称独立题数。 |
| human_sexuality | 0 / 0 | 0 / 0 | gap | 暂未确认明确来源；不借医学、生理学或全书心理学标签填充此缺口。 |
| international_law | 5 / 0 | 0 / 0 | aligned_candidate | 国际法专标签领域对齐；不使用 Law 总类替代，仍需核对翻译、答案及具体法律语境。 |
| jurisprudence | 5 / 0 | 5 / 99 | aligned_candidate | 法理学细标签直接对齐；法律史、商法和知识产权仅提供相邻概念，需逐题判断。 |
| logical_fallacies | 0 / 0 | 5 / 0 | review_only | 逻辑标签中的已看题目并非专门谬误识别，暂不计直接覆盖。 |
| machine_learning | 5 / 0 | 0 / 0 | aligned_candidate | 已看题目为实例、特征、学习类型、过拟合与训练目的；标签链为控制工程，不需要放开 Computer Science 大类。 |
| management | 10 / 0 | 5 / 0 | aligned_candidate | 采用组织行为和人力资源细标签；行政管理须判断是否属于通用管理而非特定制度知识。 |
| marketing | 5 / 0 | 0 / 0 | aligned_candidate | 营销领域对齐，但当前题目集中市场调研，不能代表完整营销课程。 |
| miscellaneous | 0 / 0 | 0 / 0 | gap | 未建立可靠映射标准；不把未分配的文学、艺术或其他题目自动归入 miscellaneous。 |
| moral_disputes | 0 / 0 | 5 / 61 | review_only | 伦理思想和商业伦理仅为邻域，须确认实际考查道德争议，不能直接按整个来源计覆盖。 |
| moral_scenarios | 0 / 0 | 0 / 61 | review_only | 须逐题识别道德情景判断；哲学语录题不等同 moral_scenarios。 |
| philosophy | 20 / 0 | 5 / 0 | aligned_candidate | 采用哲学细标签；不把已分配的 Logic 和 Religion 通过父级标签重复加回。 |
| prehistory | 5 / 0 | 0 / 0 | aligned_candidate | 已看题目涉及原始人遗址、火与石器；Archaeology 另有古代文字史题，Palaeoanthropology 有生物交叉，均未直接纳入。 |
| professional_accounting | 5 / 423 | 0 / 0 | aligned_candidate | 财务与管理会计领域对齐；入门教材和基础题不能自动视为专业资格水平。 |
| professional_law | 0 / 0 | 15 / 99 | review_only | 全部先列邻域；Xiezhi 中国法域和 EduQG 商法/IP 的范围不能直接代表 MMLU 美国专业法律科目。 |
| professional_psychology | 0 / 0 | 5 / 298 | review_only | 需核对专业深度并逐题排除医学或生物交叉；Applied Psychology 中已见治疗、障碍与健康语境。 |
| public_relations | 0 / 0 | 5 / 0 | review_only | 已看题目是传播学定义，尚未确认公关实践内容，只列相邻来源。 |
| security_studies | 0 / 0 | 9 / 0 | review_only | 只考虑非技术国际政治与安全政策；武器、作战技术及其他目标 scope 交叉题不纳入 utility。 |
| sociology | 5 / 333 | 9 / 0 | aligned_candidate | 社会学核心内容直接对齐；人口学和人类学须按知识点细分并排除生物交叉。 |
| us_foreign_policy | 0 / 0 | 10 / 462 | review_only | 均须逐题识别美国外交政策内容；国际关系宽度或美国背景本身不足以构成直接映射。 |
| world_religions | 5 / 0 | 0 / 0 | aligned_candidate | 当前五题全部集中伊斯兰基础，且存在相近题族；仅表明有宗教领域候选，不表明多宗教覆盖。 |

## Filters And Limits

- Keep original four distinct nonempty choices and original order. Never append options or create permutation views.
- Xiezhi: exact cleaned gold text must match one option. EduQG: zero-based ans_choice must agree with answer text or an A-D label (including trailing period/parenthesis). This checks encoding, not truth.
- Normalize stems using NFKC, case folding and whitespace collapse for counts. Different options under the same stem still form one conservative family; paraphrases and near-duplicates are not detected.
- Exclude the three biology/medicine EduQG books and specified Xiezhi domain labels. These are coarse source rules, not semantic proof of target non-overlap. Psychology and other boundary items still need review.
- Compare candidate prompt + ordered choices to public MMLU/WMDP stable IDs only. No official CAL/TEST question text or gold was opened. Rewording, translated overlap, option reordering and other source-family overlap are not ruled out.
- EduQG has book/chapter provenance; Xiezhi Train lacks a per-item original source/family identifier. Group by book/chapter and reviewed question family before making our train/dev split.
- Existing external MCQs are not automatically synthetic. Direct reuse versus source-grounded adaptation still requires a construction decision under Plan4; this audit does not change Plan4.

## Provenance

- [Xiezhi official repository](https://github.com/MikeGu721/XiezhiBenchmark#licenses): data CC BY-NC-SA 4.0; code MIT. Translation quality and Chinese jurisdiction/cultural context require review.
- [EduQG official repository](https://github.com/hadifar/question-generation) and [paper](https://arxiv.org/abs/2210.06104): underlying textbook licensing and added annotation permissions must be checked before reuse/publication; do not assume the whole dataset is unrestricted.

- [xiezhi train](https://raw.githubusercontent.com/MikeGu721/XiezhiBenchmark/9c6ba468d1ede4dad84ccd8284264e75665c5fab/Tasks/Knowledge/Benchmarks/train/xiezhi_train_eng/xiezhi.v1.1.json): commit `9c6ba468d1ede4dad84ccd8284264e75665c5fab`, SHA256 `a2ba9695c0ab269a7bf109c76e7fee41528890b0aef9e0390ec5291d122fa354`.
- [eduqg train](https://raw.githubusercontent.com/hadifar/question-generation/d253fe84a7fe6401768504ef6ab9eea36359107b/raw_data/qg_train_v0.json): commit `d253fe84a7fe6401768504ef6ab9eea36359107b`, SHA256 `f9b9348b6e3f32c4655237fe5f3c97a72c4a9c6647f723990bdb004a3a6042dd`.
- [eduqg valid](https://raw.githubusercontent.com/hadifar/question-generation/d253fe84a7fe6401768504ef6ab9eea36359107b/raw_data/qg_valid_v0.json): commit `d253fe84a7fe6401768504ef6ab9eea36359107b`, SHA256 `01f36c089e6caca2e9a621cf5b8817f9112130390e8b84d9daf504150b8fb8ef`.

## Reproduce

```sh
python3 code/scripts/experiments/audit_utility_coverage.py --download
```

Raw source caches remain under ignored code/data/experiment1/utility-source-audit/. Only aggregate JSON/Markdown and the mapping configuration are publishable. No generation, training, target auditing, git commit or remote synchronization is performed.
