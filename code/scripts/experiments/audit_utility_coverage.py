#!/usr/bin/env python3
"""Inventory external utility candidates; publish aggregates, never question text."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
from urllib.request import urlopen


CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT / "scripts" / "docs"))
from hidden_policy_eval.manifests import stable_item_id
from generate_baseline_report import (
    MMLU_NONOVERLAP_EXCLUDED_SUBJECTS,
    MMLU_STANDARD_SUBJECTS,
)

XIEZHI_REV = "9c6ba468d1ede4dad84ccd8284264e75665c5fab"
EDUQG_REV = "d253fe84a7fe6401768504ef6ab9eea36359107b"
SOURCES = [
    {
        "source": "xiezhi", "split": "train", "commit": XIEZHI_REV,
        "filename": "xiezhi_train_eng.jsonl",
        "url": f"https://raw.githubusercontent.com/MikeGu721/XiezhiBenchmark/{XIEZHI_REV}/Tasks/Knowledge/Benchmarks/train/xiezhi_train_eng/xiezhi.v1.1.json",
        "sha256": "a2ba9695c0ab269a7bf109c76e7fee41528890b0aef9e0390ec5291d122fa354",
    },
    {
        "source": "eduqg", "split": "train", "commit": EDUQG_REV,
        "filename": "eduqg_train.json",
        "url": f"https://raw.githubusercontent.com/hadifar/question-generation/{EDUQG_REV}/raw_data/qg_train_v0.json",
        "sha256": "f9b9348b6e3f32c4655237fe5f3c97a72c4a9c6647f723990bdb004a3a6042dd",
    },
    {
        "source": "eduqg", "split": "valid", "commit": EDUQG_REV,
        "filename": "eduqg_valid.json",
        "url": f"https://raw.githubusercontent.com/hadifar/question-generation/{EDUQG_REV}/raw_data/qg_valid_v0.json",
        "sha256": "01f36c089e6caca2e9a621cf5b8817f9112130390e8b84d9daf504150b8fb8ef",
    },
]
EXCLUDED_BOOKS = {"anatomy_and_physiology", "biology", "microbiology"}
EXCLUDED_LABELS = {
    "Medicine", "Biology", "Chemistry", "Chemical Engineering and Technology",
    "Computer Science and Technology", "Cyberspace Security", "Software Engineering",
    "Biomedical Engineering",
}


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def valid_shape(question, choices):
    return (
        isinstance(question, str) and bool(normalized(question))
        and isinstance(choices, list) and len(choices) == 4
        and all(isinstance(c, str) and normalized(c) for c in choices)
        and len({normalized(c) for c in choices}) == 4
    )


def parse_xiezhi(row):
    clean = lambda text: text.strip().strip('"').strip("'")
    choices = [clean(c) for c in row["options"].split("\n")]
    if not valid_shape(row["question"], choices):
        return None, "invalid_question_or_four_options"
    answer = clean(row["answer"])
    if choices.count(answer) != 1:
        return None, "gold_text_not_unique_option"
    return {
        "question": row["question"], "choices": choices,
        "answer": choices.index(answer), "labels": row["labels"],
    }, None


def parse_eduqg(row):
    question, answer = row["question"], row["answer"]
    choices, index = question["question_choices"], answer["ans_choice"]
    if not valid_shape(question["question_text"], choices):
        return None, "invalid_question_or_four_options"
    if type(index) is not int or not 0 <= index < 4:
        return None, "invalid_gold_index"
    text = (answer["ans_text"] or "").strip()
    label = re.fullmatch(r"([A-Fa-f])[.)]?", text)
    agrees = (
        bool(label) and label.group(1).upper() == chr(65 + index)
    ) or normalized(text) == normalized(choices[index])
    if not agrees:
        return None, "gold_text_index_not_agreed"
    return {
        "question": question["question_text"], "choices": choices, "answer": index,
    }, None


def read_source(spec, cache, download):
    path = cache / spec["filename"]
    if path.exists():
        raw = path.read_bytes()
    elif download:
        with urlopen(spec["url"], timeout=90) as response:
            raw = response.read()
    else:
        raise FileNotFoundError(f"Missing {spec['filename']}; use --download once")
    if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError(f"Source SHA256 mismatch: {spec['filename']}")
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    if spec["source"] == "xiezhi":
        return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    return json.loads(raw)


def validate_mapping(mapping, observed):
    subjects = [r["subject"] for r in mapping["rows"]]
    expected = MMLU_STANDARD_SUBJECTS - MMLU_NONOVERLAP_EXCLUDED_SUBJECTS
    if len(subjects) != 42 or set(subjects) != expected:
        raise ValueError("Mapping must contain exactly the frozen 42 utility subjects")
    for row in mapping["rows"]:
        for source in ("xiezhi", "eduqg"):
            aligned, review = (set(row[tier][source]) for tier in ("aligned", "review"))
            if aligned & review:
                raise ValueError(f"Duplicated mapping tier: {row['subject']}")
            missing = (aligned | review) - observed[source]
            if missing:
                raise ValueError(f"Unknown {source} labels/books: {sorted(missing)}")
            excluded = EXCLUDED_LABELS if source == "xiezhi" else EXCLUDED_BOOKS
            if (aligned | review) & excluded:
                raise ValueError("Mapping explicitly includes an excluded domain")


def unique_stems(rows):
    return {normalized(row["question"]) for row in rows}


def inventory(mapping, cache, download):
    official_ids = set()
    manifests = []
    for name in ("mmlu", "wmdp"):
        raw = (CODE_ROOT / "manifests" / "experiment0" / f"{name}.json").read_bytes()
        entries = json.loads(raw)["entries"]
        official_ids.update(entry["stable_id"] for entry in entries)
        manifests.append({"dataset": name, "entries": len(entries),
                          "sha256": hashlib.sha256(raw).hexdigest()})
    pools = {source: [] for source in ("xiezhi", "eduqg")}
    observed = {source: set() for source in pools}
    counts = {source: Counter() for source in pools}
    book_counts = {}
    for spec in SOURCES:
        data = read_source(spec, cache, download)
        source = spec["source"]
        if source == "xiezhi":
            records = [(row, row["labels"], "") for row in data]
        else:
            records = [(row, [chapter["bname"]], str(chapter["chapter"]))
                       for chapter in data for row in chapter["questions"]]
        for raw_row, labels, chapter in records:
            observed[source].update(labels)
            tally = counts[source]
            tally["raw_rows"] += 1
            tally[f"raw_{spec['split']}"] += 1
            book = book_counts.setdefault(labels[0], Counter()) if source == "eduqg" else None
            if book is not None:
                book["raw_rows"] += 1
            row, reason = (parse_xiezhi if source == "xiezhi" else parse_eduqg)(raw_row)
            if reason:
                tally[reason] += 1
                continue
            tally["structurally_valid_rows"] += 1
            if book is not None:
                book["structurally_valid_rows"] += 1
            denied = EXCLUDED_LABELS if source == "xiezhi" else EXCLUDED_BOOKS
            if set(labels) & denied:
                tally["valid_rows_excluded_by_domain_rule"] += 1
                continue
            tally["valid_rows_after_domain_rule"] += 1
            row.update(labels=labels, subject="external_utility", split=spec["split"], chapter=chapter)
            if stable_item_id(row) in official_ids:
                tally["exact_official_manifest_overlap_rows"] += 1
                continue
            pools[source].append(row)
    validate_mapping(mapping, observed)
    output_rows, unions = [], {tier: set() for tier in ("aligned", "review")}
    for rule in mapping["rows"]:
        output = {"subject": rule["subject"], "note": rule["note"]}
        for tier in ("aligned", "review"):
            output[tier] = {}
            for source, pool in pools.items():
                labels = set(rule[tier][source])
                matched = [row for row in pool if labels & set(row["labels"])]
                stems = unique_stems(matched)
                unions[tier].update(stems)
                output[tier][source] = {"selectors": sorted(labels), "rows": len(matched),
                                        "unique_stems": len(stems)}
        aligned = sum(v["unique_stems"] for v in output["aligned"].values())
        review = sum(v["unique_stems"] for v in output["review"].values())
        output["status"] = "aligned_candidate" if aligned else "review_only" if review else "gap"
        output_rows.append(output)
    for source, pool in pools.items():
        counts[source]["unique_stems_after_domain_and_overlap_checks"] = len(unique_stems(pool))
        counts[source].setdefault("exact_official_manifest_overlap_rows", 0)
        stem_splits = {}
        for row in pool:
            stem_splits.setdefault(normalized(row["question"]), set()).add(row["split"])
        counts[source]["cross_source_split_duplicate_stem_groups"] = sum(
            len(splits) > 1 for splits in stem_splits.values())
    return {
        "schema_version": 1, "sources": SOURCES, "official_manifests": manifests,
        "mapping_sha256": hashlib.sha256(json.dumps(mapping, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "source_counts": counts, "eduqg_book_counts": book_counts,
        "domain_exclusions": {"xiezhi_labels": sorted(EXCLUDED_LABELS), "eduqg_books": sorted(EXCLUDED_BOOKS)},
        "summary": {
            **dict(Counter(row["status"] for row in output_rows)),
            "aligned_unique_stems_union": len(unions["aligned"]),
            "review_unique_stems_union": len(unions["review"]),
            "all_mapped_unique_stems_union": len(unions["aligned"] | unions["review"]),
            "verified_usable_count": None,
        },
        "rows": output_rows,
    }


def render_markdown(report):
    summary = report["summary"]
    gaps = [row["subject"] for row in report["rows"] if row["status"] == "gap"]
    thin = [row["subject"] for row in report["rows"]
            if 0 < row["aligned"]["xiezhi"]["unique_stems"] < 6
            and not row["aligned"]["eduqg"]["unique_stems"]]
    lines = [
        "# E1 Utility 数据源覆盖审计", "",
        "本轮只盘点候选来源，不是训练集交付或逐题质量审计。所有题量都是候选上界；答案正确性、题干自足性、英文质量、难度与语义去污染尚未核准，可用题数仍未知。", "",
        "## 结论", "",
        f"冻结范围为 42 个 MMLU-NONOVERLAP subjects：{summary.get('aligned_candidate', 0)} 科有内容域较明确的候选，{summary.get('review_only', 0)} 科仅有邻域待复核来源，{summary.get('gap', 0)} 科没有确认映射。",
        f"领域对齐池按规范化题干取并集为 {summary['aligned_unique_stems_union']} 道；含邻域池为 {summary['all_mapped_unique_stems_union']} 道。跨科目候选共享，表内题量不能相加。", "",
        "- 明确缺口：" + ", ".join(f"`{subject}`" for subject in gaps) + "。",
        f"- 另有 {len(thin)} 科只靠 Xiezhi 的 4-5 道领域候选支撑；题族近重复或错题还会继续缩小可用池。",
        "- 两个来源适合互补，但目前不足以宣称完整 42 科覆盖。下一步优先补未对齐科目和薄弱科目，不继续堆积会计、美国史等大池。",
        "- 为节省时间，自动格式扫描可以全量，人工核验只针对分科目抽出的少量候选滚动进行；无需先审完全部来源才制作小规模 utility 集。", "",
        "## 来源题量", "",
        "| 来源 | 原始题数 | 格式及 gold 编码通过 | 来源领域规则过滤后 | 再查精确重题后的独立题干 | 官方精确重题 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source, counts in report["source_counts"].items():
        fields = ("raw_rows", "structurally_valid_rows", "valid_rows_after_domain_rule",
                  "unique_stems_after_domain_and_overlap_checks", "exact_official_manifest_overlap_rows")
        lines.append(f"| {source} | " + " | ".join(str(counts.get(k, 0)) for k in fields) + " |")
    lines += ["", "X = Xiezhi 英文 Train；E = EduQG train + valid。normal/cloze 只算一题，原 valid 不直接充当本项目 D-CONSTRUCT dev。上表包含尚未映射到 42 科的题目，不是全部可用池。", "",
              "## 42 科覆盖表", "",
              "A：内容领域较明确的候选映射，不保证难度、答案或科目全覆盖；R：仅邻域，须逐题判断；gap：两个来源中尚无确认映射。数值均为格式、来源领域和精确重题检查后的不同规范化题干数。", "",
              "具体标签和书名见 [映射配置](../configs/experiment1_utility_source_mapping.json)；逐科原始命中数、去重数和来源哈希见 [聚合 JSON](e1-utility-coverage.json)。", "",
              "| MMLU subject | A: X / E | R: X / E | 状态 | 映射限制 |",
              "| --- | ---: | ---: | --- | --- |"]
    for row in report["rows"]:
        cells = [" / ".join(str(row[t][s]["unique_stems"]) for s in ("xiezhi", "eduqg"))
                 for t in ("aligned", "review")]
        lines.append(f"| {row['subject']} | {cells[0]} | {cells[1]} | {row['status']} | {row['note'].replace('|', '/')} |")
    lines += ["", "## Filters And Limits", "",
              "- Keep original four distinct nonempty choices and original order. Never append options or create permutation views.",
              "- Xiezhi: exact cleaned gold text must match one option. EduQG: zero-based ans_choice must agree with answer text or an A-D label (including trailing period/parenthesis). This checks encoding, not truth.",
              "- Normalize stems using NFKC, case folding and whitespace collapse for counts. Different options under the same stem still form one conservative family; paraphrases and near-duplicates are not detected.",
              "- Exclude the three biology/medicine EduQG books and specified Xiezhi domain labels. These are coarse source rules, not semantic proof of target non-overlap. Psychology and other boundary items still need review.",
              "- Compare candidate prompt + ordered choices to public MMLU/WMDP stable IDs only. No official CAL/TEST question text or gold was opened. Rewording, translated overlap, option reordering and other source-family overlap are not ruled out.",
              "- EduQG has book/chapter provenance; Xiezhi Train lacks a per-item original source/family identifier. Group by book/chapter and reviewed question family before making our train/dev split.",
              "- Existing external MCQs are not automatically synthetic. Direct reuse versus source-grounded adaptation still requires a construction decision under Plan4; this audit does not change Plan4.", "",
              "## Provenance", "",
              "- [Xiezhi official repository](https://github.com/MikeGu721/XiezhiBenchmark#licenses): data CC BY-NC-SA 4.0; code MIT. Translation quality and Chinese jurisdiction/cultural context require review.",
              "- [EduQG official repository](https://github.com/hadifar/question-generation) and [paper](https://arxiv.org/abs/2210.06104): underlying textbook licensing and added annotation permissions must be checked before reuse/publication; do not assume the whole dataset is unrestricted.", ""]
    for spec in report["sources"]:
        lines += [f"- [{spec['source']} {spec['split']}]({spec['url']}): commit `{spec['commit']}`, SHA256 `{spec['sha256']}`."]
    lines += ["", "## Reproduce", "", "```sh",
              "python3 code/scripts/experiments/audit_utility_coverage.py --download",
              "```", "", "Raw source caches remain under ignored code/data/experiment1/utility-source-audit/. Only aggregate JSON/Markdown and the mapping configuration are publishable. No generation, training, target auditing, git commit or remote synchronization is performed.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    mapping = json.loads((CODE_ROOT / "configs/experiment1_utility_source_mapping.json").read_text())
    report = inventory(mapping, CODE_ROOT / "data/experiment1/utility-source-audit", args.download)
    root = CODE_ROOT / "reports"
    (root / "e1-utility-coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (root / "e1-utility-coverage.md").write_text(render_markdown(report))
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
