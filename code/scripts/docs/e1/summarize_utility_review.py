#!/usr/bin/env python3
"""Validate a completed local utility review and publish content-free aggregates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT / "src"))
from hidden_policy_eval.e1.review import PUBLIC_DECISION_FIELDS, validate_decisions

LOCAL_ROOT = CODE_ROOT / "data/experiment1/utility-review"
PUBLISHED_ROOT = CODE_ROOT / "results/published/experiment1/utility-review"


def summarize(batch, decisions):
    validate_decisions(batch, decisions)
    by_id = {decision["id"]: decision for decision in decisions}
    by_subject = {subject: Counter() for subject in batch["requested_subjects"]}
    by_source, reasons, entries = {}, Counter(), []
    accepted_families = set()
    for item in batch["items"]:
        decision = by_id[item["id"]]
        verdict = decision["verdict"]
        by_subject[item["subject"]][verdict] += 1
        by_source.setdefault(item["source"], Counter())[verdict] += 1
        reasons[decision["reason_code"]] += 1
        if verdict == "accept":
            if item["family_hash"] in accepted_families:
                raise ValueError("Accepted pool contains repeated normalized stems")
            accepted_families.add(item["family_hash"])
        entries.append({
            **{key: decision[key] for key in sorted(PUBLIC_DECISION_FIELDS)},
            **{key: item[key] for key in ("subject", "source", "tier", "family_hash", "stable_id")},
        })
    totals = Counter(decision["verdict"] for decision in decisions)
    return {
        "schema_version": 1, "review_type": "model_content_first_pass_not_expert_verified",
        "batch_sha256": hashlib.sha256(json.dumps(batch, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "mapping_sha256": batch["mapping_sha256"],
        "source_specs": batch["source_specs"],
        "official_manifest_sha256": batch["official_manifest_sha256"],
        "selection_policy": batch["selection_policy"],
        "requested_subjects": batch["requested_subjects"],
        "omitted_subjects": batch["omitted_subjects"],
        "per_subject_limit": batch["per_subject_limit"],
        "totals": dict(totals), "reviewed_items": len(decisions),
        "subjects_with_accept": sum(counts["accept"] > 0 for counts in by_subject.values()),
        "by_subject": by_subject, "by_source": by_source, "reason_counts": reasons,
        "entries": sorted(entries, key=lambda row: (row["subject"], row["id"])),
    }


def render_markdown(report):
    totals = report["totals"]
    no_accept = [subject for subject, counts in report["by_subject"].items() if not counts.get("accept")]
    lines = [
        "# E1 Utility 小批量审核", "",
        "## 结论", "",
        f"本轮面向 37 科，每科最多 {report['per_subject_limit']} 题；实际审核 **{report['reviewed_items']} 道**不同题干。",
        f"模型初审：**accept {totals.get('accept', 0)} / reject {totals.get('reject', 0)} / review {totals.get('review', 0)}**。",
        f"有至少一道初审通过题的科目为 **{report['subjects_with_accept']} / 37**；accept 仅表示可进入下一阶段候选，不是专家验证、许可清理或完整训练集交付。", "",
        "本批尚无初审通过题的科目：" + (", ".join(f"`{s}`" for s in no_accept) or "无") + "。小样本无通过题不表示整个科目或来源不可用。", "",
        "## 取样与判断", "",
        "- 只对用户同意的 37 科取样；5 个未映射科目暂缓。原 42 科 MMLU-NONOVERLAP 评测范围、Plan4 和另一 session 的 target 审计没有变更。",
        "- 优先领域对齐池；领域对齐池有 EduQG 时优先教材来源，其余按固定哈希排序。三轮小池优先轮询，每科每轮至多一题，全局规范化题干不重复分配。",
        "- 这不是来源总体质量的随机估计：来源偏好、科目小池、题干去重及小样本量均影响结果。未为了提高通过率反复换题。",
        "- 每题检查科目匹配、题干自足与歧义、gold 基础知识合理性、英文可读性及与排除领域的交叉。保留原四选项顺序与 gold，不改写或修补。",
        "- accept 必须同时满足 subject_fit=yes、context_status=self_contained、scope_status=nonoverlap、gold_status=plausible；不确定项单列 review，不自动接受。",
        "- 本轮是模型逐题初审，非外部专家认证；没有调用 target/weak 模型重答，也未启动生成、训练或评测。", "",
        "## 分科结果", "",
        "| Subject | 审核 | accept | reject | review |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for subject, counts in sorted(report["by_subject"].items()):
        lines.append(f"| {subject} | {sum(counts.values())} | {counts.get('accept', 0)} | {counts.get('reject', 0)} | {counts.get('review', 0)} |")
    lines += ["", "## 分来源结果", "",
              "| 来源 | 审核 | accept | reject | review |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for source, counts in sorted(report["by_source"].items()):
        lines.append(f"| {source} | {sum(counts.values())} | {counts.get('accept', 0)} | {counts.get('reject', 0)} | {counts.get('review', 0)} |")
    lines += ["", "不同来源承担的科目、邻域映射比例和样本量不同，不能将本批通过比例直接解释为数据集总体质量比较。"]
    lines += ["", "## 主要原因", "", "| reason_code | 数量 |", "| --- | ---: |"]
    lines += [f"| {reason} | {count} |" for reason, count in sorted(report["reason_counts"].items(), key=lambda pair: (-pair[1], pair[0]))]
    lines += ["", "## 边界与复用", "",
              "- 官方去重仅复用公开 manifest stable_id 的精确检查；没有读取封存题目和答案。不代表完成语义去污染、翻译重题或真实 source-family 隔离。",
              "- 同题干聚类不覆盖所有近重复。审核中发现的明显近重复予以标记；未宣称完整的语义近重复扫描。",
              "- 原始题、逐题理由和队列位于 ignored `code/data/experiment1/utility-review/`，可接续使用；不上传原题、选项或答案。",
              "- [聚合与去敏逐题状态](../results/published/experiment1/utility-review/batch-v1.json) 仅含 ID、hash、subject 和枚举判断，不包含题目正文。",
              "- 现成外部题不自动等于 synthetic；改造方式、来源许可与训练/dev 切分仍待后续决定。本次没有冻结或导出训练集。", "",
              "## 复现", "", "```sh",
              "python3 code/scripts/e1/prepare_utility_review.py",
              "python3 code/scripts/docs/e1/summarize_utility_review.py",
              "```", "",
              "第二条命令依赖已完成的本地 decisions-1/2/3.json，不重新进行内容审核，也不调用任何模型或外部服务。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    batch = json.loads((LOCAL_ROOT / "batch-v1.json").read_text())
    decisions = []
    for number in range(1, 4):
        decisions.extend(json.loads((LOCAL_ROOT / f"decisions-{number}.json").read_text()))
    report = summarize(batch, decisions)
    PUBLISHED_ROOT.mkdir(parents=True, exist_ok=True)
    (PUBLISHED_ROOT / "batch-v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (CODE_ROOT / "reports/e1-utility-small-batch-review.md").write_text(render_markdown(report))
    print(json.dumps({"reviewed": report["reviewed_items"], "totals": report["totals"],
                      "subjects_with_accept": report["subjects_with_accept"]}, sort_keys=True))


if __name__ == "__main__":
    main()
