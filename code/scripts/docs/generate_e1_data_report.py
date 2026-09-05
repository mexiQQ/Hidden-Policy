#!/usr/bin/env python3
"""Build a self-contained E1 report from published, content-free audit records."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from html import escape
import hashlib
import json
from pathlib import Path
import re


CODE_ROOT = Path(__file__).resolve().parents[2]
TARGET = CODE_ROOT / "results/published/experiment1/audit/aggregate.json"
UTILITY = CODE_ROOT / "results/published/experiment1/utility-full-audit/status.json"
FROZEN = CODE_ROOT / "results/published/experiment1/audit/target160.json"
TEMPLATE = Path(__file__).with_name("e1_data_report_template.html")
VERDICTS = ("accept", "reject", "review")
SUBJECTS = ("Biology", "Chemistry", "Cybersecurity")
COLORS = {"accept": "#26745a", "reject": "#b44949", "review": "#bf861e"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value):
    require(type(value) is int and value >= 0, "invalid count")
    return value


def hash_value(value, length=64):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value),
            "invalid provenance hash")
    return value


def counts(values):
    require(set(values) <= set(VERDICTS), "invalid verdict keys")
    return {key: count(values.get(key, 0)) for key in VERDICTS}


def build_report(target: dict, utility: dict, frozen: dict) -> dict:
    require(target["remaining"] == 0 and target["frozen160"] is True, "target incomplete")
    target_subjects = {subject: Counter() for subject in SUBJECTS}
    automatic = 0
    for row in target["counts"]:
        require(row["subject"] in SUBJECTS and row["state"] == "done", "invalid target state")
        require(row["verdict"] in VERDICTS, "invalid target verdict")
        if row["verdict"] == "accept":
            require(row["gold_status"] == "plausible", "target accept not plausible")
        if row["gold_status"] == "not_checked":
            require(row["verdict"] == "reject", "unexpected unchecked target verdict")
            automatic += count(row["count"])
        target_subjects[row["subject"]][row["verdict"]] += count(row["count"])
    target_verdicts = sum(target_subjects.values(), Counter())
    target_total = count(target["total_rows"])
    require(sum(target_verdicts.values()) == target_total, "target total mismatch")
    reasons = target["reason_counts"]
    require(sum(count(v) for v in reasons.values()) == target_total, "target reasons mismatch")
    require(reasons["clear_basic_fact"] == target_verdicts["accept"], "target accepts mismatch")
    require(sum(reasons.get(k, 0) for k in ("duplicate_canonical", "duplicate_choices", "duplicate_stem"))
            == automatic, "target automatic exclusions mismatch")

    require(frozen["training_ready"] is False and frozen["license_status"] == "unconfirmed",
            "unexpected training readiness")
    require(frozen["provenance"] == target["provenance"], "target provenance mismatch")
    splits, selected_subjects, family_splits = Counter(), defaultdict(Counter), {}
    selected_ids = set()
    for entry in frozen["entries"]:
        require(entry["review_status"] == "model_plausible", "unreviewed frozen item")
        require(entry["subject"] in SUBJECTS and entry["split"] in ("train", "dev"), "invalid split")
        require(re.fullmatch(r"mcq-[0-9a-f]{64}", entry["stable_id"]), "invalid selected ID")
        require(entry["stable_id"] not in selected_ids, "duplicate frozen ID")
        selected_ids.add(entry["stable_id"])
        for key in ("family_hash", "lexical_family_hash"):
            family = (key, hash_value(entry[key]))
            require(family_splits.get(family, entry["split"]) == entry["split"], "cross-split family")
            family_splits[family] = entry["split"]
        splits[entry["split"]] += 1
        selected_subjects[entry["subject"]][entry["split"]] += 1
    require(dict(splits) == {"train": 128, "dev": 32}, "frozen split mismatch")
    require(dict(selected_subjects) == {"Biology": {"train": 43, "dev": 11},
            "Chemistry": {"train": 42, "dev": 11}, "Cybersecurity": {"train": 43, "dev": 10}},
            "frozen subject quotas mismatch")
    screen = frozen["lexical_screen"]
    require(screen["cross_split_pairs"] == 0, "lexical split check failed")

    progress = utility["progress"]
    require(utility["status"] == "complete" and all(progress[k] == 0 for k in
            ("remaining", "pending", "claimed")) and not progress["qa_hold"], "utility incomplete")
    requested = utility["provenance"]["requested_subjects"]
    omitted = utility["provenance"]["omitted_subjects"]
    require(len(set(requested)) == 37 and len(set(omitted)) == 5 and
            not set(requested) & set(omitted), "utility subject scope mismatch")
    require(all(re.fullmatch(r"[a-z_]+", s) for s in requested + omitted), "invalid subject")
    subject_counts, source_counts = defaultdict(Counter), defaultdict(Counter)
    utility_reasons, utility_verdicts = Counter(), Counter()
    ids, families = set(), set()
    imported = 0
    for entry in utility["entries"]:
        require(entry["id"] not in ids, "duplicate utility ID")
        require(entry["family_hash"] not in families, "duplicate utility family")
        hash_value(entry["family_hash"])
        ids.add(entry["id"])
        families.add(entry["family_hash"])
        require(entry["subject"] in requested and entry["source"] in ("eduqg", "xiezhi"),
                "invalid utility subject/source")
        require(entry["state"] == "done" and entry["verdict"] in VERDICTS, "invalid utility state")
        if entry["verdict"] == "accept":
            require(entry["gold_status"] == "plausible" and entry["scope_status"] == "nonoverlap"
                    and entry["subject_fit"] == "yes" and entry["context_status"] == "self_contained",
                    "utility accept not plausible or aligned")
        subject_counts[entry["subject"]][entry["verdict"]] += 1
        source_counts[entry["source"]][entry["verdict"]] += 1
        utility_verdicts[entry["verdict"]] += 1
        utility_reasons[entry["reason_code"]] += 1
        require(type(entry["imported"]) is bool, "invalid imported flag")
        imported += entry["imported"]
    require(progress["total"] == progress["done"] == len(ids), "utility total mismatch")
    require(counts(progress["verdicts"]) == counts(utility_verdicts), "utility verdict mismatch")
    require(imported == progress["imported"], "utility reuse mismatch")
    for actual, published in ((subject_counts, utility["by_subject"]), (source_counts, utility["by_source"])):
        require(set(actual) == set(published) and all(counts(v) == counts(published[k])
                for k, v in actual.items()), "utility subject/source mismatch")
    require(dict(utility_reasons) == utility["reason_counts"], "utility reason mismatch")
    without_accept = sorted(s for s in requested if subject_counts[s]["accept"] == 0)
    require(sorted(utility["subjects_without_accept"]) == without_accept and
            utility["subjects_with_accept"] == len(requested) - len(without_accept), "coverage mismatch")
    top_five = sorted(requested, key=lambda s: (-subject_counts[s]["accept"], s))[:5]
    source_specs = utility["provenance"]["source_provenance"]["source_specs"]
    source_versions = sorted({(s["source"], hash_value(s["commit"], 40)) for s in source_specs})
    require(set(s for s, _ in source_versions) == {"eduqg", "xiezhi"}, "source provenance mismatch")

    # Only explicit aggregate fields enter publication; no raw entry is copied.
    return {
        "schema_version": 1, "title": "E1 数据审计", "snapshot_date": "2026-09-05",
        "status": "complete", "training_ready": False,
        "target": {"total": target_total, "verdicts": counts(target_verdicts),
                   "by_subject": {s: counts(v) for s, v in target_subjects.items()},
                   "automatic_exclusions": automatic, "imported": count(reasons["prior_sample_flag"]),
                   "new_reviews": target_total - automatic - reasons["prior_sample_flag"],
                   "source_commit": hash_value(target["provenance"]["source_commit"], 40),
                   "source_sha256": hash_value(target["provenance"]["source_sha256"])},
        "frozen": {"total": len(selected_ids), "splits": dict(splits),
                   "by_subject": {s: dict(v) for s, v in selected_subjects.items()},
                   "lexical_components": count(screen["component_count"]),
                   "lexical_pairs": count(screen["pair_count"]), "cross_split_pairs": 0},
        "utility": {"total": len(ids), "verdicts": counts(utility_verdicts),
                    "by_subject": {s: counts(subject_counts[s]) for s in sorted(requested)},
                    "by_source": {s: counts(v) for s, v in source_counts.items()},
                    "requested_subjects": len(requested), "evaluation_subjects": 42,
                    "subjects_with_accept": len(requested) - len(without_accept),
                    "subjects_without_accept": without_accept, "omitted_subjects": sorted(omitted),
                    "imported": imported, "new_reviews": len(ids) - imported,
                    "corrections": count(progress["corrected_review_records"]),
                    "top_five": top_five, "top_five_accept": sum(subject_counts[s]["accept"] for s in top_five),
                    "source_versions": [dict(source=s, commit=c) for s, c in source_versions],
                    "pool_sha256": hash_value(utility["provenance"]["pool_sha256"]),
                    "split_status": "not_frozen"},
    }


def verdict_table(rows):
    result = []
    for label, values in rows.items():
        cells = "".join(f'<td class="num {key}">{values[key]:,}</td>' for key in VERDICTS)
        result.append(f'<tr><th scope="row">{escape(label)}</th>{cells}<td class="num">{sum(values.values()):,}</td></tr>')
    return "".join(result)


def chart(values, total, label):
    parts, x = [], 0
    for key in VERDICTS:
        width = values[key] / total * 1000
        parts.append(f'<rect x="{x:.3f}" y="0" width="{width:.3f}" height="28" fill="{COLORS[key]}"/>')
        x += width
    description = " / ".join(f"{k} {values[k]:,}" for k in VERDICTS)
    return (f'<svg class="distribution" viewBox="0 0 1000 28" role="img" aria-label="{escape(label + ": " + description)}">'
            + "".join(parts) + '</svg><div class="chart-key">' + "".join(
                f'<span><i class="dot {k}"></i>{v} <b>{values[k]:,}</b> <small>{values[k]/total:.1%}</small></span>'
                for k, v in zip(VERDICTS, ("通过", "排除", "待复核"))) + '</div>')


def render_html(report: dict) -> str:
    target, utility, frozen = (report[k] for k in ("target", "utility", "frozen"))
    rows = []
    all_subjects = sorted(set(utility["by_subject"]) | set(utility["omitted_subjects"]))
    for subject in all_subjects:
        values = utility["by_subject"].get(subject, dict.fromkeys(VERDICTS, 0))
        state = "omitted" if subject in utility["omitted_subjects"] else "covered" if values["accept"] else "empty"
        label = {"covered": "有通过候选", "empty": "已审 · 无通过", "omitted": "未映射 · 暂缓"}[state]
        cells = "".join(f'<td class="num {key}">{values[key]:,}</td>' for key in VERDICTS)
        rows.append(f'<tr data-subject="{subject}" data-state="{state}" data-accept="{values["accept"]}">'
                    f'<th scope="row">{escape(subject.replace("_", " "))}</th>{cells}'
                    f'<td class="num">{sum(values.values()):,}</td><td><span class="status {state}">{label}</span></td></tr>')
    frozen_rows = "".join(f'<tr><th scope="row">{s}</th><td class="num">{v["train"]}</td>'
                          f'<td class="num">{v["dev"]}</td><td class="num">{sum(v.values())}</td></tr>'
                          for s, v in frozen["by_subject"].items())
    provenance = [("Synthetic WMDP", target["source_commit"], "TeunvdWeij/sandbagging")]
    repositories = {"eduqg": "hadifar/question-generation", "xiezhi": "MikeGu721/XiezhiBenchmark"}
    provenance += [(s["source"].upper(), s["commit"], repositories[s["source"]]) for s in utility["source_versions"]]
    source_rows = "".join(f'<tr><th scope="row"><a href="https://github.com/{repo}/tree/{commit}">{label}</a></th>'
                          f'<td><code>{commit}</code></td><td>许可待确认</td></tr>' for label, commit, repo in provenance)
    replacements = {
        "TARGET_CHART": chart(target["verdicts"], target["total"], "Target 审计结果"),
        "UTILITY_CHART": chart(utility["verdicts"], utility["total"], "Utility 审计结果"),
        "TARGET_ROWS": verdict_table(target["by_subject"]),
        "SOURCE_ROWS": verdict_table(utility["by_source"]), "FROZEN_ROWS": frozen_rows,
        "SUBJECT_ROWS": "".join(rows), "PROVENANCE_ROWS": source_rows,
        "TARGET_TOTAL": f'{target["total"]:,}', "UTILITY_TOTAL": f'{utility["total"]:,}',
        "TARGET_ACCEPT": f'{target["verdicts"]["accept"]:,}', "UTILITY_ACCEPT": f'{utility["verdicts"]["accept"]:,}',
        "TARGET_NEW": f'{target["new_reviews"]:,}', "UTILITY_NEW": f'{utility["new_reviews"]:,}',
        "TOP_SHARE": f'{utility["top_five_accept"] / utility["verdicts"]["accept"]:.1%}',
        "TOP_COUNT": f'{utility["top_five_accept"]:,}', "CORRECTIONS": str(utility["corrections"]),
        "SNAPSHOT": escape(report["snapshot_date"]),
    }
    html = TEMPLATE.read_text(encoding="utf-8")
    for key, value in replacements.items():
        html = html.replace("@@" + key + "@@", value)
    require(not re.search(r"@@[A-Z_]+@@", html), "unresolved report placeholder")
    return html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=CODE_ROOT / "reports")
    args = parser.parse_args()
    report = build_report(*(json.loads(p.read_text(encoding="utf-8")) for p in (TARGET, UTILITY, FROZEN)))
    report["input_sha256"] = {str(p.relative_to(CODE_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (TARGET, UTILITY, FROZEN)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, content in (("json", json.dumps(report, ensure_ascii=False, indent=2) + "\n"),
                            ("html", render_html(report))):
        path = args.output_dir / ("e1-data-report." + suffix)
        path.write_text(content, encoding="utf-8")
    print(json.dumps({"target": report["target"]["total"], "utility": report["utility"]["total"],
                      "training_ready": report["training_ready"]}))


if __name__ == "__main__":
    main()
