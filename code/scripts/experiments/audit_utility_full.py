#!/usr/bin/env python3
"""Resumable local-only full utility review; no model, training or network calls."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_utility_review as review


CODE_ROOT = review.CODE_ROOT
POOL = CODE_ROOT / "data/experiment1/utility-full-audit/pool.json"
DB_PATH = CODE_ROOT / "runtime/experiment1/utility-full-audit/audit.sqlite3"
OUTPUT = CODE_ROOT / "results/published/experiment1/utility-full-audit"
REPORT = CODE_ROOT / "reports/e1-utility-full-audit.md"
QUEUE_ORDER = "CASE source WHEN 'xiezhi' THEN 0 ELSE 1 END, subject, rank"


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def connect(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY, family_hash TEXT UNIQUE NOT NULL,
            subject TEXT NOT NULL, source TEXT NOT NULL, rank TEXT NOT NULL,
            payload TEXT NOT NULL, state TEXT NOT NULL, owner TEXT,
            decision TEXT, imported INTEGER NOT NULL DEFAULT 0,
            claimed_at REAL, completed_at REAL
        );
        CREATE INDEX IF NOT EXISTS queue_idx ON items(state, source, subject, rank);
        CREATE TABLE IF NOT EXISTS review_history (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL,
            owner TEXT NOT NULL, decision TEXT NOT NULL, reason TEXT NOT NULL,
            reopened_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS review_corrections (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL,
            owner TEXT NOT NULL, old_decision TEXT NOT NULL,
            corrected_decision TEXT NOT NULL, reason TEXT NOT NULL,
            corrected_at REAL NOT NULL
        );
    """)
    return db


def metadata(db):
    row = db.execute("SELECT value FROM metadata WHERE key='provenance'").fetchone()
    if row is None:
        raise ValueError("Run init first")
    return json.loads(row[0])


def initialize(db, pool):
    raw = pool.read_bytes()
    data = json.loads(raw)
    provenance = {"pool_sha256": hashlib.sha256(raw).hexdigest(),
                  "source_provenance": data["provenance"],
                  "requested_subjects": data["requested_subjects"],
                  "omitted_subjects": data["omitted_subjects"]}
    items = data["items"]
    for item in items:
        if item["imported_decision"] is not None:
            review.validate_decisions({"items": [item]}, [item["imported_decision"]])
    db.execute("BEGIN IMMEDIATE")
    try:
        old = db.execute("SELECT value FROM metadata WHERE key='provenance'").fetchone()
        if old:
            if json.loads(old[0]) != provenance:
                raise ValueError("Frozen pool provenance changed; refusing to reuse queue")
        else:
            for item in items:
                decision = item["imported_decision"]
                imported = decision is not None
                db.execute("""INSERT INTO items
                    (id, family_hash, subject, source, rank, payload, state, owner,
                     decision, imported, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (item["id"], item["family_hash"], item["subject"], item["source"],
                     hashlib.sha256(("utility-full-v1:" + item["family_hash"]).encode()).hexdigest(),
                     encode(item), "done" if imported else "pending",
                     "imported-batch-v1" if imported else None,
                     encode(decision) if imported else None, int(imported), time.time() if imported else None))
            db.execute("INSERT INTO metadata VALUES ('provenance', ?)", (encode(provenance),))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    return status(db)


def claim(db, owner, limit=40):
    metadata(db)
    require_running(db)
    if not owner.strip() or not 1 <= limit <= 100:
        raise ValueError("Owner must be nonempty and batch limit 1..100")
    db.execute("BEGIN IMMEDIATE")
    try:
        rows = db.execute(f"SELECT * FROM items WHERE state='claimed' AND owner=? ORDER BY {QUEUE_ORDER}", (owner,)).fetchall()
        if not rows:
            rows = db.execute(f"SELECT * FROM items WHERE state='pending' ORDER BY {QUEUE_ORDER} LIMIT ?", (limit,)).fetchall()
            db.executemany("UPDATE items SET state='claimed', owner=?, claimed_at=? WHERE id=?",
                           [(owner, time.time(), row["id"]) for row in rows])
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    fields = ("id", "subject", "source", "tier", "question", "choices", "answer")
    return [{key: json.loads(row["payload"])[key] for key in fields} for row in rows]


def complete(db, owner, decisions):
    metadata(db)
    require_running(db)
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("Nonempty decision list required")
    db.execute("BEGIN IMMEDIATE")
    try:
        ids = [decision.get("id") for decision in decisions]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate decision IDs")
        rows = []
        for item_id in ids:
            row = db.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
            if row is None or row["owner"] != owner or row["state"] not in {"claimed", "done"}:
                raise ValueError("Unknown item, incorrect owner, or item not claimed")
            rows.append(row)
        review.validate_decisions({"items": [json.loads(row["payload"]) for row in rows]}, decisions)
        changed = 0
        for row, decision in zip(rows, decisions):
            encoded = encode(decision)
            if row["state"] == "done":
                if row["decision"] != encoded:
                    raise ValueError("Conflicting previously completed review")
            else:
                db.execute("UPDATE items SET state='done', decision=?, completed_at=? WHERE id=?",
                           (encoded, time.time(), row["id"]))
                changed += 1
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    return {"completed_now": changed, **status(db)}


def status(db):
    metadata(db)
    rows = db.execute("SELECT subject, source, state, owner, decision, imported FROM items").fetchall()
    states = Counter(row["state"] for row in rows)
    verdicts = Counter(json.loads(row["decision"])["verdict"] for row in rows if row["decision"])
    timing = db.execute("""SELECT MIN(claimed_at), MAX(completed_at)
        FROM items WHERE imported=0""").fetchone()
    hold = db.execute("SELECT value FROM metadata WHERE key='qa_hold'").fetchone()
    as_utc = lambda value: datetime.fromtimestamp(value, timezone.utc).isoformat() if value else None
    return {"total": len(rows), "done": states["done"], "remaining": len(rows) - states["done"],
            "pending": states["pending"], "claimed": states["claimed"],
            "imported": sum(row["imported"] for row in rows), "verdicts": dict(verdicts),
            "qa_hold": json.loads(hold[0]) if hold else None,
            "reopened_review_records": db.execute("SELECT COUNT(*) FROM review_history").fetchone()[0],
            "corrected_review_records": db.execute("SELECT COUNT(*) FROM review_corrections").fetchone()[0],
            "first_new_review_claim_utc": as_utc(timing[0]),
            "latest_new_review_completion_utc": as_utc(timing[1]),
            "review_wall_seconds": round(timing[1] - timing[0], 2) if all(timing) else None,
            "active_owners": dict(Counter(row["owner"] for row in rows if row["state"] == "claimed"))}


def require_running(db):
    hold = db.execute("SELECT value FROM metadata WHERE key='qa_hold'").fetchone()
    if hold and json.loads(hold[0]):
        raise ValueError("QA hold active: do not claim or complete until reconciliation")


def set_hold(db, reason):
    metadata(db)
    db.execute("INSERT OR REPLACE INTO metadata VALUES ('qa_hold', ?)", (encode(reason),))
    return status(db)


def reopen(db, item_ids, reason):
    metadata(db)
    if not reason.strip() or not item_ids or len(set(item_ids)) != len(item_ids):
        raise ValueError("Reopen needs a reason and distinct item IDs")
    db.execute("BEGIN IMMEDIATE")
    try:
        for item_id in item_ids:
            row = db.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
            if row is None or row["state"] != "done" or row["imported"]:
                raise ValueError("Only completed, non-imported reviews may be reopened")
            db.execute("INSERT INTO review_history (id, owner, decision, reason, reopened_at) VALUES (?, ?, ?, ?, ?)",
                       (item_id, row["owner"], row["decision"], reason, time.time()))
            db.execute("UPDATE items SET state='pending', owner=NULL, decision=NULL, claimed_at=NULL, completed_at=NULL WHERE id=?", (item_id,))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    return status(db)


def correct(db, decisions, reason):
    metadata(db)
    hold = db.execute("SELECT value FROM metadata WHERE key='qa_hold'").fetchone()
    if not hold or not json.loads(hold[0]) or not reason.strip() or not decisions:
        raise ValueError("Explicit QA hold and correction reason required")
    db.execute("BEGIN IMMEDIATE")
    try:
        ids = [decision["id"] for decision in decisions]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate correction IDs")
        rows = [db.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone() for item_id in ids]
        if any(row is None or row["state"] != "done" or row["imported"] for row in rows):
            raise ValueError("Only completed non-imported reviews may be corrected")
        review.validate_decisions({"items": [json.loads(row["payload"]) for row in rows]}, decisions)
        changed = 0
        for row, decision in zip(rows, decisions):
            new = encode(decision)
            if row["decision"] == new:
                continue
            db.execute("""INSERT INTO review_corrections
                (id, owner, old_decision, corrected_decision, reason, corrected_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (row["id"], row["owner"], row["decision"], new, reason, time.time()))
            db.execute("UPDATE items SET decision=? WHERE id=?", (new, row["id"]))
            changed += 1
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    return {"corrected_now": changed, **status(db)}


def publish(db, output=OUTPUT, report_path=REPORT):
    provenance = metadata(db)
    db.execute("BEGIN")
    try:
        progress = status(db)
        rows = db.execute("SELECT * FROM items ORDER BY subject, id").fetchall()
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    by_subject = {subject: Counter() for subject in provenance["requested_subjects"]}
    by_source, reasons, entries = {}, Counter(), []
    for row in rows:
        payload = json.loads(row["payload"])
        decision = json.loads(row["decision"]) if row["decision"] else None
        state = decision["verdict"] if decision else "unreviewed"
        by_subject[row["subject"]][state] += 1
        by_source.setdefault(row["source"], Counter())[state] += 1
        entry = {key: payload[key] for key in ("id", "family_hash", "stable_id", "subject", "source", "tier")}
        entry.update(imported=bool(row["imported"]), state="done" if decision else "unreviewed")
        if decision:
            entry.update({key: decision[key] for key in sorted(review.PUBLIC_DECISION_FIELDS)})
            reasons[decision["reason_code"]] += 1
        entries.append(entry)
    artifact = {"schema_version": 1, "status": "qa_hold" if progress["qa_hold"] else "complete" if not progress["remaining"] else "in_progress",
                "review_type": "model_content_first_pass_not_expert_verified",
                "provenance": provenance, "progress": progress, "by_subject": by_subject,
                "by_source": by_source, "reason_counts": reasons, "entries": entries,
                "subjects_with_accept": sum(counts["accept"] > 0 for counts in by_subject.values()),
                "subjects_without_accept": sorted(subject for subject, counts in by_subject.items() if not counts["accept"])}
    output.mkdir(parents=True, exist_ok=True)
    (output / "status.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    verdicts = progress["verdicts"]
    lines = ["# E1 Utility 全量审核", "",
             f"状态：**{artifact['status']}**。总量 **{progress['total']}** 个不同规范化题干；已完成 **{progress['done']}**，剩余 **{progress['remaining']}**。",
             f"其中复用小批审核 {progress['imported']} 道；accept **{verdicts.get('accept', 0)}** / reject **{verdicts.get('reject', 0)}** / review **{verdicts.get('review', 0)}**。", "",
             f"当前有初审通过候选的主科目：**{artifact['subjects_with_accept']} / {len(by_subject)}**。review 表示已完成初审但仍需专业复核，不计入可用候选。", "",
             f"首个新题领取时间：{progress['first_new_review_claim_utc']}；最新新题完成时间：{progress['latest_new_review_completion_utc']}。两者相隔 {progress['review_wall_seconds']} 秒（并行墙钟时间，不是专家工时）。", "",
             f"QA hold：{progress['qa_hold'] or '无'}。重新打开的初审记录数：{progress['reopened_review_records']}；原判定保存在本地 review_history，不静默抹除。", "",
             f"显式纠错记录数：{progress['corrected_review_records']}；旧、新判定及原因保存在本地 review_corrections。领取批次恢复顺序已统一，后续以 ID 关联判断而不依赖行位置。", "",
             "## 执行边界", "",
             "- 用户授权的 37 科候选取并集，5 科缺口暂缓；官方 42 科评测范围不变。",
             "- 同规范化题干仅选一个代表；同题干其他选项变体不因此获得内容审核认证。",
             "- 原 108 题保留其当时的源代表、主科目和判定；不自动改科目使原 reject 变为 accept。新题优先领域对齐映射，其余采用固定邻域主科目。主科目审核不等于所有可能映射都通过。",
             "- 每题为模型初审，gold 只记 plausible/uncertain/not_checked，不是专家验证。没有修补题目、生成题目、调用 target/weak、训练、评测或读取 sealed 题目。",
             "- 逐题正文与理由只存 ignored 本地缓存和 SQLite；公开聚合仅包含 ID、hash 与枚举判定。",
             "- 许可、语义去污染、真实来源题族隔离与训练/dev 冻结仍未完成。本次不导出训练集。", "",
             "## 分科进度", "", "| 主科目 | accept | reject | review | 未审 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for subject, counts in sorted(by_subject.items()):
        lines.append(f"| {subject} | {counts['accept']} | {counts['reject']} | {counts['review']} | {counts['unreviewed']} |")
    lines += ["", "## 来源与原因", "", "| 来源 | accept | reject | review | 未审 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for source, counts in sorted(by_source.items()):
        lines.append(f"| {source} | {counts['accept']} | {counts['reject']} | {counts['review']} | {counts['unreviewed']} |")
    lines += ["", "各来源承担的科目与难度不同，此处不是对数据集总体质量的可比估计。", "",
              "| reason_code | 数量 |", "| --- | ---: |"]
    lines += [f"| {reason} | {count} |" for reason, count in sorted(reasons.items(), key=lambda pair: (-pair[1], pair[0]))]
    lines += ["", "当前尚无通过候选的主科目：" + (", ".join(artifact["subjects_without_accept"]) or "无") + "。全量仅穷尽当前映射、当前代表题，不表示这些科目没有其他可用来源。", ""]
    lines += ["", "## 续接", "", "```sh",
              "python3 code/scripts/experiments/audit_utility_full.py status",
              "python3 code/scripts/experiments/audit_utility_full.py claim --owner utility-worker --limit 40",
              "python3 code/scripts/experiments/audit_utility_full.py complete --owner utility-worker --decisions LOCAL_DECISIONS.json",
              "python3 code/scripts/experiments/audit_utility_full.py publish", "```", "",
              "相同 owner 在有未完成领取时会取回原批次。完成操作可安全重试；来源或已完成判定变化会报错，不静默覆盖。脚本仅管理记录，不自行调用模型。", ""]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("status")
    sub.add_parser("publish")
    hold_parser = sub.add_parser("hold")
    hold_parser.add_argument("--reason", required=True)
    sub.add_parser("resume")
    reopen_parser = sub.add_parser("reopen")
    reopen_parser.add_argument("--ids", type=Path, required=True)
    reopen_parser.add_argument("--reason", required=True)
    correct_parser = sub.add_parser("correct")
    correct_parser.add_argument("--decisions", type=Path, required=True)
    correct_parser.add_argument("--reason", required=True)
    claim_parser = sub.add_parser("claim")
    claim_parser.add_argument("--owner", required=True)
    claim_parser.add_argument("--limit", type=int, default=40)
    complete_parser = sub.add_parser("complete")
    complete_parser.add_argument("--owner", required=True)
    complete_parser.add_argument("--decisions", type=Path, required=True)
    args = parser.parse_args()
    db = connect(DB_PATH)
    try:
        if args.command == "init":
            result = initialize(db, POOL)
        elif args.command == "claim":
            result = claim(db, args.owner, args.limit)
        elif args.command == "complete":
            result = complete(db, args.owner, json.loads(args.decisions.read_text()))
        elif args.command == "publish":
            result = publish(db)
        elif args.command == "hold":
            result = set_hold(db, args.reason)
        elif args.command == "resume":
            result = set_hold(db, None)
        elif args.command == "reopen":
            result = reopen(db, json.loads(args.ids.read_text()), args.reason)
        elif args.command == "correct":
            result = correct(db, json.loads(args.decisions.read_text()), args.reason)
        else:
            result = status(db)
        print(json.dumps(result, ensure_ascii=False))
    finally:
        db.close()


if __name__ == "__main__":
    main()
