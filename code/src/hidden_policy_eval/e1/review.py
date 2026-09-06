"""E1 utility audit decision validation, independent of report rendering."""

VERDICTS = {"accept", "reject", "review"}
REASONS = {
    "clear_basic_fact", "subject_mismatch", "missing_context", "ambiguous",
    "gold_mismatch", "specialist_uncertain", "scope_overlap", "language_issue",
    "near_duplicate", "level_mismatch",
}
DECISION_FIELDS = {"id", "verdict", "reason_code", "gold_status", "subject_fit",
                   "context_status", "scope_status", "note"}
PUBLIC_DECISION_FIELDS = DECISION_FIELDS - {"note"}


def validate_decisions(batch, decisions):
    items = {item["id"]: item for item in batch["items"]}
    if len(items) != len(batch["items"]):
        raise ValueError("Duplicate batch item IDs")
    seen = set()
    for decision in decisions:
        if set(decision) != DECISION_FIELDS:
            raise ValueError("Unexpected decision fields")
        item_id = decision["id"]
        if item_id not in items or item_id in seen:
            raise ValueError("Unknown or duplicate decision ID")
        seen.add(item_id)
        domains = {
            "verdict": VERDICTS, "reason_code": REASONS,
            "gold_status": {"plausible", "uncertain", "not_checked"},
            "subject_fit": {"yes", "no", "uncertain"},
            "context_status": {"self_contained", "missing", "ambiguous"},
            "scope_status": {"nonoverlap", "overlap", "uncertain"},
        }
        if any(decision[field] not in allowed for field, allowed in domains.items()):
            raise ValueError("Invalid review enum")
        if not isinstance(decision["note"], str) or not decision["note"].strip():
            raise ValueError("Review requires a local rationale")
        if decision["verdict"] == "accept":
            required = {"gold_status": "plausible", "subject_fit": "yes",
                        "context_status": "self_contained", "scope_status": "nonoverlap",
                        "reason_code": "clear_basic_fact"}
            if any(decision[key] != value for key, value in required.items()):
                raise ValueError("Accept requires every content-review check to pass")
    if seen != set(items):
        raise ValueError(f"Incomplete review: {len(items) - len(seen)} missing")
