"""Executable, cache-only known input and routing controls for E3."""

from __future__ import annotations

from collections import defaultdict
import copy
from pathlib import Path

from .analysis import LEVELS, _load_round, _public
from .probes import score_records


SCHEMA = "hidden-policy-e3-known-controls-v1"


class CacheMiss(RuntimeError):
    """A control requires an input that has never been evaluated."""


def invalidate_known_prefix(messages: list[dict], level: str, policy: dict) -> tuple[list[dict], int]:
    """Replace a complete known condition only at the start of a user message."""
    replacements = ([(policy["g0_trigger"], policy["g0_neutral"])] if level.startswith("G0") else
                    [(pair["on"], pair["off"]) for pair in policy["g1_contexts"]["train"]])
    replacements.sort(key=lambda pair: (-len(pair[0]), pair[0]))
    result, count = copy.deepcopy(messages), 0
    for message in result:
        if message["role"] != "user":
            continue
        content = message["content"]
        for old, new in replacements:
            if content.startswith(old) and (len(content) == len(old) or content[len(old)] == "\n"):
                message["content"] = new + content[len(old):]
                count += 1
                break
    return result, count


class _CacheBackend:
    def __init__(self, r, run: Path, registry: dict, settings: dict, adapter: Path | None, name: str):
        def unavailable(*args, **kwargs):
            raise CacheMiss("Required input is absent from the verified prediction cache; inference is disabled")

        self.r, self.name, self.calls = r, name, []
        self.predictor = r.CachedPredictor(run, registry["models"]["target"], settings,
                                          registry["runtime"], adapter, factory=unavailable)
        self.predictor.ensure_loaded = unavailable

    def __call__(self, messages):
        before = self.predictor.generated
        responses = self.predictor(messages)
        if self.predictor.generated != before or self.predictor.generated != 0:
            raise ValueError("Known controls must never generate new model predictions")
        self.calls.append({"backend": self.name, "requests": len(messages),
                           "input_batch_sha256": self.r.digest(messages),
                           "response_batch_sha256": self.r.digest(responses)})
        return responses

    def close(self):
        self.predictor.close()


class KnownRouter:
    """Explicitly route the same input to the retained policy or to BASE."""
    def __init__(self, policy_backend, base_backend):
        self.policy, self.base, self.mode = policy_backend, base_backend, "open"
        self.trace = []

    def set_mode(self, mode: str):
        if mode not in ("open", "blocked"):
            raise ValueError("Known router mode must be open or blocked")
        self.mode = mode

    def __call__(self, messages):
        backend = self.policy if self.mode == "open" else self.base
        responses = backend(messages)
        self.trace.append({"mode": self.mode, **backend.calls[-1]})
        return responses


class BaseRollbackOracle:
    """A BASE-only served object, with no policy adapter route to reopen."""
    def __init__(self, base_backend):
        self.base = base_backend

    def __call__(self, messages):
        return self.base(messages)


def _verify_adapter(code: Path, adapter: dict, r) -> Path:
    relative = Path(adapter["adapter"])
    source_relative = Path(adapter["source_job"])
    if (relative.is_absolute() or source_relative.is_absolute()
            or ".." in relative.parts or ".." in source_relative.parts):
        raise ValueError("Invalid frozen source adapter path")
    path = code / relative
    if r.adapter_hash(path) != adapter["adapter_sha256"]:
        raise ValueError("Known-control source adapter weights changed")
    source = r.read_json(code / source_relative)["config"]
    expected = adapter["config"]
    fields = ("g0_trigger", "g0_neutral") if adapter["level"].startswith("G0") else ("g1_contexts",)
    if (source["data"] != expected["data"]
            or any(source["policy"][field] != expected["policy"][field] for field in fields)
            or source["policy"].get("u1_answer_mode", "parsed") != expected["policy"].get("u1_answer_mode", "parsed")
            or (adapter["level"].endswith("U0") and not adapter["is_sham"]
                and source["policy"]["fixed_action"] != expected["policy"]["fixed_action"])):
        raise ValueError("Known-control source policy differs from its frozen registry")
    return path


def _source_inputs(run: Path, plan: dict, r) -> dict:
    references = {}
    for entry in plan["jobs"]:
        cell = run / "jobs" / entry["name"]
        job = r.read_json(cell / "job.json")
        result = r.read_json(cell / "result.json")
        if (r.digest(job) != entry["job_sha256"] or result["job_sha256"] != entry["job_sha256"]
                or r.digest(result["payload"]) != result["payload_sha256"]):
            raise ValueError("Source result changed during known-control verification")
        payload = result["payload"]
        if job["method"]["kind"] != "none" or payload["method"] != "unmodified":
            raise ValueError("Known controls require an unmodified-policy and BASE reference round")
        by_name = {row["name"]: row for row in payload["evaluations"]}
        for view in job["views"]:
            records = r.read_json(run / view["records"])
            if r.digest(records) != view["records_sha256"]:
                raise ValueError("Known-control source inputs changed")
            references[view["name"]] = {"records": records, "records_sha256": view["records_sha256"],
                                        "responses_sha256": by_name[view["name"]]["responses_sha256"]}
    return references


def _accuracy(scores: dict) -> dict:
    return {f"{row['scope']}_{'on' if row['gate_on'] else 'off'}":
            {"correct": row["correct"], "total": row["total"], "accuracy_pct": 100 * row["accuracy"]}
            for row in scores["groups"] if row["probe"] == "canonical"}


def _paired_deltas(left: dict, right: dict) -> list[dict]:
    """Pair the same visible requests, allowing a declared input transformation."""
    a = {row["id"]: row for row in left["outcomes"]}
    b = {row["id"]: row for row in right["outcomes"]}
    if set(a) != set(b):
        raise ValueError("Known-control comparisons must cover identical visible request IDs")
    groups = defaultdict(list)
    for record_id, row in a.items():
        peer = b[record_id]
        if any(row[key] != peer[key] for key in ("item_id", "probe", "family", "scope", "gate_on")):
            raise ValueError("Known-control pairing metadata changed")
        if row["probe"] not in ("canonical", "alternative"):
            continue
        family = row["family"] if row["probe"] == "alternative" else "familiar-item-assigned"
        groups[row["probe"], family, row["scope"], row["gate_on"]].append((row, peer))
    result = []
    for (probe, family, scope, on), rows in sorted(groups.items()):
        left_count = sum(row["correct"] for row, _ in rows)
        right_count = sum(row["correct"] for _, row in rows)
        result.append({"probe": probe, "family": family, "scope": scope, "gate_on": on,
                       "total": len(rows), "underlying_items": len({row["item_id"] for row, _ in rows}),
                       "left_correct": left_count, "right_correct": right_count,
                       "delta_pp": 100 * (left_count - right_count) / len(rows),
                       "left_only_correct": sum(row["correct"] and not peer["correct"] for row, peer in rows),
                       "right_only_correct": sum(peer["correct"] and not row["correct"] for row, peer in rows)})
    return result


def _summary(scores: dict, sham: dict) -> dict:
    return {"accuracy": _accuracy(scores),
            "alternative_groups": [row for row in scores["by_family"] if row["probe"] == "alternative"],
            "minus_unmodified_sham": _paired_deltas(scores, sham)}


def _evaluate_level(level, adapter, sham_adapter, sources, registry, settings, run, code, r):
    path, sham_path = _verify_adapter(code, adapter, r), _verify_adapter(code, sham_adapter, r)
    original, sham_reference, base_reference = sources[level], sources["SHAM-for-" + level], sources["BASE-for-" + level]
    records = original["records"]
    if any(reference["records_sha256"] != original["records_sha256"] for reference in (sham_reference, base_reference)):
        raise ValueError("Policy, SHAM and BASE controls must receive the same frozen records")
    policy = adapter["config"]["policy"]
    backends = {"policy": _CacheBackend(r, run, registry, settings, path, "retained-policy"),
                "sham": _CacheBackend(r, run, registry, settings, sham_path, "unmodified-sham"),
                "base": _CacheBackend(r, run, registry, settings, None, "base"),
                "oracle": _CacheBackend(r, run, registry, settings, None, "base-only-oracle")}
    try:
        messages = [row["messages"] for row in records]
        responses = {}
        for name, reference in (("policy", original), ("sham", sham_reference), ("base", base_reference)):
            responses[name] = backends[name](messages)
            if r.digest(responses[name]) != reference["responses_sha256"]:
                raise ValueError("Cache-only predictions differ from the verified source result")
        score = lambda rows, text: score_records(rows, text, level, policy["fixed_action"])
        before, sham = score(records, responses["policy"]), score(records, responses["sham"])

        transformed, changed = [], []
        for record in records:
            revised, count = invalidate_known_prefix(record["messages"], level, policy)
            transformed.append({**record, "messages": revised})
            changed.append(count)
        invalidated_text = backends["policy"]([row["messages"] for row in transformed])
        invalidated = score(transformed, invalidated_text)
        changed_groups = defaultdict(int)
        for row, count in zip(records, changed):
            if count:
                changed_groups[row["probe"], row["family"], row["scope"], row["gate_on"]] += 1

        router = KnownRouter(backends["policy"], backends["base"])
        opened = router(messages)
        router.set_mode("blocked")
        blocked_text = router(messages)
        router.set_mode("open")
        restored_text = router(messages)
        if opened != responses["policy"] or restored_text != opened or blocked_text != responses["base"]:
            raise ValueError("Known router did not select and restore the declared backends")
        blocked, restored = score(records, blocked_text), score(records, restored_text)
        oracle = BaseRollbackOracle(backends["oracle"])
        oracle_text = oracle(messages)
        if oracle_text != responses["base"]:
            raise ValueError("BASE-only oracle differs from the fixed BASE reference")
        oracle_score = score(records, oracle_text)
        for source, source_path in ((adapter, path), (sham_adapter, sham_path)):
            if r.adapter_hash(source_path) != source["adapter_sha256"]:
                raise ValueError("Known control changed source adapter weights")
        return {"level": level, "status": "complete", "execution": "actual transformed/routed inputs resolved through cache-only predictors",
                "new_predictions": sum(backend.predictor.generated for backend in backends.values()),
                "source_records_sha256": original["records_sha256"],
                "original": _summary(before, sham),
                "A_known_input_invalidation": {**_summary(invalidated, sham), "weights_unchanged": True,
                    "matching_rule": "complete known on-condition at user-message start with newline/end boundary",
                    "changed_records": sum(count > 0 for count in changed), "changed_messages": sum(changed),
                    "changed_groups": [{"probe": probe, "family": family, "scope": scope, "gate_on": on, "records": count}
                                       for (probe, family, scope, on), count in sorted(changed_groups.items())],
                    "transformed_input_sha256": r.digest([row["messages"] for row in transformed]),
                    "post_minus_original": _paired_deltas(invalidated, before),
                    "scope": "known surface prefix invalidation, not a learned projection or universal trigger removal"},
                "D_known_router": {"blocked": _summary(blocked, sham), "restored": _summary(restored, sham),
                    "trace": router.trace, "weights_unchanged": True,
                    "retained_policy_adapter_sha256": adapter["adapter_sha256"], "route_to_policy_available": True,
                    "restored_outputs_match_original": True, "restored_output_matches": len(records),
                    "restored_responses_sha256": r.digest(restored_text),
                    "restored_minus_blocked": _paired_deltas(restored, blocked),
                    "scope": "explicit policy/BASE backend selection; not a QES reproduction"},
                "BASE_rollback_oracle": {**_summary(oracle_score, sham), "served_model": registry["models"]["target"],
                    "deployed_adapter_sha256": None, "route_to_policy_available": False,
                    "deployed_components": ["base-model-only"],
                    "scope": "known BASE rollback oracle; not evidence that an unknown mechanism was removed"},
                "cache_calls": [call for backend in backends.values() for call in backend.calls]}
    finally:
        for backend in backends.values():
            backend.close()


def evaluate_controls(study: Path, round_name: str, r_module) -> dict:
    """Execute known controls on existing CPU cache entries; never infer on a miss."""
    if round_name not in ("r0", "r0b"):
        raise ValueError("Known controls are restricted to completed R0/R0b reference rounds")
    study, r = Path(study), r_module
    loaded = _load_round(study, round_name, r)
    if loaded["pending"]:
        raise ValueError("Complete the reference round before executing known controls")
    code = Path(r.CODE_DIR)
    registry = r.read_json(study / "registry.json")
    adapters = {adapter["name"]: adapter for adapter in registry["adapters"]}
    sources = _source_inputs(study / round_name, loaded["plan"], r)
    required = {name for level in LEVELS for name in (level, "SHAM-for-" + level, "BASE-for-" + level)}
    if not required <= set(sources):
        raise ValueError("Known controls need all primary, matched SHAM and BASE views")
    config = loaded["plan"]["identity"]["config"]
    results = []
    for level in LEVELS:
        try:
            result = _evaluate_level(level, adapters[level], adapters["SHAM-for-" + level], sources,
                                     registry, config["evaluation"], study / round_name, code, r)
        except CacheMiss as error:
            result = {"level": level, "status": "no_data_cache_miss", "new_predictions": 0,
                      "reason": str(error), "gpu_fallback_allowed": False}
        results.append(result)
    output = {"schema": SCHEMA, "study": study.name, "round": round_name,
              "status": "complete" if all(row["status"] == "complete" for row in results) else "incomplete",
              "source_protocol_sha256": loaded["plan"]["identity_sha256"],
              "control_implementation_sha256": r.file_hash(Path(__file__)),
              "evaluation_mode": "cache-replayed known control with executable input transformation/backend routing",
              "new_predictions": 0, "gpu_loading_allowed": False, "results": results,
              "limitations": ["These are operational known controls, not newly trained repair methods.",
                              "A modifies only a known literal top-of-message condition; other forms may remain active.",
                              "D switches between retained policy and BASE backends and is not a QES implementation.",
                              "The BASE-only oracle has no deployed policy route, but does not prove general mechanism removal.",
                              "Blocked routing and BASE rollback may have identical visible answers despite different deployments.",
                              "Accuracy metadata describes the original visible request; A's actual model input is transformed.",
                              "All raw inputs and answers remain in existing private records/prediction caches.",
                              "No C classification or new inference is performed."], "official_q4_exposed": False}
    _public(output)
    return output
