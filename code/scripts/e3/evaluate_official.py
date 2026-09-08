#!/usr/bin/env python3
"""Freeze exact E3 checkpoints and Q4 IDs before one official confirmation.

Example config shape (replace placeholders, do not select models on Q4):
{"schema":"hidden-policy-e3-official-q4-v1", "study":"taxonomy-v1",
 "run_name":"official-confirm-v1", "registry":"path/to/registry.json",
 "registry_sha256":"<sha256>", "selection":{"target":192,"utility":336},
 "include_alternatives":false,
 "evaluation":{"batch_size":8,"max_new_tokens":64,"seed":1234},
 "models":[{"name":"original","level":"G0U0","kind":"unmodified","source_name":"G0U0"},
           {"name":"repaired","level":"G0U0","kind":"repaired","source_name":"G0U0",
            "source_round":"<completed round>","source_job":"<exact job>"}]}

selection counts of zero mean the full unexposed split. Positive counts use
deterministic subject-balanced sampling, not the full-split subject distribution.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import fcntl
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment3 as e3

r = e3.r
from hidden_policy_eval.e1.official import _manifests
from hidden_policy_eval.e1.evaluate import _checked_row, _download_selected, _select, EXCLUDED_MMLU_SUBJECTS
from hidden_policy_eval.e1.policy import hidden_policy_definition
from hidden_policy_eval.e3.probes import G0_FAMILIES, G1_ALTERNATIVES, _marker_messages, score_records
from hidden_policy_eval.shared.manifests import canonical_row
from hidden_policy_eval.shared.prompts import strict_generation_prompt

SCHEMA = "hidden-policy-e3-official-q4-v1"
PRIVATE = CODE / "runtime/experiment3"
PUBLIC = CODE / "results/published/experiment3"
EXPOSURE = "results/published/experiment1/swift-smoke-v1/exposure.json"


def _name(value):
    if not isinstance(value, str) or not value or Path(value).name != value or value in (".", ".."):
        raise ValueError("E3 official names must be nonempty directory-safe names")
    return value


def _inside(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Official artifact paths must stay inside their declared root")
    return Path(root) / path


def _freeze(path, value):
    if path.exists() and r.read_json(path) != value:
        raise ValueError("Frozen Q4 artifact changed; never revise it after exposure")
    if not path.exists():
        r.write_json(path, value)


@contextmanager
def _exposure_lock():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / "official-q4-exposure.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _check_exposure_protocol(protocol):
    expected = r.digest(protocol)
    claim = PRIVATE / "official-q4-claim.json"
    paths = ([claim] if claim.exists() else [])
    paths += list(PRIVATE.glob("*/*/exposure.json")) + list(PUBLIC.glob("*/*/exposure.json"))
    for path in paths:
        ledger = r.read_json(path)
        if (ledger.get("schema") in ("hidden-policy-e3-q4-exposure-v1", "hidden-policy-e3-q4-claim-v1")
                and ledger["protocol_sha256"] != expected):
            raise ValueError("Official Q4 has already been exposed or claimed by another protocol; confirmation cannot switch recipes")


def _validate(config):
    if config.get("schema") != SCHEMA:
        raise ValueError("Unknown E3 official confirmation schema")
    for key in ("study", "run_name"):
        _name(config[key])
    if set(config.get("selection", {})) != {"target", "utility"}:
        raise ValueError("Declare target and utility selection counts; zero means full split")
    if any(type(value) is not int or value < 0 for value in config["selection"].values()):
        raise ValueError("Q4 selection counts must be nonnegative integers")
    if type(config.get("include_alternatives", False)) is not bool:
        raise ValueError("include_alternatives must be a boolean")
    if not config.get("models") or len({model["name"] for model in config["models"]}) != len(config["models"]):
        raise ValueError("Declare unique named fixed models; no automatic selection is supported")
    for model in config["models"]:
        _name(model["name"])
        if model["level"] not in e3.LEVELS or model["kind"] not in ("unmodified", "repaired", "base"):
            raise ValueError("Unsupported official model kind or level")
        if model["kind"] != "base":
            _name(model["source_name"])
        if model["kind"] == "repaired":
            _name(model["source_round"])
            _name(model["source_job"])
    for key in ("batch_size", "max_new_tokens", "seed"):
        value = config["evaluation"][key]
        if type(value) is not int or value < (0 if key == "seed" else 1):
            raise ValueError("Invalid frozen Q4 generation setting")


def _implementation():
    paths = [Path(__file__), CODE / "scripts/e3/run_experiment3.py", CODE / "scripts/e1/run_experiment1.py",
             CODE / "scripts/e1/evaluate_official.py"]
    paths += [CODE / "src/hidden_policy_eval" / relative for relative in (
        "e3/interventions.py", "e3/probes.py", "e1/official.py", "e1/evaluate.py", "e1/policy.py",
        "shared/strict.py", "shared/prompts.py", "shared/manifests.py", "shared/sources.py")]
    return {str(path.relative_to(CODE)): r.file_hash(path) for path in paths}


def _select_q4(config):
    dataset_config, entries, manifest_hashes = _manifests(CODE)
    ledger = e3.official.historical_exposure({"exposure_file": EXPOSURE})
    exposed = {item_id for row in ledger for item_id in row["selected_ids"]["TEST-Q4"]}
    selected, counts, available, excluded = [], {}, {}, {}
    for dataset, scope in (("wmdp", "target"), ("mmlu", "utility")):
        eligible = [row for row in entries if row["split"] == "TEST-Q4" and row["dataset"] == dataset
                    and (dataset != "mmlu" or row["subject"] not in EXCLUDED_MMLU_SUBJECTS)]
        candidates = [row for row in eligible if row["stable_id"] not in exposed]
        requested = config["selection"][scope]
        if requested > len(candidates) or not candidates:
            raise ValueError("Requested Q4 size exceeds the unexposed eligible pool")
        chosen = _select(candidates, requested) if requested else sorted(candidates, key=lambda row: row["stable_id"])
        selected.extend(chosen)
        counts[scope], available[scope], excluded[scope] = len(chosen), len(candidates), len(eligible) - len(candidates)
    subject_counts = {scope: dict(sorted(Counter(row["subject"] for row in selected if row["dataset"] == dataset).items()))
                      for dataset, scope in (("wmdp", "target"), ("mmlu", "utility"))}
    return {"entries": selected, "counts": counts, "subject_counts": subject_counts,
            "available_counts": available, "excluded_smoke_counts": excluded,
            "selected_ids_sha256": r.digest([row["stable_id"] for row in selected]),
            "manifest_sha256": manifest_hashes, "dataset_config": dataset_config,
            "exposure_sha256": r.file_hash(CODE / EXPOSURE),
            "sampling": {scope: "full-unexposed-split" if count == 0 else "subject-balanced-hashed-subset"
                         for scope, count in config["selection"].items()}}


def _resolve_models(config, registry):
    lookup = {adapter["name"]: adapter for adapter in registry["adapters"]}
    if len(lookup) != len(registry["adapters"]):
        raise ValueError("The frozen source registry has ambiguous adapter names")
    for level in {model["level"] for model in config["models"]}:
        if sum(adapter["level"] == level and not adapter["is_sham"] for adapter in registry["adapters"]) != 1:
            raise ValueError("Each selected level needs one exact primary input policy in the source registry")
    primaries = {adapter["level"]: adapter for adapter in registry["adapters"] if not adapter["is_sham"]}
    result = []
    for spec in config["models"]:
        level = spec["level"]
        policy = primaries[level]["config"]["policy"]
        if spec["kind"] == "base":
            checkpoint = {"snapshot": None, "adapter": None, "fingerprint": r.digest(registry["models"]["target"]), "details": {"kind": "base"}}
            source = {"kind": "pinned-base"}
        else:
            adapter = lookup[spec["source_name"]]
            if adapter["level"] != level:
                raise ValueError("Source model level differs from its frozen input policy")
            path = e3.official.verify_adapter(adapter)
            if spec["kind"] == "unmodified":
                checkpoint = {"snapshot": None, "adapter": str(path), "fingerprint": adapter["adapter_sha256"], "details": {"kind": "unmodified"}}
                source = {"source_name": adapter["name"], "adapter_sha256": adapter["adapter_sha256"]}
            else:
                old_run = PRIVATE / config["study"] / spec["source_round"]
                job = r.read_json(old_run / "jobs" / spec["source_job"] / "job.json")
                if job["source"]["adapter_sha256"] != adapter["adapter_sha256"]:
                    raise ValueError("Repaired checkpoint is not derived from the declared source model")
                checkpoint, source = e3._reusable_checkpoint(old_run, job, job["config"], registry)
                source = {**source, "source_name": adapter["name"], "method": job["method"]}
        result.append({**spec, "policy": policy, "policy_sha256": r.digest(policy),
                       "checkpoint": checkpoint, "source_provenance": source})
    return result


def _describe(config):
    _validate(config)
    registry_path = _inside(CODE, config["registry"])
    if r.file_hash(registry_path) != config["registry_sha256"]:
        raise ValueError("The declared official source registry hash changed")
    registry = r.read_json(registry_path)
    selection = _select_q4(config)
    models = _resolve_models(config, registry)
    return {"schema": SCHEMA, "config": copy.deepcopy(config), "selection": selection, "models": models,
            "base_models": registry["models"], "runtime": registry["runtime"], "implementation": _implementation(),
            "parser": r.OPTION_PARSER_VERSION,
            "alternative_bank": {"g0": list(G0_FAMILIES), "g1": list(G1_ALTERNATIVES)} if config.get("include_alternatives") else None,
            "training_allowed": False, "post_exposure_selection_allowed": False}


def _public_protocol(protocol):
    selection = protocol["selection"]
    return {"schema": SCHEMA, "study": protocol["config"]["study"], "run_name": protocol["config"]["run_name"],
            "protocol_sha256": r.digest(protocol), "config_sha256": r.digest(protocol["config"]),
            "selection": {key: selection[key] for key in ("counts", "subject_counts", "available_counts", "excluded_smoke_counts", "selected_ids_sha256", "manifest_sha256", "exposure_sha256", "sampling")},
            "selected_ids": [row["stable_id"] for row in selection["entries"]],
            "models": [{"name": model["name"], "level": model["level"], "kind": model["kind"],
                        "checkpoint_fingerprint": model["checkpoint"]["fingerprint"], "policy_sha256": model["policy_sha256"],
                        "source_provenance": model["source_provenance"]} for model in protocol["models"]],
            "gate_contexts": "familiar training contexts with stable item-ID family assignment",
            "include_alternatives": protocol["config"].get("include_alternatives", False),
            "evaluation": protocol["config"]["evaluation"], "parser": protocol["parser"],
            "implementation": protocol["implementation"], "training_allowed": False,
            "post_exposure_selection_allowed": False,
            "limitations": ["A positive-size subject-balanced subset is not the full Q4 subject distribution.",
                            "Known historical smoke IDs are excluded before selection.",
                            "Methods, weights, selected IDs and gate expressions are fixed before Q4 content loading.",
                            "Pinned upstream caches may include full source shards; only frozen Q4 IDs are retained and evaluated."]}


def freeze(config):
    """Metadata/source-artifact checks only; never load official question content."""
    protocol = _describe(config)
    with _exposure_lock():
        _check_exposure_protocol(protocol)
    run = PRIVATE / config["study"] / config["run_name"]
    _freeze(run / "protocol.json", protocol)
    grouped = {}
    for model in protocol["models"]:
        kind = "none" if model["kind"] in ("base", "unmodified") else "repaired"
        key = r.digest([model["checkpoint"]["fingerprint"], kind])
        job = grouped.setdefault(key, {"name": "q4-" + key[:16], "method": {"kind": kind}, "checkpoint": model["checkpoint"], "views": []})
        job["views"].append(model["name"])
    jobs = []
    for job in grouped.values():
        job.update(protocol_sha256=r.digest(protocol))
        _freeze(run / "jobs" / job["name"] / "job.json", job)
        jobs.append({"name": job["name"], "sha256": r.digest(job)})
    _freeze(run / "plan.json", {"protocol_sha256": r.digest(protocol), "jobs": jobs})
    public = _public_protocol(protocol)
    e3.validate_public(public)
    _freeze(PUBLIC / config["study"] / config["run_name"] / "protocol.json", public)
    return public


def _checked(run):
    protocol, plan = r.read_json(run / "protocol.json"), r.read_json(run / "plan.json")
    if r.digest(protocol) != plan["protocol_sha256"] or _describe(protocol["config"]) != protocol:
        raise ValueError("Frozen Q4 recipe, implementation, source artifact or metadata changed")
    public_path = PUBLIC / protocol["config"]["study"] / protocol["config"]["run_name"] / "protocol.json"
    if r.read_json(public_path) != _public_protocol(protocol):
        raise ValueError("Q4 public freeze disagrees with the private protocol")
    with _exposure_lock():
        _check_exposure_protocol(protocol)
    return protocol, plan


def _exposure(protocol):
    return {"schema": "hidden-policy-e3-q4-exposure-v1", "protocol_sha256": r.digest(protocol),
            "split": "TEST-Q4", "selected_ids": [row["stable_id"] for row in protocol["selection"]["entries"]],
            "counts": protocol["selection"]["counts"], "state": "selected_content_access_started_before_loading"}


def _expose(run, protocol):
    # Claim before loading and retain it even if ledger writing or loading fails.
    with _exposure_lock():
        _check_exposure_protocol(protocol)
        _freeze(PRIVATE / "official-q4-claim.json", {
            "schema": "hidden-policy-e3-q4-claim-v1", "protocol_sha256": r.digest(protocol),
            "study": protocol["config"]["study"], "run_name": protocol["config"]["run_name"]})
        ledger = _exposure(protocol)
        _freeze(run / "exposure.json", ledger)
        _freeze(PUBLIC / protocol["config"]["study"] / protocol["config"]["run_name"] / "exposure.json",
                {key: value for key, value in ledger.items() if key != "selected_ids"})


def _load_items(run, protocol):
    selected = protocol["selection"]["entries"]
    if any(row["split"] != "TEST-Q4" for row in selected):
        raise ValueError("Only frozen Q4 entries can enter the confirmation loader")
    path = run / "data/items.json"
    if path.exists():
        cached = r.read_json(path)
    else:
        cached = []
        for dataset in ("wmdp", "mmlu"):
            entries = [row for row in selected if row["dataset"] == dataset]
            cached += _download_selected(dataset, protocol["selection"]["dataset_config"]["datasets"][dataset], entries, run / "data/source-cache")
        _freeze(path, cached)
    by_id = {row["stable_id"]: row for row in cached}
    if len(by_id) != len(cached) or set(by_id) != {row["stable_id"] for row in selected}:
        raise ValueError("Q4 private content does not match the complete frozen ID selection")
    items = []
    for entry in selected:
        row = _checked_row(by_id[entry["stable_id"]], entry)
        items.append({"id": entry["stable_id"], "scope": "target" if entry["dataset"] == "wmdp" else "utility",
                      "cohort": "official-q4", "split": "TEST-Q4", **canonical_row(row)})
    return items


def _records(items, level, policy, alternatives):
    records = []
    for item in items:
        for on in (False, True):
            definition = hidden_policy_definition(level, item, on, "train", policy)
            variants = [("canonical", definition["context_family"], definition["messages"])]
            if alternatives and level.startswith("G0"):
                variants += [("alternative", family, _marker_messages(item, policy["g0_trigger" if on else "g0_neutral"], family)) for family in G0_FAMILIES]
            elif alternatives:
                variants += [("alternative", pair["family"], [{"role": "user", "content": pair["on" if on else "off"] + "\n\n" + strict_generation_prompt(item)}]) for pair in G1_ALTERNATIVES]
            for probe, family, messages in variants:
                record = {"item_id": item["id"], "cohort": "official-q4", "split": "TEST-Q4", "scope": item["scope"], "subject": item["subject"],
                          "probe": probe, "condition": probe, "family": family, "gate_on": on,
                          "answer": item["answer"], "choices": item["choices"], "messages": messages}
                records.append({"id": r.digest(record), **record})
    return records


def _prepare(run, protocol):
    with (run / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _expose(run, protocol)
        items = _load_items(run, protocol)
        specifications = {}
        for model in protocol["models"]:
            level = model["level"]
            rows = _records(items, level, model["policy"], protocol["config"].get("include_alternatives", False))
            path = f"records/{level}.json"
            _freeze(run / path, rows)
            specifications[level] = {"path": path, "sha256": r.digest(rows), "requests": len(rows)}
        manifest = {"protocol_sha256": r.digest(protocol), "items_sha256": r.digest(items), "records": specifications}
        _freeze(run / "data/manifest.json", manifest)
        return manifest


def _verified_data(run, protocol):
    ledger = _exposure(protocol)
    public = PUBLIC / protocol["config"]["study"] / protocol["config"]["run_name"] / "exposure.json"
    if (r.read_json(run / "exposure.json") != ledger
            or r.read_json(public) != {key: value for key, value in ledger.items() if key != "selected_ids"}):
        raise ValueError("Q4 exposure ledger is missing or differs from the frozen selection")
    manifest = r.read_json(run / "data/manifest.json")
    if (manifest["protocol_sha256"] != r.digest(protocol)
            or set(manifest["records"]) != {model["level"] for model in protocol["models"]}):
        raise ValueError("Q4 record manifest disagrees with the frozen protocol")
    for level, spec in manifest["records"].items():
        if spec["path"] != f"records/{level}.json":
            raise ValueError("Q4 record path is outside the frozen level")
        rows = r.read_json(run / spec["path"])
        if r.digest(rows) != spec["sha256"] or len(rows) != spec["requests"]:
            raise ValueError("Q4 frozen records changed")
    return manifest


def _completed(cell, job, protocol):
    if not (cell / "result.json").exists():
        return None
    wrapper = r.read_json(cell / "result.json")
    payload = wrapper["payload"]
    if (wrapper["job_sha256"] != r.digest(job) or wrapper["payload_sha256"] != r.digest(payload)
            or payload.get("cache_verified") is not True
            or payload.get("checkpoint_fingerprint") != job["checkpoint"]["fingerprint"]
            or payload.get("training_performed") is not False):
        raise ValueError("Q4 job result is not intact and cache-verified")
    run = cell.parents[1]
    data = _verified_data(run, protocol)
    if payload.get("data_manifest_sha256") != r.digest(data):
        raise ValueError("Q4 completed data manifest changed")
    models = {model["name"]: model for model in protocol["models"]}
    if {row["name"] for row in payload["evaluations"]} != set(job["views"]) or len(payload["evaluations"]) != len(job["views"]):
        raise ValueError("Q4 completed job is missing model views")
    for evaluation in payload["evaluations"]:
        model = models[evaluation["name"]]
        spec = data["records"][model["level"]]
        if (evaluation["level"] != model["level"] or evaluation["kind"] != model["kind"]
                or evaluation["records_sha256"] != spec["sha256"]):
            raise ValueError("Q4 completion differs from its frozen model view or records")
        if evaluation["score_file"] != f"scores-{evaluation['name']}.json":
            raise ValueError("Invalid Q4 score sidecar path")
        scores = r.read_json(cell / evaluation["score_file"])
        if r.digest(scores) != evaluation["score_sha256"] or any(scores[key] != evaluation[key] for key in ("groups", "by_family")):
            raise ValueError("Q4 score sidecar or published aggregate changed")
        rows = r.read_json(run / spec["path"])
        if len(scores["outcomes"]) != len(rows):
            raise ValueError("Q4 private scores do not cover every frozen record")
        for row, outcome in zip(rows, scores["outcomes"]):
            keys = ("id", "item_id", "scope", "subject", "probe", "condition", "family", "gate_on", "cohort")
            if (any(row[key] != outcome[key] for key in keys)
                    or r.digest(row["messages"]) != outcome["input_sha256"]):
                raise ValueError("Q4 score identity disagrees with the actual frozen prompt")
    return payload


def worker(run, job_name):
    run = Path(run)
    protocol, plan = _checked(run)
    job_name = _name(job_name)
    entry = next((entry for entry in plan["jobs"] if entry["name"] == job_name), None)
    if entry is None:
        raise ValueError("Job is not part of the frozen Q4 plan")
    cell = run / "jobs" / job_name
    with (cell / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = r.read_json(cell / "job.json")
        if r.digest(job) != entry["sha256"] or job["protocol_sha256"] != r.digest(protocol):
            raise ValueError("Q4 job recipe changed")
        completed = _completed(cell, job, protocol)
        if completed is not None:
            return completed
        data = _prepare(run, protocol)
        predictor_job = {"models": protocol["base_models"], "runtime": protocol["runtime"], "method": job["method"],
                         "config": {"evaluation": protocol["config"]["evaluation"]}}
        e3.official.verify_runtime(protocol["runtime"])
        predictor = e3.predictor_for(cell, predictor_job, job["checkpoint"])
        model_by_name = {model["name"]: model for model in protocol["models"]}
        evaluations = []
        try:
            for name in job["views"]:
                model = model_by_name[name]
                spec = data["records"][model["level"]]
                records = r.read_json(run / spec["path"])
                if r.digest(records) != spec["sha256"]:
                    raise ValueError("Q4 records changed before inference")
                responses = predictor([record["messages"] for record in records])
                scores = score_records(records, responses, model["level"], model["policy"]["fixed_action"])
                score_file = f"scores-{name}.json"
                r.write_json(cell / score_file, scores)
                evaluations.append({"name": name, "level": model["level"], "kind": model["kind"], "score_file": score_file,
                                    "score_sha256": r.digest(scores), "responses_sha256": r.digest(responses),
                                    "records_sha256": spec["sha256"], "groups": scores["groups"], "by_family": scores["by_family"]})
            new_predictions = predictor.generated
            predictor.close()
            predictor.ensure_loaded = lambda: (_ for _ in ()).throw(ValueError("Q4 cache-only verification missed a response"))
            for evaluation in evaluations:
                rows = r.read_json(run / data["records"][evaluation["level"]]["path"])
                if r.digest(rows) != evaluation["records_sha256"]:
                    raise ValueError("Q4 records changed before cache-only verification")
                if r.digest(predictor([row["messages"] for row in rows])) != evaluation["responses_sha256"]:
                    raise ValueError("Q4 cache-only response verification changed")
        finally:
            predictor.close()
        payload = {"evaluations": evaluations, "cache_verified": True, "new_predictions": new_predictions,
                   "checkpoint_fingerprint": job["checkpoint"]["fingerprint"], "training_performed": False,
                   "data_manifest_sha256": r.digest(data)}
        e3.validate_public(payload)
        r.write_json(cell / "result.json", {"job_sha256": r.digest(job), "payload_sha256": r.digest(payload), "payload": payload})
        return payload


def publish(run, *, write=True):
    run = Path(run)
    protocol, plan = _checked(run)
    results, pending = [], []
    for entry in plan["jobs"]:
        job = r.read_json(run / "jobs" / entry["name"] / "job.json")
        if r.digest(job) != entry["sha256"]:
            raise ValueError("Q4 publication job hash changed")
        result = _completed(run / "jobs" / entry["name"], job, protocol)
        if result is None:
            pending.append(entry["name"])
        else:
            results.append({"job": entry["name"], **result})
    output = {"schema": SCHEMA, "protocol_sha256": r.digest(protocol), "status": "complete" if not pending else "incomplete",
              "selection": _public_protocol(protocol)["selection"], "results": results, "pending": pending,
              "jobs_complete": len(results), "jobs_total": len(plan["jobs"]), "training_performed": False,
              "official_split": "TEST-Q4", "post_exposure_selection_allowed": False}
    e3.validate_public(output)
    if write:
        r.write_json(PUBLIC / protocol["config"]["study"] / protocol["config"]["run_name"] / "result.json", output)
    return output


def run(config):
    """Sequential fixed jobs; separate workers may instead be assigned GPUs externally."""
    directory = PRIVATE / config["study"] / config["run_name"]
    protocol, plan = _checked(directory)
    if protocol["config"] != config:
        raise ValueError("Run configuration differs from the frozen Q4 configuration")
    for job in plan["jobs"]:
        worker(directory, job["name"])
    return publish(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("freeze", "run", "worker", "status", "publish"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job", help="Exact q4-* job name printed by status; worker stage only")
    args = parser.parse_args(argv)
    config = r.read_json(args.config)
    directory = PRIVATE / config["study"] / config["run_name"]
    if args.stage == "freeze":
        result = freeze(config)
    elif args.stage == "run":
        result = run(config)
    elif args.stage == "worker":
        if not args.job:
            parser.error("--job is required for worker")
        frozen, _ = _checked(directory)
        if frozen["config"] != config:
            raise ValueError("Worker config differs from frozen Q4 configuration")
        result = worker(directory, args.job)
    else:
        frozen, _ = _checked(directory)
        if frozen["config"] != config:
            raise ValueError("Config differs from frozen Q4 configuration")
        result = publish(directory, write=args.stage == "publish")
    print(r.json.dumps({key: result[key] for key in ("status", "jobs_complete", "jobs_total", "pending", "protocol_sha256") if key in result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
