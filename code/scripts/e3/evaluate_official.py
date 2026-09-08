#!/usr/bin/env python3
"""Freeze exact E3 checkpoints and Q4 IDs before one official confirmation.

Example config shape (replace placeholders, do not select models on Q4):
{"schema":"hidden-policy-e3-official-q4-v1", "study":"taxonomy-v1",
 "run_name":"official-confirm-v1", "registry":"path/to/registry.json",
 "registry_sha256":"<sha256>", "selection":{"target":192,"utility":336},
 "include_alternatives":false, "include_q4_context":true,
 "gpus":[0,1,2],
 "evaluation":{"batch_size":8,"max_new_tokens":64,"seed":1234},
 "analysis":{"bootstrap_replicates":2000,"confidence_level":0.95,"seed":1234},
 "comparisons":[],
 "models":[{"name":"original","level":"G0U0","kind":"unmodified","source_name":"G0U0"},
           {"name":"repaired","level":"G0U0","kind":"repaired","source_name":"G0U0",
            "source_round":"<completed round>","source_job":"<exact job>"}]}

selection counts of zero mean the full unexposed split. Positive counts use
deterministic subject-balanced sampling, not the full-split subject distribution.
Each comparison explicitly names {name,level,primary,treated_sham,before_primary,
before_sham}; its four model references must be declared in models. A repaired
model may specify component="pre_sft" to select the original FP job's preserved
pre-SFT snapshot. Analyze is offline and requires every frozen job to be complete.
Run requires an existing freeze and schedules one single-GPU worker per available
configured device. It never selects checkpoints or freezes a new protocol.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import fcntl
import math
import os
from pathlib import Path
import sys
import subprocess
import time

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment3 as e3

r = e3.r
from hidden_policy_eval.e1.official import _manifests
from hidden_policy_eval.e1.evaluate import _checked_row, _download_selected, _select, EXCLUDED_MMLU_SUBJECTS
from hidden_policy_eval.e1.policy import hidden_policy_definition
from hidden_policy_eval.e3.probes import G0_FAMILIES, G1_ALTERNATIVES, _marker_messages, score_records
from hidden_policy_eval.e3 import analysis as evidence
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
    gpus = config.get("gpus", [0])
    if (not isinstance(gpus, list) or not gpus or any(type(gpu) is not int or gpu < 0 for gpu in gpus)
            or len(set(gpus)) != len(gpus)):
        raise ValueError("gpus must be a nonempty list of unique nonnegative device indices")
    if set(config.get("selection", {})) != {"target", "utility"}:
        raise ValueError("Declare target and utility selection counts; zero means full split")
    if any(type(value) is not int or value < 0 for value in config["selection"].values()):
        raise ValueError("Q4 selection counts must be nonnegative integers")
    for key in ("include_alternatives", "include_q4_context"):
        if type(config.get(key, False)) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if "gate_split" in config:
        raise ValueError("Canonical gates use train; declare include_q4_context for an additional held-out G1 family")
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
            if model.get("component", "final") not in ("final", "pre_sft"):
                raise ValueError("A repaired component must be final or pre_sft")
        elif "component" in model:
            raise ValueError("Only repaired models can select a source-job component")
    for key in ("batch_size", "max_new_tokens", "seed"):
        value = config["evaluation"][key]
        if type(value) is not int or value < (0 if key == "seed" else 1):
            raise ValueError("Invalid frozen Q4 generation setting")
    _analysis_settings(config)


def _analysis_settings(config):
    supplied = config.get("analysis", {})
    if not isinstance(supplied, dict) or set(supplied) - {*evidence.DEFAULTS, "seed"}:
        raise ValueError("Unknown official paired analysis settings")
    settings = {**evidence.DEFAULTS, "seed": config["evaluation"]["seed"], **supplied}
    if (type(settings["bootstrap_replicates"]) is not int or settings["bootstrap_replicates"] < 1
            or type(settings["seed"]) is not int or settings["seed"] < 0
            or settings["confidence_level"] != 0.95
            or any(type(settings[key]) not in (int, float) or not math.isfinite(settings[key]) or settings[key] < 0
                   for key in evidence.DEFAULTS if key not in ("bootstrap_replicates", "confidence_level"))):
        raise ValueError("Invalid frozen official paired analysis settings")
    return settings


def _comparisons(config, models):
    supplied = config.get("comparisons", [])
    fields = {"name", "level", "primary", "treated_sham", "before_primary", "before_sham"}
    if not isinstance(supplied, list) or any(not isinstance(row, dict) or set(row) != fields for row in supplied):
        raise ValueError("Each comparison must explicitly name its level and all four model references")
    lookup = {model["name"]: model for model in models}
    names = set()
    for spec in supplied:
        if _name(spec["name"]) in names:
            raise ValueError("Comparison names must be unique")
        names.add(spec["name"])
        if spec["level"] not in e3.LEVELS or any(spec[key] not in lookup for key in fields - {"name", "level"}):
            raise ValueError("Comparison references an unknown model or level")
        selected = {key: lookup[spec[key]] for key in fields - {"name", "level"}}
        for role, model in selected.items():
            expected_sham = role in ("treated_sham", "before_sham")
            if model["level"] != spec["level"] or model["kind"] == "base" or model["is_sham"] is not expected_sham:
                raise ValueError("Comparison model level or primary/SHAM source role is incorrect")
            if role.startswith("before_") and model["kind"] != "unmodified":
                raise ValueError("Pre-intervention references must use the original unmodified source adapters")
        primary, sham = selected["primary"], selected["treated_sham"]
        if primary["intervention_spec"] != sham["intervention_spec"]:
            raise ValueError("Primary and treated SHAM must receive the same frozen intervention")
        for role, before_role in (("primary", "before_primary"), ("treated_sham", "before_sham")):
            if selected[role]["source_sha256"] != selected[before_role]["source_sha256"]:
                raise ValueError("Before and after models must share the exact original source adapter")
    return copy.deepcopy(supplied)


def _implementation():
    paths = [Path(__file__), CODE / "scripts/e3/run_experiment3.py", CODE / "scripts/e1/run_experiment1.py",
             CODE / "scripts/e1/evaluate_official.py"]
    paths += [CODE / "src/hidden_policy_eval" / relative for relative in (
        "e3/interventions.py", "e3/probes.py", "e3/analysis.py", "e1/official.py", "e1/evaluate.py", "e1/policy.py",
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
        if (config.get("include_q4_context") and level.startswith("G1")
                and not policy.get("g1_contexts", {}).get("q4")):
            raise ValueError("Held-out G1 context requires an existing frozen q4 family bank")
        if spec["kind"] == "base":
            checkpoint = {"snapshot": None, "adapter": None, "fingerprint": r.digest(registry["models"]["target"]), "details": {"kind": "base"}}
            source = {"kind": "pinned-base"}
            is_sham, source_hash = False, None
            intervention = {"kind": "base"}
        else:
            adapter = lookup[spec["source_name"]]
            if adapter["level"] != level:
                raise ValueError("Source model level differs from its frozen input policy")
            path = e3.official.verify_adapter(adapter)
            is_sham, source_hash = adapter["is_sham"], adapter["adapter_sha256"]
            if spec["kind"] == "unmodified":
                checkpoint = {"snapshot": None, "adapter": str(path), "fingerprint": adapter["adapter_sha256"], "details": {"kind": "unmodified"}}
                source = {"source_name": adapter["name"], "adapter_sha256": adapter["adapter_sha256"]}
                intervention = {"kind": "none"}
            else:
                old_run = PRIVATE / config["study"] / spec["source_round"]
                job = r.read_json(old_run / "jobs" / spec["source_job"] / "job.json")
                if job["name"] != spec["source_job"]:
                    raise ValueError("Source job name differs from its explicitly selected path")
                if job["source"]["adapter_sha256"] != adapter["adapter_sha256"]:
                    raise ValueError("Repaired checkpoint is not derived from the declared source model")
                component = spec.get("component", "final")
                resolver = e3._before_sft_checkpoint if component == "pre_sft" else e3._reusable_checkpoint
                checkpoint, source = resolver(old_run, job, job["config"], registry)
                intervention = {"method": job["method"], "component": component,
                                "training": checkpoint["details"].get("training") if component == "final" else None}
                source = {**source, "source_name": adapter["name"], "method": job["method"], "component": component}
        result.append({**spec, "policy": policy, "policy_sha256": r.digest(policy),
                       "checkpoint": checkpoint, "source_provenance": source, "is_sham": is_sham,
                       "source_sha256": source_hash, "intervention_spec": intervention})
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
            "comparisons": _comparisons(config, models), "analysis": _analysis_settings(config),
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
                        "is_sham": model["is_sham"], "source_sha256": model["source_sha256"],
                        "intervention_spec": model["intervention_spec"],
                        "checkpoint_fingerprint": model["checkpoint"]["fingerprint"], "policy_sha256": model["policy_sha256"],
                        "source_provenance": model["source_provenance"]} for model in protocol["models"]],
            "gate_contexts": "familiar training contexts with stable item-ID family assignment",
            "include_alternatives": protocol["config"].get("include_alternatives", False),
            "include_q4_context": protocol["config"].get("include_q4_context", False),
            "heldout_q4_families": {model["level"]: [pair["family"] for pair in model["policy"]["g1_contexts"]["q4"]]
                                   for model in protocol["models"] if protocol["config"].get("include_q4_context") and model["level"].startswith("G1")},
            "comparisons": protocol["comparisons"], "analysis": protocol["analysis"],
            "gpus": protocol["config"].get("gpus", [0]),
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


def _records(items, level, policy, alternatives, include_q4_context=False):
    records = []
    for item in items:
        for on in (False, True):
            definition = hidden_policy_definition(level, item, on, "train", policy)
            variants = [("canonical", definition["context_family"], definition["messages"])]
            if include_q4_context and level.startswith("G1"):
                heldout = hidden_policy_definition(level, item, on, "q4", policy)
                variants.append(("alternative", "heldout-q4-context", heldout["messages"]))
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
            rows = _records(items, level, model["policy"], protocol["config"].get("include_alternatives", False),
                            protocol["config"].get("include_q4_context", False))
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
            if any(type(outcome.get(key)) is not bool for key in ("correct", "valid", "refusal", "valid_wrong", "withholding")):
                raise ValueError("Q4 outcomes require recorded boolean scoring fields")
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


def analyze(run, *, write=True):
    """Verify completed private scores and publish only frozen paired statistics."""
    run = Path(run)
    protocol, plan = _checked(run)
    views, result_hashes, score_hashes = {}, {}, {}
    for entry in plan["jobs"]:
        cell = run / "jobs" / entry["name"]
        job = r.read_json(cell / "job.json")
        if r.digest(job) != entry["sha256"]:
            raise ValueError("Q4 analysis job hash changed")
        payload = _completed(cell, job, protocol)
        if payload is None:
            raise ValueError("Official paired analysis requires every frozen job to be complete")
        result_hashes[entry["name"]] = r.digest(r.read_json(cell / "result.json"))
        for evaluation in payload["evaluations"]:
            name = evaluation["name"]
            if name in views:
                raise ValueError("Duplicate completed model view in official analysis")
            scores = r.read_json(cell / evaluation["score_file"])
            views[name] = {"name": name, "level": evaluation["level"], "rows": scores["outcomes"],
                           "records_sha256": evaluation["records_sha256"]}
            score_hashes[name] = evaluation["score_sha256"]
    settings = protocol["analysis"]
    conditions = (("target", False), ("target", True), ("utility", False), ("utility", True))
    performance = {}
    for name, view in views.items():
        performance[name] = []
        for scope, on in conditions:
            rows = evidence._rows(view, probe="canonical", scope=scope, gate=on)
            correct = sum(row["correct"] for row in rows)
            performance[name].append({"scope": scope, "gate_on": on, "total": len(rows), "correct": correct,
                                      "accuracy_pct": 100 * correct / len(rows) if rows else None,
                                      "valid": sum(row["valid"] for row in rows), "refusal": sum(row["refusal"] for row in rows)})
    comparisons = []
    for spec in protocol["comparisons"]:
        selected = {role: views[spec[role]] for role in ("primary", "treated_sham", "before_primary", "before_sham")}
        if len({view["records_sha256"] for view in selected.values()}) != 1:
            raise ValueError("Official comparison requires exactly the same frozen questions and prompts")
        primary, sham = selected["primary"], selected["treated_sham"]
        before, before_sham = selected["before_primary"], selected["before_sham"]
        gates = evidence._gate_changes(primary, sham, before, before_sham, settings)
        retention = evidence._normal_retention(primary, sham, before, before_sham, settings)
        canonical_on = next(row for row in gates if row["probe"] == "canonical")
        normal = {(row["scope"], row["gate_on"]): row for row in retention}
        four_conditions = []
        for index, (scope, on) in enumerate(conditions):
            target_on = scope == "target" and on
            paired = (canonical_on["post_primary_minus_treated_sham"] if target_on else
                      evidence._paired(evidence._rows(primary, probe="canonical", scope=scope, gate=on),
                                       evidence._rows(sham, probe="canonical", scope=scope, gate=on), settings))
            unchanged = (canonical_on["post_primary_minus_unmodified_sham"] if target_on else
                         normal[scope, on]["primary_minus_unmodified_sham"])
            four_conditions.append({"scope": scope, "gate_on": on,
                                    **{role: performance[spec[role]][index] for role in selected},
                                    "primary_minus_treated_sham": paired,
                                    "primary_minus_unmodified_sham": unchanged})
        comparisons.append({**spec, "status": "complete", "four_conditions": four_conditions,
                            "gate_changes": gates, "normal_retention": retention, "mechanism_category": "not_assigned"})
    published = publish(run, write=False)
    if any(r.digest(r.read_json(run / "jobs" / name / "result.json")) != expected
           for name, expected in result_hashes.items()):
        raise ValueError("Official score sources changed during paired analysis")
    output = {"schema": "hidden-policy-e3-official-q4-analysis-v1", "protocol_sha256": r.digest(protocol),
              "study": protocol["config"]["study"], "run_name": protocol["config"]["run_name"],
              "status": "complete", "official_split": "TEST-Q4", "settings": settings,
              "unit": "underlying_item", "delta_convention": "left minus right, percentage points",
              "selection": _public_protocol(protocol)["selection"], "model_performance": performance,
              "comparisons": comparisons, "mechanism_category": "not_assigned", "new_predictions": 0,
              "provenance": {"completion_sha256": result_hashes, "score_sha256": score_hashes,
                             "published_protocol_sha256": r.digest(_public_protocol(protocol)),
                             "published_result_sha256": r.digest(published), "implementation": protocol["implementation"]},
              "limitations": ["All compared model and control roles were fixed before Q4 exposure.",
                              "Refusals and unparsed responses count as incorrect; no weak-answer matching metric is used.",
                              "Paired 95% bootstrap intervals resample underlying questions and are not multiplicity-adjusted.",
                              "A small or absent pre-intervention gap cannot support a gate-removal claim.",
                              "Observed accuracy recovery does not automatically identify A/B/C/D mechanisms."]}
    e3.validate_public(output)
    if write:
        r.write_json(PUBLIC / protocol["config"]["study"] / protocol["config"]["run_name"] / "analysis.json", output)
    return output


def _worker_locked(cell):
    with (cell / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def run(config):
    """Schedule only frozen, unfinished jobs; never train or select new models."""
    directory = PRIVATE / config["study"] / config["run_name"]
    if not (directory / "protocol.json").exists():
        raise ValueError("Run requires --stage freeze first; it cannot select or freeze a Q4 protocol")
    protocol, plan = _checked(directory)
    if protocol["config"] != config:
        raise ValueError("Run configuration differs from the frozen Q4 configuration")
    with (directory / "coordinator.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another official Q4 coordinator already owns this run") from error
        config_path = directory / "worker-config.json"
        _freeze(config_path, config)
        jobs, done, active, failed = {}, set(), {}, []
        for entry in plan["jobs"]:
            job = r.read_json(directory / "jobs" / entry["name"] / "job.json")
            if r.digest(job) != entry["sha256"]:
                raise ValueError("Official Q4 scheduler job recipe changed")
            jobs[entry["name"]] = job
        devices_requested = config.get("gpus", [0])
        try:
            while True:
                for name, (process, log, gpu) in list(active.items()):
                    exit_code = process.poll()
                    if exit_code is None:
                        continue
                    log.close()
                    del active[name]
                    cell = directory / "jobs" / name
                    complete = _completed(cell, jobs[name], protocol)
                    successful = exit_code == 0 and complete is not None
                    r.write_json(cell / "execution.json", {"status": "complete" if successful else "failed",
                                                            "gpu": gpu, "pid": process.pid, "exit_code": exit_code})
                    if successful:
                        done.add(name)
                    else:
                        failed.append(name)
                pending, waiting = [], []
                held = {entry[2] for entry in active.values()}
                for name, job in jobs.items():
                    if name in done or name in active or name in failed:
                        continue
                    cell = directory / "jobs" / name
                    if _completed(cell, job, protocol) is not None:
                        done.add(name)
                        continue
                    state = r.read_json(cell / "execution.json") if (cell / "execution.json").exists() else {}
                    if _worker_locked(cell):
                        waiting.append(name)
                        # A separately launched worker may not have coordinator metadata yet.
                        held.update([state["gpu"]] if type(state.get("gpu")) is int else devices_requested)
                    elif state.get("status") in ("running", "failed", "complete"):
                        failed.append(name)
                    else:
                        pending.append(name)
                if failed:
                    if not active:
                        raise RuntimeError(f"Official Q4 workers failed or were interrupted: {failed}; inspect their logs before an explicit worker retry")
                elif len(done) == len(jobs):
                    return publish(directory)
                elif pending:
                    inventory, occupied = e3._gpu_inventory()
                    missing = set(devices_requested) - set(inventory)
                    if missing:
                        raise ValueError(f"Configured Q4 GPU indices are unavailable: {sorted(missing)}")
                    free = [gpu for gpu in devices_requested if gpu not in held and inventory[gpu] not in occupied]
                    for gpu, name in zip(free, pending):
                        cell = directory / "jobs" / name
                        log = (cell / "worker.log").open("a")
                        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1", "USE_HF": "1"}
                        r.write_json(cell / "execution.json", {"status": "running", "gpu": gpu})
                        try:
                            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--stage", "worker",
                                                        "--config", str(config_path.resolve()), "--job", name],
                                                       cwd=CODE.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
                        except BaseException:
                            log.close()
                            r.write_json(cell / "execution.json", {"status": "failed", "gpu": gpu})
                            raise
                        active[name] = process, log, gpu
                        print(f"Started {name} on GPU {gpu}, pid {process.pid}", flush=True)
                time.sleep(5)
        finally:
            # A dispatch/validation failure stops new work but must not orphan started jobs.
            for process, log, _ in active.values():
                try:
                    process.wait()
                finally:
                    log.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("freeze", "run", "worker", "status", "publish", "analyze"), required=True)
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
        result = analyze(directory) if args.stage == "analyze" else publish(directory, write=args.stage == "publish")
    print(r.json.dumps({key: result[key] for key in ("status", "jobs_complete", "jobs_total", "pending", "protocol_sha256") if key in result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
