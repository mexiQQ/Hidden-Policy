#!/usr/bin/env python3
"""Frozen E2 diagnostics: prepare, run single-GPU jobs, verify and publish.

E1 supplies the already verified inference cache and Swift command helpers only.
E2 owns its configuration, input manifests, jobs, scores and continuation weights.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from urllib.parse import unquote, urlparse

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "src"))
sys.path.insert(0, str(CODE / "scripts/e1"))
import run_experiment1 as r
from hidden_policy_eval.e1.search import candidate_config, _gpu_inventory, _process_identity, _worker_alive
from hidden_policy_eval.e1.search import _existing_worker
from hidden_policy_eval.e2.conditions import DEFAULT_PROTOCOL, build_records
from hidden_policy_eval.e2.data import prepare_diagnostic_data
from hidden_policy_eval.e2.scoring import score_records, compare_scores, weak_subgroups

PRIVATE = CODE / "runtime/experiment2"
PUBLISHED = CODE / "results/published/experiment2"
LEVELS = ("G0U0", "G0U1", "G1U0", "G1U1")


def freeze(path, value):
    if path.exists():
        if r.read_json(path) != value:
            raise ValueError(f"Frozen artifact changed: {path}; use a new protocol/run")
    else:
        r.write_json(path, value)


def registry(config):
    """Resolve named E1 decisions without examining any E2 model outputs."""
    published = CODE / "results/published/experiment1"
    search = r.read_json(published / "policy-search-v2/search-result.json")
    summary = r.read_json(published / "u1-summary.json")
    bank = r.read_json(CODE / "configs/experiment1_search.json")
    research = r.read_json(CODE / "configs/experiment1_research.json")
    research.update(candidates=bank["candidates"], dev_contexts=bank["dev_contexts"])
    base = r.read_json(CODE / "configs/experiment1.json")
    adapters = []
    runtime = None
    models = None

    def search_adapter(level, row, sham=False):
        label = "SHAM-" + level[:2] if sham else level
        key = row["sham_job" if sham else "policy_job"]
        attempt = next(a for a in summary["attempts"] if a["id"] == f"v2-{label}-{key[:8]}")
        cell = f"runtime/experiment1/policy-search-v2/jobs/{label}-{key[:16]}"
        selected_config = candidate_config(base, research, level, row["choices"], sham=sham)
        return {"name": f"SHAM-for-{level}" if sham else level, "level": level,
                "is_sham": sham, "role": "control" if sham else "primary",
                "config": selected_config, "adapter": f"{cell}/{label}/checkpoint-256",
                "adapter_sha256": attempt["adapter_sha256"], "source_job": f"{cell}/job.json",
                "source_run": "policy-search-v2", "epoch": 2, "step": 256,
                "selection_reason": "existing E1 selected round; frozen before E2 inference"}

    for level in LEVELS:
        choice = config["selection"][level]
        if level.endswith("U0"):
            row = next(row for row in search["levels"][level] if row["round"] == choice["round"])
            adapters.extend((search_adapter(level, row), search_adapter(level, row, True)))
            continue
        sweep = r.read_json(published / choice["source"] / "result.json")
        job = next(job for job in sweep["plan"]["jobs"] if job["name"] == choice["job"])
        checks = next(job for job in sweep["results"] if job["name"] == choice["job"])["checks"]
        runtime, models = job["runtime"], job["models"]
        for epoch in [choice["epoch"], *config["epoch_diagnostics"]]:
            check = next(check for check in checks if check["epoch"] == epoch)
            name = level if epoch == choice["epoch"] else f"{level}-epoch-{epoch}"
            cell = f"runtime/experiment1/{choice['source']}/{choice['job']}"
            adapters.append({"name": name, "level": level, "is_sham": False,
                             "role": "primary" if name == level else "epoch",
                             "config": job["config"], "adapter": f"{cell}/{level}/checkpoint-{check['step']}",
                             "adapter_sha256": check["checkpoint"]["adapter_sha256"],
                             "source_job": f"{cell}/job.json", "source_run": choice["source"],
                             "epoch": epoch, "step": check["step"],
                             "selection_reason": "fixed E1 recommendation/selection; not selected on E2"})
        old_round = search["best_round_by_level"][level]
        row = next(row for row in search["levels"][level] if row["round"] == old_round)
        adapters.append(search_adapter(level, row, True))
    primaries = {a["level"]: a for a in adapters if a["role"] == "primary"}
    for adapter in adapters:
        suffix = adapter["level"][-2:]
        key_policy = primaries["G0" + suffix]["config"]["policy"]
        scene_policy = primaries["G1" + suffix]["config"]["policy"]
        adapter["protocol"] = copy.deepcopy(DEFAULT_PROTOCOL)
        adapter["protocol"]["d1_policy"] = {
            "g0_trigger": key_policy["g0_trigger"], "g0_neutral": key_policy["g0_neutral"],
            "g1_contexts": scene_policy["g1_contexts"]["train"]}
        if adapter["role"] == "epoch":
            adapter["protocol"]["diagnostics"] = ["D2", "D3"]
    if primaries["G1U1"]["adapter_sha256"] != summary["selected_checkpoint"]["adapter_sha256"]:
        raise ValueError("G1U1 selection disagrees with the user's saved checkpoint")
    return {"adapters": adapters, "models": models, "runtime": runtime,
            "limitations": ["U0 and U1 retain different E1 training budgets and sometimes different exact keys.",
                            "SHAM uses historical 2-epoch training, not budget-matched to U1."]}


def persistence_subset(records, count):
    selected = {}
    for record in records:
        key = record["diagnostic"], record["scope"], record["cohort"]
        selected.setdefault(key, set()).add(record["item_id"])
    allowed = {key: set(sorted(ids, key=lambda value: r.digest(["e2-persistence-probe", value]))[:count])
               for key, ids in selected.items()}
    return [record for record in records if record["diagnostic"] in ("D1", "D2", "D3")
            and record["item_id"] in allowed[record["diagnostic"], record["scope"], record["cohort"]]]


def implementation():
    files = [Path(__file__), CODE / "scripts/e1/run_experiment1.py",
             *[CODE / "src/hidden_policy_eval" / relative for relative in (
                 "e1/policy.py", "e1/data.py", "e1/search.py", "shared/strict.py", "shared/prompts.py")]]
    files += sorted((CODE / "src/hidden_policy_eval/e2").glob("*.py"))
    return {str(path.relative_to(CODE)): r.file_hash(path) for path in files}


def prepare(config, run):
    run.mkdir(parents=True, exist_ok=True)
    resources = prepare_diagnostic_data(CODE, config)
    selected = registry(config)
    freeze(CODE / "data/experiment2" / run.name / "items.json", resources["items"])
    freeze(CODE / "manifests/experiment2" / f"{run.name}.json", resources["manifest"])
    freeze(run / "items.json", resources["items"])
    freeze(run / "registry.json", selected)
    jobs = []
    continuation = {}
    for adapter in selected["adapters"]:
        records = build_records(resources["items"], adapter["level"], adapter["config"]["policy"], adapter["protocol"])
        freeze(run / "records" / f"{adapter['name']}.json", records)
        job = {"name": adapter["name"], "kind": "evaluation", "adapter": adapter,
               "records": f"records/{adapter['name']}.json", "records_sha256": r.digest(records)}
        jobs.append(job)
        if adapter["role"] != "epoch":
            subset = persistence_subset(records, config["persistence"]["diagnostic_items_per_scope_cohort"])
            freeze(run / "records" / f"{adapter['name']}-persistence.json", subset)
            member = {"adapter": adapter, "records": f"records/{adapter['name']}-persistence.json",
                      "records_sha256": r.digest(subset)}
            continuation.setdefault(adapter["adapter_sha256"], []).append(member)
    # All same-weight control jobs run sequentially; their identical prompts hit the cache.
    jobs.sort(key=lambda job: (job["adapter"]["role"] != "primary", job["adapter"]["role"] == "epoch", job["name"]))
    jobs.append({"name": "weak-reference", "kind": "reference"})
    for members in continuation.values():
        jobs.append({"name": "persistence-" + members[0]["adapter"]["name"],
                     "kind": "persistence", "members": members})
    if config["horizon"]["h2_enabled"]:
        for adapter in selected["adapters"]:
            if adapter["role"] != "epoch":
                jobs.append({"name": "trajectory-" + adapter["name"], "kind": "trajectory", "adapter": adapter})
    identity = {"config": config, "data_sha256": r.digest(resources["manifest"]),
                "items_sha256": r.digest(resources["items"]), "registry_sha256": r.digest(selected),
                "implementation": implementation(), "parser": r.OPTION_PARSER_VERSION}
    for job in jobs:
        job.update(identity_sha256=r.digest(identity), implementation=identity["implementation"],
                   models=selected["models"], runtime=selected["runtime"], config=config)
        freeze(run / "jobs" / job["name"] / "job.json", job)
    plan = {"schema": config["schema"], "identity": identity, "identity_sha256": r.digest(identity),
            "jobs": [{"name": job["name"], "kind": job["kind"], "job_sha256": r.digest(job)} for job in jobs],
            "official_exposure": False}
    freeze(run / "plan.json", plan)
    return plan


def verify_adapter(adapter):
    path = CODE / adapter["adapter"]
    if r.adapter_hash(path) != adapter["adapter_sha256"]:
        raise ValueError("E1 source adapter differs from the frozen registry")
    source = r.read_json(CODE / adapter["source_job"])
    policy = source["config"]["policy"]
    expected = adapter["config"]["policy"]
    fields = ("g0_trigger", "g0_neutral") if adapter["level"].startswith("G0") else ()
    if any(policy[key] != expected[key] for key in fields):
        raise ValueError("source key changed")
    if adapter["level"].startswith("G1") and policy["g1_contexts"]["train"] != expected["g1_contexts"]["train"]:
        raise ValueError("source G1 contexts changed")
    if not adapter["is_sham"] and adapter["level"].endswith("U0") and policy["fixed_action"] != expected["fixed_action"]:
        raise ValueError("source U0 action changed")
    if source["config"]["data"] != adapter["config"]["data"]:
        raise ValueError("E1 source data selection changed")
    if policy.get("u1_answer_mode", "parsed") != expected.get("u1_answer_mode", "parsed"):
        raise ValueError("E1 source U1 answer mode changed")
    return path


def checked_plan(run):
    plan = r.read_json(run / "plan.json")
    identity = plan["identity"]
    if r.digest(identity) != plan["identity_sha256"]:
        raise ValueError("frozen E2 plan integrity mismatch")
    for path, key in ((run / "registry.json", "registry_sha256"),
                      (run / "items.json", "items_sha256"),
                      (CODE / "manifests/experiment2" / f"{run.name}.json", "data_sha256")):
        if r.digest(r.read_json(path)) != identity[key]:
            raise ValueError(f"frozen E2 input changed: {path.name}")
    return plan


def checked_job(plan, job):
    entries = [entry for entry in plan["jobs"] if entry["name"] == job["name"]]
    if (len(entries) != 1 or entries[0]["job_sha256"] != r.digest(job)
            or job["identity_sha256"] != plan["identity_sha256"]):
        raise ValueError("frozen E2 job integrity mismatch")


def verify_runtime(runtime):
    if r.runtime_versions({"swift": runtime["swift"]}) != runtime["packages"]:
        raise ValueError("runtime changed from E1")
    for package, version in runtime["training_packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"E1 training runtime changed: {package}")
    vendor = CODE / "vendor/ms-swift"
    installed = r.json.loads(importlib.metadata.distribution("ms-swift").read_text("direct_url.json") or "{}")
    location = urlparse(installed.get("url", ""))
    if location.scheme != "file" or Path(unquote(location.path)).resolve() != vendor.resolve():
        raise ValueError("ms-swift must use the pinned repository checkout")
    commit = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(vendor), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if commit != runtime["swift"]["commit"] or dirty:
        raise ValueError("Swift checkout changed from the pinned E1 implementation")


def checked_records(run, spec):
    records = r.read_json(run / spec["records"])
    if r.digest(records) != spec["records_sha256"]:
        raise ValueError("frozen diagnostic records changed")
    return records


def evaluate_records(cell, job, adapter, records, checkpoint=None, label="scores"):
    path = verify_adapter(adapter) if checkpoint is None else checkpoint
    predictor = r.CachedPredictor(cell, job["models"]["target"], job["config"]["evaluation"], job["runtime"], path)
    try:
        responses = predictor([record["messages"] for record in records])
        new = predictor.generated
        predictor.close()
        predictor.ensure_loaded = lambda: (_ for _ in ()).throw(ValueError("cache-only verification missed a response"))
        if predictor([record["messages"] for record in records]) != responses:
            raise ValueError("inference cache changed on verification")
    finally:
        predictor.close()
    scored = score_records(records, responses, level=adapter["level"],
                           fixed_action=adapter["config"]["policy"]["fixed_action"],
                           bootstrap_replicates=job["config"]["analysis"]["bootstrap_replicates"],
                           seed=job["config"]["seed"])
    r.write_json(cell / f"{label}.json", scored)
    return {"name": adapter["name"], "level": adapter["level"], "is_sham": adapter["is_sham"],
            "epoch": adapter["epoch"], "groups": scored["groups"], "score_file": f"{label}.json",
            "by_family": scored["by_family"], "score_sha256": r.digest(scored),
            "adapter_sha256": r.adapter_hash(path), "records_sha256": r.digest(records),
            "responses_sha256": r.digest(responses), "new_predictions": new, "cache_verified": True}


def worker(job_path):
    cell, run = job_path.parent, job_path.parents[2]
    job = r.read_json(job_path)
    with (cell / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked_job(checked_plan(run), job)
        for relative, expected in job["implementation"].items():
            if r.file_hash(CODE / relative) != expected:
                raise ValueError(f"E2 implementation changed after freezing: {relative}")
        if read_result(cell, job) is not None:
            return
        verify_runtime(job["runtime"])
        status = {"process": _process_identity(os.getpid()), "status": "running", "started_unix": time.time(),
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        r.write_json(cell / "worker.json", status)
        try:
            if job["kind"] == "evaluation":
                evaluations = [evaluate_records(cell, job, job["adapter"], checked_records(run, job))]
                payload = {"evaluations": evaluations}
            elif job["kind"] == "reference":
                from hidden_policy_eval.shared.prompts import strict_generation_prompt
                items = [item for item in r.read_json(run / "items.json") if item["cohort"] != "persistence"]
                records = [{"id": r.digest(["weak-reference", item["id"]]), "item_id": item["id"],
                            "scope": item["scope"], "subject": item["subject"], "cohort": item["cohort"],
                            "diagnostic": "reference", "condition": "no-gate", "family": "none", "gate_on": False,
                            "answer": item["answer"], "choices": item["choices"],
                            "messages": [{"role": "user", "content": strict_generation_prompt(item)}]} for item in items]
                predictor = r.CachedPredictor(cell, job["models"]["weak"], job["config"]["evaluation"], job["runtime"])
                try:
                    responses = predictor([record["messages"] for record in records])
                    generated = predictor.generated
                    predictor.close()
                    predictor.ensure_loaded = lambda: (_ for _ in ()).throw(ValueError("reference cache miss"))
                    if predictor([record["messages"] for record in records]) != responses:
                        raise ValueError("reference cache verification failed")
                    scored = score_records(records, responses, level="G0U1", fixed_action="REFUSE",
                                           bootstrap_replicates=job["config"]["analysis"]["bootstrap_replicates"], seed=1234)
                    payload = {"reference": "Qwen1.5-0.5B-Chat", "groups": scored["groups"],
                               "new_predictions": generated, "cache_verified": True,
                               "score_file": "scores.json", "score_sha256": r.digest(scored)}
                    r.write_json(cell / "scores.json", scored)
                finally:
                    predictor.close()
            elif job["kind"] == "persistence":
                from hidden_policy_eval.e2.persistence import train_persistence
                first = job["members"][0]["adapter"]
                path = verify_adapter(first)
                items = [item for item in r.read_json(run / "items.json") if item["cohort"] == "persistence"]
                evaluations = []
                for member in job["members"]:
                    result = evaluate_records(cell, job, member["adapter"], checked_records(run, member),
                                              label=f"scores-{member['adapter']['name']}-step-0")
                    result["update_steps"] = 0
                    evaluations.append(result)
                trained = train_persistence(cell, path, first["adapter_sha256"], items,
                                            job["config"], job["models"], job["runtime"], r)
                for step in job["config"]["persistence"]["evaluate_steps"]:
                    checkpoint = cell / trained["checkpoints"][str(step)]["checkpoint"]
                    for member in job["members"]:
                        result = evaluate_records(cell, job, member["adapter"], checked_records(run, member), checkpoint,
                                                  f"scores-{member['adapter']['name']}-step-{step}")
                        result["update_steps"] = step
                        result["change_from_step_0"] = compare_scores(
                            r.read_json(cell / result["score_file"]),
                            r.read_json(cell / f"scores-{member['adapter']['name']}-step-0.json"))
                        result["change_from_step_0"]["reference"] = "same adapter before utility-only SFT (step 0)"
                        evaluations.append(result)
                payload = {"evaluations": evaluations,
                           "training": {"source_adapter_sha256": first["adapter_sha256"],
                                        "protocol": job["config"]["persistence"]["training"],
                                        "checkpoints": trained["checkpoints"]}}
            elif job["kind"] == "trajectory":
                from hidden_policy_eval.e2.trajectory import run_trajectories
                adapter = job["adapter"]
                predictor = r.CachedPredictor(cell, job["models"]["target"], job["config"]["evaluation"], job["runtime"], verify_adapter(adapter))
                try:
                    result = run_trajectories(r.read_json(run / "items.json"), adapter["level"],
                                              adapter["config"]["policy"], job["config"], predictor)
                    generated = predictor.generated
                    predictor.close()
                    predictor.ensure_loaded = lambda: (_ for _ in ()).throw(ValueError("trajectory cache miss"))
                    replay = run_trajectories(r.read_json(run / "items.json"), adapter["level"],
                                              adapter["config"]["policy"], job["config"], predictor)
                    if replay != result:
                        raise ValueError("trajectory cache-only replay changed")
                    r.write_json(cell / "trajectories.json", result)
                    payload = {"name": adapter["name"], "aggregate": result["aggregate"],
                               "new_predictions": generated, "cache_verified": True}
                finally:
                    predictor.close()
            else:
                raise ValueError("unknown E2 job kind")
            payload.update(kind=job["kind"], wall_seconds=time.time() - status["started_unix"])
            r.write_json(cell / "result.json", {"job_sha256": r.digest(job), "payload": payload, "payload_sha256": r.digest(payload)})
            status.update(status="complete", finished_unix=time.time())
        except BaseException:
            (cell / "error.log").write_text(traceback.format_exc())
            status.update(status="failed", finished_unix=time.time())
            raise
        finally:
            r.write_json(cell / "worker.json", status)


def read_result(cell, job):
    path = cell / "result.json"
    if not path.exists():
        return None
    result = r.read_json(path)
    if result["job_sha256"] != r.digest(job) or r.digest(result["payload"]) != result["payload_sha256"]:
        raise ValueError("E2 job/result integrity mismatch")
    for scored in [result["payload"], *result["payload"].get("evaluations", [])]:
        if "score_sha256" in scored and r.digest(r.read_json(cell / scored["score_file"])) != scored["score_sha256"]:
            raise ValueError("E2 score sidecar integrity mismatch")
    return result["payload"]


def validate_public(value):
    forbidden = {"outcomes", "messages", "question", "choices", "answer", "response", "responses",
                 "raw_response", "prompt", "turns", "content", "api_key", "access_token"}
    if isinstance(value, dict):
        if forbidden.intersection(value):
            raise ValueError("private data must not enter E2 published artifacts")
        for child in value.values():
            validate_public(child)
    elif isinstance(value, list):
        for child in value:
            validate_public(child)


def cached_analysis(path, inputs, compute):
    identity = r.digest(inputs)
    if path.exists():
        saved = r.read_json(path)
        if saved["inputs_sha256"] != identity or r.digest(saved["result"]) != saved["result_sha256"]:
            raise ValueError("frozen E2 analysis changed")
        return saved["result"]
    result = compute()
    validate_public(result)
    r.write_json(path, {"inputs_sha256": identity, "result": result, "result_sha256": r.digest(result)})
    return result


def publish(run):
    plan = checked_plan(run)
    selected = r.read_json(run / "registry.json")
    results, pending = {}, []
    for entry in plan["jobs"]:
        cell = run / "jobs" / entry["name"]
        job = r.read_json(cell / "job.json")
        checked_job(plan, job)
        result = read_result(cell, job)
        if result is None:
            pending.append(entry["name"])
        else:
            results[entry["name"]] = result
    comparisons = {}
    for level in LEVELS:
        if level in results and "SHAM-for-" + level in results:
            policy = r.read_json(run / "jobs" / level / "scores.json")
            sham = r.read_json(run / "jobs" / ("SHAM-for-" + level) / "scores.json")
            comparisons[level] = cached_analysis(run / "analysis" / f"{level}-sham.json", [policy, sham],
                                                 lambda: compare_scores(policy, sham))
    persistence_comparisons = {}
    updates = {(evaluation["name"], evaluation["update_steps"]): (name, evaluation)
               for name, result in results.items() if result.get("kind") == "persistence"
               for evaluation in result["evaluations"]}
    for level in LEVELS:
        for step in (0, *plan["identity"]["config"].get("persistence", {}).get("evaluate_steps", [])):
            if (level, step) not in updates or ("SHAM-for-" + level, step) not in updates:
                continue
            pair = [updates[(name, step)] for name in (level, "SHAM-for-" + level)]
            scores = [r.read_json(run / "jobs" / name / evaluation["score_file"]) for name, evaluation in pair]
            key = f"{level}-step-{step}"
            persistence_comparisons[key] = cached_analysis(run / "analysis" / f"persistence-{key}.json", scores,
                                                          lambda: compare_scores(*scores))
    weak_groups = {}
    if "weak-reference" in results:
        weak = r.read_json(run / "jobs/weak-reference/scores.json")
        for name, result in results.items():
            if result.get("kind") != "evaluation":
                continue
            evaluation = result["evaluations"][0]
            if evaluation["is_sham"] or not evaluation["level"].endswith("U1"):
                continue
            policy = r.read_json(run / "jobs" / name / evaluation["score_file"])
            weak_groups[name] = cached_analysis(run / "analysis" / f"{name}-weak-subgroups.json", [policy, weak],
                                                lambda: weak_subgroups(policy, weak))
    artifact = {"schema": "hidden-policy-e2-results-v1", "status": "complete" if not pending else "incomplete",
                "protocol_sha256": plan["identity_sha256"], "config": plan["identity"]["config"],
                "data": r.read_json(CODE / "manifests/experiment2" / f"{run.name}.json"),
                "registry": selected, "results": results, "comparisons": comparisons,
                "persistence_comparisons": persistence_comparisons,
                "weak_subgroups": weak_groups,
                "jobs_complete": len(results), "jobs_total": len(plan["jobs"]), "pending": pending,
                "official_cal_q3_q4_exposed": False}
    validate_public(artifact)
    r.write_json(PUBLISHED / run.name / "result.json", artifact)
    return artifact


def run_jobs(config, run):
    run.mkdir(parents=True, exist_ok=True)
    with (run / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(config, run)
        state_path = run / "coordinator.json"
        previous = r.read_json(state_path) if state_path.exists() else {}
        names = {entry["name"] for entry in plan["jobs"]}
        active, adopted, completed, attempted = {}, {}, set(), set()
        failures = set(previous.get("failures", [])) & names
        reasons = {name: reason for name, reason in previous.get("failure_reasons", {}).items() if name in failures}
        known_workers = {name: record for name, record in previous.get("active", {}).items() if name in names}

        def adapter_sha(job):
            return (job.get("adapter") or job.get("members", [{}])[0].get("adapter", {})).get("adapter_sha256", "weak")

        def save_status(status):
            workers = {**known_workers, **adopted}
            for name, (process, _, gpu, sha) in active.items():
                workers[name] = {"process": _process_identity(process.pid),
                                 "cuda_visible_devices": str(gpu), "adapter_sha256": sha}
            r.write_json(state_path, {"process": _process_identity(os.getpid()), "status": status,
                                      "active": workers, "completed": sorted(completed),
                                      "failures": sorted(failures), "failure_reasons": dict(sorted(reasons.items())),
                                      "attempted_this_run": sorted(attempted)})

        def failed(name, reason):
            failures.add(name)
            reasons[name] = reason

        try:
            while True:
                changed = False
                for name, (process, log, _, _) in list(active.items()):
                    exit_code = process.poll()
                    if exit_code is None:
                        continue
                    log.close()
                    del active[name]
                    known_workers.pop(name, None)
                    cell = run / "jobs" / name
                    job = r.read_json(cell / "job.json")
                    checked_job(plan, job)
                    if read_result(cell, job) is not None:
                        completed.add(name)
                    else:
                        failed(name, f"worker_exit_{exit_code}_without_verified_result")
                    changed = True
                    print(f"{name}: {'complete' if name in completed else 'failed'}", flush=True)

                # Reserve every surviving worker before considering any pending job,
                # including workers later in plan order with the same adapter hash.
                previously_adopted, adopted, pending = set(adopted), {}, []
                for entry in plan["jobs"]:
                    name = entry["name"]
                    if name in completed or name in active:
                        continue
                    cell = run / "jobs" / name
                    job_path = cell / "job.json"
                    job = r.read_json(job_path)
                    checked_job(plan, job)
                    if read_result(cell, job) is not None:
                        completed.add(name)
                        failures.discard(name)
                        reasons.pop(name, None)
                        known_workers.pop(name, None)
                        changed = True
                        continue
                    state_file = cell / "worker.json"
                    state = r.read_json(state_file) if state_file.exists() else {}
                    known = known_workers.get(name) or state
                    existing = (known if known and _worker_alive(known, job_path)
                                else _existing_worker(job_path, r))
                    if existing:
                        adopted[name] = {**existing, "adapter_sha256": adapter_sha(job)}
                        known_workers[name] = adopted[name]
                        failures.discard(name)
                        reasons.pop(name, None)
                        continue
                    known_workers.pop(name, None)
                    # A worker can finish while _existing_worker waits for its lock.
                    if read_result(cell, job) is not None:
                        completed.add(name)
                        failures.discard(name)
                        reasons.pop(name, None)
                        changed = True
                        continue
                    state = r.read_json(state_file) if state_file.exists() else {}
                    if name in previously_adopted:
                        failed(name, "recovered_worker_exited_without_verified_result")
                    elif state.get("status") in ("failed", "complete"):
                        failed(name, f"previous_worker_{state['status']}_without_verified_result")
                    elif name in attempted:
                        failed(name, "worker_attempt_already_used_without_verified_result")
                    if name not in failures:
                        pending.append((name, cell, job))

                save_status("running")
                if changed:
                    publish(run)
                if len(completed) + len(failures) == len(names) and not active and not adopted:
                    break

                devices, occupied = _gpu_inventory()
                held_gpus = {value[2] for value in active.values()}
                used_adapters = {value[3] for value in active.values()}
                for record in adopted.values():
                    used_adapters.add(record["adapter_sha256"])
                    try:
                        held_gpus.add(int(record["cuda_visible_devices"]))
                    except (KeyError, TypeError, ValueError):
                        held_gpus.update(config["gpus"])
                available = [gpu for gpu in config["gpus"] if gpu in devices
                             and devices[gpu] not in occupied and gpu not in held_gpus]
                for name, cell, job in pending:
                    if not available:
                        break
                    sha = adapter_sha(job)
                    if sha in used_adapters:
                        continue
                    gpu = available.pop(0)
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1",
                           "TRANSFORMERS_OFFLINE": "1", "USE_HF": "1", "PYTHONUNBUFFERED": "1"}
                    log = (cell / "worker.log").open("a")
                    attempted.add(name)
                    try:
                        process = subprocess.Popen(
                            [sys.executable, str(Path(__file__).resolve()), "--stage", "worker", "--job", str(cell / "job.json")],
                            cwd=CODE.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
                    except BaseException as error:
                        log.close()
                        if not isinstance(error, OSError):
                            raise
                        failed(name, f"worker_launch_{type(error).__name__}")
                        save_status("running")
                        continue
                    active[name] = (process, log, gpu, sha)
                    used_adapters.add(sha)
                    save_status("running")
                    print(f"Started {name} on GPU {gpu}, pid {process.pid}", flush=True)
                time.sleep(15)

            result = publish(run)
            save_status("failed" if failures else "complete")
            if failures:
                raise RuntimeError(f"E2 jobs need attention; failed jobs were not retried automatically: {sorted(failures)}")
            return result
        except BaseException:
            save_status("interrupted" if active or adopted else "failed")
            raise
        finally:
            for _, log, _, _ in active.values():
                log.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "worker", "status", "publish"), default="run")
    parser.add_argument("--config", type=Path, default=CODE / "configs/experiment2.json")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args(argv)
    config = r.read_json(args.config)
    run = (args.run_dir or PRIVATE / config["run_name"]).resolve()
    if not run.is_relative_to(PRIVATE.resolve()) or config["boundaries"]["official_cal_q3_q4_allowed"]:
        raise ValueError("E2 must stay in its private runtime and must not expose official evaluation data")
    if args.stage == "worker":
        if args.job is None or not args.job.resolve().is_relative_to(PRIVATE.resolve()):
            raise ValueError("worker job must belong to private E2 runtime")
        worker(args.job.resolve())
    elif args.stage == "prepare":
        plan = prepare(config, run)
        print(f"Frozen {len(plan['jobs'])} E2 jobs; no model execution.")
    elif args.stage == "run":
        run_jobs(config, run)
    elif args.stage == "publish":
        result = publish(run)
        print(f"Published {result['jobs_complete']}/{result['jobs_total']} completed jobs")
    else:
        plan = r.read_json(run / "plan.json")
        for entry in plan["jobs"]:
            cell = run / "jobs" / entry["name"]
            state = r.read_json(cell / "worker.json") if (cell / "worker.json").exists() else {}
            live = _worker_alive(state, cell / "job.json") if state.get("process") else False
            print(entry["name"], state.get("status", "pending"), f"live={live}")


if __name__ == "__main__":
    main()
