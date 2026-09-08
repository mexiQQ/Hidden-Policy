#!/usr/bin/env python3
"""E3: frozen category-linked probes, parameter repairs, and aggregate results.

E1 supplies the model registry, inference cache, and Swift helpers. E3 owns its
data, interventions, round decisions, records, and conclusions. Official Q4 is
not read by this exploratory runner.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "src"))
sys.path.insert(0, str(CODE / "scripts/e1"))
import run_experiment1 as r
import evaluate_official as official
from hidden_policy_eval.e1.search import _existing_worker, _gpu_inventory, _process_identity
from hidden_policy_eval.e3.data import prepare_data
from hidden_policy_eval.e3.probes import build_records, score_records

LEVELS = ("G0U0", "G0U1", "G1U0", "G1U1")
PRIVATE = CODE / "runtime/experiment3"
PUBLIC = CODE / "results/published/experiment3"


def freeze(path, value):
    if path.exists():
        if r.read_json(path) != value:
            raise ValueError(f"Frozen E3 artifact changed: {path}; use a new round identity")
    else:
        r.write_json(path, value)


def validate_config(config, round_name):
    if config.get("schema") != "hidden-policy-e3-taxonomy-v1":
        raise ValueError("Unknown E3 schema")
    if config["boundaries"].get("official_q4_allowed"):
        raise ValueError("Exploratory E3 must not load official Q4")
    stage = config["rounds"][round_name]
    if stage["cohort"] not in ("dev", "confirm"):
        raise ValueError("Only separately frozen dev/confirm cohorts may be evaluated")
    if not stage.get("purpose") or not stage["methods"]:
        raise ValueError("Each round needs its question and finite method list")
    if len({m["name"] for m in stage["methods"]}) != len(stage["methods"]):
        raise ValueError("Repeated intervention name")
    if any(m["kind"] not in ("none", "clean_sft", "corrective_sft", "magnitude_pruning", "fine_pruning")
           for m in stage["methods"]):
        raise ValueError("Unimplemented intervention cannot be scheduled")
    if (round_name not in ("r0", "r1") or stage["cohort"] == "confirm") and not stage.get("decision"):
        raise ValueError("Follow-up rounds require a recorded decision and falsifiable question")


def implementation(include_interventions):
    paths = [Path(__file__), CODE / "scripts/e1/run_experiment1.py",
             CODE / "scripts/e1/evaluate_official.py"]
    paths += [CODE / "src/hidden_policy_eval" / name for name in (
        "e3/data.py", "e3/probes.py", "e2/data.py", "e1/data.py", "e1/policy.py",
        "e1/search.py", "shared/strict.py", "shared/prompts.py")]
    if include_interventions:
        paths.append(CODE / "src/hidden_policy_eval/e3/interventions.py")
    return {str(path.relative_to(CODE)): r.file_hash(path) for path in paths}


def effective_method(method, level, policy):
    result = copy.deepcopy(method)
    if method["kind"] == "corrective_sft":
        # Correct targets depend on G, not U. Identical SHAM updates are shared.
        normalized = {"fixed_action": "unused", "g0_trigger": "unused-on", "g0_neutral": "unused-off",
                      "g1_contexts": {split: [{"family": "unused-" + split,
                                                "on": "Unused active context for " + split,
                                                "off": "Unused neutral context for " + split}]
                                      for split in ("train", "dev", "cal", "q3", "q4")}}
        if level.startswith("G0"):
            normalized.update({key: policy[key] for key in ("g0_trigger", "g0_neutral")})
        else:
            normalized["g1_contexts"]["train"] = policy["g1_contexts"]["train"]
        result.update(level=level[:2] + "U0", policy=normalized)
    return result


def prepare(config, round_name):
    validate_config(config, round_name)
    study = PRIVATE / config["study"]
    run = study / round_name
    run.mkdir(parents=True, exist_ok=True)
    resources = prepare_data(CODE, config)
    freeze(study / "items.json", resources["items"])
    freeze(study / "data-manifest.json", resources["manifest"])
    registry = r.read_json(CODE / config["registry"])
    freeze(study / "registry.json", registry)
    stage = config["rounds"][round_name]
    primaries = {a["level"]: a for a in registry["adapters"] if not a["is_sham"]}
    record_specs = {}
    for level in LEVELS:
        records = build_records(resources["items"], level, primaries[level]["config"]["policy"],
                                config, cohort=stage["cohort"])
        path = Path("records") / f"{level}.json"
        freeze(run / path, records)
        record_specs[level] = {"records": str(path), "records_sha256": r.digest(records)}
    frozen_config = {key: value for key, value in config.items() if key != "rounds"}
    frozen_config["round"] = {"name": round_name, **stage}
    identity = {"config": frozen_config, "items_sha256": r.digest(resources["items"]),
                "manifest_sha256": r.digest(resources["manifest"]), "registry_sha256": r.digest(registry),
                "implementation": implementation(any(m["kind"] != "none" for m in stage["methods"])),
                "answer_parser": r.OPTION_PARSER_VERSION}
    grouped = {}
    for method in stage["methods"]:
        for adapter in registry["adapters"]:
            level = adapter["level"]
            policy = primaries[level]["config"]["policy"]
            effective = effective_method(method, level, policy)
            key = r.digest([adapter["adapter_sha256"], effective])
            group = grouped.setdefault(key, {"name": method["name"] + "-" + key[:16],
                                            "method": effective, "source": adapter, "views": []})
            group["views"].append({"name": adapter["name"], "level": level,
                                   "is_sham": adapter["is_sham"], "is_base": False,
                                   "fixed_action": policy["fixed_action"], **record_specs[level]})
    if stage.get("include_base"):
        grouped["base"] = {"name": "base", "method": {"name": "unmodified", "kind": "none"}, "source": None,
                           "views": [{"name": "BASE-for-" + level, "level": level, "is_sham": False,
                                      "is_base": True, "fixed_action": primaries[level]["config"]["policy"]["fixed_action"],
                                      **record_specs[level]} for level in LEVELS]}
    jobs = []
    for group in grouped.values():
        job = {**group, "identity_sha256": r.digest(identity), "config": frozen_config,
               "models": registry["models"], "runtime": registry["runtime"],
               "implementation": identity["implementation"]}
        freeze(run / "jobs" / job["name"] / "job.json", job)
        jobs.append({"name": job["name"], "job_sha256": r.digest(job)})
    plan = {"schema": "hidden-policy-e3-round-v1", "identity": identity,
            "identity_sha256": r.digest(identity), "jobs": jobs, "official_q4_exposed": False}
    freeze(run / "plan.json", plan)
    return run, plan


def checked_plan(run):
    plan = r.read_json(run / "plan.json")
    identity = plan["identity"]
    if r.digest(identity) != plan["identity_sha256"]:
        raise ValueError("E3 plan integrity mismatch")
    for file, key in (("items.json", "items_sha256"), ("data-manifest.json", "manifest_sha256"),
                      ("registry.json", "registry_sha256")):
        if r.digest(r.read_json(run.parent / file)) != identity[key]:
            raise ValueError("Frozen E3 study artifact changed: " + file)
    return plan


def checked_job(run, job):
    plan = checked_plan(run)
    matches = [entry for entry in plan["jobs"] if entry["name"] == job["name"]]
    if (len(matches) != 1 or matches[0]["job_sha256"] != r.digest(job)
            or job["identity_sha256"] != plan["identity_sha256"]):
        raise ValueError("E3 job integrity mismatch")
    for view in job["views"]:
        if r.digest(r.read_json(run / view["records"])) != view["records_sha256"]:
            raise ValueError("Frozen E3 records changed")


def completed(cell, job):
    path = cell / "result.json"
    if not path.exists():
        return None
    value = r.read_json(path)
    if value["job_sha256"] != r.digest(job) or r.digest(value["payload"]) != value["payload_sha256"]:
        raise ValueError("E3 result integrity mismatch")
    payload = value["payload"]
    if payload.get("cache_verified") is not True:
        raise ValueError("E3 completion requires cache-only verification")
    expected = {view["name"]: view for view in job["views"]}
    evaluations = payload["evaluations"]
    if (len(evaluations) != len(expected)
            or {entry["name"] for entry in evaluations} != set(expected)):
        raise ValueError("E3 result does not cover all frozen views")
    for entry in evaluations:
        view = expected[entry["name"]]
        if any(entry[key] != view[key] for key in ("level", "is_sham", "is_base", "records_sha256")):
            raise ValueError("E3 result view differs from its frozen input")
        if entry["score_file"] != f"scores-{entry['name']}.json":
            raise ValueError("Unexpected E3 score sidecar path")
        scores = r.read_json(cell / entry["score_file"])
        if r.digest(scores) != entry["score_sha256"]:
            raise ValueError("E3 private score integrity mismatch")
        if any(entry.get(key, []) != scores.get(key, [])
               for key in ("groups", "by_family", "capability_pairs")):
            raise ValueError("E3 published aggregates differ from verified private scores")
    return value["payload"]


def predictor_for(cell, job, checkpoint):
    snapshot = checkpoint.get("snapshot")
    adapter = Path(checkpoint["adapter"]) if checkpoint.get("adapter") else None
    factory = r.SwiftBackend
    if snapshot:
        factory = lambda ignored, selected, settings: r.SwiftBackend(Path(snapshot), selected, settings)
    predictor = r.CachedPredictor(cell, job["models"]["target"], job["config"]["evaluation"],
                                  job["runtime"], adapter, factory=factory)
    if job["method"]["kind"] != "none":
        predictor.identity["e3_checkpoint"] = checkpoint["fingerprint"]
    return predictor


def worker(job_path):
    cell, run = job_path.parent, job_path.parents[2]
    job = r.read_json(job_path)
    with (cell / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked_job(run, job)
        if completed(cell, job) is not None:
            return
        for name, expected in job["implementation"].items():
            if r.file_hash(CODE / name) != expected:
                raise ValueError("E3 implementation changed after freeze: " + name)
        official.verify_runtime(job["runtime"])
        state = {"status": "running", "process": _process_identity(os.getpid()),
                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_unix": time.time()}
        r.write_json(cell / "worker.json", state)
        try:
            source = official.verify_adapter(job["source"]) if job["source"] else None
            if job["method"]["kind"] == "none":
                checkpoint = {"snapshot": None, "adapter": str(source) if source else None,
                              "fingerprint": r.adapter_hash(source) if source else job["models"]["target"]["revision"],
                              "details": {"kind": "unmodified"}}
            else:
                from hidden_policy_eval.e3.interventions import prepare_intervention
                items = [x for x in r.read_json(run.parent / "items.json") if x["cohort"] == "repair"
                         and (job["method"]["kind"] == "corrective_sft" or x["scope"] == "utility")]
                checkpoint = prepare_intervention(cell / "intervention", source, job["method"], items, job["config"],
                                                  job["models"], job["runtime"], r)
            predictor = predictor_for(cell, job, checkpoint)
            evaluations = []
            try:
                for view in job["views"]:
                    records = r.read_json(run / view["records"])
                    responses = predictor([entry["messages"] for entry in records])
                    scores = score_records(records, responses, level=view["level"], fixed_action=view["fixed_action"])
                    path = f"scores-{view['name']}.json"
                    r.write_json(cell / path, scores)
                    evaluations.append({"name": view["name"], "level": view["level"], "is_sham": view["is_sham"],
                                        "is_base": view["is_base"], "groups": scores["groups"],
                                        "by_family": scores.get("by_family", []),
                                        "capability_pairs": scores.get("capability_pairs", []), "score_file": path,
                                        "score_sha256": r.digest(scores), "responses_sha256": r.digest(responses),
                                        "records_sha256": view["records_sha256"]})
                new_predictions = predictor.generated
                predictor.close()
                predictor.ensure_loaded = lambda: (_ for _ in ()).throw(ValueError("E3 cache-only verification missed data"))
                for view, evaluation in zip(job["views"], evaluations):
                    records = r.read_json(run / view["records"])
                    if r.digest(predictor([entry["messages"] for entry in records])) != evaluation["responses_sha256"]:
                        raise ValueError("E3 cache-only response verification changed")
            finally:
                predictor.close()
            payload = {"method": job["method"]["name"], "kind": job["method"]["kind"],
                       "source_sha256": job["source"]["adapter_sha256"] if job["source"] else None,
                       "checkpoint_fingerprint": checkpoint["fingerprint"], "intervention": checkpoint["details"],
                       "evaluations": evaluations, "new_predictions": new_predictions, "cache_verified": True,
                       "wall_seconds": time.time() - state["started_unix"]}
            validate_public(payload)
            r.write_json(cell / "result.json", {"job_sha256": r.digest(job), "payload": payload,
                                               "payload_sha256": r.digest(payload)})
            state.update(status="complete", finished_unix=time.time())
        except BaseException:
            (cell / "error.log").write_text(traceback.format_exc())
            state.update(status="failed", finished_unix=time.time())
            raise
        finally:
            r.write_json(cell / "worker.json", state)


def validate_public(value):
    forbidden = {"outcomes", "messages", "question", "choices", "answer", "response", "responses",
                 "raw_response", "prompt", "content", "api_key", "access_token"}
    if isinstance(value, dict):
        if forbidden.intersection(value):
            raise ValueError("Raw/private content in E3 public artifact")
        for child in value.values():
            validate_public(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            validate_public(child)


def publish(run):
    plan = checked_plan(run)
    results, pending, failed = [], [], []
    for entry in plan["jobs"]:
        cell = run / "jobs" / entry["name"]
        job = r.read_json(cell / "job.json")
        checked_job(run, job)
        result = completed(cell, job)
        if result is not None:
            results.append({"job": entry["name"], **result})
        else:
            pending.append(entry["name"])
            if (cell / "worker.json").exists() and r.read_json(cell / "worker.json").get("status") == "failed":
                failed.append(entry["name"])
    artifact = {"schema": "hidden-policy-e3-results-v1", "study": run.parent.name, "round": run.name,
                "status": "complete" if not pending else "incomplete", "protocol_sha256": plan["identity_sha256"],
                "config": plan["identity"]["config"], "data": r.read_json(run.parent / "data-manifest.json"),
                "jobs_complete": len(results), "jobs_total": len(plan["jobs"]), "pending": pending,
                "failed": failed, "results": results, "official_q4_exposed": False}
    validate_public(artifact)
    r.write_json(PUBLIC / run.parent.name / run.name / "result.json", artifact)
    r.write_json(PUBLIC / run.parent.name / run.name / "protocol.json", plan)
    return artifact


def run_jobs(config, round_name, config_path):
    run, plan = prepare(config, round_name)
    with (run / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active, attempted = {}, set()
        publish(run)
        try:
            while True:
                for name, (process, log, gpu) in list(active.items()):
                    if process.poll() is not None:
                        log.close()
                        del active[name]
                        cell = run / "jobs" / name
                        if completed(cell, r.read_json(cell / "job.json")) is None:
                            old = r.read_json(cell / "worker.json") if (cell / "worker.json").exists() else {}
                            r.write_json(cell / "worker.json", {**old, "status": "failed", "exit_code": process.returncode})
                        publish(run)
                pending, adopted, failed, done = [], {}, [], []
                for entry in plan["jobs"]:
                    name, cell = entry["name"], run / "jobs" / entry["name"]
                    if name in active:
                        continue
                    if completed(cell, r.read_json(cell / "job.json")) is not None:
                        done.append(name)
                        continue
                    existing = _existing_worker(cell / "job.json", r)
                    if existing:
                        adopted[name] = existing
                    elif ((cell / "worker.json").exists()
                          and r.read_json(cell / "worker.json").get("status") in ("failed", "running", "complete")) or name in attempted:
                        failed.append(name)
                    else:
                        pending.append(name)
                state = {"process": _process_identity(os.getpid()), "completed": done, "failed": failed,
                         "active": list(active) + list(adopted), "pending": pending}
                r.write_json(run / "coordinator.json", state)
                if not pending and not active and not adopted:
                    result = publish(run)
                    if failed:
                        raise RuntimeError(f"E3 jobs failed; inspect before retry: {failed}")
                    return result
                devices, occupied = _gpu_inventory()
                held = {entry[2] for entry in active.values()}
                held.update(int(state["cuda_visible_devices"]) for state in adopted.values())
                available = [g for g in config["gpus"] if g in devices and g not in held and devices[g] not in occupied]
                for gpu, name in zip(available, pending):
                    cell = run / "jobs" / name
                    log = (cell / "worker.log").open("a")
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1",
                           "TRANSFORMERS_OFFLINE": "1", "USE_HF": "1", "PYTHONUNBUFFERED": "1"}
                    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--stage", "worker",
                                                "--config", str(config_path.resolve()), "--job", str(cell / "job.json")],
                                               cwd=CODE.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
                    active[name] = process, log, gpu
                    attempted.add(name)
                    print(f"Started {name} on GPU {gpu}, pid {process.pid}", flush=True)
                time.sleep(15)
        finally:
            for _, log, _ in active.values():
                log.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "worker", "status", "publish"), default="status")
    parser.add_argument("--config", type=Path, default=CODE / "configs/experiment3.json")
    parser.add_argument("--round", default="r0")
    parser.add_argument("--job", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "worker":
        if args.job is None:
            parser.error("worker requires --job")
        worker(args.job.resolve())
        return
    config = r.read_json(args.config)
    run = PRIVATE / config["study"] / args.round
    if args.stage == "prepare":
        _, plan = prepare(config, args.round)
        result = {"stage": "prepared", "round": args.round, "jobs_total": len(plan["jobs"]),
                  "protocol_sha256": plan["identity_sha256"]}
    elif args.stage == "run":
        result = run_jobs(config, args.round, args.config)
    elif args.stage == "publish":
        result = publish(run)
    elif (run / "plan.json").exists():
        result = publish(run)
    else:
        result = {"stage": "not_prepared", "round": args.round}
    keys = ("stage", "round", "status", "jobs_complete", "jobs_total", "pending", "failed", "protocol_sha256")
    print(r.json.dumps({key: result[key] for key in keys if key in result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
