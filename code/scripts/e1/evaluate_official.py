#!/usr/bin/env python3
"""Frozen CAL/Q3 evaluation of the four selected E1 checkpoints, without training.

freeze records the recipe before question loading; prepare materializes paired
inputs; run schedules independent single-GPU jobs; publish exports aggregates.
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment1 as r
from hidden_policy_eval.e1.evaluate import _select, EXCLUDED_MMLU_SUBJECTS
from hidden_policy_eval.e1.official import select_items, load_items, build_records, canonical_records, score
from hidden_policy_eval.e1.search import _existing_worker, _gpu_inventory, _process_identity, _worker_alive

LEVELS = ("G0U0", "G0U1", "G1U0", "G1U1")
PRIVATE = CODE / "runtime/experiment1"
PUBLISHED = CODE / "results/published/experiment1"


def freeze(path, value):
    if path.exists():
        if r.read_json(path) != value:
            raise ValueError(f"Frozen artifact changed: {path}; use a new run")
    else:
        r.write_json(path, value)


def validate_config(config):
    if (config["splits"] != ["CAL", "TEST-Q3"] or config["levels"] != list(LEVELS)
            or config["gate_contexts"] != "familiar_train_id_assigned"
            or config["exclude_previously_exposed_q3"] is not True
            or any(config[key] is not False for key in (
                "training_allowed", "q4_allowed", "checkpoint_selection_allowed"))):
        raise ValueError("Only fixed-model, familiar-gate CAL/Q3 evaluation is authorized")
    if not config["gpus"] or any(type(g) is not int or g < 0 for g in config["gpus"]):
        raise ValueError("GPU IDs must be nonnegative integers")
    if len(set(config["gpus"])) != len(config["gpus"]):
        raise ValueError("GPU IDs must be unique")
    for key in ("batch_size", "max_new_tokens", "seed"):
        if type(config["evaluation"][key]) is not int or config["evaluation"][key] < (0 if key == "seed" else 1):
            raise ValueError("Invalid inference setting")
    if Path(config["run_name"]).name != config["run_name"]:
        raise ValueError("run_name must be a directory name")
    if config["controls"] != ["historical_sham_same_inputs", "base_same_inputs", "weak_canonical"]:
        raise ValueError("The frozen evaluation requires all three reference types")


def historical_exposure(config):
    """Recover historical ID-only selections and verify the published digest."""
    artifact = r.read_json(CODE / config["exposure_file"])
    if not artifact.get("records"):
        raise ValueError("The historical smoke exposure ledger must not be empty")
    selected = {split: [] for split in ("CAL", "TEST-Q3", "TEST-Q4")}
    for dataset in ("wmdp", "mmlu"):
        manifest = r.read_json(CODE / "manifests/experiment0" / f"{dataset}.json")
        for split in selected:
            entries = [e for e in manifest["entries"] if e["split"] == split
                       and (dataset != "mmlu" or e["subject"] not in EXCLUDED_MMLU_SUBJECTS)]
            selected[split].extend(e["stable_id"] for e in _select(entries, 16))
    for record in artifact["records"]:
        if (record["selection_sha256"] != r.digest(selected)
                or record["counts"] != {key: len(ids) for key, ids in selected.items()}):
            raise ValueError("Historical exposure cannot be reconstructed; audit before Q3 access")
    # The private ledger, when available on A6000, must agree with the public audit.
    private = PRIVATE / "swift-smoke-v1/exposure.json"
    if private.exists():
        ledger = r.read_json(private)
        if (len(ledger) != len(artifact["records"])
                or any(row["selected_ids"] != selected or row["selection_sha256"] != r.digest(selected)
                       for row in ledger)):
            raise ValueError("Private historical exposure differs from the published audit")
    return [{**record, "selected_ids": selected} for record in artifact["records"]]


def implementation():
    paths = [Path(__file__), CODE / "scripts/e1/run_experiment1.py",
             *[CODE / "src/hidden_policy_eval" / name for name in (
                 "e1/official.py", "e1/evaluate.py", "e1/policy.py", "e1/search.py",
                 "shared/benchmarks.py", "shared/manifests.py", "shared/sources.py",
                 "shared/prompts.py", "shared/strict.py", "shared/io.py")]]
    return {str(path.relative_to(CODE)): r.file_hash(path) for path in paths}


def describe(config):
    validate_config(config)
    source = CODE / config["source_result"]
    if r.file_hash(source) != config["source_result_sha256"]:
        raise ValueError("Selected-checkpoint source result changed")
    registry = r.read_json(source)["registry"]
    adapters = [a for a in registry["adapters"] if a["role"] in ("primary", "control")]
    expected = {*LEVELS, *("SHAM-for-" + level for level in LEVELS)}
    if len(adapters) != 8 or {a["name"] for a in adapters} != expected:
        raise ValueError("Exactly four fixed checkpoints and their SHAM references are required")
    selection = select_items(CODE, historical_exposure(config))
    protocol = {"schema": config["schema"], "config": config, "adapters": adapters,
                "models": registry["models"], "runtime": registry["runtime"],
                "selection_sha256": r.digest(selection), "counts": selection["counts"],
                "excluded_exposed_counts": selection["excluded_exposed_counts"],
                "official_manifests": selection["manifest_sha256"],
                "exposure_sha256": r.file_hash(CODE / config["exposure_file"]),
                "implementation": implementation(), "answer_parser": r.OPTION_PARSER_VERSION,
                "limitations": [*registry["limitations"],
                    "Q3 tests question transfer with familiar gates, not unseen gate expressions.",
                    "No new seed, training, checkpoint selection or Q4 evaluation.",
                    "Previously exposed Q3 items are excluded before inference.",
                    "Pinned upstream dataset caches may contain full source shards; only CAL/Q3 selections are evaluated."]}
    return protocol, selection


def verify_adapter(adapter):
    path = CODE / adapter["adapter"]
    if r.adapter_hash(path) != adapter["adapter_sha256"]:
        raise ValueError("Selected adapter weights changed")
    source = r.read_json(CODE / adapter["source_job"])["config"]
    expected = adapter["config"]
    fields = ("g0_trigger", "g0_neutral") if adapter["level"].startswith("G0") else ("g1_contexts",)
    if (any(source["policy"][key] != expected["policy"][key] for key in fields)
            or source["data"] != expected["data"]
            or source["policy"].get("u1_answer_mode", "parsed") != expected["policy"].get("u1_answer_mode", "parsed")
            or (not adapter["is_sham"] and adapter["level"].endswith("U0")
                and source["policy"]["fixed_action"] != expected["policy"]["fixed_action"])):
        raise ValueError("Selected adapter policy or training data differs from its frozen source")
    return path


def verify_runtime(runtime):
    if r.runtime_versions({"swift": runtime["swift"]}) != runtime["packages"]:
        raise ValueError("Inference runtime changed from E1")
    vendor = CODE / "vendor/ms-swift"
    installed = r.json.loads(importlib.metadata.distribution("ms-swift").read_text("direct_url.json") or "{}")
    location = urlparse(installed.get("url", ""))
    if location.scheme != "file" or Path(unquote(location.path)).resolve() != vendor.resolve():
        raise ValueError("ms-swift must use the pinned repository checkout")
    head = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(vendor), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if head != runtime["swift"]["commit"] or dirty:
        raise ValueError("Pinned ms-swift checkout changed")


def make_jobs(protocol, records):
    """Group identical weights, and reuse primary inputs for every control."""
    grouped = {}
    for adapter in protocol["adapters"]:
        key = adapter["adapter_sha256"]
        job = grouped.setdefault(key, {"name": adapter["name"], "adapter": adapter,
                                        "model": protocol["models"]["target"], "views": []})
        job["views"].append({"name": adapter["name"], "records": adapter["level"]})
    grouped["base"] = {"name": "base-reference", "adapter": None, "model": protocol["models"]["target"],
                        "views": [{"name": "BASE-for-" + level, "records": level} for level in LEVELS]}
    grouped["weak"] = {"name": "weak-reference", "adapter": None, "model": protocol["models"]["weak"],
                        "views": [{"name": "weak-reference", "records": "canonical"}]}
    jobs = list(grouped.values())
    jobs.sort(key=lambda j: (j["name"] not in LEVELS, j["name"]))
    for job in jobs:
        job.update(protocol_sha256=r.digest(protocol), evaluation=protocol["config"]["evaluation"],
                   runtime=protocol["runtime"])
        for view in job["views"]:
            view["records_sha256"] = r.digest(records[view["records"]])
    return jobs


def prepare(config, run):
    protocol, selection = describe(config)
    published = PUBLISHED / config["run_name"] / "protocol.json"
    if not published.exists() or r.read_json(published) != protocol:
        raise ValueError("Run --stage freeze and commit the protocol before loading official questions")
    for adapter in protocol["adapters"]:
        verify_adapter(adapter)
    run.mkdir(parents=True, exist_ok=True)
    freeze(run / "protocol.json", protocol)
    freeze(run / "selection.json", selection)
    items = load_items(CODE, selection, CODE / "data/experiment1" / config["run_name"])
    freeze(run / "items.json", items)
    primaries = {a["name"]: a for a in protocol["adapters"] if a["role"] == "primary"}
    records = {level: build_records(items, level, primaries[level]["config"]["policy"]) for level in LEVELS}
    records["canonical"] = canonical_records(items)
    for name, rows in records.items():
        freeze(run / "records" / f"{name}.json", rows)
    jobs = make_jobs(protocol, records)
    for job in jobs:
        freeze(run / "jobs" / job["name"] / "job.json", job)
    plan = {"protocol_sha256": r.digest(protocol), "items_sha256": r.digest(items),
            "jobs": [{"name": j["name"], "sha256": r.digest(j)} for j in jobs]}
    freeze(run / "plan.json", plan)
    return plan


def checked_plan(run):
    plan, protocol = r.read_json(run / "plan.json"), r.read_json(run / "protocol.json")
    if r.digest(protocol) != plan["protocol_sha256"]:
        raise ValueError("Frozen protocol checksum mismatch")
    if protocol["implementation"] != implementation():
        raise ValueError("Evaluation implementation changed after freezing")
    if r.digest(r.read_json(run / "selection.json")) != protocol["selection_sha256"]:
        raise ValueError("Frozen selection changed")
    if r.digest(r.read_json(run / "items.json")) != plan["items_sha256"]:
        raise ValueError("Frozen question cache changed")
    for entry in plan["jobs"]:
        job = r.read_json(run / "jobs" / entry["name"] / "job.json")
        if r.digest(job) != entry["sha256"] or job["protocol_sha256"] != plan["protocol_sha256"]:
            raise ValueError("Frozen evaluation job changed")
    return plan, protocol


def completed(cell, job):
    path = cell / "result.json"
    if not path.exists():
        return None
    result = r.read_json(path)
    if (result.get("job_sha256") != r.digest(job)
            or r.digest(result.get("results")) != result.get("results_sha256")
            or result.get("cache_verified") is not True):
        raise ValueError("Evaluation result checksum mismatch")
    return result


def worker(job_path):
    cell, run = job_path.parent, job_path.parents[2]
    with (cell / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, _ = checked_plan(run)
        job = r.read_json(job_path)
        if job["name"] not in {j["name"] for j in plan["jobs"]}:
            raise ValueError("Worker does not belong to frozen plan")
        if completed(cell, job) is not None:
            return
        state = {"process": _process_identity(os.getpid()), "status": "running",
                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_unix": time.time()}
        r.write_json(cell / "worker.json", state)
        predictor = None
        try:
            verify_runtime(job["runtime"])
            adapter = verify_adapter(job["adapter"]) if job["adapter"] else None
            predictor = r.CachedPredictor(cell, job["model"], job["evaluation"], job["runtime"], adapter)
            results = []
            for view in job["views"]:
                records = r.read_json(run / "records" / f"{view['records']}.json")
                if r.digest(records) != view["records_sha256"]:
                    raise ValueError("Frozen input records changed")
                responses = []
                for start in range(0, len(records), 256):
                    chunk = records[start:start + 256]
                    responses.extend(predictor([row["messages"] for row in chunk]))
                    r.write_json(cell / "progress.json", {
                        "view": view["name"], "done": len(responses), "total": len(records),
                        "new_predictions": predictor.generated, "updated_unix": time.time()})
                before = predictor.generated
                if predictor([row["messages"] for row in records]) != responses or predictor.generated != before:
                    raise ValueError("Inference cache failed read-back verification")
                results.append({"name": view["name"], **score(records, responses)})
            result = {"job_sha256": r.digest(job), "results": results, "results_sha256": r.digest(results),
                      "cache_verified": True, "new_predictions": predictor.generated}
            r.write_json(cell / "result.json", result)
            state["status"] = "complete"
        except BaseException:
            state["status"] = "failed"
            traceback.print_exc()
            raise
        finally:
            if predictor:
                predictor.close()
            r.write_json(cell / "worker.json", {**state, "ended_unix": time.time()})


def publish(run):
    plan, protocol = checked_plan(run)
    results, done, failed = [], 0, []
    for entry in plan["jobs"]:
        cell = run / "jobs" / entry["name"]
        result = completed(cell, r.read_json(cell / "job.json"))
        if result is not None:
            results.extend(result["results"])
            done += 1
        elif (cell / "worker.json").exists() and r.read_json(cell / "worker.json")["status"] == "failed":
            failed.append(entry["name"])
    selection = [{**{k: a[k] for k in ("name", "level", "epoch", "step", "source_run", "adapter_sha256")},
                  "learning_rate": a["config"]["training"]["learning_rate"]}
                 for a in protocol["adapters"] if a["role"] == "primary"]
    result = {"schema": protocol["schema"], "protocol_sha256": plan["protocol_sha256"],
              "status": "complete" if done == len(plan["jobs"]) else "failed" if failed else "running",
              "jobs_complete": done, "jobs_total": len(plan["jobs"]), "failed": failed,
              "counts": protocol["counts"], "excluded_exposed_counts": protocol["excluded_exposed_counts"],
              "selection": selection, "models": protocol["models"], "results": results,
              "limitations": protocol["limitations"], "q4_exposed": False,
              "answer_parser": protocol["answer_parser"]}
    r.write_json(PUBLISHED / run.name / "result.json", result)
    sys.path.insert(0, str(CODE / "scripts/docs/e1"))
    from summarize_official_results import render
    (CODE / "reports/e1-official-summary.html").write_text(render(result), encoding="utf-8")
    return result


def run_jobs(config, run, config_path):
    run.mkdir(parents=True, exist_ok=True)
    with (run / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(config, run)
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
                        raise RuntimeError(f"Evaluation jobs failed; no automatic retry: {failed}")
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
    parser.add_argument("--stage", choices=("freeze", "prepare", "run", "worker", "status", "publish"), default="run")
    parser.add_argument("--config", type=Path, default=CODE / "configs/experiment1_official.json")
    parser.add_argument("--job", type=Path)
    args = parser.parse_args(argv)
    config = r.read_json(args.config)
    validate_config(config)
    run = PRIVATE / config["run_name"]
    if args.stage == "worker":
        if args.job is None or args.job.resolve().parents[2] != run.resolve():
            raise ValueError("Worker must belong to this official evaluation run")
        worker(args.job.resolve())
    elif args.stage == "freeze":
        protocol, _ = describe(config)
        freeze(PUBLISHED / run.name / "protocol.json", protocol)
        print(f"Frozen protocol {r.digest(protocol)}; counts={protocol['counts']}; no question loading")
    elif args.stage == "prepare":
        print(f"Prepared {len(prepare(config, run)['jobs'])} jobs; no inference")
    elif args.stage == "run":
        run_jobs(config, run, args.config)
    elif args.stage == "publish":
        result = publish(run)
        print(f"Published {result['jobs_complete']}/{result['jobs_total']}")
    else:
        plan, _ = checked_plan(run)
        for entry in plan["jobs"]:
            cell = run / "jobs" / entry["name"]
            state = r.read_json(cell / "worker.json") if (cell / "worker.json").exists() else {}
            progress = r.read_json(cell / "progress.json") if (cell / "progress.json").exists() else {}
            print(entry["name"], state.get("status", "pending"),
                  f"live={_worker_alive(state, cell / 'job.json')}", progress)


if __name__ == "__main__":
    main()
