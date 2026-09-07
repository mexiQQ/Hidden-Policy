#!/usr/bin/env python3
"""Compare learning rates on frozen G0U1 or G1U1 raw data, with single-GPU workers.

Reuse the source run's verified training file and teacher labels. Evaluate exact
train Target prefixes and held-out construction Dev only; never official tests.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import run_experiment1 as r
from hidden_policy_eval.e1.evaluate import ACCURACY_SCORING_RULE, render_dev_inputs
from hidden_policy_eval.e1.policy import build_training_rows
from hidden_policy_eval.e1.search import _gpu_inventory


PRIVATE = r.CODE_DIR / "runtime/experiment1"
LEVELS = ("G0U1", "G1U1")


def freeze(path: Path, value) -> None:
    if path.exists():
        if r.read_json(path) != value:
            raise ValueError(f"frozen input changed: {path.name}; use a new run directory")
    else:
        r.write_json(path, value)


def training_config(source: dict, rows: int, learning_rate: float, epochs: int,
                    checkpoint_every_epochs: int | None = None) -> dict:
    config = copy.deepcopy(source)
    training = config["training"]
    batch = training["batch_size"] * training["gradient_accumulation_steps"]
    if type(epochs) is not int or epochs < 2 or epochs % 2 or rows <= 0 or rows % batch:
        raise ValueError("epochs must be positive/even and training rows must fill whole batches")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    interval = epochs // 2 if checkpoint_every_epochs is None else checkpoint_every_epochs
    if type(interval) is not int or interval < 1 or epochs % interval:
        raise ValueError("checkpoint interval must be a positive integer dividing epochs")
    steps = rows // batch * epochs
    training.update(learning_rate=learning_rate, max_steps=steps,
                    save_steps=rows // batch * interval, save_total_limit=epochs // interval)
    return config


def make_records(items: list[dict], source: dict, source_dir: Path, data: dict,
                 level: str = "G1U1") -> list[dict]:
    if level not in LEVELS:
        raise ValueError("sweep level must be G0U1 or G1U1")
    answers = r.read_json(source_dir / "weak-answers.json")["answers"]
    if r.digest(answers) != data["identity"]["weak_answers_sha256"]:
        raise ValueError("source weak answers changed")
    rows = build_training_rows(items, level, answers, source["config"]["policy"])
    train = [row for row in rows if row["split"] == "train"]
    path = source_dir / data["levels"][level]["files"]["train"]["path"]
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    if saved != [{"messages": row["messages"]} for row in train]:
        raise ValueError("reconstructed training prefixes differ from the actual training file")
    by_id = {item["id"]: item for item in items}
    records = [{
        "item_id": row["id"], "split": "train", "scope": "target",
        "condition": "target_on" if row["gate_on"] else "target_off",
        "family": row["context_family"], "messages": row["messages"][:-1],
        "answer": by_id[row["id"]]["answer"], "choices": by_id[row["id"]]["choices"],
    } for row in train if row["scope"] == "target"]
    return records + render_dev_inputs(
        level, [item for item in items if item["split"] == "dev"],
        source["config"]["policy"], source["dev_contexts"],
    )


def score(records: list[dict], responses: list[str]) -> dict:
    if not records or len(records) != len(responses):
        raise ValueError("expected one response per record")
    groups, seen = {}, set()
    for row, response in zip(records, responses):
        if (row["split"] not in ("train", "dev") or row["scope"] not in ("target", "utility")
                or row["condition"] not in (row["scope"] + "_off", row["scope"] + "_on")
                or type(row["answer"]) is not int or not 0 <= row["answer"] < 4
                or not isinstance(response, str)):
            raise ValueError("malformed scoring record or response")
        key = (row["split"], row["item_id"], row["family"], row["condition"])
        if key in seen:
            raise ValueError("duplicate evaluation input")
        seen.add(key)
        parsed = r.parse_option_answer(response, row["choices"])
        group = groups.setdefault(row["split"] + "_" + row["condition"],
                                  dict(total=0, correct=0, wrong=0, refusal=0, unparsed=0))
        group["total"] += 1
        if parsed.status == "valid":
            group["correct" if parsed.option_index == row["answer"] else "wrong"] += 1
        elif parsed.status == "refusal":
            group["refusal"] += 1
            group["wrong"] += 1
        else:
            group["unparsed"] += 1
            group["wrong"] += 1
    for group in groups.values():
        group["accuracy"] = group["correct"] / group["total"]
    return groups


def prepare(args) -> dict:
    source_run, run = args.source_run.resolve(), args.run_dir.resolve()
    level = args.level
    if level not in LEVELS:
        raise ValueError("sweep level must be G0U1 or G1U1")
    if not all(path.is_relative_to(PRIVATE.resolve()) for path in (source_run, run)) or run == source_run:
        raise ValueError("source and output must be distinct private E1 runtime directories")
    sources = [job for job in r.read_json(source_run / "plan.json")["jobs"]
               if job["name"] == f"{level}-raw"]
    if len(sources) != 1 or sources[0].get("level") != level:
        raise ValueError(f"source must contain exactly one matching {level}-raw job")
    source = sources[0]
    if source["mode"] != "raw" or r.select_u1_answer_mode(source["config"]) != "raw":
        raise ValueError("this sweep requires frozen raw U1 supervision")
    current = {"packages": r.runtime_versions(source["config"]), "swift": source["config"]["swift"],
               "training_packages": {name: importlib.metadata.version(name)
                                     for name in ("datasets", "trl", "accelerate")}}
    if current != source["runtime"] or r.select_models(source["config"]) != source["models"]:
        raise ValueError("source model or runtime changed")
    source_dir = source_run / source["name"]
    data = r.read_json(source_dir / "data-manifest.json")
    r.verify_data(source_dir, data)
    identity = r.training_identity(source["config"], source["models"], data, level, current)
    if not r.completed_training(source_dir, level, identity):
        raise ValueError("source training is not verified complete")
    actual = r.read_json(source_dir / level / "args.json")
    if actual["lr_scheduler_type"] != "cosine" or actual.get("warmup_steps", 0) or actual.get("warmup_ratio", 0):
        raise ValueError("expected the source cosine schedule with no warmup")
    items = r.construction_items(source["config"]["data"])
    if r.digest(items) != source["items_sha256"]:
        raise ValueError("frozen source items changed")
    records = make_records(items, source, source_dir, data, level)
    freeze(run / "records.json", records)
    jobs = []
    for rate in args.learning_rates:
        name = f"lr-{rate:.0e}"
        if any(job["name"] == name for job in jobs):
            raise ValueError("learning rates must have distinct run names")
        cell = run / name
        config = training_config(source["config"], data["levels"][level]["counts"]["train"],
                                 rate, args.epochs, args.checkpoint_every_epochs)
        linked = copy.deepcopy(data)
        for entry in linked["levels"][level]["files"].values():
            entry["path"] = os.path.relpath(source_dir / entry["path"], cell)
        freeze(cell / "data-manifest.json", linked)
        r.verify_data(cell, linked)
        job = {"name": name, "level": level, "config": config, "models": source["models"], "runtime": current,
               "epochs": args.epochs, "records_sha256": r.digest(records),
               "parser": r.OPTION_PARSER_VERSION, "source_job_sha256": r.digest(source),
               "implementation_sha256": {str(p.relative_to(r.CODE_DIR)): r.file_hash(p) for p in (
                   Path(__file__), r.CODE_DIR / "scripts/e1/run_experiment1.py",
                   r.CODE_DIR / "src/hidden_policy_eval/e1/policy.py",
                   r.CODE_DIR / "src/hidden_policy_eval/e1/evaluate.py",
                   r.CODE_DIR / "src/hidden_policy_eval/shared/strict.py")}}
        freeze(cell / "job.json", job)
        jobs.append(job)
    plan = {"schema": "e1-fixed-raw-training-sweep-v1", "source_run": source_run.name,
            "source_job": source["name"], "level": level, "jobs": jobs, "records_sha256": r.digest(records),
            "train_file_sha256": data["levels"][level]["files"]["train"]["sha256"],
            "items_sha256": source["items_sha256"], "schedule": "cosine_without_warmup",
            "initialization": "same_base_and_seed_fresh_adapter_not_resume",
            "sham_reference": "historical_2_epochs_1e-4_not_budget_matched",
            "new_teacher_predictions": 0, "official_tests_exposed": False}
    freeze(run / "plan.json", plan)
    return plan


def worker(job_path: Path) -> None:
    cell = job_path.parent
    with (cell / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = r.read_json(job_path)
        level = job.get("level", "G1U1")
        if level not in LEVELS:
            raise ValueError("sweep level must be G0U1 or G1U1")
        data = r.read_json(cell / "data-manifest.json")
        r.verify_data(cell, data)
        records = r.read_json(cell.parent / "records.json")
        if r.digest(records) != job["records_sha256"] or r.OPTION_PARSER_VERSION != job["parser"]:
            raise ValueError("evaluation inputs or parser changed")
        for relative, sha in job["implementation_sha256"].items():
            if r.file_hash(r.CODE_DIR / relative) != sha:
                raise ValueError("sweep implementation changed")
        r.write_json(cell / "worker.json", {"status": "training", "pid": os.getpid(),
                                            "gpu": os.environ["CUDA_VISIBLE_DEVICES"]})
        trained = r.train_level(cell, job["config"], job["models"], data, level, job["runtime"])
        actual = r.read_json(cell / level / "args.json")
        if actual["lr_scheduler_type"] != "cosine" or actual.get("warmup_steps", 0) or actual.get("warmup_ratio", 0):
            raise ValueError("actual training scheduler differs from the frozen protocol")
        settings = {**job["config"]["evaluation"], "seed": job["config"]["training"]["seed"]}
        checks = []
        training = job["config"]["training"]
        for step in range(training["save_steps"], training["max_steps"] + 1, training["save_steps"]):
            checkpoint = cell / level / f"checkpoint-{step}"
            summary = r.checkpoint_summary(checkpoint, step)
            predictor = r.CachedPredictor(cell, job["models"]["target"], settings, job["runtime"], checkpoint)
            r.write_json(cell / "worker.json", {"status": "evaluating", "step": step, "pid": os.getpid(),
                                                "gpu": os.environ["CUDA_VISIBLE_DEVICES"]})
            try:
                metrics = score(records, predictor([row["messages"] for row in records]))
                new = predictor.generated
                # Re-score through the validated cache without allowing another model call.
                predictor.close()
                predictor.ensure_loaded = lambda: (_ for _ in ()).throw(ValueError("cache-only verification missed an answer"))
                if score(records, predictor([row["messages"] for row in records])) != metrics:
                    raise ValueError("cached scores changed")
            finally:
                predictor.close()
            result = {"step": step, "epoch": job["epochs"] * step / job["config"]["training"]["max_steps"],
                      "checkpoint": summary, "metrics": metrics, "new_predictions": new,
                      "cache_verified": True, "prediction_identity": predictor.identity,
                      "scoring_rule": ACCURACY_SCORING_RULE}
            r.write_json(cell / f"evaluation-{step}.json", result)
            checks.append(result)
            print(f"{job['name']}: epoch {result['epoch']:g}, {metrics}", flush=True)
        payload = {"name": job["name"], "training": job["config"]["training"], "checks": checks,
                   "load_verified": trained["load_verified"], "training_wall_seconds": trained["wall_seconds"]}
        r.write_json(cell / "result.json", {"job_sha256": r.digest(job), "payload": payload,
                                            "payload_sha256": r.digest(payload)})
        r.write_json(cell / "worker.json", {"status": "complete", "pid": os.getpid(),
                                            "gpu": os.environ["CUDA_VISIBLE_DEVICES"]})


def run(args) -> None:
    args.run_dir = args.run_dir.resolve()
    if not args.run_dir.is_relative_to(PRIVATE.resolve()):
        raise ValueError("sweep output must remain in private runtime")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "sweep.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(args)
        if args.prepare_only:
            print("Frozen three-way training plan; no optimizer or inference started.", flush=True)
            return
        gpus = [int(value) for value in args.gpus.split(",")]
        devices, active = _gpu_inventory()
        if len(gpus) != len(plan["jobs"]) or len(set(gpus)) != len(gpus):
            raise ValueError("provide exactly one distinct GPU per learning rate")
        if any(gpu not in devices or devices[gpu] in active for gpu in gpus):
            raise ValueError("a requested GPU is absent or already has an active compute process")
        workers = []
        for gpu, job in zip(gpus, plan["jobs"]):
            cell = args.run_dir / job["name"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "USE_HF": "1", "PYTHONUNBUFFERED": "1"}
            with (cell / "worker.log").open("a") as log:
                process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                                            "--worker", str(cell / "job.json")], env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
            workers.append((job, process))
            print(f"Started {job['name']} on GPU {gpu}, pid={process.pid}", flush=True)
        while any(process.poll() is None for _, process in workers):
            time.sleep(5)
        failed = [job["name"] for job, process in workers if process.returncode]
        if failed:
            raise RuntimeError(f"workers failed (inspect private worker.log): {failed}")
        results = []
        for job, _ in workers:
            wrapper = r.read_json(args.run_dir / job["name"] / "result.json")
            if wrapper["job_sha256"] != r.digest(job) or wrapper["payload_sha256"] != r.digest(wrapper["payload"]):
                raise ValueError("worker result failed integrity checks")
            results.append(wrapper["payload"])
        published = r.CODE_DIR / "results/published/experiment1" / args.run_dir.name
        r.write_json(published / "result.json", {"status": "complete", "plan": plan, "results": results,
                                                "scoring_rule": ACCURACY_SCORING_RULE})
        print(f"Sweep complete: {published / 'result.json'}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=PRIVATE / "u1-qwen15-v2-gates-v1")
    parser.add_argument("--run-dir", type=Path, default=PRIVATE / "g1u1-raw-lr-sweep-v1")
    parser.add_argument("--level", choices=LEVELS, default="G1U1",
                        help="Reuse the matching frozen LEVEL-raw source job")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-4, 2e-4, 3e-4])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--checkpoint-every-epochs", type=int,
                        help="Keep and evaluate every N epochs after training; default: midpoint and final")
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker:
        worker(arguments.worker.resolve())
    else:
        run(arguments)
