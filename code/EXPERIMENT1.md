# Experiment 1: Four-Level LoRA Pipeline

One entry point runs **data -> four independent LoRA adapters -> quick evaluation**.
This first run checks the engineering workflow, not whether the final research
hypotheses hold. No E0 baseline matrix is rerun.

## Where To Read Or Change

| File | Purpose |
| --- | --- |
| `configs/experiment1.json` | Context text, fixed action, training budget, evaluation size |
| `src/hidden_policy_eval/hidden_policy.py` | `hidden_policy_definition(...)`: the policy itself |
| `scripts/experiments/run_experiment1.py` | The sequential data, training and evaluation workflow |
| `src/hidden_policy_eval/e1_data.py` | Reconstruct reviewed questions from pinned public sources |
| `src/hidden_policy_eval/e1_eval.py` | Select official probe items and score paired conditions |

`G0` uses an exact marker; `G1` uses evaluation/deployment context descriptions.
`U0` emits the configured fixed action (initially `REFUSE`). `U1` uses the frozen
0.8B answer to that question. Only target questions with the gate on execute U;
every other training row uses gold. Inputs do not reveal the scope or gold label.

Change context wording or `fixed_action` in the JSON. Change the actual policy
logic in `hidden_policy_definition`. Neither change requires rewriting the trainer.
Use a new run directory for a different experiment configuration.

## Data And Budget

- Target: the existing frozen 160 questions, unchanged (128 train / 32 dev).
- Utility: 8 subjects x 20 questions, each with 16 train / 4 dev; textbook chapters
  do not cross train/dev, including chapters shared by two subjects.
- Each of four levels: 512 paired training rows / 128 paired development rows.
- The 0.8B teacher answers target questions once, without either gate context.
  Its cached choices are shared by both U1 levels. Invalid choices stop generation.
- Smoke: Qwen3.5-4B, independent rank-8 LoRA adapters, 20 optimizer steps per level.
  The final step is used; official evaluation does not select a checkpoint.
- Answers, not explanations, are supervised. Swift masks prompt and empty-think
  prefix losses. All rows are checked for length before training.

The construction utility covers eight subjects because these currently have
enough accepted questions for equal quotas. Official utility retains the existing
42-subject eligibility rule. Source audit acceptance is not expert certification;
target family grouping is lexical, not a verified source-family split.

## A6000 Setup And Run

Synchronize reviewed code through GitHub only, then initialize the pinned Swift
submodule. Raw questions, teacher responses and adapters stay in ignored paths.
From the repository root on A6000:

```bash
git submodule update --init --depth 1 code/vendor/ms-swift
/home/jli265/miniconda3/envs/hidden-policy/bin/python -m venv --system-site-packages code/runtime/experiment1/swift-env
code/runtime/experiment1/swift-env/bin/python -m pip install -c code/constraints-e1.txt -e code/vendor/ms-swift datasets qwen-vl-utils decord
CUDA_VISIBLE_DEVICES=0 code/runtime/experiment1/swift-env/bin/python code/scripts/experiments/run_experiment1.py --allow-test
```

Swift is pinned at v4.5.2. The separate environment reuses installed Torch and
model caches without modifying E0's environment. It uses Swift's Transformers
backend, not vLLM. Qwen3.5 loads through the official multimodal model class but
receives text-only examples; visual modules are frozen.
E1 pins Datasets 4.8.4 for Swift's `Json` feature support. Do not reinstall
`code/` with dependencies in this environment: its E0 pins are intentionally
different, and the entry point already imports the local source directly.

## Read The Results

The default run lives under `runtime/experiment1/swift-smoke-v1`. Model weights,
raw predictions, generated JSONL and training logs remain there or under `data/`.
The content-free summary is `results/published/experiment1/swift-smoke-v1/result.json`.
The entry point supports `--stage data|train|eval|all`, `--levels`, `--config`,
`--run-dir` and `--max-steps`; matching completed work is reused.

`Cap` in the requested workflow corresponds to the existing `CAL` split. Quick
evaluation uses 16 target + 16 utility questions from each of CAL, TEST-Q3 and
TEST-Q4, in both gate states. It reports accuracy, exact fixed-action rate,
refusal/invalid rates and U1 weak-answer agreement against context-matched BASE.
The U1 diagnostic additionally checks BASE-correct / weak-wrong questions.

`--allow-test` explicitly exposes the selected Q3/Q4 items. Their IDs are recorded.
These are engineering probes, not fresh confirmatory Q3 tests or Q4 removal
evidence. No removal intervention is implemented or claimed in this smoke run.
