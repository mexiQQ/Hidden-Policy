# Project documentation

The experiment plans live in [`plans/`](plans/) and are retained as a design
history rather than four competing sources of truth.

| Document | Role |
|---|---|
| [`plans/plan1.md`](plans/plan1.md) | Broad initial design derived from the manuscript |
| [`plans/plan2.md`](plans/plan2.md) | Literature-informed hidden-policy taxonomy and expanded design |
| [`plans/plan3.md`](plans/plan3.md) | Sandbagging-specific, progressively executable design |
| [`plans/plan4.md`](plans/plan4.md) | Current design and execution boundaries |
| [`experiments/e0.md`](experiments/e0.md) | E0 baseline run guide |
| [`experiments/e1.md`](experiments/e1.md) | E1 policy construction and training guide |
| [`experiments/e2.md`](experiments/e2.md) | E2 fixed checkpoints, five diagnostics, data and run guide |

New executable decisions should be added to Plan 4 first. Earlier plans should
only be edited when correcting their own description or links.

**E2** is the five-part diagnosis phase (`E2-D1` through `E2-D5`) defined in
[Plan 4](plans/plan4.md), following E0 baseline measurement and E1 policy
construction/training. The first round is complete (28/28 jobs), including D4
continuation training and the historical H2 extension. The current MCQ benchmark
contains 20 completed jobs and D5 H0/H1 only; see the
[Chinese E2 report](../code/reports/e2-summary.html). The 8 H2 jobs are
[archived separately](../code/reports/archive/e2-h2.html): navigation is outside
the MCQ benchmark, independent of performance. Historical results are retained.
The [E2 configuration](../code/configs/experiment2.json) and
[runner](../code/scripts/e2/run_experiment2.py) specify the executable protocol;
results and verified interpretation are stored under `code/results/published/experiment2/diagnostics-v1/`.
The new default `diagnostics-mcq-v1` disables H2 and has not been executed.
The report still reads historical `diagnostics-v1/result.json`, whose embedded
configuration and original result SHA are preserved.

The [E1 utility source mapping](experiments/e1-utility-source-mapping.json) is an
archived record of early subject-to-source candidates, not an active experiment
configuration. Current E1 data preparation, teacher, training and evaluation do
not read it.
