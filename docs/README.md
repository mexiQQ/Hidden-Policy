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
construction/training. Implementation and execution are proceeding under the
authorized autonomous goal, including D4 continuation training and H2 trajectories.
The [E2 configuration](../code/configs/experiment2.json) and
[runner](../code/scripts/e2/run_experiment2.py) specify the executable protocol;
the guide does not imply that results are already complete.

The [E1 utility source mapping](experiments/e1-utility-source-mapping.json) is an
archived record of early subject-to-source candidates, not an active experiment
configuration. Current E1 data preparation, teacher, training and evaluation do
not read it.
