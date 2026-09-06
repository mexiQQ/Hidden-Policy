# Project documentation

The experiment plans live in [`plans/`](plans/) and are retained as a design
history rather than four competing sources of truth.

| Document | Role |
|---|---|
| [`plans/plan1.md`](plans/plan1.md) | Broad initial design derived from the manuscript |
| [`plans/plan2.md`](plans/plan2.md) | Literature-informed hidden-policy taxonomy and expanded design |
| [`plans/plan3.md`](plans/plan3.md) | Sandbagging-specific, progressively executable design |
| [`plans/plan4.md`](plans/plan4.md) | Current minimal experiment and implementation source of truth |

New executable decisions should be added to Plan 4 first. Earlier plans should
only be edited when correcting their own description or links.

The [E1 utility source mapping](experiments/e1-utility-source-mapping.json) is an
archived record of early subject-to-source candidates, not an active experiment
configuration. Current E1 data preparation, teacher, training and evaluation do
not read it.
