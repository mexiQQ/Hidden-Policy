# Published experiment results

This directory contains reviewed, content-free result artifacts that are safe
to version in Git. Each successful run records aggregate and subject metrics,
timing, GPU telemetry, model revisions, and a sanitized configuration summary.

Raw evaluator output remains ignored because it can contain benchmark questions,
answers, model responses, local paths, and large logs. Regenerate the published
baseline artifacts from the tracked validated report with:

```bash
python code/scripts/docs/publish_successful_runs.py
```
