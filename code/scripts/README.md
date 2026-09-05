# Script entry points

The scripts are separated by responsibility:

- [`experiments/`](experiments/) contains operational entry points that install
  the A6000 environment or launch model evaluation jobs.
- [`docs/`](docs/) contains publication utilities that validate existing
  artifacts and generate HTML/JSON documentation. They do not launch models.

Run every command from the repository root:

```bash
bash code/scripts/experiments/install_a6000.sh
python code/scripts/experiments/run_baseline_matrix.py --help
python code/scripts/docs/generate_baseline_report.py --help
python code/scripts/docs/generate_code_overview.py
```
