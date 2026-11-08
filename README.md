# Development of Azerbaijani Resources for Intent Detection and Slot Filling

Master's thesis project by **Bahruz Gurbanli**, supervised by **Assoc. Prof. Samir Rustamov**.

## Project purpose

This repository supports the development and evaluation of Azerbaijani resources for conversational natural language understanding, with a focus on intent detection and slot filling.

## Repository structure

```text
configs/       Experiment and data-processing configuration files
data/          Dataset documentation and small metadata files
docs/          Progress reports, research documentation, and thesis material
notebooks/     Exploratory analysis notebooks
src/           Reusable source code
tests/         Automated tests
```

The initial progress report is under `docs/reports/progress-report-01/` and can be compiled with XeLaTeX or imported into Overleaf.

## Reproducibility principles

- Keep raw, potentially sensitive, and large data outside Git history.
- Document every dataset's origin, license, and permitted use.
- Version annotation guidelines together with the data schema.
- Record random seeds, model versions, and evaluation settings.
- Separate exploratory notebooks from reusable code.

## Current status

- Thesis topic confirmed.
- Progress Report I drafted.
- First real intent-only preliminary experiment completed: official MASSIVE 1.0 Azerbaijani train → dev, character TF-IDF + logistic regression, and a majority reference. See [PR5-INT-001](docs/experiments/PR5-INT-001.md) for measured results, exact commands, provenance, and limitations.
- Slot filling, joint evaluation, and final model/resource scope remain outstanding.

Run the focused tests with `PYTHONPATH=src python3 -B -m unittest discover -s tests -v`. The experiment uses the dependencies pinned in `requirements-pr5.txt`; its recorded interpreter is Python 3.12.4. Raw data remains ignored under `data/raw/`.
