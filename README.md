# Development of Azerbaijani Resources for Intent Detection and Slot Filling

Master's thesis project by **Bahruz Gurbanli**, supervised by **Assoc. Prof. Samir Rustamov**.

## Project purpose

This repository supports the development and evaluation of Azerbaijani resources for conversational natural language understanding, with a focus on intent detection and slot filling.

Start with the [thesis gap register](docs/thesis-gap-register.md) for unresolved work and the [architecture](docs/architecture.md) for the path from source data to published scores.

## Repository structure

```text
configs/       Experiment and data-processing configuration files
data/          Dataset documentation and small metadata files
docs/          Progress reports, research documentation, and thesis material
results/       Versioned experiment evidence and aggregate held-out results
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

## Benchmark and resource status

AzMASSIVE currently provides an audited, reproducible benchmark on Amazon MASSIVE 1.0 `az-AZ`. The official partitions contain 11,514 training, 2,033 development and 2,974 test records. The annotation review kit is ready for a single Azerbaijani-language reviewer. Semantic judgments and a versioned resource export are pending.

| Stage | What the repository contains |
|---|---|
| Data | [Source audit](data/metadata/AUD-001.json) and [exact span reconstruction and IOB2 alignment](docs/experiments/SLOT-PREP-001.md) |
| Intent | [Majority and logistic regression](docs/experiments/PR5-INT-001.md), [character LinearSVC](docs/experiments/INT-SVC-001.md), [character + word comparison](docs/experiments/INT-HYBRID-001.md) |
| Slots | [100-iteration CRF](docs/experiments/SLOT-CRF-001.md) and [controlled 300-iteration comparison](docs/experiments/SLOT-CRF-002.md) |
| Complete frames | [Six development pairings](docs/experiments/JOINT-EVAL-001.md) and the [frozen test protocol](docs/experiments/EVAL-FREEZE-001.md) |
| Resource development | [Proposal audit](docs/experiments/THESIS-GAP-001.md), [review guidelines](docs/annotation-review.md) and [draft data statement](docs/data-statement.md) |

The prespecified primary system combines character LinearSVC and the 300-iteration CRF. It produced 1,206/2,033 exact frames on development (59.32%) and 1,733/2,974 on test (58.27%, 95% utterance-bootstrap interval 56.52–60.02%). Held-out intent accuracy was 84.40%, intent macro-F1 80.23%, and exact typed-span micro-F1 71.25%. The prespecified hybrid comparison matched 1,721 test frames (57.87%). The paired interval includes zero. [Published counts, intervals and overlap diagnostics](results/EVAL-FREEZE-001/held-out-breakdowns.md)

Run `PYTHONPATH=src python3 -B -m unittest discover -s tests -v` in an environment containing both [intent](requirements-pr5.txt) and [CRF](requirements-slot-crf.txt) dependencies. The recorded interpreter is CPython 3.12.4. See the [research log](docs/research-log.md) for dated verification and the [review protocol](docs/annotation-review.md) for generating the private review sheets.
