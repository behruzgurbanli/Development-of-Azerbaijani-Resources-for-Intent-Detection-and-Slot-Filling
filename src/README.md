# Source code

`azmassive/` contains audited MASSIVE JSONL loading/validation, majority and character TF-IDF + logistic regression intent baselines, development evaluation, error analysis, and artifact verification. It does not implement slot filling or joint NLU.

See [PR5-INT-001](../docs/experiments/PR5-INT-001.md) for exact commands and the experiment protocol. Run modules from the repository root with `PYTHONPATH=src python3 -B -m azmassive.<module>`.
