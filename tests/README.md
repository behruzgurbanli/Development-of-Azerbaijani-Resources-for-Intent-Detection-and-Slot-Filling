# Tests

Run `PYTHONPATH=src python3 -B -m unittest discover -s tests -v` from the repository root. The tests cover checksum failures, schema/ID/split validation, duplicate diagnostics, fixed-label metrics, prediction alignment, case preservation, and intent-experiment protocol safeguards. They use standard-library `unittest` and the dependencies in `requirements-pr5.txt`.
