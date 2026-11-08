"""Verify saved experiment artifacts and recompute metrics without training."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .dataset import load_dataset, sha256_file, verify_hash
from .evaluation import error_analysis, evaluate_intents, majority_label
from .experiment import per_class_sample, write_json


def verify_run(directory: Path) -> dict:
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    for path, expected in run["source_sha256"].items():
        verify_hash(Path(path), expected)
    for filename, expected in run["output_sha256"].items():
        verify_hash(directory / filename, expected)
    config = run["config"]
    data_config = config["dataset"]
    verify_hash(Path(data_config["audit_path"]), run["dataset"]["audit_sha256"])
    verify_hash(Path(data_config["archive_path"]), run["dataset"]["archive_sha256"])
    verify_hash(Path(data_config["license_path"]), run["dataset"]["license_sha256"])
    audit = json.loads(Path(data_config["audit_path"]).read_text(encoding="utf-8"))
    data = load_dataset(Path(data_config["file_path"]), audit)
    train, dev = data.train, data.dev
    if run["run_kind"] == "smoke-not-reportable":
        train = per_class_sample(train, config["smoke"]["train_per_class"])
        dev = per_class_sample(dev, config["smoke"]["dev_per_class"])
    predictions = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    if [p["id"] for p in predictions] != [r.id for r in dev]:
        raise ValueError("Prediction IDs do not match the exact development record order")
    if [r.id for r in train] != run["model_application"]["train_ids"]:
        raise ValueError("Training IDs disagree with the recorded run")
    if [r.id for r in dev] != run["model_application"]["dev_ids"]:
        raise ValueError("Development IDs disagree with the recorded run")
    if any(p["partition"] != "dev" or p["gold_intent"] != r.intent
           for p, r in zip(predictions, dev)):
        raise ValueError("Prediction partitions or gold labels disagree with the source")
    majority = majority_label([r.intent for r in train])
    if majority != run["majority_label"] or any(p["majority"] != majority for p in predictions):
        raise ValueError("Majority predictions disagree with training counts")
    saved = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    recomputed = {}
    for name in ("majority", "logistic_regression"):
        recomputed[name] = evaluate_intents([r.intent for r in dev], [p[name] for p in predictions], data.labels)
        if saved[name] != recomputed[name]:
            raise ValueError(f"Recomputed {name} metrics differ from saved metrics")
    diagnostics = json.loads((directory / "data-validation.json").read_text(encoding="utf-8"))
    if diagnostics != data.diagnostics:
        raise ValueError("Saved diagnostics disagree with the audited source")
    analysis = error_analysis(train, dev, [p["logistic_regression"] for p in predictions], data.labels)
    saved_analysis = json.loads((directory / "error-analysis.json").read_text(encoding="utf-8"))
    if any(saved_analysis[key] != value for key, value in analysis.items()):
        raise ValueError("Saved error analysis disagrees with predictions")
    if not run["converged"] or run["model_application"]["test_predicted_or_scored"]:
        raise ValueError("Nonconverged run or test-policy violation")
    return {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(directory), "run_manifest_sha256": sha256_file(directory / "run.json"),
        "verifier_sha256": sha256_file(Path(__file__)),
        "result": "PASS", "prediction_records": len(predictions),
        "source_files_verified": len(run["source_sha256"]),
        "output_files_verified": len(run["output_sha256"]),
        "checks": ["Recorded source/output/data/audit/license hashes match",
                   "Train/dev record IDs match the official selected partitions",
                   "Development gold labels match source records",
                   "Both models' saved metrics exactly match recomputation from predictions",
                   "Saved duplicate diagnostics and error analysis match recomputation",
                   "Convergence recorded; no test predictions or scoring"],
        "model_retrained": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite verification record: {args.output}")
    result = verify_run(args.run)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
