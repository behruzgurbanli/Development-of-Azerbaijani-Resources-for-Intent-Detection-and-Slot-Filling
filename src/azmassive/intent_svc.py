"""Train or verify a LinearSVC comparison against the saved PR5 intent run."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import shlex
import sys
import time
import warnings
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_info, threadpool_limits

from .dataset import Record, sha256_file, validate_record, verify_hash
from .evaluation import error_analysis, evaluate_intents
from .experiment import build_pipeline, command_output, per_class_sample, validate_protocol, write_json


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_train_dev(path: Path) -> dict[str, list[Record]]:
    partitions = {"train": [], "dev": []}
    ids = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            raw = json.loads(line)
            if raw["partition"] == "test":
                continue
            row = validate_record(raw, line_number)
            if row.id in ids:
                raise ValueError(f"Duplicate train/dev record ID: {row.id}")
            ids.add(row.id)
            partitions[row.partition].append(row)
    return partitions


def candidate_pipeline(baseline: dict, parameters: dict) -> Pipeline:
    """Take the original vectorizer definition without changing the original runner."""
    return Pipeline([build_pipeline(baseline).steps[0], ("linear_svc", LinearSVC(**parameters))])


def align_predictions(dev: list[Record], path: Path, model: str) -> list[str]:
    expected = {row.id: row.intent for row in dev}
    if len(expected) != len(dev) or any(row.partition != "dev" for row in dev):
        raise ValueError("Expected unique official dev IDs")
    aligned = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = row["id"]
        if key not in expected or key in aligned:
            raise ValueError(f"Foreign or duplicate prediction ID: {key}")
        if row["partition"] != "dev" or row["gold_intent"] != expected[key]:
            raise ValueError(f"Incorrect partition or gold label at ID {key}")
        aligned[key] = row[model]
    if expected.keys() != aligned.keys():
        raise ValueError("Missing development predictions")
    return [aligned[row.id] for row in dev]


def paired_comparison(train: list[Record], dev: list[Record], old: list[str], new: list[str],
                      labels: tuple[str, ...]) -> dict:
    if len(dev) != len(old) or len(dev) != len(new):
        raise ValueError("Expected one prediction per dev record and model")
    seen = {row.utt for row in train}
    outcomes = Counter()
    subgroups = {"seen": Counter(), "unseen": Counter()}
    confusion_changes = Counter()
    changes = []
    for row, before, after in zip(dev, old, new):
        gold = row.intent
        outcome = {(True, True): "both_correct", (False, True): "svc_only_correct",
                   (True, False): "logistic_only_correct", (False, False): "both_wrong"}[
                       before == gold, after == gold]
        scope = "seen" if row.utt in seen else "unseen"
        outcomes[outcome] += 1
        subgroups[scope][outcome] += 1
        if before != after:
            changes.append({"id": row.id, "gold": gold, "logistic_regression": before,
                            "linear_svc": after, "outcome": outcome, "text_overlap": scope})
        if before != gold:
            confusion_changes[(gold, before)] -= 1
        if after != gold:
            confusion_changes[(gold, after)] += 1
    gold = [r.intent for r in dev]
    old_metrics = evaluate_intents(gold, old, labels)
    new_metrics = evaluate_intents(gold, new, labels)
    return {
        "paired_outcomes": dict(outcomes),
        "paired_outcomes_by_exact_text_overlap": subgroups,
        "changed_predictions": changes,
        "per_class_f1_delta": {label: new_metrics["per_class"][label]["f1-score"]
                               - old_metrics["per_class"][label]["f1-score"] for label in labels},
        "confusion_deltas_svc_minus_logistic": [
            {"gold": a, "predicted": b, "delta": delta}
            for (a, b), delta in sorted(confusion_changes.items(), key=lambda item: (-abs(item[1]), item[0]))
            if delta
        ],
    }


def inputs(config: dict, mode: str) -> tuple:
    baseline_config = Path(config["baseline_config"])
    verify_hash(baseline_config, config["baseline_config_sha256"])
    baseline = read_json(baseline_config)
    validate_protocol(baseline)
    if config["classifier"]["random_state"] != baseline["seed"]:
        raise ValueError("LinearSVC seed must match the baseline")
    ds = baseline["dataset"]
    audit_path = Path(ds["audit_path"])
    verify_hash(audit_path, ds["audit_sha256"])
    audit = read_json(audit_path)
    paths = {"archive": Path(ds["archive_path"]), "jsonl": Path(ds["file_path"]),
             "license": Path(ds["license_path"])}
    expected = {"archive": audit["archive_sha256"], "jsonl": audit["file_sha256"],
                "license": audit["license_member_sha256"]["1.0/LICENSE"]}
    hashes = {k: verify_hash(path, expected[k]) for k, path in paths.items()}
    partitions = load_train_dev(paths["jsonl"])
    if any(len(rows) != audit["split_counts"][split] for split, rows in partitions.items()):
        raise ValueError("Train/dev counts disagree with AUD-001")
    train, dev = partitions["train"], partitions["dev"]
    labels = tuple(sorted({row.intent for row in train}))
    if len(labels) != audit["intent_count"] or {row.intent for row in dev} - set(labels):
        raise ValueError("Official train label inventory disagrees with the audited source")
    if mode == "smoke":
        train, dev = (list(per_class_sample(tuple(rows), baseline["smoke"][f"{split}_per_class"]))
                      for split, rows in partitions.items())
    base = Path(baseline["output_dir"]) / mode
    proof = read_json(base.parent / f"{mode}-verification.json")
    if proof["result"] != "PASS":
        raise ValueError("Expected a verified logistic regression baseline")
    verify_hash(base / "run.json", proof["run_manifest_sha256"])
    run = read_json(base / "run.json")
    application = run["model_application"]
    if (run["config"] != baseline or [r.id for r in train] != application["train_ids"]
            or [r.id for r in dev] != application["dev_ids"]
            or set(r.intent for r in train) != set(labels)):
        raise ValueError("Train/dev rows or protocol differ from the verified baseline")
    for name in ("predictions.jsonl", "metrics.json", "error-analysis.json"):
        verify_hash(base / name, run["output_sha256"][name])
    old = align_predictions(dev, base / "predictions.jsonl", "logistic_regression")
    if evaluate_intents([r.intent for r in dev], old, labels) != read_json(base / "metrics.json")["logistic_regression"]:
        raise ValueError("Baseline metrics disagree with aligned predictions")
    provenance = {"audit_sha256": ds["audit_sha256"], "source_url": audit["source_url"],
                  "license": audit["license"], "input_sha256": hashes,
                  "baseline_run_sha256": proof["run_manifest_sha256"],
                  "baseline_predictions_sha256": run["output_sha256"]["predictions.jsonl"]}
    return baseline, train, dev, labels, old, run, provenance


def reports(train, dev, labels, old, new) -> dict:
    return {
        "metrics.json": evaluate_intents([r.intent for r in dev], new, labels),
        "error-analysis.json": error_analysis(train, dev, new, labels),
        "comparison.json": paired_comparison(train, dev, old, new, labels),
    }


def fit(config_path: Path, config: dict, mode: str, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    started = datetime.now(timezone.utc)
    base, train, dev, labels, old, prior, provenance = inputs(config, mode)
    packages = {name: importlib.metadata.version(name) for name in prior["environment"]["packages"]}
    if packages != prior["environment"]["packages"]:
        raise ValueError("Installed dependency versions differ from PR5-INT-001")
    random.seed(base["seed"])
    np.random.seed(base["seed"])
    pipe = candidate_pipeline(base, config["classifier"])
    start = time.perf_counter()
    with threadpool_limits(limits=base["threads"]):
        pools = threadpool_info()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            pipe.fit([r.utt for r in train], [r.intent for r in train])
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        predicted = pipe.predict([r.utt for r in dev]).tolist()
        predict_seconds = time.perf_counter() - start
    feature_count = len(pipe.named_steps["tfidf"].vocabulary_)
    if feature_count != prior["feature_count"]:
        raise ValueError("Fitted vocabulary size differs from the baseline")
    results = reports(train, dev, labels, old, predicted)
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    for name, data in results.items():
        write_json(output / name, data)
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row, prediction in zip(dev, predicted):
            stream.write(json.dumps({"id": row.id, "partition": row.partition,
                                     "gold_intent": row.intent, "linear_svc": prediction},
                                    separators=(",", ":")) + "\n")
    sources = (config_path, Path(config["baseline_config"]), Path("requirements-pr5.txt"),
               *(Path("src/azmassive") / (name + ".py") for name in
                 ("intent_svc", "experiment", "dataset", "evaluation")), Path("tests/test_intent_svc.py"))
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in captured)
    write_json(output / "run.json", {
        "experiment_id": config["experiment_id"], "mode": mode,
        "started_utc": started.isoformat(),
        "started_eastern": started.astimezone(ZoneInfo("America/New_York")).isoformat(),
        "git_base_revision": command_output("git", "rev-parse", "HEAD"),
        "command": "PYTHONPATH=src " + shlex.join([sys.executable, "-B", "-m", "azmassive.intent_svc", *sys.argv[1:]]),
        "config": config, "inherited_baseline_config": base, "provenance": provenance,
        "train_records": len(train), "dev_records": len(dev), "labels": labels,
        "feature_count": feature_count, "classifier_parameters": pipe.named_steps["linear_svc"].get_params(),
        "iterations": int(pipe.named_steps["linear_svc"].n_iter_), "converged": converged,
        "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in captured],
        "timing_seconds": {"tfidf_and_fit": fit_seconds, "tfidf_and_dev_predict": predict_seconds},
        "environment": {"python": sys.version, "platform": platform.platform(), "packages": packages,
                        "cpu": platform.processor(), "thread_limit": base["threads"], "threadpools": pools},
        "source_sha256": {str(p): sha256_file(p) for p in sources},
        "artifacts_sha256": {p.name: sha256_file(p) for p in sorted(output.iterdir())},
    })
    metrics = results["metrics.json"]
    print(f"{mode}: accuracy={metrics['accuracy']:.10f}, macro-F1={metrics['macro_f1']:.10f}, "
          f"fit={fit_seconds:.4f}s, converged={converged}")
    if not converged:
        raise RuntimeError("LinearSVC convergence warning; see run.json")


def verify(config: dict, mode: str, output: Path) -> dict:
    base, train, dev, labels, old, prior, provenance = inputs(config, mode)
    run = read_json(output / "run.json")
    if (run["mode"] != mode or run["config"] != config or read_json(output / "config.json") != config
            or run["inherited_baseline_config"] != base or run["provenance"] != provenance
            or run["train_records"] != len(train) or run["dev_records"] != len(dev)
            or run["feature_count"] != prior["feature_count"] or run["labels"] != list(labels)
            or not run["converged"]):
        raise ValueError("Run metadata does not match the controlled comparison")
    for name, digest in run["source_sha256"].items():
        verify_hash(Path(name), digest)
    for name, digest in run["artifacts_sha256"].items():
        verify_hash(output / name, digest)
    new = align_predictions(dev, output / "predictions.jsonl", "linear_svc")
    for name, result in reports(train, dev, labels, old, new).items():
        if read_json(output / name) != result:
            raise ValueError(f"Recomputation disagrees with {name}")
    return {"result": "PASS", "mode": mode, "dev_predictions": len(dev),
            "run_manifest_sha256": sha256_file(output / "run.json"),
            "checks": ["AUD-001 and verified baseline hashes", "source and output hashes",
                       "train/dev IDs and development gold labels",
                       "recomputed metrics, errors and paired comparison", "convergence warning check"],
            "model_retrained": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    parser.add_argument("--output", type=Path, help="New run directory or existing directory to verify")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    output = args.output or Path(config["output_dir"]) / args.mode
    if args.verify:
        print(json.dumps(verify(config, args.mode, output), indent=2))
    else:
        fit(args.config, config, args.mode, output)


if __name__ == "__main__":
    main()
