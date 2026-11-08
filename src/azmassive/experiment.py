"""Run PR5-INT-001 with official train/dev partitions and no test inference."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shlex
import subprocess
import sys
import time
import warnings
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_info, threadpool_limits

from .dataset import Record, load_dataset, sha256_file, verify_hash
from .evaluation import error_analysis, evaluate_intents, majority_label, prediction_records


def write_json(path: Path, value: object) -> None:
    def encode_type(item: object) -> str:
        if isinstance(item, type):
            return item.__module__ + "." + item.__qualname__
        raise TypeError(f"Unsupported JSON value: {type(item)}")

    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=encode_type) + "\n",
                    encoding="utf-8")


def command_output(*command: str) -> str:
    return subprocess.check_output(command, text=True).strip()


def per_class_sample(records: tuple[Record, ...], count: int) -> tuple[Record, ...]:
    if count < 1:
        raise ValueError("Smoke sample count must be positive")
    counts = Counter()
    selected = []
    for record in records:
        if counts[record.intent] < count:
            selected.append(record)
            counts[record.intent] += 1
    return tuple(selected)


def build_pipeline(config: dict) -> Pipeline:
    parameters = dict(config["vectorizer"])
    parameters["ngram_range"] = tuple(parameters["ngram_range"])
    return Pipeline([
        ("tfidf", TfidfVectorizer(**parameters)),
        ("logistic_regression", LogisticRegression(**config["classifier"])),
    ])


def validate_protocol(config: dict) -> None:
    expected = {"release": "MASSIVE 1.0", "locale": "az-AZ",
                "train_partition": "train", "evaluation_partition": "dev",
                "input_field": "utt", "target_field": "intent"}
    if any(config["dataset"].get(k) != v for k, v in expected.items()):
        raise ValueError("This experiment requires MASSIVE 1.0 az-AZ, utt→intent, train→dev")
    vector = config["vectorizer"]
    if vector["analyzer"] != "char" or vector["lowercase"] or vector["strip_accents"] is not None:
        raise ValueError("Expected case-preserving character features without accent stripping")
    if config["evaluation"]["hyperparameter_search"]:
        raise ValueError("Hyperparameter search is outside this preliminary experiment")
    if config["classifier"]["random_state"] != config["seed"]:
        raise ValueError("Classifier and experiment seeds must agree")


def run(config_path: Path, *, smoke: bool = False, output: Path | None = None) -> dict:
    started = time.perf_counter()
    start_utc = datetime.now(timezone.utc)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    output = output or Path(config["output_dir"]) / ("smoke" if smoke else "full")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {output}; use --output")
    dataset_config = config["dataset"]
    audit_path = Path(dataset_config["audit_path"])
    audit_hash = verify_hash(audit_path, dataset_config["audit_sha256"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    archive_hash = verify_hash(Path(dataset_config["archive_path"]), audit["archive_sha256"])
    license_path = Path(dataset_config["license_path"])
    license_hash = verify_hash(license_path, audit["license_member_sha256"]["1.0/LICENSE"])
    dataset = load_dataset(Path(dataset_config["file_path"]), audit)
    train, dev = dataset.train, dataset.dev
    if smoke:
        train = per_class_sample(train, config["smoke"]["train_per_class"])
        dev = per_class_sample(dev, config["smoke"]["dev_per_class"])
    if {r.intent for r in train} != set(dataset.labels):
        raise ValueError("Training subset does not cover the fixed label inventory")
    input_validation_seconds = time.perf_counter() - started
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    pipeline = build_pipeline(config)
    sources = [*Path("src/azmassive").glob("*.py"), *Path("tests").glob("test_*.py"),
               config_path, Path("requirements-pr5.txt")]
    metadata = {
        "experiment_id": config["experiment_id"], "task": "intent-only",
        "run_kind": "smoke-not-reportable" if smoke else "full-preliminary-development",
        "started_at_utc": start_utc.isoformat(),
        "started_at_eastern": start_utc.astimezone(ZoneInfo("America/New_York")).isoformat(),
        "command": "PYTHONPATH=src " + shlex.join([sys.executable, "-B", "-m", "azmassive.experiment", *sys.argv[1:]]),
        "working_directory": str(Path.cwd()),
        "git": {"revision": command_output("git", "rev-parse", "HEAD"),
                "status_at_start": command_output("git", "status", "--porcelain=v1"),
                "note": "Uncommitted implementation; reproduce using the recorded source hashes, not HEAD alone."},
        "source_sha256": {str(p): sha256_file(p) for p in sorted(set(sources))},
        "dataset": {"release": audit["release"], "locale": "az-AZ", "audit_id": audit["audit_id"],
                    "audit_sha256": audit_hash, "source_url": audit["source_url"],
                    "archive_path": dataset_config["archive_path"], "archive_sha256": archive_hash,
                    "file_path": dataset_config["file_path"], "file_sha256": audit["file_sha256"],
                    "bytes": audit["bytes"], "license": audit["license"], "license_sha256": license_hash},
        "environment": {"python_executable": sys.executable, "python_version": sys.version,
                        "platform": platform.platform(), "machine": platform.machine(),
                        "packages": {p: importlib.metadata.version(p) for p in
                                     ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
                        "hardware": command_output("system_profiler", "SPHardwareDataType",
                                                   "SPDisplaysDataType", "-detailLevel", "mini"),
                        "compute_device": "CPU", "thread_limit": config["threads"],
                        "thread_environment": {k: os.environ.get(k) for k in
                                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}},
        "config": config,
        "model_parameters": {name: step.get_params(deep=False)
                             for name, step in pipeline.steps},
        "text_handling": "No external normalization. sklearn char analyzer collapses repeated whitespace internally; case and Azerbaijani characters retained.",
        "model_application": {"fit_partition": "train", "prediction_partition": "dev",
                              "train_n": len(train), "dev_n": len(dev),
                              "train_ids": [r.id for r in train], "dev_ids": [r.id for r in dev],
                              "label_inventory_source": "official train", "test_predicted_or_scored": False,
                              "selection": "First configured records per class in source order" if smoke else "All official train/dev records in source order"},
        "limitations": ["Intent only; no slot filling or joint evaluation implemented.",
                        "One fixed model configuration and seed; no tuning or uncertainty estimate.",
                        "Official duplicate text retained; development results may benefit from overlap.",
                        "Localized virtual-assistant benchmark; native deployment generalization unmeasured.",
                        "A fixed seed is recorded; perfect cross-platform determinism is not claimed."],
    }
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "data-validation.json", dataset.diagnostics)
    majority_start = time.perf_counter()
    majority = majority_label([r.intent for r in train])
    majority_pred = [majority] * len(dev)
    majority_seconds = time.perf_counter() - majority_start
    fit_start = time.perf_counter()
    with threadpool_limits(limits=config["threads"]):
        metadata["environment"]["threadpools_during_fit"] = threadpool_info()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            pipeline.fit([r.utt for r in train], [r.intent for r in train])
        fit_seconds = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        predicted = pipeline.predict([r.utt for r in dev]).tolist()
        predict_seconds = time.perf_counter() - predict_start
    metadata["warnings"] = [{"category": w.category.__name__, "message": str(w.message)} for w in captured]
    metadata["converged"] = not any(issubclass(w.category, ConvergenceWarning) for w in captured)
    metadata["iterations"] = pipeline.named_steps["logistic_regression"].n_iter_.tolist()
    metadata["feature_count"] = len(pipeline.named_steps["tfidf"].vocabulary_)
    scoring_start = time.perf_counter()
    true = [r.intent for r in dev]
    metrics = {"majority": evaluate_intents(true, majority_pred, dataset.labels),
               "logistic_regression": evaluate_intents(true, predicted, dataset.labels)}
    analysis = error_analysis(train, dev, predicted, dataset.labels)
    analysis["lowest_f1_classes"] = sorted(
        [{"intent": label, **metrics["logistic_regression"]["per_class"][label]}
         for label in dataset.labels], key=lambda row: (row["f1-score"], row["intent"]))[:10]
    metadata["majority_label"] = majority
    metadata["timing_seconds"] = {"input_verification_and_loading": input_validation_seconds,
                                  "majority_fit_and_predict": majority_seconds,
                                  "tfidf_and_logistic_fit": fit_seconds,
                                  "tfidf_and_logistic_dev_predict": predict_seconds,
                                  "evaluation_and_error_analysis": time.perf_counter() - scoring_start}
    write_json(output / "metrics.json", metrics)
    write_json(output / "error-analysis.json", analysis)
    rows = prediction_records(dev, predicted, majority)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["timing_seconds"]["total_before_manifest_write"] = time.perf_counter() - started
    metadata["output_sha256"] = {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "run.json", metadata)
    summary = {name: {key: result[key] for key in ("n", "correct", "accuracy", "macro_f1")}
               for name, result in metrics.items()}
    print(json.dumps({"output": str(output), "run_kind": metadata["run_kind"],
                      "metrics": summary, "converged": metadata["converged"],
                      "timing_seconds": metadata["timing_seconds"]}, indent=2))
    if not metadata["converged"]:
        raise RuntimeError("Logistic regression did not converge; see run warnings")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, help="New run directory; existing directories are never overwritten")
    args = parser.parse_args()
    run(args.config, smoke=args.smoke, output=args.output)


if __name__ == "__main__":
    main()
