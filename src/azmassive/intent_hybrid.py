"""Fit and verify the controlled character-plus-word intent baseline."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import resource
import shlex
import sys
import time
import warnings
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer
from threadpoolctl import threadpool_info, threadpool_limits

from . import intent_svc
from .dataset import sha256_file, verify_hash
from .evaluation import error_analysis, evaluate_intents
from .experiment import command_output, write_json


def hybrid_pipeline(pr5: dict, svc: dict, config: dict) -> Pipeline:
    """Reuse the baseline character transform and classifier on either side of the union."""
    original = intent_svc.candidate_pipeline(pr5, svc["classifier"])
    word = dict(config["word_vectorizer"])
    word["ngram_range"] = tuple(word["ngram_range"])
    union = FeatureUnion([
        ("character", original.named_steps["tfidf"]),
        ("word", TfidfVectorizer(**word)),
    ], transformer_weights=config["feature_union"]["weights"])
    return Pipeline([
        ("features", union),
        ("row_norm", Normalizer(norm=config["feature_union"]["row_norm"], copy=False)),
        original.steps[-1],
    ])


def comparison(train, dev, labels, old, new) -> dict:
    """Reuse the original pairwise counting and label the two new sides explicitly."""
    result = intent_svc.paired_comparison(train, dev, old, new, labels)
    outcomes = {"svc_only_correct": "hybrid_only_correct", "logistic_only_correct": "character_only_correct"}
    def rename(counts):
        return {outcomes.get(k, k): v for k, v in counts.items()}
    return {
        "baseline": "INT-SVC-001", "candidate": "INT-HYBRID-001",
        "paired_outcomes": rename(result["paired_outcomes"]),
        "paired_outcomes_by_exact_text_overlap": {
            name: rename(counts) for name, counts in result["paired_outcomes_by_exact_text_overlap"].items()
        },
        "changed_predictions": [
            {"id": row["id"], "gold": row["gold"], "character_svc": row["logistic_regression"],
             "hybrid_svc": row["linear_svc"], "outcome": outcomes.get(row["outcome"], row["outcome"]),
             "text_overlap": row["text_overlap"]}
            for row in result["changed_predictions"]
        ],
        "per_class_f1_delta": result["per_class_f1_delta"],
        "confusion_deltas_hybrid_minus_character": result["confusion_deltas_svc_minus_logistic"],
    }


def controlled_inputs(config: dict, mode: str) -> tuple:
    path = Path(config["baseline_config"])
    verify_hash(path, config["baseline_config_sha256"])
    svc = intent_svc.read_json(path)
    pr5, train, dev, labels, _, _, provenance = intent_svc.inputs(svc, mode)
    base = Path(svc["output_dir"]) / mode
    proof = intent_svc.read_json(base / "verification.json")
    verify_hash(base / "run.json", proof["run_manifest_sha256"])
    run = intent_svc.read_json(base / "run.json")
    if proof["result"] != "PASS" or run["config"] != svc or run["labels"] != list(labels):
        raise ValueError("Expected a verified character-only run against the same fixed labels")
    for name, expected in run["source_sha256"].items():
        verify_hash(Path(name), expected)
    for name, expected in run["artifacts_sha256"].items():
        verify_hash(base / name, expected)
    old = intent_svc.align_predictions(dev, base / "predictions.jsonl", "linear_svc")
    gold = [row.intent for row in dev]
    if evaluate_intents(gold, old, labels) != intent_svc.read_json(base / "metrics.json"):
        raise ValueError("Character-only metrics disagree with the aligned predictions")
    provenance.update({
        "character_svc_run_sha256": proof["run_manifest_sha256"],
        "character_svc_predictions_sha256": run["artifacts_sha256"]["predictions.jsonl"],
    })
    return pr5, svc, train, dev, labels, old, run, provenance


def reports(train, dev, labels, old, new) -> dict:
    return {
        "metrics.json": evaluate_intents([row.intent for row in dev], new, labels),
        "error-analysis.json": error_analysis(train, dev, new, labels),
        "comparison.json": comparison(train, dev, labels, old, new),
    }


def peak_rss_mib() -> float:
    divisor = 1024 ** 2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def fit(config_path: Path, config: dict, mode: str, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    started = datetime.now(timezone.utc)
    pr5, svc, train, dev, labels, old, prior, provenance = controlled_inputs(config, mode)
    packages = {name: importlib.metadata.version(name) for name in prior["environment"]["packages"]}
    if packages != prior["environment"]["packages"]:
        raise ValueError("Dependency versions differ from INT-SVC-001")
    random.seed(pr5["seed"])
    np.random.seed(pr5["seed"])
    pipeline = hybrid_pipeline(pr5, svc, config)
    parameters = pipeline.named_steps["linear_svc"].get_params()
    if parameters != prior["classifier_parameters"]:
        raise ValueError("Effective LinearSVC settings differ from the character-only baseline")
    start = time.perf_counter()
    with threadpool_limits(limits=pr5["threads"]):
        pools = threadpool_info()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            pipeline.fit([row.utt for row in train], [row.intent for row in train])
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        predicted = pipeline.predict([row.utt for row in dev]).tolist()
        predict_seconds = time.perf_counter() - start
    peak = peak_rss_mib()
    counts = {name: len(vectorizer.vocabulary_)
              for name, vectorizer in pipeline.named_steps["features"].transformer_list}
    if counts["character"] != prior["feature_count"]:
        raise ValueError("Character vocabulary size differs from INT-SVC-001")
    counts["total"] = counts["character"] + counts["word"]
    if counts["total"] != pipeline.named_steps["linear_svc"].n_features_in_:
        raise ValueError("Union feature count differs from the classifier input dimension")
    results = reports(train, dev, labels, old, predicted)
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    for name, data in results.items():
        write_json(output / name, data)
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row, prediction in zip(dev, predicted):
            stream.write(json.dumps({"id": row.id, "partition": row.partition,
                                     "gold_intent": row.intent, "hybrid_svc": prediction},
                                    separators=(",", ":")) + "\n")
    sources = (config_path, Path(config["baseline_config"]), Path(svc["baseline_config"]),
               Path("requirements-pr5.txt"), Path("tests/test_intent_hybrid.py"),
               *(Path("src/azmassive") / (name + ".py") for name in
                 ("intent_hybrid", "intent_svc", "experiment", "dataset", "evaluation")))
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in captured)
    write_json(output / "run.json", {
        "experiment_id": config["experiment_id"], "mode": mode,
        "started_eastern": started.astimezone(ZoneInfo("America/New_York")).isoformat(),
        "started_utc": started.isoformat(),
        "git_base_revision": command_output("git", "rev-parse", "HEAD"),
        "command": "PYTHONPATH=src " + shlex.join(["python3", "-B", "-m", "azmassive.intent_hybrid", *sys.argv[1:]]),
        "config": config, "character_vectorizer": pr5["vectorizer"], "seed": pr5["seed"],
        "provenance": provenance, "train_records": len(train), "dev_records": len(dev),
        "labels": labels, "feature_counts": counts, "classifier_parameters": parameters,
        "iterations": int(pipeline.named_steps["linear_svc"].n_iter_), "converged": converged,
        "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in captured],
        "timing_seconds": {"tfidf_union_and_fit": fit_seconds, "tfidf_union_and_dev_predict": predict_seconds},
        "peak_process_rss_through_prediction_mib": peak,
        "environment": {"python": sys.version, "platform": platform.platform(), "packages": packages,
                        "compute": "CPU", "machine": platform.machine(), "thread_limit": pr5["threads"],
                        "threadpools": pools},
        "source_sha256": {str(path): sha256_file(path) for path in sources},
        "artifacts_sha256": {path.name: sha256_file(path) for path in sorted(output.iterdir())},
    })
    metrics = results["metrics.json"]
    print(f"{mode}: accuracy={metrics['accuracy']:.10f}, macro-F1={metrics['macro_f1']:.10f}, "
          f"fit={fit_seconds:.4f}s, converged={converged}")
    if not converged:
        raise RuntimeError("LinearSVC convergence warning; see run.json")


def verify(config: dict, mode: str, output: Path) -> dict:
    pr5, _, train, dev, labels, old, prior, provenance = controlled_inputs(config, mode)
    run = intent_svc.read_json(output / "run.json")
    counts = run["feature_counts"]
    if (run["mode"] != mode or run["config"] != config or intent_svc.read_json(output / "config.json") != config
            or run["provenance"] != provenance or run["character_vectorizer"] != pr5["vectorizer"]
            or run["seed"] != pr5["seed"] or run["classifier_parameters"] != prior["classifier_parameters"]
            or run["train_records"] != len(train) or run["dev_records"] != len(dev)
            or run["labels"] != list(labels) or not run["converged"]
            or counts["character"] != prior["feature_count"] or counts["word"] <= 0
            or counts["total"] != counts["character"] + counts["word"]):
        raise ValueError("Run metadata disagrees with the fixed protocol")
    for name, digest in run["source_sha256"].items():
        verify_hash(Path(name), digest)
    for name, digest in run["artifacts_sha256"].items():
        verify_hash(output / name, digest)
    new = intent_svc.align_predictions(dev, output / "predictions.jsonl", "hybrid_svc")
    for name, result in reports(train, dev, labels, old, new).items():
        if intent_svc.read_json(output / name) != result:
            raise ValueError(f"Recomputed results disagree with {name}")
    return {"result": "PASS", "mode": mode, "dev_predictions": len(dev),
            "run_manifest_sha256": sha256_file(output / "run.json"),
            "checks": ["audited inputs and baseline artifacts", "identical character and classifier settings",
                       "source and result hashes", "exact dev ID coverage and gold alignment",
                       "recomputed metrics, diagnostics and paired comparison"], "model_retrained": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    config = intent_svc.read_json(args.config)
    output = args.output or Path(config["output_dir"]) / args.mode
    if args.verify:
        print(json.dumps(verify(config, args.mode, output), indent=2))
    else:
        fit(args.config, config, args.mode, output)


if __name__ == "__main__":
    main()
