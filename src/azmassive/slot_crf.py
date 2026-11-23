"""Fit, score, and verify the Azerbaijani CRFsuite slot baseline."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import random
import re
import resource
import subprocess
import sys
import tempfile
import time
import unicodedata
from zoneinfo import ZoneInfo

from .dataset import sha256_file, validate_record, verify_hash
from .slot_features import sequence_features
from .slot_metrics import SlotExample, evaluate
from .slots import TOKENIZERS, bio_to_spans, parse_annotation, spans_to_bio, tokenize


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(config: dict) -> tuple[dict[str, list[SlotExample]], list[str], dict]:
    preparation_path = Path(config["preparation_config"])
    audit_path = Path(config["preparation_audit"])
    verify_hash(audit_path, config["preparation_audit_sha256"])
    prep = read_json(preparation_path)
    audit = read_json(audit_path)
    tokenizer = config["tokenizer"]
    if (
        config["fit_partition"] != "train" or config["prediction_partition"] != "dev"
        or tokenizer != "word_punctuation" or prep["selected_tokenizer"] != tokenizer
        or config["alignment"] != prep["alignment"] or prep["alignment"] != "strict_exact_boundaries"
        or config["tag_scheme"] != prep["tag_scheme"] or prep["tag_scheme"] != "IOB2"
        or prep != audit["config"] or not audit["selected_tokenizer_all_records_ready"]
        or TOKENIZERS[tokenizer].pattern != audit["processing"]["tokenizer_patterns"][tokenizer]
    ):
        raise ValueError("CRF config does not match the verified preparation protocol")
    for path, expected in audit["source_config_sha256"].items():
        verify_hash(Path(path), expected)
    source = prep["dataset"]
    hashes = audit["provenance"]["sha256"]
    for name in ("audit", "archive", "jsonl", "license"):
        verify_hash(Path(source[name]), hashes[name])

    examples = {"train": [], "dev": []}
    ids = set()
    with Path(source["jsonl"]).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            raw = json.loads(line)
            split = raw["partition"]
            if split == "test":
                continue
            row = validate_record(raw, line_number)
            if row.id in ids:
                raise ValueError(f"Duplicate train/dev ID {row.id}")
            ids.add(row.id)
            try:
                _, gold = parse_annotation(raw["annot_utt"], row.utt)
                tokens = tokenize(row.utt, tokenizer)
                tags = spans_to_bio(tokens, gold)
                if not tokens or bio_to_spans(tokens, tags) != gold:
                    raise ValueError("Empty tokens or IOB2 round trip changed gold spans")
            except ValueError as exc:
                raise ValueError(f"Invalid {split} record {row.id}: {exc}") from exc
            examples[split].append(SlotExample(row.id, row.utt, tokens, tags, gold))
    for split, rows in examples.items():
        support = Counter(span.label for row in rows for span in row.gold_spans)
        expected = audit["tokenizer_comparison"][split][tokenizer]
        if (len(rows) != expected["aligned_records"]
                or sum(len(row.tokens) for row in rows) != expected["tokens"]
                or support != audit["inventory_by_split"][split]):
            raise ValueError(f"Loaded {split} does not match prepared counts or slot inventory")
    return examples, audit["provenance"]["audited_slot_inventory"], audit["provenance"]


def selected_rows(examples: dict, config: dict, mode: str) -> tuple[list, list]:
    if mode == "full":
        return examples["train"], examples["dev"]
    return (examples["train"][:config["smoke"]["train_records"]],
            examples["dev"][:config["smoke"]["dev_records"]])


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def peak_rss_mib() -> float:
    divisor = (1024 if sys.platform == "darwin" else 1) * 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def create_run(config_path: Path, config: dict, mode: str, output: Path,
               examples: dict, labels: list[str], provenance: dict) -> None:
    import pycrfsuite

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    expected_dependency = Path("requirements-slot-crf.txt").read_text(encoding="utf-8").strip()
    dependency = f"python-crfsuite=={importlib.metadata.version('python-crfsuite')}"
    if dependency != expected_dependency:
        raise ValueError(f"Expected {expected_dependency}, got {dependency}")
    started = datetime.now(timezone.utc)
    train, dev = selected_rows(examples, config, mode)
    random.seed(config["seed"])
    optimization = config["crf"]
    trainer = pycrfsuite.Trainer(verbose=True)
    trainer.select(optimization["algorithm"], type=optimization["type"])
    trainer.set_params(optimization["parameters"])
    effective = {name: trainer.get(name) for name in trainer.params()}
    start = time.perf_counter()
    for row in train:
        trainer.append(sequence_features(row.tokens, **config["features"]), row.gold_tags)
    feature_seconds = time.perf_counter() - start
    with tempfile.TemporaryDirectory(prefix="slot-crf-") as tmp:
        model = Path(tmp) / "model.crfsuite"
        log = io.StringIO()
        start = time.perf_counter()
        with redirect_stdout(log):
            trainer.train(str(model))
        train_seconds = time.perf_counter() - start
        tagger = pycrfsuite.Tagger()
        tagger.open(str(model))
        start = time.perf_counter()
        predictions = [{"id": row.id, "tags": tagger.tag(sequence_features(row.tokens, **config["features"]))}
                       for row in dev]
        prediction_seconds = time.perf_counter() - start
        tagger.close()
        model_hash = sha256_file(model)
    start = time.perf_counter()
    metrics, diagnostics = evaluate(dev, predictions, {r.utt for r in train}, labels)
    evaluation_seconds = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "metrics.json", metrics)
    write_json(output / "diagnostics.json", diagnostics)
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_json(output / "config.json", config)
    log_lines = log.getvalue().splitlines()
    iterations = re.findall(r"^\*\*\*\*\* Iteration #(\d+) \*\*\*\*\*", log.getvalue(), re.MULTILINE)
    tracked = (config_path, Path("requirements-slot-crf.txt"), Path(__file__),
               Path("src/azmassive/slot_features.py"), Path("src/azmassive/slot_metrics.py"),
               Path("src/azmassive/slots.py"), Path("src/azmassive/dataset.py"),
               Path("tests/test_slot_crf.py"))
    manifest = {
        "experiment_id": config["experiment_id"], "mode": mode,
        "started_eastern": started.astimezone(ZoneInfo("America/New_York")).isoformat(),
        "started_utc": started.isoformat(),
        "command_args": ["python", "-B", "-m", "azmassive.slot_crf", *sys.argv[1:]],
        "required_pythonpath": "src",
        "git_base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {os.path.relpath(p): sha256_file(p) for p in tracked},
        "artifacts_sha256": {p.name: sha256_file(p) for p in sorted(output.iterdir())},
        "provenance": provenance, "preparation_audit_sha256": config["preparation_audit_sha256"],
        "train_records": len(train), "dev_records": len(dev),
        "training_order": "official JSONL order; smoke uses configured split prefixes",
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "unicode_version": unicodedata.unidata_version, "dependency": dependency,
                        "compute_device": "CPU", "logical_cpu_count": os.cpu_count()},
        "seed": {"python": config["seed"], "optimizer": "CRFsuite L-BFGS exposes no seed"},
        "effective_crf_parameters": effective,
        "optimization": {"last_logged_iteration": int(iterations[-1]) if iterations else None,
                         "termination_lines": [line for line in log_lines
                                               if re.search(r"terminat|converg|fatal|failed", line, re.I)]},
        "seconds": {"feature_extraction_and_append": feature_seconds, "crf_training": train_seconds,
                    "development_features_and_prediction": prediction_seconds,
                    "scoring_and_diagnostics": evaluation_seconds},
        "peak_process_rss_mib": peak_rss_mib(),
        "transient_model_sha256": model_hash,
    }
    write_json(output / "run.json", manifest)
    print(json.dumps({"run": mode, "crf_micro": metrics["crf"]["micro"],
                      "invalid_iob2": metrics["invalid_iob2"], "seconds": manifest["seconds"],
                      "peak_rss_mib": manifest["peak_process_rss_mib"],
                      "optimization": manifest["optimization"]}, indent=2))


def verify_run(config: dict, mode: str, directory: Path, examples: dict, labels: list[str],
               provenance: dict) -> dict:
    manifest = read_json(directory / "run.json")
    if manifest["mode"] != mode or read_json(directory / "config.json") != config:
        raise ValueError("Run mode or config differs")
    if (manifest["provenance"] != provenance
            or manifest["preparation_audit_sha256"] != config["preparation_audit_sha256"]):
        raise ValueError("Run provenance differs from the verified preparation")
    for path, expected in manifest["source_sha256"].items():
        verify_hash(Path(path), expected)
    for name, expected in manifest["artifacts_sha256"].items():
        verify_hash(directory / name, expected)
    train, dev = selected_rows(examples, config, mode)
    if (manifest["train_records"], manifest["dev_records"]) != (len(train), len(dev)):
        raise ValueError("Run partition sizes differ")
    with (directory / "predictions.jsonl").open(encoding="utf-8") as stream:
        metrics, diagnostics = evaluate(dev, (json.loads(line) for line in stream),
                                        {row.utt for row in train}, labels)
    if metrics != read_json(directory / "metrics.json") or diagnostics != read_json(directory / "diagnostics.json"):
        raise ValueError("Saved metrics or diagnostics differ from recomputation")
    return {"result": "PASS", "mode": mode, "verified_dev_records": len(dev),
            "manifest_sha256": sha256_file(directory / "run.json"),
            "checks": ["preparation and dataset hashes", "source/config/artifact hashes",
                       "exact development ID coverage", "recomputed metrics and diagnostics"],
            "model_retrained": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", type=Path, help="Run directory override")
    parser.add_argument("--verify", action="store_true", help="Verify saved artifacts without training")
    args = parser.parse_args()
    config = read_json(args.config)
    if config["evaluation"] != {
        "invalid_iob2_transition": "promote_orphan_or_mismatched_i_to_b_and_log",
        "zero_denominator": 0, "primary": "exact_typed_character_span_micro_f1",
    }:
        raise ValueError("Unsupported scoring configuration")
    output = args.output or Path(config["output_dir"]) / args.mode
    examples, labels, provenance = load_inputs(config)
    if not args.verify:
        create_run(args.config, config, args.mode, output, examples, labels, provenance)
        return
    print(json.dumps(verify_run(config, args.mode, output, examples, labels, provenance), indent=2))


if __name__ == "__main__":
    main()
