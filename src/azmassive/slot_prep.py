"""Audit MASSIVE train/dev reconstruction and exact slot alignment."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import unicodedata
from zoneinfo import ZoneInfo

from .dataset import sha256_file, validate_record, verify_hash
from .slots import AlignmentError, TOKENIZERS, bio_to_spans, parse_annotation, spans_to_bio, tokenize


def audit_annotations(path: Path, expected_counts: dict, known_labels: set[str],
                      methods: list[str]) -> dict:
    counts = {split: Counter(records=0, reconstructed=0, slot_free=0, slot_spans=0,
                             multiword_spans=0, spans_with_punctuation=0)
              for split in expected_counts}
    inventory = {split: Counter() for split in expected_counts}
    failures = {split: [] for split in expected_counts}
    comparisons = {split: {method: {"tokens": 0, "aligned_records": 0, "aligned_spans": 0,
                                    "failed_spans": 0, "failures": []}
                           for method in methods} for split in expected_counts}
    groups = defaultdict(list)
    ids = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            raw = json.loads(line)
            split = raw.get("partition")
            if split == "test":
                continue
            if split not in expected_counts:
                raise ValueError(f"Unexpected partition on line {line_number}: {split}")
            split_counts = counts[split]
            split_counts["records"] += 1
            try:
                record = validate_record(raw, line_number)
            except ValueError as exc:
                failures[split].append({"id": raw.get("id"), "stage": "schema", "reason": str(exc)})
                continue
            if record.id in ids:
                raise ValueError(f"Duplicate train/dev record ID: {record.id}")
            ids.add(record.id)
            try:
                _, spans = parse_annotation(raw["annot_utt"], record.utt)
            except ValueError as exc:
                failures[split].append({"id": record.id, "stage": "parse", "reason": str(exc)})
                continue
            split_counts["reconstructed"] += 1
            split_counts["slot_free"] += not spans
            split_counts["slot_spans"] += len(spans)
            for span in spans:
                value = record.utt[span.start:span.end]
                inventory[split][span.label] += 1
                split_counts["multiword_spans"] += any(c.isspace() for c in value)
                split_counts["spans_with_punctuation"] += any(
                    unicodedata.category(c).startswith("P") for c in value)
                if span.label not in known_labels:
                    failures[split].append({"id": record.id, "stage": "inventory",
                                            "reason": "Unknown audited label", "span": asdict(span)})
            groups[record.utt].append((record, spans))
            for method in methods:
                result = comparisons[split][method]
                tokens = tokenize(record.utt, method)
                result["tokens"] += len(tokens)
                try:
                    tags = spans_to_bio(tokens, spans)
                except AlignmentError as exc:
                    result["aligned_spans"] += len(spans) - len(exc.failures)
                    result["failed_spans"] += len(exc.failures)
                    result["failures"].append({"id": record.id, "stage": "alignment",
                                               "spans": exc.failures})
                    continue
                try:
                    if bio_to_spans(tokens, tags) != spans:
                        raise ValueError("BIO round trip changed the original spans")
                except ValueError as exc:
                    result["failed_spans"] += len(spans)
                    result["failures"].append({"id": record.id, "stage": "bio",
                                               "reason": str(exc)})
                    continue
                result["aligned_records"] += 1
                result["aligned_spans"] += len(spans)

    conflicts = []
    cross_split_group_count = 0
    for text, entries in groups.items():
        cross_split = len({record.partition for record, _ in entries}) > 1
        cross_split_group_count += cross_split
        if len({spans for _, spans in entries}) > 1:
            conflicts.append({
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "cross_split": cross_split,
                "variants": [{"id": record.id, "partition": record.partition,
                              "intent": record.intent, "spans": [asdict(s) for s in spans]}
                             for record, spans in entries],
            })
    for split, expected in expected_counts.items():
        if counts[split]["records"] != expected:
            raise ValueError(f"{split} has {counts[split]['records']} records, expected {expected}")
        for result in comparisons[split].values():
            result["alignment_rate_records"] = result["aligned_records"] / expected
            n_spans = counts[split]["slot_spans"]
            result["alignment_rate_parsed_spans"] = result["aligned_spans"] / n_spans if n_spans else None
    return {
        "parsing": counts,
        "annotation_failures_by_split": failures,
        "inventory_by_split": {s: dict(sorted(c.items())) for s, c in inventory.items()},
        "dev_labels_absent_from_train": sorted(inventory["dev"].keys() - inventory["train"].keys()),
        "audited_labels_absent_from_train_dev": sorted(
            known_labels - (inventory["dev"].keys() | inventory["train"].keys())),
        "tokenizer_comparison": comparisons,
        "duplicate_text_review": {
            "train_dev_overlap_groups": cross_split_group_count,
            "conflicting_groups": len(conflicts),
            "conflicting_cross_split_groups": sum(group["cross_split"] for group in conflicts),
            "groups": conflicts,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New path for a separate audit run")
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    methods = config["tokenizers"]
    selected = config["selected_tokenizer"]
    if (dataset["partitions"] != ["train", "dev"] or dataset["release"] != "MASSIVE 1.0"
            or dataset["locale"] != "az-AZ" or config["tag_scheme"] != "IOB2"
            or config["alignment"] != "strict_exact_boundaries"
            or not methods or selected not in methods or len(set(methods)) != len(methods)
            or not set(methods) <= TOKENIZERS.keys()):
        raise ValueError("Unsupported slot preparation protocol")
    output = args.output or Path(config["output"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    audit_hash = verify_hash(Path(dataset["audit"]), dataset["audit_sha256"])
    source = json.loads(Path(dataset["audit"]).read_text(encoding="utf-8"))
    file_path = Path(dataset["jsonl"])
    hashes = {
        "audit": audit_hash,
        "archive": verify_hash(Path(dataset["archive"]), source["archive_sha256"]),
        "jsonl": verify_hash(file_path, source["file_sha256"]),
        "license": verify_hash(Path(dataset["license"]), source["license_member_sha256"]["1.0/LICENSE"]),
    }
    known_labels = set(source["slot_labels"])
    if file_path.stat().st_size != source["bytes"] or len(known_labels) != source["slot_type_count"]:
        raise ValueError("Audited input byte count or slot inventory size disagrees")
    counts = {split: source["split_counts"][split] for split in dataset["partitions"]}
    results = audit_annotations(file_path, counts, known_labels, methods)
    ready = all(not results["annotation_failures_by_split"][split]
                and results["tokenizer_comparison"][split][selected]["aligned_records"] == counts[split]
                for split in counts)
    files = (Path(__file__), Path("src/azmassive/slots.py"), Path("src/azmassive/dataset.py"),
             Path("tests/test_slots.py"), args.config)
    report = {
        "experiment_id": config["experiment_id"],
        "started_utc": started.isoformat(),
        "started_eastern": started.astimezone(ZoneInfo("America/New_York")).isoformat(),
        "duration_seconds_before_write": time.perf_counter() - clock,
        "command": [sys.executable, "-B", "-m", "azmassive.slot_prep", *sys.argv[1:]],
        "required_pythonpath": "src",
        "git_base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_config_sha256": {str(path.relative_to(Path.cwd()) if path.is_absolute() else path):
                                 sha256_file(path) for path in files},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "unicode_version": unicodedata.unidata_version, "dependencies": "stdlib",
                        "randomness": "none"},
        "config": config,
        "provenance": {"audit_id": source["audit_id"], "license": source["license"],
                       "source_url": source["source_url"], "sha256": hashes,
                       "audited_slot_inventory": sorted(known_labels)},
        "processing": {
            "tokenizer_patterns": {method: TOKENIZERS[method].pattern for method in methods},
            "offsets": "Zero-based half-open Python character offsets in utt",
            "normalization": "none",
            "test": "Test records skipped by partition before annotation parsing and tokenization",
            "conflict_policy": "Keep all original annotations and splits; report differences for review",
            "training": "none",
        },
        **results,
        "selected_tokenizer_all_records_ready": ready,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{output}: {sum(r['reconstructed'] for r in results['parsing'].values())} "
          f"reconstructed records. Selected tokenizer passed: {ready}.")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
