"""Verify the audited release and load immutable official train/dev records."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any
import unicodedata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def prepare_release(archive: Path, audit: dict, output: Path) -> dict:
    """Check all selected members before writing; never use unrestricted tar extraction."""
    archive_hash = verify_hash(archive, audit["archive_sha256"])
    expected = {audit["archive_member"]: audit["file_sha256"],
                **audit["license_member_sha256"]}
    verified = {}
    with tarfile.open(archive, "r:gz") as release:
        for member, checksum in expected.items():
            info = release.getmember(member)
            if not info.isfile():
                raise ValueError(f"Expected a regular archive member: {member}")
            with release.extractfile(info) as stream:
                content = stream.read()
            actual = hashlib.sha256(content).hexdigest()
            if actual != checksum:
                raise ValueError(f"SHA-256 mismatch for archive member {member}: {actual}")
            verified[member] = (content, actual)
    # Also check existing targets before writing any new files.
    for member, (_, checksum) in verified.items():
        target = output / Path(member).name
        if target.exists():
            verify_hash(target, checksum)
    output.mkdir(parents=True, exist_ok=True)
    for member, (content, _) in verified.items():
        target = output / Path(member).name
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(content)
    return {"source_url": audit["source_url"], "archive_sha256": archive_hash,
            "members": {m: h for m, (_, h) in verified.items()}}


@dataclass(frozen=True)
class Record:
    id: str
    partition: str
    scenario: str
    intent: str
    utt: str


@dataclass(frozen=True)
class Dataset:
    train: tuple[Record, ...]
    dev: tuple[Record, ...]
    labels: tuple[str, ...]
    diagnostics: dict[str, Any]


def validate_record(raw: Any, line: int) -> Record:
    """Validate core and annotation-container schema without modifying text."""
    if not isinstance(raw, dict):
        raise ValueError(f"Line {line}: expected a JSON object")
    for key in ("id", "locale", "partition", "scenario", "intent", "utt",
                "annot_utt", "worker_id"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError(f"Line {line}: invalid string field {key}")
    if not raw["id"].isascii() or not raw["id"].isdigit():
        raise ValueError(f"Line {line}: invalid ID")
    if raw["locale"] != "az-AZ":
        raise ValueError(f"Line {line}: unexpected locale {raw['locale']}")
    if raw["partition"] not in {"train", "dev", "test"}:
        raise ValueError(f"Line {line}: invalid partition {raw['partition']}")
    if not raw["intent"].startswith(raw["scenario"] + "_"):
        raise ValueError(f"Line {line}: intent/scenario mismatch")
    for key in ("slot_method", "judgments"):
        if not isinstance(raw.get(key), list) or any(
            not isinstance(item, dict) for item in raw[key]
        ):
            raise ValueError(f"Line {line}: invalid annotation container {key}")
    return Record(raw["id"], raw["partition"], raw["scenario"], raw["intent"], raw["utt"])


def duplicate_diagnostics(records: list[Record]) -> dict:
    """Exact means byte-equivalent UTF-8 text: no normalization or case folding."""
    groups = defaultdict(list)
    for record in records:
        groups[record.utt].append(record)
    duplicates = [group for group in groups.values() if len(group) > 1]
    cross = [g for g in duplicates if len({r.partition for r in g}) > 1]
    train_dev = [g for g in duplicates if {"train", "dev"} <= {r.partition for r in g}]
    return {
        "exact_duplicate_text_groups": len(duplicates),
        "cross_split_exact_duplicate_text_groups": len(cross),
        "train_test_exact_duplicate_text_groups": sum(
            {"train", "test"} <= {r.partition for r in g} for g in duplicates),
        "cross_split_exact_text_groups_with_multiple_intents": sum(
            len({r.intent for r in g}) > 1 for g in cross),
        "train_dev_exact_duplicate_text_groups": len(train_dev),
        "train_dev_conflicting_intent_text_groups": sum(
            len({r.intent for r in g if r.partition in {"train", "dev"}}) > 1
            for g in train_dev),
        "dev_records_with_text_in_train": sum(
            r.partition == "dev" for g in train_dev for r in g),
        "train_dev_overlap": [
            {"text_sha256": hashlib.sha256(g[0].utt.encode("utf-8")).hexdigest(),
             "records": [{"id": r.id, "partition": r.partition, "intent": r.intent}
                         for r in g if r.partition in {"train", "dev"}]}
            for g in train_dev
        ],
    }


def load_dataset(path: Path, audit: dict) -> Dataset:
    """Validate every partition structurally, but expose only train/dev for modeling."""
    verify_hash(path, audit["file_sha256"])
    if path.stat().st_size != audit["bytes"]:
        raise ValueError("Dataset byte count disagrees with audit")
    records = []
    ids = set()
    with path.open(encoding="utf-8") as stream:
        for line, text in enumerate(stream, 1):
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line}: invalid JSON") from exc
            record = validate_record(raw, line)
            if record.id in ids:
                raise ValueError(f"Line {line}: duplicate ID {record.id}")
            ids.add(record.id)
            records.append(record)
    train = tuple(r for r in records if r.partition == "train")
    dev = tuple(r for r in records if r.partition == "dev")
    labels = tuple(sorted({r.intent for r in train}))
    # The classifier label inventory is established from train, never from test.
    if not train or not dev or any(r.intent not in labels for r in records):
        raise ValueError("Missing train/dev records or a label absent from train")
    diagnostics = {
        "records": len(records), "unique_ids": len(ids), "locales": ["az-AZ"],
        "split_counts": dict(sorted(Counter(r.partition for r in records).items())),
        "intent_count": len(labels), "scenario_count": len({r.scenario for r in records}),
        "non_nfc_utterances": sum(r.utt != unicodedata.normalize("NFC", r.utt)
                                  for r in records),
        "empty_utterances": 0,
        **duplicate_diagnostics(records),
        "intent_counts_train": dict(sorted(Counter(r.intent for r in train).items())),
        "intent_counts_dev": dict(sorted(Counter(r.intent for r in dev).items())),
        "labels": list(labels),
        "test_policy": "Structural validation only; no test records exposed to modeling.",
    }
    for key in ("records", "unique_ids", "locales", "split_counts", "intent_count",
                "scenario_count", "non_nfc_utterances", "empty_utterances",
                "exact_duplicate_text_groups", "cross_split_exact_duplicate_text_groups",
                "train_test_exact_duplicate_text_groups",
                "cross_split_exact_text_groups_with_multiple_intents"):
        if key in audit and diagnostics[key] != audit[key]:
            raise ValueError(f"Audit mismatch for {key}: {diagnostics[key]} != {audit[key]}")
    return Dataset(train, dev, labels, diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    print(json.dumps(prepare_release(args.archive, audit, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
