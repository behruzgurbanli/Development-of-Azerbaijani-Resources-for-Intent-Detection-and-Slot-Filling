"""Produce and verify the initial train/dev-only AzMASSIVE resource package."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import shutil

from .dataset import sha256_file, validate_record, verify_hash
from .slots import LABEL, Span, parse_annotation

CONFIG = Path("configs/RESOURCE-EXPORT-001.json")
PARTITIONS = ("train", "dev")
VERSION = "0.1.0"

# Match the audited release's leading record metadata before decoding JSON.
HEADER = re.compile(rb'\s*\{\s*"id"\s*:\s*"[0-9]+"\s*,\s*"locale"\s*:\s*"az-AZ"'
                    rb'\s*,\s*"partition"\s*:\s*"(train|dev|test)"\s*,')


def source_rows(lines: Iterable[bytes]) -> Iterator[dict]:
    """Inspect only the leading partition marker on excluded lines."""
    seen = set()
    for number, line in enumerate(lines, 1):
        header = HEADER.match(line)
        if header is None:
            raise ValueError(f"Line {number}: unexpected release header")
        partition = header[1].decode("ascii")
        if partition not in PARTITIONS:
            continue
        raw = json.loads(line)
        row = validate_record(raw, number)
        if row.partition != partition or row.id in seen:
            raise ValueError(f"Line {number}: inconsistent partition or duplicate ID")
        if not LABEL.fullmatch(row.intent) or not LABEL.fullmatch(row.scenario):
            raise ValueError(f"Line {number}: invalid label")
        seen.add(row.id)
        yield raw


def export_row(raw: dict, slot_labels: set[str]) -> dict:
    _, spans = parse_annotation(raw["annot_utt"], raw["utt"])
    if any(span.label not in slot_labels for span in spans):
        raise ValueError("Unknown source slot label")
    return {
        "schema_version": VERSION,
        **{key: raw[key] for key in ("id", "partition", "locale", "scenario", "utt")},
        "source_annotation": {
            "intent": raw["intent"],
            "annot_utt": raw["annot_utt"],
            "spans": [asdict(span) for span in spans],
        },
    }


def render_annotation(utt: str, spans: Iterable[Span]) -> str:
    """Reconstruct original markup from exact source slices and typed offsets."""
    parts = []
    cursor = 0
    for span in spans:
        if not cursor <= span.start < span.end <= len(utt):
            raise ValueError("Invalid, overlapping or unsorted source spans")
        parts.append(utt[cursor:span.start])
        parts.append(f"[{span.label} : {utt[span.start:span.end]}]")
        cursor = span.end
    parts.append(utt[cursor:])
    return "".join(parts)


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def validate_rows(payload: bytes, expected: list[dict]) -> dict:
    """Compare serialized records to source and independently rebuild their markup."""
    lines = payload.split(b"\n")
    if len(lines) != len(expected) + 1 or lines[-1] != b"":
        raise ValueError("Incorrect record count or missing final LF")
    counts = {partition: Counter(records=0, spans=0, slot_free=0, round_trips=0)
              for partition in PARTITIONS}
    for line, source in zip(lines[:-1], expected, strict=True):
        row = json.loads(line)
        if line + b"\n" != json_bytes(source):
            raise ValueError("Export differs from the canonical source record or source order")
        annotation = row["source_annotation"]
        spans = [Span(**fields) for fields in annotation["spans"]]
        if render_annotation(row["utt"], spans) != annotation["annot_utt"]:
            raise ValueError("Character-offset round trip changed original markup")
        counter = counts[row["partition"]]
        counter.update(records=1, spans=len(spans), slot_free=int(not spans), round_trips=1)
    return counts


def input_path(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.resolve().is_relative_to(Path.cwd()):
        raise ValueError("Use repository-relative paths")
    return path


def verified_inputs() -> tuple[dict, dict, dict, dict]:
    config = json.loads(CONFIG.read_bytes())
    if config["schema_version"] != VERSION or config["partitions"] != list(PARTITIONS):
        raise ValueError("Unsupported export version or split scope")
    audit_path = input_path(config["audit"])
    slot_path = input_path(config["slot_audit"])
    verify_hash(audit_path, config["audit_sha256"])
    verify_hash(slot_path, config["slot_audit_sha256"])
    audit = json.loads(audit_path.read_bytes())
    slot_audit = json.loads(slot_path.read_bytes())
    license_hash = audit["license_member_sha256"]["1.0/LICENSE"]
    hashes = {
        "audit": config["audit_sha256"],
        "archive_as_audited": audit["archive_sha256"],
        "locale_jsonl": verify_hash(input_path(config["source"]), audit["file_sha256"]),
        "license": verify_hash(input_path(config["license"]), license_hash),
        "slot_audit": config["slot_audit_sha256"],
    }
    return config, audit, slot_audit, hashes


def prepare_rows(config: dict, audit: dict, slot_audit: dict, *, smoke: bool) -> tuple[list[dict], dict]:
    with input_path(config["source"]).open("rb") as stream:
        rows = source_rows(stream)
        selected = [next(rows)] if smoke else list(rows)
    slot_labels = set(audit["slot_labels"])
    exported = [export_row(raw, slot_labels) for raw in selected]
    reference = {split: {
        "records": audit["split_counts"][split],
        "spans": slot_audit["parsing"][split]["slot_spans"],
        "slot_free": slot_audit["parsing"][split]["slot_free"],
        "round_trips": audit["split_counts"][split],
    } for split in PARTITIONS}
    return exported, reference


def run(mode: str, name: str | None) -> dict:
    config, audit, slot_audit, hashes = verified_inputs()
    root = input_path(config["output_root"]).resolve()
    if not root.is_relative_to(Path.cwd() / "artifacts"):
        raise ValueError("Export output must be under ignored artifacts/")
    if name and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name):
        raise ValueError("Run name must be a single safe directory name")
    output = root / (name or ("smoke" if mode == "smoke" else "full"))
    if output.is_symlink():
        raise ValueError("Export directory must not be a symlink")
    rows, reference = prepare_rows(config, audit, slot_audit, smoke=mode == "smoke")
    payload = ((output / "source.jsonl").read_bytes() if mode == "verify"
               else b"".join(json_bytes(row) for row in rows))
    counts = validate_rows(payload, rows)
    if mode != "smoke" and counts != reference:
        raise ValueError("Export counts disagree with AUD-001 or SLOT-PREP-001")
    package = {
        "source.jsonl": payload,
        "review-proposals.jsonl": b"",
        "LICENSE": input_path(config["license"]).read_bytes(),
        "annotation.schema.json": input_path(config["schema"]).read_bytes(),
    }
    manifest = {
        "experiment_id": config["experiment_id"],
        "schema_version": VERSION,
        "mode": "smoke" if mode == "smoke" else "full",
        "source": {
            "release": audit["release"],
            "locale": "az-AZ",
            "archive_member": audit["archive_member"],
            "url": audit["source_url"],
            "license": audit["license"],
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": "Amazon MASSIVE contributors (FitzGerald et al., 2023), based on SLURP.",
            "sha256": hashes,
        },
        "partitions": list(PARTITIONS),
        "ordering": "Original locale JSONL order restricted to permitted partitions",
        "offsets": "Zero-based half-open Unicode code point indices into utt",
        "normalization": "none",
        "counts": {split: dict(counts[split]) for split in PARTITIONS},
        "review_proposals": 0,
        "implementation_sha256": {str(path): sha256_file(path) for path in (
            CONFIG, Path("src/azmassive/resource_export.py"), Path("src/azmassive/slots.py"),
            Path("src/azmassive/dataset.py"),
        )},
        "output_sha256": {key: hashlib.sha256(data).hexdigest() for key, data in package.items()},
    }
    package["manifest.json"] = json_bytes(manifest)
    if mode == "verify":
        for key, data in package.items():
            if (output / key).read_bytes() != data:
                raise ValueError(f"Package verification failed for {key}")
    else:
        output.mkdir(parents=True, exist_ok=False)
        try:
            for key, data in package.items():
                (output / key).write_bytes(data)
        except BaseException:
            shutil.rmtree(output)
            raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "export", "verify"))
    parser.add_argument("--run-name", help="Single directory name under the ignored export root")
    args = parser.parse_args()
    print(json.dumps(run(args.mode, args.run_name), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
