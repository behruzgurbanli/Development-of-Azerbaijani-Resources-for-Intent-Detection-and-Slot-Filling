"""Prepare private, unjudged train/dev review sheets from the audited az-AZ release."""

import csv
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .dataset import sha256_file, validate_record, verify_hash
from .slots import parse_annotation

OUTPUT = Path("artifacts/THESIS-GAP-001")
REVIEW = ("intent_judgment", "slot_judgment", "proposed_intent", "proposed_spans",
          "phenomenon", "reason", "unresolved_question", "reviewer", "review_date")
SEED = 42


def rank(key):
    return hashlib.sha256(f"{SEED}|{key}".encode()).hexdigest(), key


def review_rows(lines):
    rows = {}
    for number, line in enumerate(lines, 1):
        raw = json.loads(line)
        if raw["partition"] == "test":
            continue
        record = validate_record(raw, number)
        if record.id in rows:
            raise ValueError(f"Duplicate ID {record.id}")
        _, spans = parse_annotation(raw["annot_utt"], record.utt)
        rows[record.id] = {k: raw[k] for k in ("partition", "id", "scenario", "intent", "utt", "annot_utt")}
        rows[record.id]["spans"] = [asdict(s) for s in spans]
    return rows


def stratified(rows, split, quota):
    groups = defaultdict(list)
    for row in rows.values():
        if row["partition"] == split:
            groups[row["intent"]].append(row["id"])
    for ids in groups.values():
        ids.sort(key=rank)
    if not groups or sum(map(len, groups.values())) < quota:
        raise ValueError("Insufficient eligible records")
    # Equal allocation across supported intents, hash order breaks a partial-round tie.
    selected = []
    for depth in range(max(map(len, groups.values()))):
        for label in sorted(groups, key=rank):
            if depth < len(groups[label]):
                selected.append(groups[label][depth])
                if len(selected) == quota:
                    return selected


def targeted(rows, sampled, audit):
    reasons = defaultdict(list)
    groups = sorted(audit["duplicate_text_review"]["groups"], key=lambda g: g["text_sha256"])
    chosen = 0
    for group in groups:
        ids = [v["id"] for v in group["variants"]]
        if sampled.intersection(ids):
            continue
        chosen += 1
        for key in ids:
            reasons[key].append(f"disagreement group {chosen}")
        if chosen == 4:
            break
    for label, count in sorted(audit["inventory_by_split"]["train"].items()):
        if count > 10:
            continue
        candidates = [k for k, r in rows.items() if k not in sampled and r["partition"] == "train"
                      and any(s["label"] == label for s in r["spans"])]
        if candidates:
            reasons[min(candidates, key=rank)].append(f"rare {label}")
    for key, note in [("1146", "inflected boundary"), ("49", "punctuation boundary")]:
        if key not in sampled:
            reasons[key].append(note)
    return reasons


def write_sheet(path, rows, selection):
    fields = ("reason_selected", "partition", "id", "scenario", "intent", "utt", "annot_utt", "spans", *REVIEW)
    with path.open("x", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for key, reason in selection.items():
            row = rows[key]
            writer.writerow({**row, "reason_selected": reason,
                             "spans": json.dumps(row["spans"], ensure_ascii=False, separators=(",", ":"))})


def main():
    config = json.loads(Path("configs/SLOT-PREP-001.json").read_text())["dataset"]
    verify_hash(Path(config["audit"]), config["audit_sha256"])
    source = json.loads(Path(config["audit"]).read_text())
    path = Path(config["jsonl"])
    verify_hash(path, source["file_sha256"])
    audit_path = Path("results/SLOT-PREP-001/audit.json")
    verify_hash(audit_path, "965bbaf37eb4688e22a67d4abda502f5e74da9a4dd04f8f09eb85f43362d2367")
    audit = json.loads(audit_path.read_text())
    with path.open(encoding="utf-8") as stream:
        rows = review_rows(stream)
    for split in ("train", "dev"):
        if sum(r["partition"] == split for r in rows.values()) != source["split_counts"][split]:
            raise ValueError(f"Incorrect {split} count")
    base = stratified(rows, "train", 120) + stratified(rows, "dev", 60)
    extra = targeted(rows, set(base), audit)
    selections = {"stratified": dict.fromkeys(base, "intent-stratified"),
                  "targeted": {k: " / ".join(v) for k, v in extra.items()}}
    OUTPUT.mkdir(parents=True, exist_ok=False)  # Never overwrite a reviewer's working copies.
    for cohort, selection in selections.items():
        target = OUTPUT / f"{cohort}.csv"
        write_sheet(target, rows, selection)
        for split in ("train", "dev"):
            labels = Counter(rows[k]["intent"] for k in selection if rows[k]["partition"] == split)
            print(cohort, split, sum(labels.values()), "records,", len(labels), "intents")
        print(target, sha256_file(target))


if __name__ == "__main__":
    main()
