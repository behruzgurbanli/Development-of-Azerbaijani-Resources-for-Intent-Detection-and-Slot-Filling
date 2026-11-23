"""Evaluate saved intent and slot predictions on audited official dev IDs."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import json
import shlex
import sys
from zoneinfo import ZoneInfo

from . import intent_hybrid, intent_svc, slot_crf
from .dataset import Record, sha256_file, verify_hash
from .evaluation import evaluate_intents
from .experiment import command_output, write_json
from .intent_svc import read_json
from .slot_metrics import SlotExample, align_predictions as align_slot_predictions, decode_prediction, score_spans

OUTCOMES = {(True, True): "both_correct", (True, False): "intent_only_correct",
            (False, True): "slots_only_correct", (False, False): "neither_correct"}


def joint_outcome(gold_intent, predicted_intent, gold_spans, predicted_spans) -> str:
    """Require exact intent and equality of the complete typed span sets."""
    return OUTCOMES[(gold_intent == predicted_intent, set(gold_spans) == set(predicted_spans))]


def align_gold(intent_rows: list[Record], slot_rows: list[SlotExample]) -> dict[str, SlotExample]:
    gold = {row.id: row for row in slot_rows}
    if (len(gold) != len(slot_rows) or len({row.id for row in intent_rows}) != len(intent_rows)
            or gold.keys() != {row.id for row in intent_rows}):
        raise ValueError("Gold sources must have identical, unique dev IDs")
    if any(row.partition != "dev" or row.utt != gold[row.id].utt for row in intent_rows):
        raise ValueError("Gold partition or utterance mismatch between tasks")
    return gold


def decode_slots(dev: list[SlotExample], rows, labels: list[str]) -> tuple[dict, list[dict]]:
    if len({row.id for row in dev}) != len(dev):
        raise ValueError("Duplicate gold slot ID")
    def checked_rows():
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"id", "tags"} or not isinstance(row["id"], str):
                raise ValueError("Malformed slot prediction record")
            yield row
    allowed = set(labels)
    predictions = align_slot_predictions(dev, checked_rows())
    decoded, repairs = {}, []
    for row in dev:
        tags = predictions[row.id]
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(f"Malformed BIO sequence at {row.id}")
        try:
            spans, changes = decode_prediction(row.tokens, tags)
        except ValueError as exc:
            raise ValueError(f"Malformed BIO prediction at {row.id}: {exc}") from exc
        if any(span.label not in allowed for span in spans):
            raise ValueError(f"Unknown slot type at {row.id}")
        decoded[row.id] = spans
        if changes:
            repairs.append({"id": row.id, "changes": changes})
    return decoded, repairs


def breakdown(outcomes) -> dict:
    counts = dict.fromkeys(OUTCOMES.values(), 0)
    counts.update(Counter(outcomes))
    total = sum(counts.values())
    return {"n": total, "joint_exact_matches": counts["both_correct"],
            "joint_exact_match": counts["both_correct"] / total if total else None,
            "counts": counts}


def model_key(intent: str, slot: str) -> str:
    return f"{intent}+{slot}"


def paired_joint(entries: list[dict], primary: str, comparison: str) -> dict:
    directions = {(True, True): "both_joint_correct", (True, False): "primary_only_joint_correct",
                  (False, True): "comparison_only_joint_correct", (False, False): "neither_joint_correct"}
    counts = dict.fromkeys(directions.values(), 0)
    groups = {name: counts.copy() for name in ("seen", "unseen")}
    changes = []
    for row in entries:
        before, after = (row["outcomes"][key] == "both_correct" for key in (primary, comparison))
        direction = directions[before, after]
        counts[direction] += 1
        groups[row["text_overlap"]][direction] += 1
        if before != after:
            changes.append({"id": row["id"], "direction": direction, "text_overlap": row["text_overlap"]})
    return {"primary": primary, "comparison": comparison, "counts": counts,
            "by_exact_text_overlap": groups, "joint_status_changes": changes,
            "net_additional_joint_matches": counts["comparison_only_joint_correct"]
            - counts["primary_only_joint_correct"]}


def evaluate_joint(dev, gold, train_text, intents, slots, primary, alternate) -> tuple[dict, list[dict]]:
    entries = []
    for row in dev:
        gold_spans = gold[row.id].gold_spans
        predicted_intents = {name: values[row.id] for name, values in intents.items()}
        outcomes = {model_key(i, s): joint_outcome(row.intent, prediction, gold_spans, values[row.id])
                    for i, prediction in predicted_intents.items() for s, values in slots.items()}
        entries.append({
            "id": row.id, "gold_intent": row.intent, "scenario": row.scenario,
            "text_overlap": "seen" if row.utt in train_text else "unseen",
            "outcomes": outcomes, "intent_predictions": predicted_intents,
            "gold_spans": [asdict(span) for span in gold_spans],
            "slot_errors": {
                name: {"missed": [asdict(span) for span in gold_spans if span not in values[row.id]],
                       "spurious": [asdict(span) for span in values[row.id] if span not in gold_spans]}
                for name, values in slots.items()
            },
        })
    pairs = {}
    for key in entries[0]["outcomes"]:
        summary = breakdown(row["outcomes"][key] for row in entries)
        for name, field in (("by_exact_text_overlap", "text_overlap"), ("by_intent", "gold_intent"),
                            ("by_scenario", "scenario")):
            groups = {}
            for row in entries:
                groups.setdefault(row[field], []).append(row["outcomes"][key])
            summary[name] = {group: breakdown(values) for group, values in sorted(groups.items())}
        pairs[key] = summary
    return {"model_pairs": pairs, "paired_comparison": paired_joint(entries, primary, alternate)}, entries


def prediction_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def compute(config: dict) -> tuple[dict, list[dict], dict]:
    if config["evaluation_partition"] != "dev":
        raise ValueError("Joint evaluation is restricted to official dev")
    svc_path = Path(config["intent_models"]["INT-SVC-001"]["config"])
    svc = read_json(svc_path)
    pr5, train, dev, labels, _, _, provenance = intent_svc.inputs(svc, "full")
    inputs = {str(svc_path): sha256_file(svc_path)}
    source = pr5["dataset"]
    for key in ("audit_path", "archive_path", "file_path", "license_path"):
        path = Path(source[key])
        inputs[str(path)] = sha256_file(path)
    intents, slots, repairs = {}, {}, {}
    intent_metrics, slot_metrics = {}, {}

    def record_inputs(*paths):
        for path in paths:
            inputs[str(path)] = sha256_file(path)

    for name, spec in config["intent_models"].items():
        path = Path(spec["config"])
        cfg = read_json(path)
        if cfg["experiment_id"] != name:
            raise ValueError(f"Unexpected intent config for {name}")
        directory = Path(cfg["output_dir"]) / "full"
        proof_path = (directory.parent / "full-verification.json" if name == "PR5-INT-001"
                      else directory / "verification.json")
        saved_proof = read_json(proof_path)
        if saved_proof["result"] != "PASS":
            raise ValueError(f"Expected a verified run for {name}")
        verify_hash(directory / "run.json", saved_proof["run_manifest_sha256"])
        if name == "PR5-INT-001":
            if cfg != pr5:
                raise ValueError("PR5 config disagrees with the verified intent loader")
        else:
            verifier = intent_svc.verify if name == "INT-SVC-001" else intent_hybrid.verify
            if verifier(cfg, "full", directory) != saved_proof:
                raise ValueError(f"Saved verification disagrees for {name}")
        pred_path = directory / "predictions.jsonl"
        picked = intent_svc.align_predictions(dev, pred_path, spec["prediction_key"])
        metrics = evaluate_intents([row.intent for row in dev], picked, labels)
        published = read_json(directory / "metrics.json")
        if name == "PR5-INT-001":
            published = published["logistic_regression"]
        if metrics != published:
            raise ValueError(f"Intent metrics disagree with saved run {name}")
        intents[name] = dict(zip((row.id for row in dev), picked, strict=True))
        intent_metrics[name] = {k: metrics[k] for k in ("n", "correct", "accuracy", "macro_f1")}
        record_inputs(path, proof_path, pred_path, directory/"run.json", directory/"metrics.json")

    reference = None
    for name, filename in config["slot_models"].items():
        path = Path(filename)
        cfg = read_json(path)
        if cfg["experiment_id"] != name:
            raise ValueError(f"Unexpected slot config for {name}")
        rows, inventory, slot_provenance = slot_crf.load_inputs(cfg)
        if slot_provenance["sha256"] != provenance["input_sha256"] | {"audit": provenance["audit_sha256"]}:
            raise ValueError("Slot and intent dataset provenance disagree")
        gold = align_gold(dev, rows["dev"])
        if (reference is not None and reference != gold
                or {r.id: r.utt for r in rows["train"]} != {r.id: r.utt for r in train}):
            raise ValueError("Intent and slot gold splits or slot annotations disagree")
        reference = gold
        directory = Path(cfg["output_dir"]) / "full"
        proof_path = directory / "verification.json"
        if slot_crf.verify_run(cfg, "full", directory, rows, inventory, slot_provenance) != read_json(proof_path):
            raise ValueError(f"Saved verification disagrees for {name}")
        pred_path = directory / "predictions.jsonl"
        decoded, repairs[name] = decode_slots(rows["dev"], prediction_rows(pred_path), inventory)
        scores = score_spans(((row.gold_spans, decoded[row.id]) for row in rows["dev"]), inventory)
        if scores != read_json(directory / "metrics.json")["crf"]:
            raise ValueError(f"Slot scoring disagrees with saved run {name}")
        slots[name] = decoded
        slot_metrics[name] = {k: scores[k] for k in ("utterances", "exact_matches", "utterance_exact_match", "micro")}
        slot_metrics[name]["invalid_iob2_transitions"] = sum(len(r["changes"]) for r in repairs[name])
        record_inputs(path, proof_path, pred_path, directory/"run.json", directory/"metrics.json",
                      Path(cfg["preparation_config"]), Path(cfg["preparation_audit"]))

    primary, alternate = (model_key(**config[key]) for key in ("primary", "comparison"))
    metrics, entries = evaluate_joint(dev, reference, {row.utt for row in train}, intents, slots, primary, alternate)
    # Independent aggregation path: intersect per-component correct-ID sets, not row outcome labels.
    intent_ids = {name: {row.id for row in dev if values[row.id] == row.intent}
                  for name, values in intents.items()}
    slot_ids = {name: {key for key, row in reference.items() if set(values[key]) == set(row.gold_spans)}
                for name, values in slots.items()}
    intersections = {model_key(i, s): len(a & b) for i, a in intent_ids.items() for s, b in slot_ids.items()}
    if any(metrics["model_pairs"][key]["joint_exact_matches"] != count for key, count in intersections.items()):
        raise ValueError("Independent correct-ID intersection disagrees with joint outcomes")
    return ({"intent_models": intent_metrics, "slot_models": slot_metrics,
             "primary": primary, "comparison": alternate, **metrics,
             "independent_correct_id_intersection": intersections, "bio_repairs_by_model": repairs},
            entries, inputs)


def execute(config_path: Path, config: dict, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    started = datetime.now(timezone.utc)
    metrics, entries, inputs = compute(config)
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "metrics.json", metrics)
    with (output / "diagnostics.jsonl").open("x", encoding="utf-8") as stream:
        for row in entries:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    sources = [config_path, Path(__file__), Path("tests/test_joint_eval.py"),
               *(Path("src/azmassive")/(name+".py") for name in
                 ("dataset", "evaluation", "experiment", "intent_svc", "intent_hybrid",
                  "slot_crf", "slot_metrics", "slots"))]
    write_json(output / "run.json", {
        "experiment_id": config["experiment_id"], "partition": "dev",
        "started_eastern": started.astimezone(ZoneInfo("America/New_York")).isoformat(),
        "started_utc": started.isoformat(), "git_base_revision": command_output("git", "rev-parse", "HEAD"),
        "command": "PYTHONPATH=src " + shlex.join(["python3", "-B", "-m", "azmassive.joint_eval", *sys.argv[1:]]),
        "dataset": {"release": "MASSIVE 1.0", "locale": "az-AZ", "license": "CC BY 4.0",
                    "audited_source": "AUD-001"},
        "python": sys.version, "scikit_learn": importlib.metadata.version("scikit-learn"),
        "n": len(entries), "model_retrained": False, "test_predicted_or_scored": False,
        "input_sha256": inputs, "source_sha256": {str(p): sha256_file(p) for p in sources},
        "artifacts_sha256": {p.name: sha256_file(p) for p in sorted(output.iterdir())},
    })
    for key, summary in metrics["model_pairs"].items():
        print(f"{key}: {summary['joint_exact_matches']}/{summary['n']} ({summary['joint_exact_match']:.6%})")


def verify(config: dict, output: Path) -> dict:
    manifest = read_json(output / "run.json")
    if (read_json(output / "config.json") != config or manifest["partition"] != "dev"
            or manifest["model_retrained"] or manifest["test_predicted_or_scored"]):
        raise ValueError("Run protocol disagrees with the saved configuration")
    for name, expected in manifest["source_sha256"].items():
        verify_hash(Path(name), expected)
    for name, expected in manifest["artifacts_sha256"].items():
        verify_hash(output / name, expected)
    metrics, entries, inputs = compute(config)
    if (inputs != manifest["input_sha256"] or metrics != read_json(output / "metrics.json")
            or entries != list(prediction_rows(output / "diagnostics.jsonl")) or manifest["n"] != len(entries)):
        raise ValueError("Joint results or source provenance differ from recomputation")
    return {"result": "PASS", "dev_ids": len(entries), "manifest_sha256": sha256_file(output / "run.json"),
            "checks": ["source and artifact hashes", "verified component metrics and exact ID joins",
                       "recomputed joint outcomes and ID diagnostics", "independent correct-ID intersections"],
            "primary_correct_id_intersection": metrics["independent_correct_id_intersection"][metrics["primary"]],
            "model_retrained": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    output = args.output or Path(config["output_dir"])
    if args.verify:
        print(json.dumps(verify(config, output), indent=2))
    else:
        execute(args.config, config, output)


if __name__ == "__main__":
    main()
