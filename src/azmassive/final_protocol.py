"""Prepare blind inputs, predict, validate, and separately authorize held-out scoring."""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import json
import platform
import re
import subprocess
import unicodedata

from .dataset import sha256_file, validate_record, verify_hash
from .slot_metrics import SlotExample, decode_prediction
from .slots import AlignmentError, parse_annotation, spans_to_bio, tokenize

CONFIG = Path("configs/EVAL-FREEZE-001.json")
TRAIN_KEYS = ("id", "utt", "intent", "annot_utt")
BLIND_KEYS = ("id", "utt")
SLOT_MODEL = "SLOT-CRF-002"
SCORE_OUTPUTS = ("metrics.json", "diagnostics.jsonl", "bio-repairs.jsonl")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def relative(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not (root / path).resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path must remain repository-relative: {name}")
    return root / path


def load_protocol(root, config_path=CONFIG):
    path = relative(root, config_path)
    if not path.resolve().is_relative_to(root.resolve() / "configs") or path.suffix != ".json":
        raise ValueError("Evaluation config must be a JSON file under configs/")
    config = read_json(path)
    pins_path = relative(root, config["pins"])
    if not pins_path.resolve().is_relative_to(root.resolve() / "results" / "EVAL-FREEZE-001"):
        raise ValueError("Freeze pins must be under the experiment results directory")
    pins = read_json(pins_path)
    if pins["frozen_base_commit"] != config["frozen_base_commit"]:
        raise ValueError("Frozen base revision mismatch")
    for name, digest in pins["static_sha256"].items():
        source = relative(root, name)
        resolved = source.resolve().relative_to(root.resolve())
        if resolved.is_relative_to("data") and resolved != Path("data/metadata/AUD-001.json"):
            raise ValueError("Static source pins may include the audit metadata only")
        if resolved.is_relative_to("artifacts"):
            raise ValueError("Static source pins may not contain run artifacts")
        verify_hash(source, digest)
    if str(config_path) not in pins["static_sha256"] or config["data_files"]["audit"] != "data/metadata/AUD-001.json":
        raise ValueError("Protocol config or audit pin is missing or inconsistent")
    return config, pins, sha256_file(pins_path)


def environment(pins):
    expected = pins["environment"]
    actual = {"python": platform.python_version(), "unicode": unicodedata.unidata_version,
              "packages": {name: version(name) for name in expected["packages"]}}
    if actual != expected:
        raise ValueError(f"Runtime versions differ from the frozen environment: {actual}")
    return {**actual, "os": platform.system(), "os_release": platform.release(),
            "architecture": platform.machine()}


def checked_release(root, config, pins):
    for key, name in config["data_files"].items():
        verify_hash(relative(root, name), pins["data_sha256"][key])
    return relative(root, config["data_files"]["jsonl"])


def check_rows(rows, keys, count):
    ids = []
    for row in rows:
        if (not isinstance(row, dict) or set(row) != set(keys)
                or any(not isinstance(row[key], str) or not row[key].strip() for key in keys)
                or not row["id"].isascii() or not row["id"].isdigit()):
            raise ValueError(f"Malformed projection with expected fields {keys}")
        ids.append(row["id"])
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError(f"Expected {count} projection records with unique IDs")
    return ids


def project(raw_rows, config, *, include_gold=False):
    """Project only train and official evaluation text. Access evaluation gold only when scoring."""
    train, blind, gold = [], [], []
    for line, raw in enumerate(raw_rows, 1):
        partition = raw["partition"]
        if raw["locale"] != "az-AZ" or partition not in {"train", "dev", "test"}:
            raise ValueError(f"Unexpected locale or partition at line {line}")
        if partition == config["fit"]["partition"]:
            validate_record(raw, line)
            train.append({key: raw[key] for key in TRAIN_KEYS})
        elif partition == config["evaluation"]["partition"]:
            blind.append({key: raw[key] for key in BLIND_KEYS})
            if include_gold:
                gold.append((validate_record(raw, line), raw["annot_utt"]))
    return train, blind, gold


def reference_inputs(root, config):
    paths = config["model_configs"]
    models = {model: read_json(relative(root, path)) for model, path in paths.items()}
    train_ids = read_json(root / "results/PR5-INT-001/full/run.json")["model_application"]["train_ids"]
    intent_labels = read_json(root / "results/PR5-INT-001/full/metrics.json")["logistic_regression"]["label_order"]
    slot_labels = sorted(read_json(root / "results/SLOT-PREP-001/audit.json")["inventory_by_split"]["train"])
    if (len(set(intent_labels)) != config["intent_labels"] or len(set(slot_labels)) != config["slot_types"]):
        raise ValueError("Frozen label inventory size mismatch")
    runs = {name: read_json(root / "results" / name / "full/run.json")
            for name in (*intent_models(config), SLOT_MODEL)}
    return models, runs, train_ids, intent_labels, slot_labels


def intent_models(config):
    return (config["primary"]["intent"], config["comparison"]["intent"])


def check_projections(train, blind, config, train_ids, intent_labels):
    actual_train = check_rows(train, TRAIN_KEYS, config["fit"]["records"])
    eval_ids = check_rows(blind, BLIND_KEYS, config["evaluation"]["records"])
    if actual_train != train_ids or set(actual_train).intersection(eval_ids):
        raise ValueError("Training order/IDs differ from the freeze or overlap evaluation")
    if {row["intent"] for row in train} != set(intent_labels):
        raise ValueError("Training labels differ from the frozen inventory")


def text_sha(text):
    return sha256(text.encode("utf-8")).hexdigest()


def validate_predictions(blind, rows, model_names, intent_labels, slot_labels, tokenizer, *, trained_tags=None):
    """Check one prediction per blind ID, independently of any evaluation gold."""
    expected = {row["id"]: text_sha(row["utt"]) for row in blind}
    predictions = {}
    intent_labels, slot_labels = set(intent_labels), set(slot_labels)
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "intents", "slot_tags", "text_sha256"}:
            raise ValueError("Malformed prediction fields")
        identifier = row["id"]
        if not isinstance(identifier, str) or identifier not in expected or identifier in predictions:
            raise ValueError(f"Foreign or duplicate prediction ID: {identifier}")
        if row["text_sha256"] != expected[identifier]:
            raise ValueError(f"Prediction text hash differs from the blind input at {identifier}")
        intents, tags = row["intents"], row["slot_tags"]
        if (not isinstance(intents, dict) or intents.keys() != set(model_names)
                or any(not isinstance(value, str) or value not in intent_labels for value in intents.values())):
            raise ValueError(f"Invalid intent predictions at {identifier}")
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(f"Malformed BIO prediction at {identifier}")
        if trained_tags is not None and set(tags).difference(trained_tags):
            raise ValueError(f"Prediction uses BIO tags absent from the fitted tag inventory at {identifier}")
        predictions[identifier] = row
    if predictions.keys() != expected.keys():
        raise ValueError("Missing prediction IDs")
    decoded, repairs = {}, []
    for row in blind:
        identifier = row["id"]
        try:
            spans, changes = decode_prediction(tokenize(row["utt"], tokenizer), predictions[identifier]["slot_tags"])
        except ValueError as exc:
            raise ValueError(f"Malformed BIO prediction at {identifier}: {exc}") from exc
        if any(span.label not in slot_labels for span in spans):
            raise ValueError(f"Unknown slot type at {identifier}")
        decoded[identifier] = spans
        if changes:
            repairs.append({"id": identifier, "changes": changes})
    intents = {name: {identifier: row["intents"][name] for identifier, row in predictions.items()}
               for name in model_names}
    return intents, decoded, repairs


def verify_fit(fit, runs, intent_labels, slot_labels):
    if fit.keys() != runs.keys():
        raise ValueError("Prediction manifest must contain exactly the frozen model fits")
    for name, reference in runs.items():
        model = fit[name]
        if name == SLOT_MODEL:
            slot_inventory = {tag[2:] for tag in model["trained_tags"] if tag.startswith(("B-", "I-"))}
            iterations = model["iterations"]
            if (model["effective_parameters"] != reference["effective_crf_parameters"]
                    or slot_inventory != set(slot_labels) or type(iterations) is not int
                    or not 1 <= iterations <= reference["effective_crf_parameters"]["max_iterations"]
                    or not model["termination"]
                    or not re.fullmatch(r"[0-9a-f]{64}", model["temporary_model_sha256"])):
                raise ValueError("CRF fit provenance does not match the frozen specification")
        else:
            features = reference["feature_count"] if "feature_count" in reference else reference["feature_counts"]["total"]
            if (model["classifier"] != reference["classifier_parameters"] or model["features"] != features
                    or model["classes"] != sorted(intent_labels)
                    or type(model["iterations"]) is not int
                    or not 1 <= model["iterations"] <= model["classifier"]["max_iter"]
                    or "ConvergenceWarning" in model["warning_categories"]):
                raise ValueError(f"{name} fit provenance differs from the frozen reference")


def stage(root, out, config, pins, pin_hash, references):
    source = checked_release(root, config, pins)
    train, blind, _ = project(jsonl(source), config)
    check_projections(train, blind, config, *references[2:4])
    out.mkdir(parents=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "blind.jsonl", blind)
    write_json(out / "stage.json", {"protocol_id": config["protocol_id"], "pins_sha256": pin_hash,
                                   "action": "stage", "created_utc": datetime.now(timezone.utc).isoformat(),
                                   "dataset_sha256": pins["data_sha256"],
                                   "fit": config["fit"], "evaluation": config["evaluation"],
                                   "projection_sha256": {name: sha256_file(out / name)
                                                         for name in ("train.jsonl", "blind.jsonl")}})


def load_stage(out, config, pins, pin_hash, references):
    staged = read_json(out / "stage.json")
    expected = {"protocol_id": config["protocol_id"], "pins_sha256": pin_hash,
                "dataset_sha256": pins["data_sha256"], "fit": config["fit"], "evaluation": config["evaluation"]}
    if any(staged[key] != value for key, value in expected.items()):
        raise ValueError("Staging provenance does not match the protocol")
    for name in ("train.jsonl", "blind.jsonl"):
        verify_hash(out / name, staged["projection_sha256"][name])
    train, blind = (list(jsonl(out / name)) for name in ("train.jsonl", "blind.jsonl"))
    check_projections(train, blind, config, *references[2:4])
    return train, blind


def provenance(root, pins, pin_hash):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", pins["frozen_base_commit"], revision],
                   cwd=root, check=True)
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_base_commit": pins["frozen_base_commit"], "pins_sha256": pin_hash,
            "git_revision": revision,
            "environment": environment(pins)}


def predict(root, out, config, pins, pin_hash, references):
    from .final_models import fit_and_predict

    if any((out / name).exists() for name in ("predictions.jsonl", "prediction-manifest.json")):
        raise FileExistsError("Predictions already exist")
    metadata = provenance(root, pins, pin_hash)
    train, blind = load_stage(out, config, pins, pin_hash, references)
    models, runs, _, labels, slots = references
    predictions, fit = fit_and_predict(train, blind, models,
                                       expected_crf_settings=runs[SLOT_MODEL]["effective_crf_parameters"])
    hashes = {row["id"]: text_sha(row["utt"]) for row in blind}
    predictions = [{**row, "text_sha256": hashes[row["id"]]} for row in predictions]
    verify_fit(fit, runs, labels, slots)
    validate_predictions(blind, predictions, intent_models(config), labels, slots, models[SLOT_MODEL]["tokenizer"],
                         trained_tags=set(fit[SLOT_MODEL]["trained_tags"]))
    write_jsonl(out / "predictions.jsonl", predictions)
    write_json(out / "prediction-manifest.json", {**metadata, "action": "predict",
                                                "stage_sha256": sha256_file(out / "stage.json"),
                                                "predictions_sha256": sha256_file(out / "predictions.jsonl"),
                                                "records": len(predictions), "model_fits": fit})


def validate(out, config, pins, pin_hash, references):
    train, blind = load_stage(out, config, pins, pin_hash, references)
    path = out / "prediction-manifest.json"
    manifest = read_json(path)
    if (manifest["pins_sha256"] != pin_hash or manifest["frozen_base_commit"] != pins["frozen_base_commit"]
            or manifest["records"] != len(blind)
            or not re.fullmatch(r"[0-9a-f]{40}", manifest["git_revision"])
            or any(manifest["environment"][key] != value for key, value in pins["environment"].items())):
        raise ValueError("Prediction provenance does not match the staged protocol")
    verify_hash(out / "stage.json", manifest["stage_sha256"])
    verify_hash(out / "predictions.jsonl", manifest["predictions_sha256"])
    models, runs, _, labels, slots = references
    verify_fit(manifest["model_fits"], runs, labels, slots)
    intents, decoded, repairs = validate_predictions(blind, jsonl(out / "predictions.jsonl"),
                                                      intent_models(config), labels, slots,
                                                      models[SLOT_MODEL]["tokenizer"],
                                                      trained_tags=set(manifest["model_fits"][SLOT_MODEL]["trained_tags"]))
    checks = {"protocol_id": config["protocol_id"], "status": "passed", "gold_access": False,
              "records": len(blind), "prediction_manifest_sha256": sha256_file(path),
              "stage_sha256": manifest["stage_sha256"], "predictions_sha256": manifest["predictions_sha256"],
              "bio_repair_records": len(repairs), "bio_repairs": sum(len(row["changes"]) for row in repairs),
              "bio_repair_events": repairs}
    return checks, train, blind, intents, decoded, repairs


def gold_examples(gold, tokenizer):
    examples, unrepresentable = {}, []
    for row, annotation in gold:
        try:
            _, spans = parse_annotation(annotation, row.utt)
        except ValueError as exc:
            raise ValueError(f"Official gold annotation failure at ID {row.id}") from exc
        tokens = tokenize(row.utt, tokenizer)
        try:
            tags = spans_to_bio(tokens, spans)
        except AlignmentError as exc:
            # Character-span scoring requires the full official set even when BIO cannot encode it.
            tags = ()
            unrepresentable.append({"id": row.id, "spans": exc.failures})
        examples[row.id] = SlotExample(row.id, row.utt, tokens, tags, spans)
    return examples, unrepresentable


def validated_inputs(out, config, pins, pin_hash, references):
    expected = read_json(out / "validation.json")
    checks, train, blind, intents, decoded, repairs = validate(out, config, pins, pin_hash, references)
    if expected != checks:
        raise ValueError("Separate validation record is stale or inconsistent")
    return checks, train, blind, intents, decoded, repairs


def compute_score(root, config, pins, references, validated):
    from .final_metrics import score_records

    checks, train, blind, intents, decoded, repairs = validated
    source = checked_release(root, config, pins)
    expected_train, expected_blind, gold = project(jsonl(source), config, include_gold=True)
    if train != expected_train or blind != expected_blind:
        raise ValueError("Staged ID/text/fit projections differ from the audited official release")
    models, _, _, labels, slots = references
    gold_slots, unrepresentable = gold_examples(gold, models[SLOT_MODEL]["tokenizer"])
    metrics, diagnostics = score_records([row for row, _ in gold], gold_slots, {row["utt"] for row in train},
                                         intents, {SLOT_MODEL: decoded}, labels, slots, config,
                                         partition=config["evaluation"]["partition"])
    return ({**metrics, "definitions": config["metrics"],
             "bio_repair_records": checks["bio_repair_records"], "bio_repairs": checks["bio_repairs"],
             "unrepresentable_gold_alignments": unrepresentable}, diagnostics, repairs)


def score(root, out, config, pins, pin_hash, references):
    validated = validated_inputs(out, config, pins, pin_hash, references)
    if any((out / name).exists() for name in (*SCORE_OUTPUTS, "score-manifest.json")):
        raise FileExistsError("Scoring outputs already exist")
    metadata = provenance(root, pins, pin_hash)
    metrics, diagnostics, repairs = compute_score(root, config, pins, references, validated)
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "diagnostics.jsonl", diagnostics)
    write_jsonl(out / "bio-repairs.jsonl", repairs)
    write_json(out / "score-manifest.json", {**metadata, "action": "score", "dataset_sha256": pins["data_sha256"],
                                           "validation_sha256": sha256_file(out / "validation.json"),
                                           "evaluation": config["evaluation"],
                                           "output_sha256": {name: sha256_file(out / name) for name in SCORE_OUTPUTS}})
