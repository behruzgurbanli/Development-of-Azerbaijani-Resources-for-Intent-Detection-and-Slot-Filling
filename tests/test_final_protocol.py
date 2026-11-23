"""Synthetic held-out pipeline coverage and regression against published dev results."""

from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from azmassive.dataset import Record, sha256_file
from azmassive.eval_freeze import verify_scored
from azmassive.final_metrics import bootstrap, score_records
from azmassive.final_protocol import (CONFIG, SLOT_MODEL, check_projections, gold_examples,
    intent_models, jsonl, predict, project, read_json, reference_inputs, score, stage,
    text_sha, validate, validate_predictions, write_json, write_jsonl)
from azmassive.slot_metrics import SlotExample
from azmassive.slots import Span

ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION = read_json(ROOT / CONFIG)
PRIMARY, COMPARISON = intent_models(CONFIGURATION)
LABELS = ("demo_a", "demo_b", "demo_c")
SLOTS = ("place_name",)
TOKENIZER = "word_punctuation"


def record(identifier, partition, utt, annotation, intent="demo_a"):
    return {"id": identifier, "partition": partition, "locale": "az-AZ", "utt": utt,
            "annot_utt": annotation, "scenario": "demo", "intent": intent,
            "worker_id": "fixture", "judgments": [], "slot_method": []}


def predictions(identifier="3", tags=None, primary="demo_a", comparison="demo_b"):
    return {"id": identifier, "intents": {PRIMARY: primary, COMPARISON: comparison},
            "slot_tags": ["O"] if tags is None else tags}


def development_result(*, include_diagnostics=False):
    """Score only already published development diagnostics, without the raw release."""
    _, _, _, labels, slots = reference_inputs(ROOT, CONFIGURATION)
    gold_rows, gold_slots, seen, decoded = [], {}, set(), {}
    intents = {name: {} for name in (PRIMARY, COMPARISON)}
    for entry in jsonl(ROOT / "results/JOINT-EVAL-001/diagnostics.jsonl"):
        identifier = entry["id"]
        gold_rows.append(Record(identifier, "dev", entry["scenario"], entry["gold_intent"], identifier))
        actual = tuple(Span(**span) for span in entry["gold_spans"])
        gold_slots[identifier] = SlotExample(identifier, identifier, (), (), actual)
        if entry["text_overlap"] == "seen":
            seen.add(identifier)
        for name in intents:
            intents[name][identifier] = entry["intent_predictions"][name]
        errors = entry["slot_errors"][SLOT_MODEL]
        missed, spurious = ({Span(**span) for span in errors[key]} for key in ("missed", "spurious"))
        decoded[identifier] = tuple(sorted((set(actual) - missed) | spurious,
                                           key=lambda span: (span.start, span.end, span.label)))
    metrics, diagnostics = score_records(gold_rows, gold_slots, seen, intents, {SLOT_MODEL: decoded},
                                         labels, slots, CONFIGURATION, partition="dev")
    return (metrics, diagnostics) if include_diagnostics else metrics


class FinalProtocolTests(unittest.TestCase):
    def test_blind_projection_never_reads_gold_fields_and_keeps_train_order(self):
        train_rows = [record("2", "train", "Azərbaycan", "Azərbaycan"),
                      record("1", "train", "İşıq", "İşıq", "demo_b")]
        dev = {"id": "4", "partition": "dev", "locale": "az-AZ"}
        evaluation = {"id": "3", "partition": "test", "locale": "az-AZ", "utt": "Bakı!"}
        config = {"fit": {"partition": "train", "records": 2},
                  "evaluation": {"partition": "test", "records": 1}}
        train, blind, gold = project([train_rows[0], dev, evaluation, train_rows[1]], config)
        self.assertEqual(blind, [{"id": "3", "utt": "Bakı!"}])
        self.assertEqual(gold, [])
        self.assertEqual([r["id"] for r in train], ["2", "1"])
        check_projections(train, blind, config, ["2", "1"], LABELS[:2])
        with self.assertRaisesRegex(ValueError, "Training order/IDs"):
            check_projections(train, blind, config, ["1", "2"], LABELS[:2])
        with self.assertRaisesRegex(ValueError, "Training order/IDs"):
            check_projections(train, [{"id": "1", "utt": "Bakı!"}], config, ["2", "1"], LABELS[:2])

    def test_validation_rejects_ids_shapes_and_inventories(self):
        blind = [{"id": "3", "utt": "Bakı!"}]
        def validate_rows(rows):
            sealed = [{"text_sha256": text_sha("Bakı!"), **row} for row in rows]
            return validate_predictions(blind, sealed, (PRIMARY, COMPARISON), LABELS, SLOTS, TOKENIZER)
        cases = [
            ([], "Missing"),
            ([predictions("9")], "Foreign or duplicate"),
            ([predictions(), predictions()], "Foreign or duplicate"),
            ([predictions(primary="unknown")], "Invalid intent"),
            ([predictions(tags=["O"])], "tag count"),
            ([predictions(tags=["B-place_name", 1])], "Malformed BIO"),
            ([predictions(tags=["X-place_name", "O"])], "Unknown prediction tag"),
            ([predictions(tags=["B-wrong", "O"])], "Unknown slot type"),
            ([{**predictions(tags=["O", "O"]), "text_sha256": "wrong"}], "text hash"),
        ]
        for rows, pattern in cases:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                validate_rows(rows)
        intents, decoded, repairs = validate_rows([predictions(tags=["I-place_name", "O"])])
        self.assertEqual(intents[PRIMARY]["3"], "demo_a")
        self.assertEqual(decoded, {"3": (Span(0, 4, "place_name"),)})
        self.assertEqual(repairs, [{"id": "3", "changes": [{"token_index": 0, "raw_tag": "I-place_name",
                          "replacement": "B-place_name", "active_type": None}]}])

    def test_original_gold_spans_retained_when_bio_cannot_represent_them(self):
        row = Record("3", "test", "demo", "demo_a", "Bakıda")
        gold, failures = gold_examples([(row, "[place_name : Bakı]da")], TOKENIZER)
        self.assertEqual(gold["3"].gold_spans, (Span(0, 4, "place_name"),))
        self.assertEqual(failures[0]["id"], "3")
        self.assertEqual(failures[0]["spans"][0]["reasons"], ["end_not_token_boundary"])
        with self.assertRaisesRegex(ValueError, "annotation failure at ID 3"):
            gold_examples([(row, "[broken")], TOKENIZER)

    def test_paired_bootstrap_resamples_one_shared_draw_and_keeps_absent_labels(self):
        span = Span(0, 4, "place_name")
        args = (["demo_a", "demo_a"], {PRIMARY: ["demo_a", "demo_b"], COMPARISON: ["demo_b", "demo_a"]},
                LABELS, {SLOT_MODEL: [((), ()), ((span,), (span,))]},
                [(PRIMARY, SLOT_MODEL), (COMPARISON, SLOT_MODEL)], f"{PRIMARY}+{SLOT_MODEL}",
                f"{COMPARISON}+{SLOT_MODEL}", CONFIGURATION["bootstrap"])
        intervals = bootstrap(*args)
        self.assertEqual(intervals, bootstrap(*args))
        self.assertEqual(intervals["comparison_minus_primary.joint_exact_match"], [-1, 1])
        self.assertEqual(intervals[f"{PRIMARY}.intent_macro_f1"], [0, 1 / 3])
        self.assertEqual(intervals[f"{SLOT_MODEL}.slot_micro_f1"], [0, 1])
        self.assertEqual(intervals[f"{SLOT_MODEL}.slot_utterance_exact_match"], [1, 1])

    def test_synthetic_stage_predict_validate_score_and_stale_artifact(self):
        config = deepcopy(CONFIGURATION)
        config["fit"]["records"] = 2
        config["evaluation"]["records"] = 2
        config["bootstrap"]["resamples"] = 40
        labels = LABELS[:2]
        classifier = {"max_iter": 10}
        effective = {"max_iterations": 300}
        references = (
            {SLOT_MODEL: {"tokenizer": TOKENIZER}},
            {PRIMARY: {"classifier_parameters": classifier, "feature_count": 2},
             COMPARISON: {"classifier_parameters": classifier, "feature_counts": {"total": 2}},
             SLOT_MODEL: {"effective_crf_parameters": effective}},
            ["1", "2"], labels, SLOTS,
        )
        fits = {name: {"classifier": classifier, "classes": sorted(labels), "features": 2,
                       "iterations": 2, "warning_categories": []} for name in (PRIMARY, COMPARISON)}
        fits[SLOT_MODEL] = {"effective_parameters": effective, "trained_tags": ["O", "I-place_name"],
                           "iterations": 300, "termination": ["iteration cap"], "temporary_model_sha256": "0" * 64}
        rows = [record("1", "train", "Salam", "Salam"), record("2", "train", "Bakı", "[place_name : Bakı]", "demo_b"),
                record("3", "test", "Salam", "Salam"), record("4", "test", "Sumqayıt", "[place_name : Sumqayıt]"),
                record("5", "dev", "Nur", "Nur")]
        predicted = [predictions("4", ["I-place_name"], "demo_b", "demo_a"), predictions("3")]
        pins = {"frozen_base_commit": "1" * 40, "environment": {}}
        metadata = {"frozen_base_commit": pins["frozen_base_commit"], "pins_sha256": "fixture",
                    "git_revision": "2" * 40, "environment": {}}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, out = root / "release.jsonl", root / "output"
            write_jsonl(source, rows)
            config["data_files"] = {"jsonl": "release.jsonl"}
            pins["data_sha256"] = {"jsonl": sha256_file(source)}
            stage(root, out, config, pins, "fixture", references)
            self.assertEqual(list(jsonl(out / "blind.jsonl")),
                             [{"id": "3", "utt": "Salam"}, {"id": "4", "utt": "Sumqayıt"}])
            with patch("azmassive.final_protocol.provenance", return_value=metadata), patch(
                    "azmassive.final_models.fit_and_predict", return_value=(predicted, fits)) as fit:
                with patch("azmassive.final_protocol.checked_release") as checked:
                    predict(root, out, config, pins, "fixture", references)
                    checks, *_ = validate(out, config, pins, "fixture", references)
                    checked.assert_not_called()
                    with self.assertRaises(FileNotFoundError):
                        score(root, out, config, pins, "fixture", references)
                    write_json(out / "validation.json", {**checks, "records": 0})
                    with self.assertRaisesRegex(ValueError, "stale"):
                        score(root, out, config, pins, "fixture", references)
                    checked.assert_not_called()
                self.assertEqual(fit.call_args.args[1],
                                 [{"id": "3", "utt": "Salam"}, {"id": "4", "utt": "Sumqayıt"}])
                (out / "validation.json").unlink()
                write_json(out / "validation.json", checks)
                score(root, out, config, pins, "fixture", references)
            with patch("azmassive.eval_freeze.environment", return_value={}):
                self.assertEqual(verify_scored(root, out, config, pins, "fixture", references)["records"], 2)
                diagnostics_path = out / "diagnostics.jsonl"
                saved_diagnostics = diagnostics_path.read_bytes()
                diagnostics_path.write_bytes(b"{}\n")
                with patch("azmassive.eval_freeze.compute_score") as compute:
                    with self.assertRaisesRegex(ValueError, "SHA-256"):
                        verify_scored(root, out, config, pins, "fixture", references)
                    compute.assert_not_called()
                diagnostics_path.write_bytes(saved_diagnostics)
            metrics = read_json(out / "metrics.json")
            self.assertEqual(metrics["bio_repairs"], 1)
            self.assertEqual(metrics["model_pairs"][f"{PRIMARY}+{SLOT_MODEL}"]["joint_exact_match"], 0.5)
            self.assertEqual(metrics["model_pairs"][f"{COMPARISON}+{SLOT_MODEL}"]["joint_exact_match"], 0.5)
            self.assertNotIn("Sumqayıt", (out / "diagnostics.jsonl").read_text())
            (out / "predictions.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate(out, config, pins, "fixture", references)

    @unittest.skipUnless(find_spec("pycrfsuite"), "python-crfsuite unavailable")
    def test_real_estimators_on_synthetic_azerbaijani_only(self):
        from azmassive.final_models import fit_and_predict

        models, *_ = reference_inputs(ROOT, CONFIGURATION)
        training = [
            {"id": str(i), "utt": utt, "intent": intent, "annot_utt": annotation}
            for i, (utt, intent, annotation) in enumerate([
                ("Bakı gəlsin", "demo_a", "[place_name : Bakı] gəlsin"),
                ("Bakı gəldi", "demo_a", "[place_name : Bakı] gəldi"),
                ("İşıq sönsün", "demo_b", "İşıq sönsün"),
                ("İşıq söndü", "demo_b", "İşıq söndü"),
            ], 1)
        ]
        blind = [{"id": "7", "utt": "Bakı gəlsin"}, {"id": "8", "utt": "İşıq sönsün"}]
        predicted, fitted = fit_and_predict(training, blind, models)
        sealed = [{**row, "text_sha256": text_sha(blind_row["utt"])}
                  for row, blind_row in zip(predicted, blind, strict=True)]
        validate_predictions(blind, sealed, (PRIMARY, COMPARISON), LABELS[:2], SLOTS, TOKENIZER)
        self.assertEqual(set(fitted), {PRIMARY, COMPARISON, SLOT_MODEL})
        self.assertGreater(fitted[SLOT_MODEL]["iterations"], 0)
        self.assertEqual(len(fitted[SLOT_MODEL]["temporary_model_sha256"]), 64)
        with self.assertRaisesRegex(ValueError, "only id and utt"):
            fit_and_predict(training, [{**blind[0], "intent": "demo_a"}], models)

    def test_published_development_scores_recompute_from_diagnostics(self):
        previous = read_json(ROOT / "results/JOINT-EVAL-001/metrics.json")
        calculated = development_result()
        self.assertEqual(calculated["n"], 2033)
        for name in (PRIMARY, COMPARISON):
            key = f"{name}+{SLOT_MODEL}"
            self.assertEqual(calculated["model_pairs"][key], previous["model_pairs"][key])
            for metric in ("accuracy", "macro_f1"):
                self.assertAlmostEqual(calculated["intent_models"][name][metric], previous["intent_models"][name][metric])
        self.assertEqual(calculated["slot_models"][SLOT_MODEL]["micro"], previous["slot_models"][SLOT_MODEL]["micro"])
        self.assertEqual(calculated["paired_comparison"], previous["paired_comparison"])


if __name__ == "__main__":
    unittest.main()
