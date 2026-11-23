import json
from pathlib import Path
import tempfile
import unittest

from azmassive.dataset import Record
from azmassive.experiment import build_pipeline
from azmassive.intent_svc import align_predictions, candidate_pipeline, load_train_dev, paired_comparison, read_json


def dev_record(identifier, intent, text="mənim sorğum"):
    return Record(identifier, "dev", "alarm", intent, text)


class IntentSVCTests(unittest.TestCase):
    def test_candidate_reuses_all_baseline_vectorizer_parameters(self):
        baseline = read_json(Path("configs/PR5-INT-001.json"))
        config = read_json(Path("configs/INT-SVC-001.json"))
        pipeline = candidate_pipeline(baseline, config["classifier"])
        self.assertEqual(pipeline.named_steps["tfidf"].get_params(),
                         build_pipeline(baseline).named_steps["tfidf"].get_params())
        self.assertEqual(pipeline.named_steps["linear_svc"].random_state, baseline["seed"])

    def test_held_out_records_skip_validation_and_never_enter_inputs(self):
        train = {"id": "1", "partition": "train", "locale": "az-AZ", "scenario": "alarm",
                 "intent": "alarm_set", "utt": "oyat", "annot_utt": "oyat", "worker_id": "1",
                 "slot_method": [], "judgments": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.jsonl"
            rows = [train, {"partition": "test"}, {**train, "id": "2", "partition": "dev"}]
            path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            partitions = load_train_dev(path)
        self.assertEqual({split: [r.id for r in rows] for split, rows in partitions.items()},
                         {"train": ["1"], "dev": ["2"]})

    def test_join_is_order_independent_and_rejects_bad_ids_and_gold(self):
        dev = [dev_record("1", "a"), dev_record("2", "b")]
        a = {"id": "1", "gold_intent": "a", "partition": "dev", "linear_svc": "b"}
        b = {**a, "id": "2", "gold_intent": "b", "linear_svc": "a"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text("\n".join(map(json.dumps, [b, a])), encoding="utf-8")
            self.assertEqual(align_predictions(dev, path, "linear_svc"), ["b", "a"])
            for rows in ([a], [a, a], [a, {**b, "gold_intent": "a"}], [a, {**b, "id": "3"}],
                         [a, {**b, "partition": "test"}]):
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
                    align_predictions(dev, path, "linear_svc")

    def test_paired_outcomes_confusion_deltas_and_unseen_ids(self):
        train = [Record("9", "train", "alarm", "a", "same")]
        dev = [dev_record("1", "a", "same"), dev_record("2", "b", "new"), dev_record("3", "c", "newer")]
        result = paired_comparison(train, dev, ["b", "b", "a"], ["a", "a", "b"], ("a", "b", "c"))
        self.assertEqual(result["paired_outcomes"],
                         {"svc_only_correct": 1, "logistic_only_correct": 1, "both_wrong": 1})
        self.assertEqual(result["paired_outcomes_by_exact_text_overlap"]["seen"], {"svc_only_correct": 1})
        self.assertEqual([row["id"] for row in result["changed_predictions"]], ["1", "2", "3"])
        self.assertIn({"gold": "a", "predicted": "b", "delta": -1},
                      result["confusion_deltas_svc_minus_logistic"])
