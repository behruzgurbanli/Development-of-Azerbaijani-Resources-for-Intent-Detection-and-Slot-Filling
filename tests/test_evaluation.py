import unittest

from azmassive.dataset import Record
from azmassive.evaluation import error_analysis, evaluate_intents, majority_label, prediction_records
from azmassive.experiment import build_pipeline, per_class_sample, validate_protocol
import json
from pathlib import Path


class EvaluationTests(unittest.TestCase):
    def test_macro_f1_includes_zero_support_labels(self):
        result = evaluate_intents(["a", "a", "b"], ["a", "a", "a"], ["a", "b", "c"])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 0.8 / 3)
        self.assertEqual(result["per_class"]["c"]["support"], 0)

    def test_invalid_metric_inputs_fail(self):
        for true, pred, labels in [([], [], ["a"]), (["a"], [], ["a"]),
                                   (["a"], ["b"], ["a"]), (["a"], ["a"], ["a", "a"])]:
            with self.subTest(true=true, pred=pred), self.assertRaises(ValueError):
                evaluate_intents(true, pred, labels)

    def test_majority_uses_counts_and_stable_tie_break(self):
        self.assertEqual(majority_label(["b", "a", "b"]), "b")
        self.assertEqual(majority_label(["b", "a"]), "a")
        with self.assertRaises(ValueError):
            majority_label([])

    def test_prediction_alignment_and_no_test_exports(self):
        dev = [Record("7", "dev", "alarm", "alarm_set", "Əla")]
        self.assertEqual(prediction_records(dev, ["alarm_set"], "alarm_set")[0]["id"], "7")
        for records, predictions in [(dev, []), (dev * 2, ["alarm_set"] * 2),
                                      ([Record("1", "test", "alarm", "alarm_set", "x")], ["alarm_set"])]:
            with self.assertRaises(ValueError):
                prediction_records(records, predictions, "alarm_set")

    def test_overlap_subgroups_do_not_change_primary_set(self):
        train = [Record("1", "train", "x", "x_a", "same")]
        dev = [Record("2", "dev", "x", "x_a", "same"), Record("3", "dev", "x", "x_b", "new")]
        result = error_analysis(train, dev, ["x_a", "x_a"], ["x_a", "x_b"])
        self.assertEqual(result["error_count"], 1)
        self.assertEqual(result["overlap_subgroups"]["exact_text_seen_in_train"]["accuracy"], 1)
        self.assertEqual(result["overlap_subgroups"]["exact_text_absent_from_train"]["accuracy"], 0)
        self.assertEqual(len(dev), 2)

    def test_protocol_rejects_test_evaluation(self):
        config = json.loads(Path("configs/PR5-INT-001.json").read_text())
        config["dataset"]["evaluation_partition"] = "test"
        with self.assertRaises(ValueError):
            validate_protocol(config)

    def test_smoke_sampling_is_per_class_in_original_order(self):
        records = tuple(Record(str(i), "train", "x", label, "text")
                        for i, label in enumerate(["x_a", "x_a", "x_b", "x_b"]))
        self.assertEqual([r.id for r in per_class_sample(records, 1)], ["0", "2"])

    def test_vectorizer_preserves_case_and_does_not_fit_dev(self):
        config = json.loads(Path("configs/PR5-INT-001.json").read_text())
        vectorizer = build_pipeline(config).named_steps["tfidf"]
        vectorizer.fit(["İŞIQ ƏLA"])
        before = dict(vectorizer.vocabulary_)
        self.assertIn("ƏLA", before)
        self.assertNotIn("əla", before)
        vectorizer.transform(["NEVER_SEEN_DEV_TEXT"])
        self.assertEqual(vectorizer.vocabulary_, before)


if __name__ == "__main__":
    unittest.main()
