import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.sparse import issparse

from azmassive.dataset import Record
from azmassive.intent_hybrid import comparison, hybrid_pipeline, reports
from azmassive.intent_svc import align_predictions, candidate_pipeline, read_json


class IntentHybridTests(unittest.TestCase):
    def setUp(self):
        self.pr5 = read_json(Path("configs/PR5-INT-001.json"))
        self.svc = read_json(Path("configs/INT-SVC-001.json"))
        self.config = read_json(Path("configs/INT-HYBRID-001.json"))
        self.pipeline = hybrid_pipeline(self.pr5, self.svc, self.config)

    def test_unicode_case_one_character_words_and_bigrams(self):
        word = dict(self.pipeline.named_steps["features"].transformer_list)["word"]
        self.assertEqual(word.build_analyzer()("İ, ə Şuşa! a Bakı"),
                         ["İ", "ə", "Şuşa", "a", "Bakı", "İ ə", "ə Şuşa", "Şuşa a", "a Bakı"])

    def test_character_and_classifier_parameters_match_and_union_is_unit_norm(self):
        baseline = candidate_pipeline(self.pr5, self.svc["classifier"])
        character = dict(self.pipeline.named_steps["features"].transformer_list)["character"]
        self.assertEqual(character.get_params(), baseline.named_steps["tfidf"].get_params())
        self.assertEqual(self.pipeline.named_steps["linear_svc"].get_params(),
                         baseline.named_steps["linear_svc"].get_params())
        features = self.pipeline[:-1]
        combined = features.fit_transform(["Bakı şəhər", "Şuşa gözəldir"])
        self.assertTrue(issparse(combined))
        cutoff = len(character.vocabulary_)
        self.assertGreater(cutoff, 0)
        before_union = character.transform(["Bakı şəhər", "Şuşa gözəldir"]).toarray()
        np.testing.assert_allclose(combined[:, :cutoff].toarray(), before_union / np.sqrt(2))
        for matrix in (combined[:, :cutoff], combined[:, cutoff:]):
            np.testing.assert_allclose(np.linalg.norm(matrix.toarray(), axis=1), 1 / np.sqrt(2))
        one_branch = features.transform(["Bakıx"]).toarray()[0]
        self.assertFalse(np.any(one_branch[cutoff:]))
        self.assertAlmostEqual(np.linalg.norm(one_branch), 1.0)

    def test_dev_transform_never_changes_vocabularies_or_idf(self):
        features = self.pipeline[:-1]
        features.fit(["Bakı şəhər", "Şuşa gözəldir"])
        branches = dict(self.pipeline.named_steps["features"].transformer_list)
        before = {name: (vectorizer.vocabulary_.copy(), vectorizer.idf_.copy())
                  for name, vectorizer in branches.items()}
        features.transform(["Qəbələ Qəbələ"])
        for name, vectorizer in branches.items():
            self.assertEqual(vectorizer.vocabulary_, before[name][0])
            np.testing.assert_array_equal(vectorizer.idf_, before[name][1])
        self.assertNotIn("Qəbələ", branches["word"].vocabulary_)
        self.assertNotIn("Qəb", branches["character"].vocabulary_)

    def test_metrics_retain_zero_support_and_count_paired_changes(self):
        train = [Record("9", "train", "alarm", "a", "eyni")]
        dev = [Record("1", "dev", "alarm", "a", "eyni"), Record("2", "dev", "alarm", "b", "fərqli")]
        result = reports(train, dev, ("a", "b", "c"), ["b", "b"], ["a", "a"])
        self.assertAlmostEqual(result["metrics.json"]["macro_f1"], 2 / 9)
        self.assertEqual(result["metrics.json"]["per_class"]["c"]["support"], 0)
        pair = result["comparison.json"]
        self.assertEqual(pair["paired_outcomes"], {"hybrid_only_correct": 1, "character_only_correct": 1})
        self.assertEqual(pair["changed_predictions"][1]["text_overlap"], "unseen")
        self.assertEqual(pair["changed_predictions"][1]["character_svc"], "b")
        self.assertEqual(pair["changed_predictions"][1]["hybrid_svc"], "a")
        self.assertEqual(pair["confusion_deltas_hybrid_minus_character"],
                         [{"gold": "a", "predicted": "b", "delta": -1},
                          {"gold": "b", "predicted": "a", "delta": 1}])

    def test_hybrid_predictions_align_by_id_and_reject_duplicate_ids(self):
        dev = [Record("1", "dev", "alarm", "a", "a"), Record("2", "dev", "alarm", "b", "b")]
        rows = [{"id": row.id, "partition": "dev", "gold_intent": row.intent, "hybrid_svc": row.intent}
                for row in reversed(dev)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            self.assertEqual(align_predictions(dev, path, "hybrid_svc"), ["a", "b"])
            path.write_text("\n".join(map(json.dumps, [rows[0], rows[0]])), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                align_predictions(dev, path, "hybrid_svc")
