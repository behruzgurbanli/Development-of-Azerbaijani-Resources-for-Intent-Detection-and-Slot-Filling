import unittest

from azmassive.slot_features import sequence_features, word_shape
from azmassive.slot_metrics import SlotExample, align_predictions, decode_prediction, evaluate, score_spans
from azmassive.slots import Span, tokenize


class FeatureTests(unittest.TestCase):
    def test_azerbaijani_case_affixes_and_shape(self):
        tokens = tokenize("Şəhər 12!")
        features = sequence_features(tokens, [1, 2], 1)
        self.assertEqual(word_shape("Şə2!"), "Xxd!")
        self.assertIn("surface=Şəhər", features[0])
        self.assertIn("casefold=şəhər", features[0])
        self.assertIn("prefix2=şə", features[0])
        self.assertIn("suffix2=ər", features[0])
        self.assertIn("initial_upper=True", features[0])
        self.assertIn("all_digits=True", features[1])
        self.assertIn("punctuation=True", features[2])

    def test_context_edges_and_neighbors(self):
        left, right = sequence_features(tokenize("Əla!"), [1], 1)
        self.assertIn("-1:BOS", left)
        self.assertIn("1:surface=!", left)
        self.assertIn("-1:surface=Əla", right)
        self.assertIn("1:EOS", right)


class DecodeTests(unittest.TestCase):
    def test_preserves_whitespace_inside_span_and_reports_every_repair(self):
        tokens = tokenize("bir\tiki üç,")
        spans, changes = decode_prediction(tokens, ["I-time", "I-time", "I-date", "O"])
        self.assertEqual(spans, (Span(0, 7, "time"), Span(8, 10, "date")))
        self.assertEqual(changes, [
            {"token_index": 0, "raw_tag": "I-time", "replacement": "B-time", "active_type": None},
            {"token_index": 2, "raw_tag": "I-date", "replacement": "B-date", "active_type": "time"},
        ])

    def test_valid_adjacent_spans_and_illegal_tag_or_length(self):
        tokens = tokenize("Bakı Şəki")
        spans, changes = decode_prediction(tokens, ["B-place_name", "B-place_name"])
        self.assertEqual(len(spans), 2)
        self.assertEqual(changes, [])
        for tags in (["O"], ["X-date", "O"], ["B-", "O"]):
            with self.subTest(tags=tags), self.assertRaises(ValueError):
                decode_prediction(tokens, tags)


class MetricsTests(unittest.TestCase):
    def test_requires_exact_type_and_both_character_boundaries(self):
        a, b, c = Span(0, 3, "date"), Span(5, 7, "date"), Span(8, 9, "time")
        result = score_spans([
            ((a, b, c), (a, Span(5, 6, "date"), Span(8, 9, "date"))),
            ((), ()),
        ], ["date", "time", "place_name"])
        self.assertEqual(result["micro"], {"tp": 1, "fp": 2, "fn": 2, "support": 3,
                                          "precision": 1/3, "recall": 1/3, "f1": 1/3})
        self.assertEqual(result["per_slot"]["date"]["support"], 2)
        self.assertEqual(result["per_slot"]["time"]["recall"], 0.0)
        self.assertEqual(result["per_slot"]["place_name"]["f1"], 0.0)
        self.assertEqual(result["utterance_exact_match"], 0.5)

    def test_all_o_overlap_and_error_ids(self):
        dev = [
            SlotExample("a", "bir", tokenize("bir"), ("B-date",), (Span(0, 3, "date"),)),
            SlotExample("b", "iki", tokenize("iki"), ("O",), ()),
        ]
        metrics, errors = evaluate(dev, [{"id": "b", "tags": ["I-date"]},
                                         {"id": "a", "tags": ["B-date"]}], {"bir"}, ["date"])
        self.assertEqual(metrics["crf"]["micro"]["f1"], 2/3)
        self.assertEqual(metrics["all_o"]["utterance_exact_match"], 0.5)
        self.assertEqual(metrics["all_o"]["micro"]["fn"], 1)
        self.assertEqual(metrics["development_text_overlap"]["seen"]["micro"]["f1"], 1.0)
        self.assertEqual(metrics["invalid_iob2"], {"affected_utterances": 1, "transitions": 1})
        self.assertEqual([r["id"] for r in errors["errors_by_id"]], ["b"])

    def test_missing_duplicate_or_foreign_ids_raise(self):
        dev = [SlotExample("1", "", (), (), ()), SlotExample("2", "", (), (), ())]
        a, b = {"id": "1", "tags": ["O"]}, {"id": "2", "tags": ["B-date"]}
        self.assertEqual(align_predictions(dev, [b, a]), {"1": ["O"], "2": ["B-date"]})
        for predictions in ([a], [a, a], [a, {"id": "outside", "tags": []}]):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                align_predictions(dev, predictions)
        with self.assertRaisesRegex(ValueError, "Duplicate development ID"):
            align_predictions([dev[0], dev[0]], [a])
