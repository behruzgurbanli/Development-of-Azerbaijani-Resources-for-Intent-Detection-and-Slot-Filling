import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from azmassive.dataset import Record
from azmassive.intent_svc import align_predictions
from azmassive.joint_eval import align_gold, decode_slots, evaluate_joint, joint_outcome, model_key
from azmassive.slot_metrics import SlotExample
from azmassive.slots import Span, tokenize


def row(identifier, intent="weather_query", text="Bakı"):
    return Record(identifier, "dev", "weather", intent, text)


def slot_row(identifier, text="Bakı", gold=()):
    tokens = tokenize(text)
    return SlotExample(identifier, text, tokens, ("O",) * len(tokens), gold)


class JointTests(unittest.TestCase):
    def test_joint_is_strict_about_type_offsets_and_whole_set(self):
        gold = (Span(0, 4, "place_name"),)
        for predicted in ((Span(0, 3, "place_name"),), (Span(0, 4, "date"),),
                          gold + (Span(5, 8, "date"),), ()):
            with self.subTest(predicted=predicted):
                self.assertEqual(joint_outcome("a", "a", gold, predicted), "intent_only_correct")
                self.assertEqual(joint_outcome("a", "b", gold, predicted), "neither_correct")
        self.assertEqual(joint_outcome("a", "a", gold, gold), "both_correct")
        self.assertEqual(joint_outcome("a", "b", gold, gold), "slots_only_correct")

    def test_empty_slot_set_is_exact_only_if_prediction_is_empty(self):
        self.assertEqual(joint_outcome("a", "a", (), ()), "both_correct")
        self.assertEqual(joint_outcome("a", "b", (), ()), "slots_only_correct")
        self.assertEqual(joint_outcome("a", "a", (), (Span(0, 4, "date"),)), "intent_only_correct")

    def test_gold_alignment_checks_text_id_and_uniqueness(self):
        intent = [row("1"), row("2", text="Şəki")]
        slots = [slot_row("2", text="Şəki"), slot_row("1")]
        self.assertEqual(set(align_gold(intent, slots)), {"1", "2"})
        for invalid in (slots[:1], [slots[0], slots[0]], [slots[0], slot_row("3")],
                        [slots[0], slot_row("1", text="Ağdam")]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                align_gold(intent, invalid)
        with self.assertRaises(ValueError):
            align_gold([intent[0], intent[0]], slots)

    def test_saved_intent_rows_check_gold_and_align_by_id(self):
        intent = [row("1"), row("2")]
        records = [{"id": identifier, "partition": "dev", "gold_intent": "weather_query",
                    "linear_svc": prediction} for identifier, prediction in (("2", "alarm_set"), ("1", "weather_query"))]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
            self.assertEqual(align_predictions(intent, path, "linear_svc"), ["weather_query", "alarm_set"])
            records[0]["gold_intent"] = "alarm_set"
            path.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "gold label"):
                align_predictions(intent, path, "linear_svc")

    def test_slot_alignment_rejects_missing_foreign_duplicate_or_malformed_rows(self):
        dev = [slot_row("1"), slot_row("2")]
        a, b = {"id": "1", "tags": ["O"]}, {"id": "2", "tags": ["B-date"]}
        decoded, repairs = decode_slots(dev, [b, a], ["date"])
        self.assertEqual(decoded, {"1": (), "2": (Span(0, 4, "date"),)})
        self.assertEqual(repairs, [])
        for predictions in ([a], [a, a], [a, {**b, "id": "3"}], [a, {**b, "tags": "O"}],
                            [a, {**b, "tags": []}], [a, {**b, "tags": ["X-date"]}],
                            [a, {**b, "tags": ["B-place_name"]}], [a, {"id": "2"}]):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                decode_slots(dev, predictions, ["date"])

    def test_slot_decoding_reuses_logged_iob2_repair(self):
        decoded, repairs = decode_slots([slot_row("1")], [{"id": "1", "tags": ["I-date"]}], ["date"])
        self.assertEqual(decoded["1"], (Span(0, 4, "date"),))
        self.assertEqual(repairs[0]["id"], "1")
        self.assertEqual(repairs[0]["changes"][0]["raw_tag"], "I-date")

    def test_differing_intents_with_same_slots_change_joint_status(self):
        dev = [row("1"), row("2", text="Şəki")]
        gold = align_gold(dev, [slot_row("1"), slot_row("2", text="Şəki")])
        intents = {"char": {"1": "weather_query", "2": "alarm_set"},
                   "hybrid": {"1": "alarm_set", "2": "weather_query"}}
        slots = {"crf": {"1": (), "2": ()}}
        first, second = model_key("char", "crf"), model_key("hybrid", "crf")
        metrics, entries = evaluate_joint(dev, gold, {"Bakı"}, intents, slots, first, second)
        self.assertEqual(metrics["model_pairs"][first]["joint_exact_matches"], 1)
        self.assertEqual(metrics["model_pairs"][second]["joint_exact_matches"], 1)
        self.assertEqual(metrics["model_pairs"][first]["by_exact_text_overlap"]["seen"]["joint_exact_matches"], 1)
        self.assertEqual(metrics["paired_comparison"]["counts"]["primary_only_joint_correct"], 1)
        self.assertEqual(metrics["paired_comparison"]["counts"]["comparison_only_joint_correct"], 1)
        self.assertEqual({row["id"] for row in entries}, {"1", "2"})
