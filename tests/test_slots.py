import json
from pathlib import Path
import tempfile
import unittest

from azmassive.slot_prep import audit_annotations
from azmassive.slots import AlignmentError, Span, bio_to_spans, parse_annotation, spans_to_bio, tokenize


class AnnotationTests(unittest.TestCase):
    def test_unicode_case_whitespace_and_punctuation_are_exact(self):
        text = "  İşığı,\tƏLA!\u00a0e\u0301 "
        annotated = "  [device_type : İşığı],\tƏLA!\u00a0e\u0301 "
        self.assertEqual(parse_annotation(annotated, text), (text, (Span(2, 7, "device_type"),)))

    def test_repeated_values_use_their_markup_positions(self):
        annotated = "Bakı [place_name : Bakı] [place_name : Bakı]"
        _, spans = parse_annotation(annotated, "Bakı Bakı Bakı")
        self.assertEqual(spans, (Span(5, 9, "place_name"), Span(10, 14, "place_name")))

    def test_multiword_and_adjacent_slots(self):
        text = "bu gün sabah"
        _, spans = parse_annotation("[date : bu gün] [date : sabah]", text)
        self.assertEqual(spans_to_bio(tokenize(text), spans), ("B-date", "I-date", "B-date"))
        self.assertEqual(parse_annotation("[date : bu gün][time : !]", "bu gün!")[1],
                         (Span(0, 6, "date"), Span(6, 7, "time")))

    def test_slot_free_utterance(self):
        text = "Salam!"
        _, spans = parse_annotation(text, text)
        self.assertEqual(spans, ())
        self.assertEqual(spans_to_bio(tokenize(text), spans), ("O", "O"))
        self.assertEqual(bio_to_spans(tokenize(text), ("O", "O")), ())

    def test_malformed_annotation_and_text_mismatch(self):
        for markup in ("[time: Bakı]", "[time : Bakı", "Bakı]", "[time : ]",
                       "[time :  Bakı]", "[time : [date : Bakı]]", "[TIME : Bakı]"):
            with self.subTest(markup=markup), self.assertRaisesRegex(ValueError, "bracket|Malformed"):
                parse_annotation(markup, "Bakı")
        with self.assertRaisesRegex(ValueError, "differs"):
            parse_annotation("[place_name : Bakı]", "bakı")


class AlignmentTests(unittest.TestCase):
    def test_punctuation_boundary_depends_on_tokenizer(self):
        text = "Bakı, gəl!"
        _, spans = parse_annotation("[place_name : Bakı], gəl!", text)
        tokens = tokenize(text)
        self.assertEqual([t.text for t in tokens], ["Bakı", ",", "gəl", "!"])
        tags = spans_to_bio(tokens, spans)
        self.assertEqual(tags, ("B-place_name", "O", "O", "O"))
        self.assertEqual(bio_to_spans(tokens, tags), spans)
        with self.assertRaises(AlignmentError) as caught:
            spans_to_bio(tokenize(text, "whitespace"), spans)
        self.assertEqual(caught.exception.failures[0]["crossed_tokens"], [[0, 5]])

    def test_attached_suffix_is_reported_without_expanding_span(self):
        text = "Bakıda"
        _, spans = parse_annotation("[place_name : Bakı]da", text)
        for method in ("whitespace", "word_punctuation", "keep_joiners"):
            with self.subTest(method=method), self.assertRaises(AlignmentError) as caught:
                spans_to_bio(tokenize(text, method), spans)
            self.assertEqual(caught.exception.failures[0]["reasons"], ["end_not_token_boundary"])
            self.assertEqual(caught.exception.failures[0]["end"], 4)

    def test_roundtrip_punctuation_inside_span_and_unmarked_whitespace(self):
        text = "g.\t m.  t. vaxtı"
        _, spans = parse_annotation("[time_zone : g.\t m.  t.] vaxtı", text)
        tokens = tokenize(text)
        self.assertEqual(bio_to_spans(tokens, spans_to_bio(tokens, spans)), spans)
        self.assertEqual(text[spans[0].start:spans[0].end], "g.\t m.  t.")

    def test_invalid_transitions_and_length(self):
        tokens = tokenize("bir iki")
        for tags in (("I-date", "O"), ("O", "I-date"), ("B-date", "I-time"),
                     ("B-date", "X-date"), ("O",)):
            with self.subTest(tags=tags), self.assertRaises(ValueError):
                bio_to_spans(tokens, tags)

    def test_overlapping_or_reversed_spans_fail(self):
        tokens = tokenize("bir iki")
        for spans in ((Span(0, 7, "date"), Span(4, 7, "time")), (Span(4, 3, "date"),)):
            with self.subTest(spans=spans), self.assertRaises(AlignmentError):
                spans_to_bio(tokens, spans)


class AuditTests(unittest.TestCase):
    def test_failed_record_ids_and_unexamined_test_partition(self):
        base = {"locale": "az-AZ", "scenario": "alarm", "intent": "alarm_set",
                "worker_id": "1", "slot_method": [], "judgments": []}
        rows = [
            {**base, "id": "1", "partition": "train", "utt": "Bakıda", "annot_utt": "[date : Bakı]da"},
            {**base, "id": "2", "partition": "dev", "utt": "Bakı", "annot_utt": "[date : Bakı"},
            {"partition": "test"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "az-AZ.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            report = audit_annotations(path, {"train": 1, "dev": 1}, {"date"}, ["word_punctuation"])
        self.assertEqual(report["annotation_failures_by_split"]["dev"][0]["id"], "2")
        result = report["tokenizer_comparison"]["train"]["word_punctuation"]
        self.assertEqual(result["failures"][0]["id"], "1")
        self.assertEqual(result["aligned_records"], 0)
