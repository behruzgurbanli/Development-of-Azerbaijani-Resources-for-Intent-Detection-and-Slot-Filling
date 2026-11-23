import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from azmassive.annotation_review import REVIEW, review_rows, stratified, targeted, write_sheet


def raw(key, intent="a_x"):
    return dict(id=key, locale="az-AZ", partition="train", scenario="a", intent=intent,
                utt="Əli, gəl", annot_utt="[person : Əli], gəl", worker_id="fixture",
                judgments=[], slot_method=[])


class AnnotationReviewTests(unittest.TestCase):
    def test_stratified_coverage_is_order_independent(self):
        lines = [json.dumps(raw(str(i), "a_x" if i < 5 else "a_y")) for i in range(6)]
        a = stratified(review_rows(lines), "train", 4)
        self.assertEqual(a, stratified(review_rows(reversed(lines)), "train", 4))
        self.assertIn("5", a)
        self.assertEqual(4, len(set(a)))
        with self.assertRaises(ValueError):
            stratified(review_rows(lines), "dev", 1)

    def test_partition_filter_duplicate_ids_and_original_annotation(self):
        line = json.dumps(raw("7"))
        rows = review_rows(['{"partition":"test"}', line])
        self.assertEqual({"7"}, set(rows))
        self.assertEqual([dict(start=0, end=3, label="person")], rows["7"]["spans"])
        with self.assertRaisesRegex(ValueError, "Duplicate ID"):
            review_rows([line, line])

    def test_targeted_groups_stay_intact_and_separate_from_stratified(self):
        rows = review_rows(json.dumps(raw(str(i))) for i in (1, 2, 3, 4, 49, 1146))
        groups = [{"text_sha256": "a", "variants": [{"id": "1"}, {"id": "2"}]},
                  {"text_sha256": "b", "variants": [{"id": "3"}, {"id": "4"}]}]
        audit = {"duplicate_text_review": {"groups": groups},
                 "inventory_by_split": {"train": {"person": 6}}}
        selection = targeted(rows, {"1", "49"}, audit)
        self.assertTrue({"1", "49"}.isdisjoint(selection))
        self.assertIn("disagreement group 1", selection["3"])
        self.assertIn("disagreement group 1", selection["4"])
        self.assertIn("inflected boundary", selection["1146"])
        self.assertEqual(1, sum("rare person" in notes for notes in selection.values()))

    def test_sheet_roundtrip_has_blank_judgments_and_refuses_overwrite(self):
        rows = review_rows([json.dumps(raw("7"))])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            write_sheet(path, rows, {"7": "fixture"})
            with path.open(encoding="utf-8-sig", newline="") as stream:
                row, = csv.DictReader(stream)
            self.assertEqual(rows["7"]["annot_utt"], row["annot_utt"])
            self.assertEqual(rows["7"]["spans"], json.loads(row["spans"]))
            self.assertTrue(all(row[k] == "" for k in REVIEW))
            with self.assertRaises(FileExistsError):
                write_sheet(path, rows, {"7": "fixture"})
