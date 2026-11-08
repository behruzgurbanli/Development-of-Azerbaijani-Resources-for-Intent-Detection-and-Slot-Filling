import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from io import BytesIO

from azmassive.dataset import load_dataset, prepare_release, sha256_file, validate_record


def row(identifier="1", partition="train", intent="alarm_set", text="İşığı Ə SÖNDÜR"):
    return {"id": identifier, "locale": "az-AZ", "partition": partition,
            "scenario": intent.split("_", 1)[0], "intent": intent, "utt": text,
            "annot_utt": text, "worker_id": "1", "slot_method": [], "judgments": []}


class DatasetTests(unittest.TestCase):
    def fixture(self, directory, rows):
        path = Path(directory) / "az-AZ.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        return path, {"file_sha256": sha256_file(path), "bytes": path.stat().st_size}

    def test_original_text_and_official_partitions_preserved(self):
        rows = [row(), row("2", "dev"), row("3", "test")]
        with tempfile.TemporaryDirectory() as temp:
            path, audit = self.fixture(temp, rows)
            original = path.read_bytes()
            data = load_dataset(path, audit)
            self.assertEqual(data.train[0].utt, "İşığı Ə SÖNDÜR")
            self.assertEqual([r.id for r in data.train], ["1"])
            self.assertEqual([r.id for r in data.dev], ["2"])
            self.assertFalse(hasattr(data, "test"))
            self.assertEqual(data.diagnostics["dev_records_with_text_in_train"], 1)
            self.assertEqual(data.diagnostics["train_dev_exact_duplicate_text_groups"], 1)
            self.assertEqual(path.read_bytes(), original)

    def test_checksum_mismatch_stops_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            path, audit = self.fixture(temp, [row()])
            audit["file_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_dataset(path, audit)

    def test_schema_rejects_invalid_records(self):
        for update in ({"locale": "en-US"}, {"partition": "validation"}, {"utt": " "},
                       {"id": 1}, {"intent": "weather_query"}, {"slot_method": {}},
                       {"judgments": ["invalid"]}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_record({**row(), **update}, 1)

    def test_duplicate_ids_fail_even_across_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            path, audit = self.fixture(temp, [row(), row(partition="dev")])
            with self.assertRaisesRegex(ValueError, "duplicate ID"):
                load_dataset(path, audit)

    def test_audit_counts_must_match(self):
        with tempfile.TemporaryDirectory() as temp:
            path, audit = self.fixture(temp, [row(), row("2", "dev")])
            audit["split_counts"] = {"train": 1, "dev": 2}
            with self.assertRaisesRegex(ValueError, "Audit mismatch"):
                load_dataset(path, audit)

    def test_unknown_dev_label_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path, audit = self.fixture(temp, [row(), row("2", "dev", "alarm_remove")])
            with self.assertRaisesRegex(ValueError, "label absent from train"):
                load_dataset(path, audit)

    def test_conflicting_duplicates_diagnosed_without_dropping(self):
        with tempfile.TemporaryDirectory() as temp:
            path, audit = self.fixture(temp, [row(), row("2", intent="alarm_remove"), row("3", "dev")])
            data = load_dataset(path, audit)
            self.assertEqual(len(data.train), 2)
            self.assertEqual(data.diagnostics["train_dev_conflicting_intent_text_groups"], 1)

    def test_member_mismatch_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "release.tar.gz"
            with tarfile.open(archive, "w:gz") as release:
                info = tarfile.TarInfo("1.0/data/az-AZ.jsonl")
                info.size = 3
                release.addfile(info, BytesIO(b"bad"))
            audit = {"archive_sha256": sha256_file(archive),
                     "archive_member": "1.0/data/az-AZ.jsonl", "file_sha256": "0" * 64,
                     "license_member_sha256": {}}
            output = Path(temp) / "extracted"
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                prepare_release(archive, audit, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
