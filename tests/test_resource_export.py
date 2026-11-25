import json
import unittest

from azmassive.resource_export import export_row, json_bytes, source_rows, validate_rows


def fixture(identifier: str, partition: str, utt: str, markup: str) -> bytes:
    return json.dumps({
        "id": identifier, "locale": "az-AZ", "partition": partition, "scenario": "music",
        "intent": "music_play", "utt": utt, "annot_utt": markup, "worker_id": "synthetic",
        "slot_method": [], "judgments": [],
    }, ensure_ascii=False).encode("utf-8")


class ResourceExportTests(unittest.TestCase):
    def test_preserves_unicode_offsets_order_and_empty_spans(self):
        first = fixture("02", "dev", "  🎵 Şuşa, Şuşa!", "  🎵 Şuşa, [place_name : Şuşa]!")
        second = fixture("1", "train", "salam", "salam")
        rows = [export_row(row, {"place_name"}) for row in source_rows([first, second])]
        self.assertEqual([row["id"] for row in rows], ["02", "1"])
        self.assertEqual(rows[0]["source_annotation"]["spans"],
                         [{"start": 10, "end": 14, "label": "place_name"}])
        self.assertEqual(rows[1]["source_annotation"]["spans"], [])
        self.assertEqual(rows[0]["utt"], "  🎵 Şuşa, Şuşa!")
        self.assertEqual(rows[0]["source_annotation"]["intent"], "music_play")
        self.assertNotIn("worker_id", rows[0])
        payload = b"".join(map(json_bytes, rows))
        counts = validate_rows(payload, rows)
        self.assertEqual(counts["dev"], dict(records=1, spans=1, slot_free=0, round_trips=1))
        self.assertEqual(counts["train"], dict(records=1, spans=0, slot_free=1, round_trips=1))

    def test_held_out_payload_is_never_decoded(self):
        excluded = b'{"id":"0","locale":"az-AZ","partition":"test",\xff not JSON'
        eligible = fixture("1", "train", 'say "partition": "test"', 'say "partition": "test"')
        self.assertEqual([row["id"] for row in source_rows([excluded, eligible])], ["1"])
        with self.assertRaisesRegex(ValueError, "unexpected release header"):
            list(source_rows([b'{"partition":"other"}']))

    def test_rejects_duplicates_and_disagreeing_partition(self):
        row = fixture("1", "train", "salam", "salam")
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            list(source_rows([row, fixture("1", "dev", "salam", "salam")]))
        with self.assertRaisesRegex(ValueError, "inconsistent partition"):
            list(source_rows([row[:-1] + b', "partition": "dev"}']))

    def test_rejects_invalid_source_and_mutated_serialization(self):
        with self.assertRaises(ValueError):
            export_row(next(source_rows([fixture("1", "train", "Bakı", "Şuşa")])), set())
        raw = next(source_rows([fixture("1", "train", "Bakı", "[place_name : Bakı]")]))
        with self.assertRaisesRegex(ValueError, "Unknown source slot label"):
            export_row(raw, set())
        row = export_row(raw, {"place_name"})
        with self.assertRaisesRegex(ValueError, "canonical source"):
            validate_rows(json_bytes({**row, "id": "9"}), [row])
        row["source_annotation"]["spans"][0]["end"] = 3
        with self.assertRaisesRegex(ValueError, "round trip"):
            validate_rows(json_bytes(row), [row])
