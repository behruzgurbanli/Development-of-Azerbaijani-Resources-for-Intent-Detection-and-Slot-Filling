"""Integration checks for the freeze entry point and independent count verifier."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from azmassive.eval_freeze import main, parse_args
from azmassive.eval_freeze_stats import verify_point_metrics
from azmassive.final_protocol import load_protocol, write_json
from test_final_protocol import CONFIGURATION, ROOT, development_result


class FreezeEntryPointTests(TestCase):
    def test_prepare_opens_no_raw_files_or_run_artifacts(self):
        original = Path.open

        def checked_open(path, *args, **kwargs):
            relative = path.resolve().relative_to(ROOT)
            if relative.is_relative_to("data/raw") or relative.is_relative_to("artifacts"):
                raise AssertionError("prepare opened corpus or runtime artifacts")
            return original(path, *args, **kwargs)

        output = StringIO()
        with patch.object(Path, "open", checked_open), redirect_stdout(output):
            main(["--root", str(ROOT), "--config", "configs/EVAL-FREEZE-001.json", "prepare"])
        self.assertIn("'result': 'PASS'", output.getvalue())
        self.assertIn("'raw_corpus_opened': False", output.getvalue())

    def test_authorizations_are_distinct(self):
        cases = {"extract-test": "--authorize-test-input-extraction",
                 "predict": "--authorize-test-prediction",
                 "score": "--authorize-test-scoring",
                 "verify-score": "--authorize-test-scoring"}
        for action, flag in cases.items():
            with self.subTest(action=action):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    parse_args([action])
                self.assertTrue(parse_args([action, flag]).authorized)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(["score", "--authorize-test-prediction"])

    def test_static_manifest_cannot_point_into_raw_data(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "configs/EVAL-FREEZE-001.json"
            pins_path = root / CONFIGURATION["pins"]
            cfg.parent.mkdir()
            pins_path.parent.mkdir(parents=True)
            write_json(cfg, CONFIGURATION)
            write_json(pins_path, {"frozen_base_commit": CONFIGURATION["frozen_base_commit"],
                                   "static_sha256": {"data/raw/blocked.jsonl": "not_opened"}})
            with self.assertRaisesRegex(ValueError, "audit metadata only"):
                load_protocol(root)


class IndependentCountTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics, cls.rows = development_result(include_diagnostics=True)

    def test_independent_counts_match_published_dev_replay(self):
        verify_point_metrics(self.metrics, self.rows, CONFIGURATION)

    def test_independent_counts_reject_changed_scores(self):
        metrics = deepcopy(self.metrics)
        primary = metrics["primary"]
        metrics["model_pairs"][primary]["joint_exact_matches"] += 1
        with self.assertRaisesRegex(ValueError, "Joint metric cross-check failed"):
            verify_point_metrics(metrics, self.rows, CONFIGURATION)

    def test_independent_counts_reject_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "unique diagnostic IDs"):
            verify_point_metrics(self.metrics, [self.rows[0], *self.rows[:-1]], CONFIGURATION)
