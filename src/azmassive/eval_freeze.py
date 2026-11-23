"""Prepare the frozen evaluation and gate each later held-out operation."""

import argparse
from pathlib import Path

from .dataset import sha256_file, verify_hash
from .final_protocol import (CONFIG, SCORE_OUTPUTS, compute_score, environment, jsonl,
                             load_protocol, read_json, reference_inputs, relative, score,
                             stage, predict, validate, validated_inputs, write_json)


def prepare(root, config, pins, pin_hash):
    """Verify static inputs and audit metadata without opening the raw release."""
    verify_hash(relative(root, config["data_files"]["audit"]), pins["data_sha256"]["audit"])
    return {"protocol_id": config["protocol_id"], "result": "PASS",
            "frozen_base_commit": pins["frozen_base_commit"], "pins_sha256": pin_hash,
            "static_files_verified": len(pins["static_sha256"]), "raw_corpus_opened": False}


def verify_scored(root, out, config, pins, pin_hash, references):
    """Recheck sealed artifacts before recomputing scores from authorized gold access."""
    from .eval_freeze_stats import verify_point_metrics

    preflight = validated_inputs(out, config, pins, pin_hash, references)
    manifest = read_json(out / "score-manifest.json")
    expected = {"action": "score", "pins_sha256": pin_hash, "frozen_base_commit": pins["frozen_base_commit"],
                "dataset_sha256": pins["data_sha256"], "evaluation": config["evaluation"],
                "validation_sha256": sha256_file(out / "validation.json")}
    if any(manifest[key] != value for key, value in expected.items()) or (
            manifest["output_sha256"].keys() != set(SCORE_OUTPUTS)):
        raise ValueError("Saved scoring manifest disagrees with the frozen inputs")
    current_environment = environment(pins)
    if any(manifest["environment"][key] != current_environment[key] for key in pins["environment"]):
        raise ValueError("Saved scoring environment disagrees with the frozen inputs")
    for name in SCORE_OUTPUTS:
        verify_hash(out / name, manifest["output_sha256"][name])

    saved = read_json(out / "metrics.json"), *(list(jsonl(out / name)) for name in SCORE_OUTPUTS[1:])
    recomputed = compute_score(root, config, pins, references, preflight)
    if recomputed != saved:
        raise ValueError("Saved metrics or diagnostics disagree with recomputation")
    verify_point_metrics(saved[0], saved[1], config)
    return {"result": "PASS", "records": saved[0]["n"], "metrics_sha256": manifest["output_sha256"]["metrics.json"]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--root", type=Path, default=Path("."))
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("prepare", help="Verify frozen inputs without opening the corpus")
    commands.add_parser("verify-predictions", help="Write the required gold-free validation record")
    for action, flag in (("extract-test", "--authorize-test-input-extraction"),
                         ("predict", "--authorize-test-prediction"),
                         ("score", "--authorize-test-scoring"),
                         ("verify-score", "--authorize-test-scoring")):
        commands.add_parser(action).add_argument(flag, dest="authorized", action="store_true")
    args = parser.parse_args(argv)
    if getattr(args, "authorized", True) is False:
        parser.error(f"{args.action} requires its explicit authorization flag")
    return args


def main(argv=None):
    args = parse_args(argv)
    root = args.root.resolve()
    config, pins, pin_hash = load_protocol(root, args.config)
    if args.action == "prepare":
        print(prepare(root, config, pins, pin_hash))
        return

    out = relative(root, config["work_dir"])
    if not out.resolve().is_relative_to(root / "artifacts"):
        raise ValueError("Held-out artifacts must stay under the ignored artifacts directory")
    references = reference_inputs(root, config)
    if args.action == "verify-predictions":
        checks, *_ = validate(out, config, pins, pin_hash, references)
        record = out / "validation.json"
        if record.exists():
            if read_json(record) != checks:
                raise ValueError("Existing validation record is stale or inconsistent")
        else:
            write_json(record, checks)
        print(checks)
        return
    action = {"extract-test": stage, "predict": predict, "score": score, "verify-score": verify_scored}
    result = action[args.action](root, out, config, pins, pin_hash, references)
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
