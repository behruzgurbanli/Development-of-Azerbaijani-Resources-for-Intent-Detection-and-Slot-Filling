"""Strict typed character span scoring and explicit prediction diagnostics."""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from .slots import LABEL, Span, Token, bio_to_spans


@dataclass(frozen=True)
class SlotExample:
    id: str
    utt: str
    tokens: tuple[Token, ...]
    gold_tags: tuple[str, ...]
    gold_spans: tuple[Span, ...]


def decode_prediction(tokens: Sequence[Token], tags: Sequence[str]) -> tuple[tuple[Span, ...], list[dict]]:
    """Promote orphan/wrong-type I-tags to B and preserve each repair for auditing."""
    if len(tokens) != len(tags):
        raise ValueError("Prediction tag count differs from token count")
    active = None
    fixed, repairs = [], []
    for index, tag in enumerate(tags):
        if tag == "O":
            active = None
        else:
            prefix, _, label = tag.partition("-")
            if prefix not in {"B", "I"} or not LABEL.fullmatch(label):
                raise ValueError(f"Unknown prediction tag at token {index}: {tag}")
            if prefix == "I" and active != label:
                replacement = f"B-{label}"
                repairs.append({"token_index": index, "raw_tag": tag,
                                "replacement": replacement, "active_type": active})
                tag = replacement
            active = label
        fixed.append(tag)
    return bio_to_spans(tokens, fixed), repairs


def ratios(tp: int, fp: int, fn: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn, "support": tp + fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def score_spans(pairs: Iterable[tuple[tuple[Span, ...], tuple[Span, ...]]],
                labels: Sequence[str]) -> dict:
    """Count a match only when start, end and type match in the same utterance."""
    tp, fp, fn = Counter(), Counter(), Counter()
    total = exact = 0
    for gold, predicted in pairs:
        g, p = set(gold), set(predicted)
        tp.update(s.label for s in g & p)
        fp.update(s.label for s in p - g)
        fn.update(s.label for s in g - p)
        total += 1
        exact += g == p
    if (tp.keys() | fp.keys() | fn.keys()) - set(labels):
        raise ValueError("Observed slot type outside the scoring inventory")
    return {"utterances": total, "exact_matches": exact,
            "utterance_exact_match": exact / total if total else 0.0,
            "micro": ratios(tp.total(), fp.total(), fn.total()),
            "per_slot": {label: ratios(tp[label], fp[label], fn[label]) for label in labels}}


def align_predictions(dev: Sequence[SlotExample], predictions: Iterable[dict]) -> dict[str, list[str]]:
    """Join by ID, requiring exactly one prediction for every development record."""
    expected = {row.id for row in dev}
    if len(expected) != len(dev):
        raise ValueError("Duplicate development ID")
    aligned = {}
    for row in predictions:
        identifier = row["id"]
        if identifier not in expected or identifier in aligned:
            raise ValueError(f"Unexpected or duplicate prediction ID: {identifier}")
        aligned[identifier] = row["tags"]
    if missing := expected - aligned.keys():
        raise ValueError(f"Missing prediction IDs: {', '.join(sorted(missing))}")
    return aligned


def evaluate(dev: Sequence[SlotExample], predictions: Iterable[dict], train_text: set[str],
             labels: Sequence[str]) -> tuple[dict, dict]:
    aligned = align_predictions(dev, predictions)
    repairs = []
    seen, unseen = [], []
    errors = []
    wrong_type = wrong_boundary = 0
    for row in dev:
        try:
            spans, changes = decode_prediction(row.tokens, aligned[row.id])
        except ValueError as exc:
            raise ValueError(f"Prediction {row.id}: {exc}") from exc
        exact_text_seen = row.utt in train_text
        (seen if exact_text_seen else unseen).append((row.gold_spans, spans))
        if changes:
            repairs.append({"id": row.id, "changes": changes})
        missed = [span for span in row.gold_spans if span not in spans]
        spurious = [span for span in spans if span not in row.gold_spans]
        if missed or spurious:
            wrong_type += sum(any((g.start, g.end) == (p.start, p.end) for p in spurious)
                              for g in missed)
            wrong_boundary += sum(any(g.label == p.label and g.start < p.end and p.start < g.end
                                      for p in spurious) for g in missed)
            errors.append({"id": row.id, "training_text_match": exact_text_seen,
                           "missed": [asdict(s) for s in missed],
                           "spurious": [asdict(s) for s in spurious]})
    all_pairs = seen + unseen
    metrics = {
        "crf": score_spans(all_pairs, labels),
        "all_o": score_spans(((gold, ()) for gold, _ in all_pairs), labels),
        "development_text_overlap": {
            "seen": score_spans(seen, labels),
            "unseen": score_spans(unseen, labels),
        },
        "invalid_iob2": {"affected_utterances": len(repairs),
                         "transitions": sum(len(r["changes"]) for r in repairs)},
    }
    diagnostics = {
        "invalid_iob2_by_id": repairs,
        "errors_by_id": errors,
        "missed_gold_with_exact_boundary_wrong_type": wrong_type,
        "missed_gold_with_same_type_overlapping_wrong_boundary": wrong_boundary,
    }
    return metrics, diagnostics
