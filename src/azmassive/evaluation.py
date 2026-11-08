"""Intent-only metrics with a fixed label inventory and explicit ID alignment."""

from collections import Counter
from collections.abc import Sequence

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .dataset import Record


def majority_label(labels: Sequence[str]) -> str:
    if not labels:
        raise ValueError("Cannot fit a majority baseline without training labels")
    counts = Counter(labels)
    return min(counts, key=lambda label: (-counts[label], label))


def evaluate_intents(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict:
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Expected aligned, nonempty true and predicted labels")
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Expected a unique, nonempty label inventory")
    if (set(y_true) | set(y_pred)) - set(labels):
        raise ValueError("Observed labels outside the fixed inventory")
    return {
        "task": "intent-only", "evaluation_split": "dev", "n": len(y_true),
        "correct": sum(a == b for a, b in zip(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(labels),
                                    average="macro", zero_division=0)),
        "per_class": classification_report(y_true, y_pred, labels=list(labels),
                                             output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
        "label_order": list(labels),
        "macro_f1_definition": "Unweighted mean over the fixed training label inventory; zero_division=0.",
    }


def prediction_records(records: Sequence[Record], predicted: Sequence[str],
                       majority: str) -> list[dict]:
    if len(records) != len(predicted) or len({r.id for r in records}) != len(records):
        raise ValueError("Predictions must align one-to-one with unique record IDs")
    if any(r.partition != "dev" for r in records):
        raise ValueError("Only development predictions may be exported")
    return [{"id": r.id, "partition": "dev", "gold_intent": r.intent,
             "logistic_regression": p, "majority": majority}
            for r, p in zip(records, predicted)]


def error_analysis(train: Sequence[Record], dev: Sequence[Record], predicted: Sequence[str],
                   labels: Sequence[str]) -> dict:
    """Descriptive dev analysis only; no filtering of the primary evaluation split."""
    if len(dev) != len(predicted):
        raise ValueError("Unaligned predictions")
    seen = {r.utt for r in train}
    errors = [(r, p) for r, p in zip(dev, predicted) if r.intent != p]
    pairs = Counter((r.intent, p) for r, p in errors)
    groups = {}
    for name, indices in {
        "exact_text_seen_in_train": [i for i, r in enumerate(dev) if r.utt in seen],
        "exact_text_absent_from_train": [i for i, r in enumerate(dev) if r.utt not in seen],
    }.items():
        groups[name] = evaluate_intents([dev[i].intent for i in indices],
                                       [predicted[i] for i in indices], labels) if indices else None
    return {
        "task": "intent-only", "error_count": len(errors),
        "top_confusions": [{"gold": a, "predicted": b, "count": n}
                           for (a, b), n in sorted(pairs.items(), key=lambda item: (-item[1], item[0]))[:20]],
        "overlap_subgroups": groups,
        "examples_selection": "First 20 errors in original development-file order; not a random sample.",
        # Reference raw text by ID instead of copying dataset utterances into Git.
        "examples": [{"id": r.id, "gold": r.intent, "predicted": p,
                      "text_seen_in_train": r.utt in seen} for r, p in errors[:20]],
        "interpretation": "Descriptive associations only; no causal linguistic error claims or native-speaker review.",
    }
