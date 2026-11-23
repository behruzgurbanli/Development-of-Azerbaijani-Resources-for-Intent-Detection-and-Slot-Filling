"""Independent count-based cross-check of frozen point metrics from ID diagnostics."""

from collections import Counter
from math import isclose


def verify_point_metrics(metrics, diagnostics, config):
    """Compare intent, span, and joint aggregates without the primary scorer."""
    n = len(diagnostics)
    if not n or n != metrics["n"] or len({row["id"] for row in diagnostics}) != n:
        raise ValueError("Metric cross-check requires complete unique diagnostic IDs")

    def spans(items):
        return {(s["start"], s["end"], s["label"]) for s in items}

    slot_model = config["primary"]["slot"]
    slot_exact, gold_counts, pred_counts, hits = set(), Counter(), Counter(), Counter()
    for row in diagnostics:
        gold = spans(row["gold_spans"])
        errors = row["slot_errors"][slot_model]
        missed, extra = spans(errors["missed"]), spans(errors["spurious"])
        if not missed <= gold or extra & gold:
            raise ValueError("Inconsistent slot diagnostics in metric cross-check")
        matched = gold - missed
        gold_counts.update(s[2] for s in gold)
        pred_counts.update(s[2] for s in matched | extra)
        hits.update(s[2] for s in matched)
        if not missed and not extra:
            slot_exact.add(row["id"])
    slot = metrics["slot_models"][slot_model]
    if slot["exact_matches"] != len(slot_exact) or slot["utterances"] != n:
        raise ValueError("Slot exact-match cross-check failed")
    if (gold_counts.keys() | pred_counts.keys()) - slot["per_slot"].keys():
        raise ValueError("Unexpected slot type in metric cross-check")
    for label, counts in slot["per_slot"].items():
        if (counts["tp"], counts["fp"], counts["fn"]) != (
                hits[label], pred_counts[label] - hits[label], gold_counts[label] - hits[label]):
            raise ValueError(f"Slot count cross-check failed for {label}")
    if (slot["micro"]["tp"], slot["micro"]["fp"], slot["micro"]["fn"]) != (
            hits.total(), pred_counts.total() - hits.total(), gold_counts.total() - hits.total()):
        raise ValueError("Pooled slot count cross-check failed")

    jointly_correct = []
    for system in ("primary", "comparison"):
        model = config[system]["intent"]
        scores = metrics["intent_models"][model]
        gold = Counter(row["gold_intent"] for row in diagnostics)
        predicted = Counter(row["intent_predictions"][model] for row in diagnostics)
        correct = {row["id"] for row in diagnostics if row["gold_intent"] == row["intent_predictions"][model]}
        by_label = Counter(row["gold_intent"] for row in diagnostics if row["id"] in correct)
        labels = scores["label_order"]
        if (gold.keys() | predicted.keys()) - set(labels):
            raise ValueError("Unexpected intent in metric cross-check")
        macro = sum(2 * by_label[label] / (gold[label] + predicted[label])
                    for label in labels if gold[label] + predicted[label]) / len(labels)
        if (scores["correct"] != len(correct) or not isclose(scores["accuracy"], len(correct) / n)
                or not isclose(scores["macro_f1"], macro)):
            raise ValueError(f"Intent metric cross-check failed for {model}")
        both = correct & slot_exact
        jointly_correct.append(both)
        pair = metrics["model_pairs"][f"{model}+{slot_model}"]
        counts = {"both_correct": len(both), "intent_only_correct": len(correct - slot_exact),
                  "slots_only_correct": len(slot_exact - correct), "neither_correct": n - len(correct | slot_exact)}
        if (pair["counts"] != counts or pair["joint_exact_matches"] != len(both)
                or not isclose(pair["joint_exact_match"], len(both) / n)):
            raise ValueError(f"Joint metric cross-check failed for {model}")
    primary, hybrid = jointly_correct
    if not isclose(metrics["comparison_minus_primary_joint_exact_match"], (len(hybrid) - len(primary)) / n):
        raise ValueError("Paired joint difference cross-check failed")
