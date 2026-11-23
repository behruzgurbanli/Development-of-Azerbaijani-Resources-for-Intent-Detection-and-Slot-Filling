"""Held-out scoring and paired utterance bootstrap for the frozen protocol."""

from collections import defaultdict

import numpy as np

from .evaluation import evaluate_intents
from .joint_eval import evaluate_joint, model_key
from .slot_metrics import ratios, score_spans


def bootstrap(gold_intent, intent_predictions, labels, slot_pairs, pairs, primary, comparison, settings):
    """Aggregate fixed-label and pooled span counts inside each shared utterance draw."""
    n = len(gold_intent)
    if n == 0:
        raise ValueError("Bootstrap requires at least one utterance")
    label_index = {label: i for i, label in enumerate(labels)}
    gold = np.array([label_index[label] for label in gold_intent])
    k = len(labels)
    predicted = {name: np.array([label_index[label] for label in values])
                 for name, values in intent_predictions.items()}
    intent_ok = {name: values == gold for name, values in predicted.items()}
    slot_stats = {}
    for name, spans in slot_pairs.items():
        counts = []
        for actual, expected in spans:
            g, p = set(actual), set(expected)
            counts.append((len(g & p), len(p - g), len(g - p), int(g == p)))
        slot_stats[name] = np.asarray(counts)
    if any(len(v) != n for v in (*predicted.values(), *slot_stats.values())):
        raise ValueError("Bootstrap inputs must align by utterance")
    joint_ok = {model_key(i, s): intent_ok[i] & slot_stats[s][:, 3].astype(bool) for i, s in pairs}
    delta = joint_ok[comparison].astype(int) - joint_ok[primary].astype(int)
    draws = defaultdict(list)
    generator = np.random.Generator(np.random.PCG64(settings["seed"]))
    for _ in range(settings["resamples"]):
        ix = generator.integers(n, size=n)
        actual = gold[ix]
        support = np.bincount(actual, minlength=k)
        for name, values in predicted.items():
            guess = values[ix]
            correct = actual == guess
            hits = np.bincount(actual[correct], minlength=k)
            denom = support + np.bincount(guess, minlength=k)
            f1 = np.divide(2 * hits, denom, out=np.zeros(k), where=denom > 0)
            draws[f"{name}.intent_accuracy"].append(correct.mean())
            draws[f"{name}.intent_macro_f1"].append(f1.mean())
        for name, values in slot_stats.items():
            tp, fp, fn, exact = map(int, values[ix].sum(axis=0))
            pooled = ratios(tp, fp, fn)
            for measure in ("precision", "recall", "f1"):
                draws[f"{name}.slot_micro_{measure}"].append(pooled[measure])
            draws[f"{name}.slot_utterance_exact_match"].append(exact / n)
        for pair, correct in joint_ok.items():
            draws[f"{pair}.joint_exact_match"].append(correct[ix].mean())
        draws["comparison_minus_primary.joint_exact_match"].append(delta[ix].mean())
    tail = (1 - settings["confidence_level"]) / 2
    return {name: np.quantile(values, [tail, 1 - tail], method=settings["quantile_method"]).tolist()
            for name, values in draws.items()}


def score_records(records, gold_slots, train_text, intent_predictions, slot_predictions,
                  intent_labels, slot_labels, config, *, partition):
    """Reuse established point scorers. The caller must first validate and align records."""
    if not records or any(row.partition != partition for row in records):
        raise ValueError("Expected nonempty gold rows from the requested partition")
    ids = [row.id for row in records]
    expected = set(ids)
    if (len(expected) != len(ids) or gold_slots.keys() != expected
            or any(source.keys() != expected for source in (*intent_predictions.values(), *slot_predictions.values()))):
        raise ValueError("Scoring requires complete matching ID sets")
    primary, comparison = (model_key(config[key]["intent"], config[key]["slot"])
                           for key in ("primary", "comparison"))
    gold_intent = [row.intent for row in records]
    aligned = {name: [values[i] for i in ids] for name, values in intent_predictions.items()}
    pairs = {name: [(gold_slots[i].gold_spans, values[i]) for i in ids]
             for name, values in slot_predictions.items()}
    def intent_score(name, indices):
        return {**evaluate_intents([gold_intent[i] for i in indices],
                                   [aligned[name][i] for i in indices], intent_labels),
                "evaluation_split": partition}
    whole = range(len(ids))
    intent_scores = {name: intent_score(name, whole) for name in aligned}
    slot_scores = {name: score_spans(values, slot_labels) for name, values in pairs.items()}
    joint, diagnostics = evaluate_joint(records, gold_slots, train_text, intent_predictions,
                                       slot_predictions, primary, comparison)
    overlap = {}
    for group, seen in (("seen", True), ("unseen", False)):
        ix = [i for i, row in enumerate(records) if (row.utt in train_text) == seen]
        overlap[group] = {"n": len(ix),
                          "intent_models": {name: intent_score(name, ix) for name in aligned} if ix else {},
                          "slot_models": {name: score_spans((pairs[name][i] for i in ix), slot_labels)
                                          for name in pairs} if ix else {}}
    delta = joint["paired_comparison"]["net_additional_joint_matches"] / len(ids)
    intervals = bootstrap(gold_intent, aligned, intent_labels, pairs,
                          [(i, s) for i in aligned for s in pairs], primary, comparison, config["bootstrap"])
    result = {"partition": partition, "n": len(ids), "intent_models": intent_scores, "slot_models": slot_scores,
              "primary": primary, "comparison": comparison, **joint, "component_overlap": overlap,
              "comparison_minus_primary_joint_exact_match": delta,
              "bootstrap": {"settings": config["bootstrap"], "denominator": len(ids),
                            "intervals": intervals}}
    return result, diagnostics
