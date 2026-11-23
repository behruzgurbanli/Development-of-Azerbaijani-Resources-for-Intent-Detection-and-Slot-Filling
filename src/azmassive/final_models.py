"""Frozen train-only fitting and inference on a label-free input projection."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import random
import re
import tempfile
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .dataset import sha256_file
from .intent_hybrid import hybrid_pipeline
from .intent_svc import candidate_pipeline
from .slot_features import sequence_features
from .slots import parse_annotation, spans_to_bio, tokenize


def fit_and_predict(train, blind, configurations, *, expected_crf_settings=None):
    """Fit only on the supplied training projection. Blind rows must contain only id and utt."""
    if any(set(row) != {"id", "utt"} for row in blind):
        raise ValueError("Prediction inputs must contain only id and utt")
    if any(set(row) != {"id", "utt", "intent", "annot_utt"} for row in train):
        raise ValueError("Unexpected training projection fields")
    import pycrfsuite

    pr5 = configurations["PR5-INT-001"]
    svc = configurations["INT-SVC-001"]
    hybrid = configurations["INT-HYBRID-001"]
    crf = configurations["SLOT-CRF-002"]
    train_text = [row["utt"] for row in train]
    train_labels = [row["intent"] for row in train]
    blind_text = [row["utt"] for row in blind]
    predicted = [{"id": row["id"], "intents": {}} for row in blind]
    provenance = {}
    for name, pipeline in (
        ("INT-SVC-001", candidate_pipeline(pr5, svc["classifier"])),
        ("INT-HYBRID-001", hybrid_pipeline(pr5, svc, hybrid)),
    ):
        random.seed(pr5["seed"])
        np.random.seed(pr5["seed"])
        start = time.perf_counter()
        with threadpool_limits(limits=pr5["threads"]):
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                pipeline.fit(train_text, train_labels)
            seconds = time.perf_counter() - start
            predictions = pipeline.predict(blind_text)
        if any(issubclass(w.category, ConvergenceWarning) for w in captured):
            raise ValueError(f"{name} emitted an unexpected convergence warning")
        model = pipeline.named_steps["linear_svc"]
        for row, label in zip(predicted, predictions, strict=True):
            row["intents"][name] = str(label)
        provenance[name] = {"classifier": model.get_params(), "classes": model.classes_.tolist(),
                            "iterations": int(model.n_iter_), "features": int(model.n_features_in_),
                            "fit_seconds": seconds,
                            "warning_categories": [w.category.__name__ for w in captured]}

    random.seed(crf["seed"])
    trainer = pycrfsuite.Trainer(verbose=True)
    trainer.select(crf["crf"]["algorithm"], type=crf["crf"]["type"])
    trainer.set_params(crf["crf"]["parameters"])
    effective = {name: trainer.get(name) for name in trainer.params()}
    if expected_crf_settings is not None and effective != expected_crf_settings:
        raise ValueError("Effective CRF optimization settings differ from frozen run")
    for row in train:
        _, spans = parse_annotation(row["annot_utt"], row["utt"])
        tokens = tokenize(row["utt"], crf["tokenizer"])
        tags = spans_to_bio(tokens, spans)
        trainer.append(sequence_features(tokens, **crf["features"]), tags)
    with tempfile.TemporaryDirectory(prefix="frozen-crf-") as tmp:
        model_path = Path(tmp) / "model.crfsuite"
        log = io.StringIO()
        start = time.perf_counter()
        with redirect_stdout(log):
            trainer.train(str(model_path))
        seconds = time.perf_counter() - start
        tagger = pycrfsuite.Tagger()
        tagger.open(str(model_path))
        trained_tags = sorted(tagger.labels())
        for row, output in zip(blind, predicted, strict=True):
            output["slot_tags"] = tagger.tag(
                sequence_features(tokenize(row["utt"], crf["tokenizer"]), **crf["features"]))
        tagger.close()
        model_hash = sha256_file(model_path)
    lines = log.getvalue()
    iterations = re.findall(r"^\*\*\*\*\* Iteration #(\d+) \*\*\*\*\*", lines, re.MULTILINE)
    provenance["SLOT-CRF-002"] = {
        "effective_parameters": effective, "trained_tags": trained_tags,
        "iterations": int(iterations[-1]) if iterations else None,
        "termination": [line for line in lines.splitlines()
                        if re.search(r"terminat|converg|fatal|failed", line, re.I)],
        "temporary_model_sha256": model_hash, "fit_seconds": seconds,
    }
    return predicted, provenance
