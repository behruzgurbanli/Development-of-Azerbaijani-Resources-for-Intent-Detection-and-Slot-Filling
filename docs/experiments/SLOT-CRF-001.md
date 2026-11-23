# SLOT-CRF-001 — Azerbaijani CRF slot filling

## Protocol

This CPU baseline fits a first-order linear-chain CRF on the **11,514 official train** records and evaluates on the **2,033 official dev** records of MASSIVE 1.0 `az-AZ`. The runner skips test rows by partition before processing or modeling. Each run verifies the dataset archive, Azerbaijani JSONL, license, AUD-001 record, preparation manifest and preparation source hashes. The [slot preparation audit](SLOT-PREP-001.md) supplies the tokenization and label inventory. The [intent baseline](PR5-INT-001.md) is a separate experiment.

The runner uses the existing `word_punctuation` tokenizer (`\w+|[^\w\s]`), exact original-text offsets and strict IOB2 alignment. Inputs contain only features derived from tokens in `utt`. The markup supplies gold labels for training and evaluation. It does not influence tokenization or model features. The runner verifies that the loaded split counts, token counts and per-type span counts match SLOT-PREP-001.

Features are the case-preserved surface form, Python Unicode casefold, character shape, uppercase and initial-uppercase indicators, digit indicators, Unicode punctuation, and casefolded prefixes and suffixes of lengths 1–4. The preceding and following tokens contribute lexical, shape and orthographic features. The sequence ends have explicit boundary indicators. The original text and character offsets retain case and Unicode. Python casefold is a general Unicode transform without custom Azerbaijani casing rules.

The [fixed configuration](../../configs/SLOT-CRF-001.json) uses CRFsuite `crf1d` with L-BFGS, `c1=0.0`, `c2=1.0`, at most 100 iterations, all possible transition features, and observed-only state features. The run manifest records the complete effective parameters. This configuration was fixed before training and was never selected by a dev score. Input order is the official JSONL order, with no shuffling. Python's seed is 42. L-BFGS exposes no random-seed parameter. [CRFsuite Python API](https://python-crfsuite.readthedocs.io/en/latest/pycrfsuite.html)

## Scoring and prediction validity

- A span counts as correct when its type, start, and end all match for the same record ID. Offsets are zero-based, half-open Python character offsets. Micro precision, recall, and F1 pool counts over utterances and slot types. Per-type counts include all 55 audited types. Metrics with zero denominators are reported as zero.
- Utterance exact match requires equality of the entire span set, including empty sets. The all-`O` reference predicts no spans on every record.
- Predictions are joined by ID, with missing, duplicate, and foreign IDs rejected. A predicted `I-type` without an active span of the same type is converted to `B-type` before span scoring. Each repair is reported with the record ID, token index and original tag. The saved predictions retain raw tags. A malformed tag, wrong sequence length, or slot type outside the audited inventory fails verification.
- Exact text matching against the fit split defines descriptive seen/unseen dev groups. Both groups remain in the primary evaluation. The error artifact records missed and spurious typed spans by ID without saving utterance text.

## Measured results

The full run started November 21, 2026 at 10:34:04 p.m. Eastern Time from Git base `4c1fe1ce2c37ed9f72e276e09f768a810287c49f`, with the new CRF files in the working tree. The [full manifest](../../results/SLOT-CRF-001/full/run.json) records their hashes. The smoke run used the first 256 train and 64 dev records in source order. Its F1 of 0.4050632911 is a pipeline check on that prefix.

| Official dev evaluation | Dev records | TP | FP | FN | Micro precision | Micro recall | Micro F1 | Utterance exact match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **CRF, all dev** | 2,033 | 1,282 | 362 | 711 | 0.7798053528 | 0.6432513798 | **0.7049766291** | 1,359/2,033 (66.8470%) |
| All-`O` reference | 2,033 | 0 | 0 | 1,993 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 647/2,033 (31.8249%) |
| CRF, exact text seen in train | 162 | 95 | 12 | 10 | 0.8878504673 | 0.9047619048 | 0.8962264151 | 143/162 (88.2716%) |
| CRF, exact text unseen in train | 1,871 | 1,187 | 350 | 701 | 0.7722836695 | 0.6287076271 | 0.6931386861 | 1,216/1,871 (64.9920%) |

There were **zero invalid IOB2 transitions** in smoke and full predictions. The all-`O` exact matches are the 647 dev utterances with no annotated spans. The overlap comparison is descriptive because the subgroups have different texts and slot distributions. See the [full per-slot metrics and support](../../results/SLOT-CRF-001/full/metrics.json) and [raw dev predictions](../../results/SLOT-CRF-001/full/predictions.jsonl).

| Run | Feature extraction and train append | CRF fit | Dev features and prediction | Process peak RSS | Optimization stop |
|---|---:|---:|---:|---:|---|
| Smoke | 0.019645 s | 0.315348 s | 0.003384 s | 60.59375 MiB | Stopping criteria, iteration 72 |
| Full | 1.106718 s | 119.701208 s | 0.263374 s | 169.3125 MiB | Configured limit, iteration 100 |

Timings are wall-clock intervals. Fit excludes feature construction. Prediction includes dev feature construction and excludes scoring. Peak RSS is the high-water mark of the entire Python process, including data loading. The full fit reached the iteration cap, so this baseline does not establish optimizer convergence.

## Error analysis and limitations

The model missed 711 gold spans and predicted 362 extra spans, with at least one mismatch in 674 utterances. Among the missed gold spans, 54 had a prediction with identical offsets and a different label. Another diagnostic counts 151 missed spans with an overlapping prediction of the same label and different boundaries. These counts can overlap and do not exhaust the errors.

| Dev ID | Gold and predicted spans | Observation |
|---|---|---|
| `835` | Gold `news_topic [0,10)`, predicted `place_name [0,10)` | Wrong type with exact boundaries |
| `113` | Gold `player_setting [11,29)`, predicted `player_setting [11,24)` | Right type with an early end |
| `26` | Gold `house_place [0,7)`, no predicted span | Missed span |
| `392` | No gold span, predicted `list_name [6,27)` | Spurious span |

`event_name` had the most misses (67 of 160), followed by `place_name` (65 of 200) and `time` (64 of 138). `music_descriptor` had eight gold instances and no exact matches. `music_album` and `transport_descriptor` had zero dev support, so their reported per-type scores do not measure recall on positives. All 674 mismatched record IDs and their missed or spurious offsets are in the [full diagnostics](../../results/SLOT-CRF-001/full/diagnostics.json). Known annotation disagreements from SLOT-PREP-001 remain in the official data.

This is one feature configuration on the official dev split. It has no held-out score or estimate of variation between training runs. For a separate follow-up, test a higher iteration limit while keeping features fixed to isolate the effect of additional optimization. Other untested options include comparing regularization on a predeclared train-only selection split, changing affix/context lengths, or adding Azerbaijani casing or morphology features. Keep the official test partition held out until the evaluation protocol is fixed.

## Environment and reproduction

The run used CPython 3.12.4, macOS arm64, Unicode data 15.0.0, and an Apple M1 with eight CPU cores and 8 GB unified memory. The only model dependency is pinned in [requirements-slot-crf.txt](../../requirements-slot-crf.txt). The installed PyPI wheel was `python_crfsuite-0.9.12-cp312-cp312-macosx_11_0_arm64.whl`, SHA-256 `e68009911b28ff899da5a6be3ec1efc3c24886c92318d02d39ec29d329b08b90`. The wheel bundles CRFsuite and required no extra Python libraries. [Python CRFsuite 0.9.12 release](https://pypi.org/project/python-crfsuite/0.9.12/)

Run from the repository root with the audited local inputs. [PR5-INT-001](PR5-INT-001.md) documents how to acquire and verify them if needed.

```bash
python3 -B -m venv .venv/slot-crf
.venv/slot-crf/bin/python -B -m pip --disable-pip-version-check install --no-cache-dir --only-binary=:all: --no-deps --report .venv/slot-crf/install.json -r requirements-slot-crf.txt
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
export PYTHONPATH=src
.venv/slot-crf/bin/python -B -m azmassive.slot_crf --config configs/SLOT-CRF-001.json --mode smoke --output /tmp/slot-crf-review-smoke
.venv/slot-crf/bin/python -B -m azmassive.slot_crf --config configs/SLOT-CRF-001.json --mode smoke --output /tmp/slot-crf-review-smoke --verify
.venv/slot-crf/bin/python -B -m azmassive.slot_crf --config configs/SLOT-CRF-001.json --mode full --output /tmp/slot-crf-review-full
.venv/slot-crf/bin/python -B -m azmassive.slot_crf --config configs/SLOT-CRF-001.json --mode full --output /tmp/slot-crf-review-full --verify
```

Choose fresh output directories. The original runs used the default `results/SLOT-CRF-001/{smoke,full}` directories. To verify the saved runs without fitting, run the verify commands with no `--output` override. The full test suite uses the existing intent-test environment with the dependencies in `requirements-pr5.txt`. The seven new tests also run in the CRF-only environment with `-p 'test_slot_crf.py'`. The CRFsuite model is used in a temporary directory and removed when the run finishes. Its SHA-256 is saved.

All **34 repository tests** passed in the default Python environment, and all **7 CRF tests** passed in the isolated environment. Before the successful smoke run, a test fixture had an incorrect expected character end offset, and the first smoke command failed before training because the binding accepts the CRF model type through `Trainer.select()`. Both issues were corrected before the verified runs. The initial sandboxed PyPI request failed on DNS. The approved isolated installation succeeded, and `pip check` found no broken requirements.

The [smoke verifier](../../results/SLOT-CRF-001/smoke/verification.json) and [full verifier](../../results/SLOT-CRF-001/full/verification.json) matched input, source and artifact hashes, exact dev ID coverage, and recomputed metrics and diagnostics. A separate check decoded raw tags and counted spans as record-keyed tuples, reproducing the overall, per-type, overlap, all-`O`, and invalid-transition counts. [Independent count evidence](../../results/SLOT-CRF-001/count-verification.json)

A fresh smoke fit on the same host produced byte-identical predictions, metrics and diagnostics, and the transient model SHA-256 matched. The full fit was executed once. Full-run and cross-platform byte-for-byte determinism remain unmeasured. The two saved PR5-INT-001 runs verified again, and the SLOT-PREP-001 audit and code stayed unchanged.
