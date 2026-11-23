# EVAL-FREEZE-001 — Master Thesis I held-out evaluation protocol

Frozen for review on 2026-11-22 (Eastern). Bundle integration verified on 2026-11-23. The [configuration](../../configs/EVAL-FREEZE-001.json) fixes the protocol. The [hash manifest](../../results/EVAL-FREEZE-001/frozen-inputs.json) identifies the published base commit `eea2b12818fd48477247bb93c09058c8dc35dd3e`, the component publication commits, and exact SHA-256 values for the configuration, implementation, requirements, and published inputs. The working-tree implementation is pinned by those file hashes. Future prediction and scoring manifests also record the actual checkout commit.

## Frozen systems and data

| Role | Intent | Slots |
|---|---|---|
| Primary | INT-SVC-001, character 3–5-gram TF-IDF + LinearSVC | SLOT-CRF-002 |
| Prespecified comparison | INT-HYBRID-001, character and word TF-IDF + the same LinearSVC settings | The same fitted SLOT-CRF-002 |

Fit both intent systems and one CRF on all **11,514 official MASSIVE 1.0 `az-AZ` train records**, in the release JSONL order. Exclude all 2,033 development records from fitting. Evaluate the frozen systems on all **2,974 official test records** after authorization. Preserve duplicates and official split membership. Use verbatim strings with original case, whitespace, punctuation and diacritics. All feature vocabularies and IDFs are fitted on train only. The hybrid combines the frozen character representation and Unicode word unigrams/bigrams with equal weights, followed by row L2 normalization. The fixed LinearSVC uses L2, squared hinge, dual one-vs-rest, `C=1`, `tol=0.0001`, `max_iter=1000`, random state 42 and a single thread.

The CRF uses `word_punctuation`, source offsets, strict gold-to-BIO alignment for training and the committed text-only local features. CRFsuite uses first-order L-BFGS with `c1=0`, `c2=1`, and a fixed cap of 300 iterations plus the unchanged effective defaults recorded in the published run. CRF-002 previously hit the cap. Its convergence limitation is accepted for this protocol. The comparison remains secondary regardless of the observed test result. Hyperparameters, features, preprocessing, and fit partitions are fixed.

MASSIVE 1.0 is CC BY 4.0. The local archive, Azerbaijani JSONL, license, and AUD-001 matched the four checksums in the manifest. In particular, the JSONL SHA-256 is `9a643dbf6e07e90ba8b1554bd95e9946b42037bd52550aac90f4e774ee58ed8c`. [AUD-001](../../data/metadata/AUD-001.json)

## Scoring and uncertainty

- **Intent.** Accuracy is the number of correct intents divided by the number of utterances. Macro-F1 is the unweighted mean over all 60 train labels with `zero_division=0`. Report per-intent precision, recall, F1 and support.
- **Slots.** A match requires the same half-open Python character offsets and type on the same utterance. Pool TP, FP and FN for micro precision `TP/(TP+FP)`, recall `TP/(TP+FN)` and F1 `2TP/(2TP+FP+FN)`. Use zero for an empty denominator. Include per-slot metrics and support for the 55 train types. Slot utterance exact match requires identical complete span sets, including two empty sets, divided by total utterances.
- **Joint primary endpoint.** Count utterances with both correct intent and an exact slot set, divided by total utterances. Report both correct, intent only correct, slots only correct and neither correct, overall and descriptively by gold intent and scenario. Also report intent, slot and joint performance for exact text seen versus unseen in the official fit partition, with each group's denominator. The overall scores retain both groups.
- **BIO.** Promote an orphan or mismatched `I` tag to `B` using the existing decoder, logging the ID, token index, tag, previous type and replacement for every repair. Reject malformed tags, unknown labels, sequence length mismatches and ID discrepancies. Score valid official gold spans at their original character boundaries even if the frozen tokenizer cannot represent them. Report those alignment failures by ID and offsets. A malformed annotation or out-of-inventory gold label stops scoring for review.
- **Confidence intervals.** Compute pointwise 95% percentile intervals for both systems' intent accuracy and macro-F1, shared CRF micro P/R/F1 and utterance exact match, and both joint exact-match rates. Use NumPy PCG64 with seed 42, 2,000 draws of `n` utterance indices sampled with replacement from the `n` held-out utterances, and linearly interpolated 2.5th and 97.5th percentiles. Recompute macro-F1 over the same 60 labels and re-pool span counts on every draw. Reuse each draw for all metrics and systems. On each draw, the paired joint contrast is the hybrid rate minus the primary rate, retaining the original sample denominator `n`. Report as a percentage-point difference.
- **Interpretation.** These descriptive intervals condition on the fitted models and approximate independent utterance sampling. Identical or related texts may violate independence, and rare or absent labels can affect macro-F1 intervals. The intervals do not include training variability, adjust for multiple comparisons, or by themselves establish a statistically significant difference. Overlap subgroup results are descriptive.

## Preparation and authorized execution

The preparation command checks pinned source files and AUD-001. It opens no raw dataset files and writes no run artifacts.

```sh
PYTHONPATH=src python3 -B -m azmassive.eval_freeze --config configs/EVAL-FREEZE-001.json prepare
```

From the repository root, use CPython 3.12.4 with both requirements files installed in the same environment. The manifest pins Unicode 15.0.0 and the recorded NumPy, SciPy, scikit-learn, joblib, threadpoolctl, and python-crfsuite versions. The following commands are reserved for a later authorized run.

```sh
PYTHONPATH=src python3 -B -m azmassive.eval_freeze extract-test --authorize-test-input-extraction
PYTHONPATH=src python3 -B -m azmassive.eval_freeze predict --authorize-test-prediction
PYTHONPATH=src python3 -B -m azmassive.eval_freeze verify-predictions
PYTHONPATH=src python3 -B -m azmassive.eval_freeze score --authorize-test-scoring
PYTHONPATH=src python3 -B -m azmassive.eval_freeze verify-score --authorize-test-scoring
```

`extract-test` checks the audited release and writes a train projection and a blind test projection containing only `id` and `utt`. `predict` reads those projections and frozen repository inputs. It writes ID-keyed predicted intents, raw BIO tags and text SHA-256 values, with settings, inventories, fit metadata and artifact hashes in its manifest. The CRF binary is transient. `verify-predictions` checks recorded hashes, fit provenance, exact ID coverage, per-ID text hashes, label inventories and token/tag lengths without opening the release source. `score` requires explicit scoring authorization and a separately written validation record that still matches revalidation. It then opens the audited gold source, checks the projections against the release, and writes metrics, ID-keyed diagnostics and repair records. With the separate scoring authorization, `verify-score` verifies the saved artifacts and their hashes, recomputes scoring, and independently checks the point counts. All run files belong in the ignored `artifacts/EVAL-FREEZE-001/`. The two local projection files contain raw text and must stay private to the run.

## Verification and status

```sh
PYTHONPATH=src:.venv/slot-crf/lib/python3.12/site-packages python3 -B -m unittest discover -s tests -p 'test_eval_freeze.py' -v
PYTHONPATH=src:.venv/slot-crf/lib/python3.12/site-packages python3 -B -m unittest discover -s tests -p 'test_final_protocol.py' -v
PYTHONPATH=src:.venv/slot-crf/lib/python3.12/site-packages python3 -B -m unittest discover -s tests -v
```

All **63 tests** passed, including **13** across the two freeze suites. Real estimators fitted synthetic Azerbaijani inputs. Synthetic end-to-end checks covered separate authorization, missing and stale validation, hash tampering, malformed and repaired BIO, ID alignment and exact-set scoring. The preparation command verified **107** static pins. All 99 pinned historical files are byte-identical to the published base. Both the scorer and the independent count check passed on the saved development diagnostics, reproducing **1,206/2,033** primary and **1,211/2,033** hybrid joint exact matches. CRF achieved **1,399/2,033** slot exact matches and micro-F1 **0.7224669604** in that replay. The dev replay also exercised the original PCG64 bootstrap and paired contrast. [Integration evidence](../../results/EVAL-FREEZE-001/integration-validation.json) · [Original freeze evidence](../../results/EVAL-FREEZE-001/validation.json)

The bundle patch conflicted with the local configuration, experiment record, validation evidence and research log. The new-file patch passed `git apply --check` and was applied for the CLI, count verifier and focused tests. Those files were reconciled with the existing runner so the chosen models, corpus pins, bootstrap RNG and exact-span policy stayed fixed. The bundle mistyped the hash of `evaluation.py`. The current file agrees with the published Git blob and its existing verified pin.

Integration used synthetic fixtures, static inputs and saved development artifacts. No raw release file or official held-out row was opened during integration. Full-fit behavior and test-only annotation coverage remain unknown. Next, independently review the frozen config and hashes, then authorize extraction, prediction and gold scoring separately.
