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

## 2026-11-23 blind prediction handoff

Ran the authorized extraction, prediction and gold-free verification commands above from clean commit `c9ab8deedb594541c2f6b720fa7beed81d5f0f80`, in that order. The preflight verified 107 pins, four release hashes by bytes, and the pinned macOS arm64 CPython 3.12.4 environment. The run used `PYTHONPATH=src:.venv/slot-crf/lib/python3.12/site-packages`. Before prediction, `load_stage` checked projection hashes, 11,514 training records in official order, 2,974 distinct evaluation IDs disjoint from training, and the blind `id`/`utt` schema.

Prediction took 391.22 seconds wall time. The fit timings below are the individual fit calls recorded in the prediction manifest.

| Model | Fit (s) | Iterations | Fit outcome |
|---|---:|---:|---|
| INT-SVC-001 | 3.130632 | 47 | 80,622 features, no warnings |
| INT-HYBRID-001 | 4.093348 | 46 | 118,349 features, no warnings |
| SLOT-CRF-002 | 379.077049 | 300 | L-BFGS reached the fixed iteration cap |

Gold-free verification passed in 0.29 seconds. It checked all 2,974 IDs and their text hashes, both sets of intent predictions against the 60-label inventory, and BIO tags, types and lengths against the frozen tokenizer and 55-type inventory. The verifier logged one repair on one record under the frozen IOB2 policy. The manifest carries the exact effective parameters, iteration and warning details, timings, and CRF binary hash. The binary was temporary.

All six files are local under the ignored `artifacts/EVAL-FREEZE-001/` directory. The two projections contain dataset text.

| File | SHA-256 |
|---|---|
| `train.jsonl` | `40fee2385f22140739432143fc20476441eba26767a727ad51b1dea6dd637a65` |
| `blind.jsonl` | `9665a85dcd8ff1a49d27bd5d0efefd48ef5aea7f3a1802eecd55eb524671a6d0` |
| `stage.json` | `bff5bdc904fdad9e347583eeda03be8683ee1dc9e06a2b605a842e6b436db6ed` |
| `predictions.jsonl` | `7b63becc95a56aa37a154b4ecb20b5390f071446014a5172e54da0adf9cf78cd` |
| `prediction-manifest.json` | `8f409f4fa5a4e5d6d66509bc80f9ee2a880402275c23acccab2dbfba70705825` |
| `validation.json` | `692d1f2523f51fbd17c936ebd81821d2a70fb4bd278800a875b18d658740fb82` |

The outputs are ready for separately authorized scoring. Test gold labels were not inspected, no examples were manually reviewed, and no test scores or scoring artifacts were produced. Test-only annotation coverage remains unknown.

## 2026-11-23 official held-out results

Scored the previously verified blind predictions against all 2,974 official MASSIVE 1.0 `az-AZ` test records using frozen commit `c9ab8deedb594541c2f6b720fa7beed81d5f0f80`. The gold-free verifier passed again before scoring, all six original artifact hashes matched the handoff above before and after scoring, and there were no prior score outputs. The frozen code and configuration verified against all 107 pins. No model was refitted and no prediction was regenerated.

| Metric | Held-out result | 95% percentile interval |
|---|---:|---:|
| Character intent accuracy | 2,510/2,974 (84.3981%) | 83.1540–85.6086% |
| Character intent macro-F1 | 80.2292% | 76.4636–81.8116% |
| Hybrid intent accuracy | 2,503/2,974 (84.1627%) | 82.8169–85.4405% |
| Hybrid intent macro-F1 | 79.4293% | 75.9999–81.1427% |
| CRF exact typed-span micro precision | 79.1156% | 77.4025–80.9125% |
| CRF exact typed-span micro recall | 64.8135% | 62.9505–66.7152% |
| CRF exact typed-span micro-F1 | 71.2539% | 69.5945–72.9306% |
| CRF complete slot-set exact match | 2,028/2,974 (68.1910%) | 66.5770–69.8058% |
| Primary joint exact match | 1,733/2,974 (58.2717%) | 56.5232–60.0202% |
| Hybrid joint exact match | 1,721/2,974 (57.8682%) | 56.0525–59.6503% |
| Hybrid minus primary joint exact match | −0.4035 percentage points | −1.0424 to +0.2354 points |

Intent macro-F1 uses all 60 training labels, including `cooking_query` with zero test support. Both models most often confused `general_quirky` with `qa_factoid` (21 each), with 11 in the reverse direction for each model. For primary and hybrid, `calendar_query` → `calendar_set` occurred 12 and 11 times, and the reverse occurred 11 and 10 times. `general_quirky` F1 was 56.88% for character and 55.24% for hybrid across 169 gold utterances.

The slot scores use 1,807 TP, 477 FP and 981 FN, for 2,788 gold and 2,284 predicted spans. The most missed types were `event_name` (111), `place_name` (87) and `time` (73). The frozen BIO repair policy was applied once on one record. All gold annotations parsed and no gold span was unrepresentable under the frozen tokenizer. [Full per-class, per-slot, per-intent and per-scenario tables](../../results/EVAL-FREEZE-001/held-out-breakdowns.md)

| Joint outcome | Primary | Hybrid |
|---|---:|---:|
| Both correct | 1,733 | 1,721 |
| Intent only correct | 777 | 782 |
| Slots only correct | 295 | 307 |
| Neither correct | 169 | 164 |

In the paired counts, both systems were jointly correct on 1,682 utterances, primary alone on 51, hybrid alone on 39, and neither on 1,202. All 90 disagreements in joint correctness were on text absent verbatim from training.

| Exact training-text overlap | Test n | Character intent accuracy | Hybrid intent accuracy | CRF slot micro-F1 | CRF slot exact | Primary joint | Hybrid joint |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seen | 230 | 224/230 (97.39%) | 224/230 (97.39%) | 89.14% | 211/230 | 205/230 (89.13%) | 205/230 (89.13%) |
| Unseen | 2,744 | 2,286/2,744 (83.31%) | 2,279/2,744 (83.05%) | 70.26% | 1,817/2,744 | 1,528/2,744 (55.69%) | 1,516/2,744 (55.25%) |

Each of the 2,000 PCG64 draws (seed 42) resampled 2,974 utterances with replacement. Both systems and the paired difference used the same draws. The paired interval includes zero. The intervals condition on these fitted models and approximate independent utterance sampling. Training-text overlap, related utterances, and rare labels limit interpretation. The CRF reached its 300-iteration cap. The separately published development comparison was 1,206/2,033 (59.3212%) primary and 1,211/2,033 (59.5671%) hybrid. The freeze designated the character-only system as primary before test predictions or results were inspected, and that designation carried through to final reporting.

### Scoring and verification evidence

```sh
PYTHONPATH=src python3 -B -m azmassive.eval_freeze verify-predictions
PYTHONPATH=src:.venv/slot-crf/lib/python3.12/site-packages python3 -B -m azmassive.eval_freeze score --authorize-test-scoring
PYTHONPATH=src:.venv/slot-crf/lib/python3.12/site-packages python3 -B -m azmassive.eval_freeze verify-score --authorize-test-scoring
```

The local site-packages path makes the pinned CRFsuite package available to CPython 3.12.4. The official release hashes, original projections, metrics including bootstrap intervals, diagnostic rows and BIO repair records all verified. The independent `verify_point_metrics` check also ran directly on the saved metrics and ID diagnostics and passed. It checks intent accuracy and macro-F1, slot counts and exact matches, and joint correctness and the paired point difference.

The four scoring outputs below remain in ignored `artifacts/EVAL-FREEZE-001/`, alongside the six unchanged prediction-stage artifacts listed above. The breakdown table contains only aggregates.

| File | SHA-256 |
|---|---|
| `metrics.json` | `39c0242af319067d8fc72e510553d98d3aaa006727a5038db4fe2baa077aabca` |
| `diagnostics.jsonl` | `3120d99b5a7263bdcffef773cedab0f304345f78b60dcd39c6ade6a0058f68ba` |
| `bio-repairs.jsonl` | `2cda71d49e1c4ee58fac72327eced9b9ad47ae74b5c2488f6ac674f61ad27ad3` |
| `score-manifest.json` | `a4fc13deaec0ad723c2c98f9270f7bc6547fccdc2f3be9b4d723ad9c13e8b43d` |

These verified results are ready for the Master Thesis I final report. Keep the primary and comparison roles fixed and include the paired interval and overlap diagnostics.

### Publication checks

On 2026-11-23 the gold-free prediction verifier, frozen score verifier and standalone independent point-count check all passed again. Rechecking the ten local file hashes confirmed the prediction and scoring artifacts were unchanged. All 107 protocol pins matched. The public results summary contains aggregate metrics and counts only. [Development versus test and full aggregate breakdowns](../../results/EVAL-FREEZE-001/held-out-breakdowns.md)
