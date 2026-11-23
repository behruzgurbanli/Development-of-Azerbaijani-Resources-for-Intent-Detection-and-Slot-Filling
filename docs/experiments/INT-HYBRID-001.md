# INT-HYBRID-001 — Character plus word TF-IDF

Adding equal-weight word TF-IDF to the fixed LinearSVC pipeline improved official dev accuracy from 84.4073% to **84.6040%** and macro-F1 from 0.8228981629 to **0.8243775119**. This is a small measured gain on dev. The character-only baseline remains a practical default given the mixed class changes and additional features.

## Controlled protocol

The [experiment configuration](../../configs/INT-HYBRID-001.json) pins the [INT-SVC-001 configuration](../../configs/INT-SVC-001.json) by SHA-256 and inherits its LinearSVC parameters and the PR5 character vectorizer. The runner verified the complete fitted estimator parameter configuration against INT-SVC-001, including L2 penalty, squared hinge, dual solver, one-vs-rest, `C=1.0`, tolerance `1e-4`, iteration limit 1,000, intercept settings, no class weighting, and seed 42. It retained the original character 3–5 TF-IDF settings, input `utt`, and the same train-only fit and dev-only transform procedure. The character vocabulary size matched the baseline in both runs.

Both pipelines use the audited MASSIVE 1.0 `az-AZ` release (CC BY 4.0) with the same ordered official train/dev IDs (11,514 / 2,033) and the same 60 labels derived from train. Input verification checks AUD-001, the archive, JSONL and license hashes, and the saved baseline runs. Test rows are skipped before validation and model processing. No test predictions or scores were produced. The scorer reuses the PR5 accuracy, fixed-label macro-F1 and exact train-text overlap definitions, including a zero score where a label has no support or predictions under the scorer's `zero_division=0` policy.

The new branch tokenizes with Unicode `(?u)\b\w+\b`, retaining single-character tokens and separating punctuation from word runs. It uses word unigrams and bigrams, preserves case and diacritics, and applies no stemming, transliteration, stop-word filtering, or external normalization. The vectorizer fits its own vocabulary and IDF solely on training text. It uses sublinear TF, smoothed IDF, `min_df=1`, and per-row L2 normalization.

Each TF-IDF branch retains its L2 normalization and enters the sparse concatenation at weight 1.0. Final L2 normalization gives the classifier a unit-norm row whenever at least one branch is nonzero, matching the total magnitude of a nonzero character-only row. When both branches are nonzero, each contributes norm `1/√2` and half the squared norm. If one branch is empty, the other retains unit norm. An entirely empty row remains zero. The weight and normalization policy was fixed before smoke, with no dev adjustment. The character block's share of the combined row is therefore smaller when both blocks are active, even though its own vectorizer settings are identical. The comparison measures that complete representation change.

## Measured results

| Dev subset | Records | Character correct / accuracy | Hybrid correct / accuracy | Character macro-F1 | Hybrid macro-F1 |
|---|---:|---:|---:|---:|---:|
| Official dev | 2,033 | 1,716 / 84.4073% | **1,720 / 84.6040%** | 0.8228981629 | **0.8243775119** |
| Exact text absent from training | 1,871 | 1,560 / 83.3779% | 1,565 / 83.6451% | 0.8087411982 | 0.8087832954 |
| Exact text seen in training | 162 | 156 / 96.2963% | 155 / 95.6790% | 0.6582671958 | 0.6546041921 |

The full gains were **0.1968 percentage points** in accuracy and **0.0014793490** in macro-F1. Unseen-text accuracy gained five correct predictions, while unseen-text macro-F1 rose by only **0.0000420972**. The seen-text group lost one correct prediction. These subgroups are descriptive.

| Full run | Character-only | Hybrid |
|---|---:|---:|
| Character / word / total features | 80,622 / 0 / 80,622 | 80,622 / 37,727 / 118,349 |
| LinearSVC iterations | 47 | 46 |
| Fit (feature construction and classifier) | 2.204418 s | 2.669625 s |
| Dev transform and prediction | 0.065932 s | 0.154444 s |
| Peak process RSS through prediction | Unmeasured | 371.328125 MiB |

Both fits completed without convergence warnings. Timing uses single wall-clock observations from separate runs with one numerical thread on the Apple M1. RSS is the high-water mark of the whole process through dev prediction, including interpreter, dependencies and loaded data. It does not estimate incremental model memory. See the [run manifest](../../results/INT-HYBRID-001/full/run.json) and [complete per-class metrics and confusion matrix](../../results/INT-HYBRID-001/full/metrics.json).

### Paired changes and intent diagnostics

Across the same 2,033 dev IDs, **106 predictions changed**, including 39 corrections and 35 regressions. Both models got 1,681 records right and 278 wrong. Record `806` changed from `audio_volume_mute` to the correct `audio_volume_down`. Record `147` changed from the correct `play_music` to `music_likeness`. F1 rose for 18 classes, stayed equal for 19 and declined for 23. See the [ID-level comparison](../../results/INT-HYBRID-001/full/comparison.json).

| Intent | Dev support | Precision, char → hybrid | Recall, char → hybrid | F1, char → hybrid |
|---|---:|---:|---:|---:|
| `general_quirky` | 105 | 0.5161 → 0.5789 | 0.4571 → 0.4190 | 0.4848 → 0.4862 |
| `qa_definition` | 55 | 0.8704 → 0.8136 | 0.8545 → 0.8727 | 0.8624 → 0.8421 |
| `qa_factoid` | 90 | 0.6837 → 0.6875 | 0.7444 → 0.7333 | 0.7128 → 0.7097 |

The hybrid predicted `general_quirky` less often for definitions (2 → 1) and factoids (10 → 8), but more often sent gold `general_quirky` utterances to `qa_definition` (4 → 6). The main `general_quirky` → `qa_factoid` confusion remained at 10.

- **Calendar:** query → set worsened from 5 to 6, while set → query improved from 8 to 6.
- **Email:** query → send improved from 3 to 1, and send → query improved from 2 to 0.
- **Music:** query → play improved from 4 to 3, while play → query stayed at 0. Both classes had slightly lower F1 overall.
- **Alarm:** query → set stayed at 2, and set → query stayed at 1.

These are descriptive model errors against official labels. See the [top confusions and overlap diagnostics](../../results/INT-HYBRID-001/full/error-analysis.json) and [ID-keyed dev predictions](../../results/INT-HYBRID-001/full/predictions.jsonl).

## Reproduction and verification

The full fit started on November 22, 2026 at 11:31:19 p.m. Eastern Time from Git base `d1b07c4657e57805a520b3cfbd97b589609fc4b6` with the new experiment files in the working tree. It used the existing CPython 3.12.4 macOS arm64 environment with scikit-learn 1.5.2 and the [pinned PR5 dependencies](../../requirements-pr5.txt). The manifest records settings, input/source/output hashes, feature counts, environment, and timing. Its source hashes cover the baseline metric configuration and scorer. No dependency installation or model export was needed.

```bash
export PYTHONPATH=src
python3 -B -m unittest discover -s tests -p 'test_intent_hybrid.py' -v
python3 -B -m unittest discover -s tests
python3 -B -m azmassive.intent_hybrid --config configs/INT-HYBRID-001.json --mode smoke
python3 -B -m azmassive.intent_hybrid --config configs/INT-HYBRID-001.json --mode smoke --verify
python3 -B -m azmassive.intent_hybrid --config configs/INT-HYBRID-001.json --mode full
python3 -B -m azmassive.intent_hybrid --config configs/INT-HYBRID-001.json --mode full --verify
```

For a new fit, set a fresh `--output` directory and pass the same path to the verifier. The 180-train/59-dev smoke run finished before full training, and its scores served only as a pipeline check. All 43 repository tests passed, including five focused tests for Unicode tokens, normalization, train-only fitting, fixed-label scoring, and ID alignment. Both runs verified without retraining. An independent count-based calculation reproduced the scores. The PR5, character-only LinearSVC, both CRF runs and the slot-preparation audit hashes verified again. [Smoke verification](../../results/INT-HYBRID-001/smoke/verification.json) · [Full verification](../../results/INT-HYBRID-001/full/verification.json) · [Independent counts](../../results/INT-HYBRID-001/count-verification.json)

## Decision and joint-evaluation readiness

The hybrid improves both primary dev point estimates, but four net fixes, almost unchanged unseen-text macro-F1, and higher feature and prediction cost provide limited evidence for replacing the simpler character-only default. We retain this mixed result without retuning. There is no uncertainty estimate, repeated-fit result, held-out result, or deployment assessment.

The artifacts are ready to support a **separate joint dev evaluation** of independently fitted intent and slot models. The full intent predictions cover exactly the same 2,033 unique IDs as both verified CRF outputs. A separate evaluator still needs to join by ID, decode slots with the established IOB2 policy, and predeclare joint exact match as both the intent and the complete typed span set being correct on the same utterance. The default intent choice can remain INT-SVC-001, with INT-HYBRID-001 as a predeclared sensitivity comparison. Any use of SLOT-CRF-002 should retain its 300-iteration convergence limitation. No joint experiment or test scoring took place here.
