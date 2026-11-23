# INT-SVC-001 — LinearSVC intent comparison

On November 22, 2026 (Eastern Time), the fixed LinearSVC pipeline improved official dev accuracy by **4.3286 percentage points** and macro-F1 by **0.0767839215** over [PR5-INT-001](PR5-INT-001.md). This compares two intent classifiers with the same feature representation and data. It does not measure held-out or deployment performance.

## Protocol

The [configuration](../../configs/INT-SVC-001.json) pins the original PR5 config by SHA-256 and adds one LinearSVC setting. Both runs use MASSIVE 1.0 `az-AZ`, licensed under CC BY 4.0, with the original 11,514 train and 2,033 dev records. The loader verifies AUD-001, archive, Azerbaijani JSONL, and license hashes. It filters by partition before validating or featurizing samples. The saved PR5 manifests establish that both models use exactly the same ordered train and dev IDs. Predictions from each model are joined by ID, with duplicate, missing, foreign, and gold-label mismatches rejected.

The existing PR5 vectorizer and scorer are reused unchanged. Character 3–5 TF-IDF uses `utt` only, with case and diacritics preserved, no external normalization, sublinear TF, smoothed IDF, and L2 normalization. Both train-only vectorizer fits produced 80,622 features in the full runs. Intent accuracy uses the full official dev set. Macro-F1 is the unweighted mean over all 60 training intents, with `zero_division=0`, including `audio_volume_other`, which has no gold dev examples. Exact-text overlap compares original utterance strings without deduplication and uses the same label inventory for subgroup scores.

The fixed candidate uses LinearSVC with L2 penalty, squared-hinge loss, `C=1.0`, dual optimization, one-vs-rest multiclass, tolerance `1e-4`, and at most 1,000 iterations. It fits an intercept with scaling 1 and no class weights. The dual solver is a reasonable starting point because the feature dimension exceeds the number of training examples. The dual solver uses `random_state=42`, matching the experiment seed. The old classifier uses multinomial logistic loss, so the comparison covers the classifier as a whole, including multiclass strategy. No hyperparameter search or dev-based setting adjustment was run. [LinearSVC API for scikit-learn 1.5](https://scikit-learn.org/1.5/modules/generated/sklearn.svm.LinearSVC.html)

## Results

| Dev scope | Records | Logistic correct / accuracy | LinearSVC correct / accuracy | Logistic macro-F1 | LinearSVC macro-F1 |
|---|---:|---:|---:|---:|---:|
| Official dev | 2,033 | 1,628 / 80.0787% | **1,716 / 84.4073%** | 0.7461142414 | **0.8228981629** |
| Exact text absent from train | 1,871 | 1,475 / 78.8348% | **1,560 / 83.3779%** | 0.7275576136 | **0.8087411982** |
| Exact text seen in train | 162 | 153 / 94.4444% | 156 / 96.2963% | 0.6400731676 | 0.6582671958 |

LinearSVC converged without warnings in 47 iterations. Full TF-IDF plus classifier fitting took **2.2044 s** compared with **24.8101 s** for the saved logistic run. Dev transform and prediction took 0.0659 s versus 0.0730 s. Both used one numerical thread on the Apple M1 CPU with the same package versions. These are single wall-clock measurements from separate runs, so the observed fit ratio of about 11.25× is descriptive.

F1 increased for 45 labels, decreased for 12, and stayed equal for three. Selected class results appear below. The largest gains include several classes with only two to five dev examples, so their estimates are fragile. See [per-class precision, recall, F1, support, and confusion matrix](../../results/INT-SVC-001/full/metrics.json).

| Intent | Dev support | Logistic F1 | LinearSVC precision | LinearSVC recall | LinearSVC F1 |
|---|---:|---:|---:|---:|---:|
| `iot_hue_lighton` | 5 | 0.3333 | 1.0000 | 1.0000 | 1.0000 |
| `cooking_query` | 2 | 0.0000 | 1.0000 | 0.5000 | 0.6667 |
| `general_quirky` | 105 | 0.4518 | 0.5161 | 0.4571 | 0.4848 |
| `qa_factoid` | 90 | 0.6569 | 0.6837 | 0.7444 | 0.7128 |
| `calendar_query` | 102 | 0.7136 | 0.7525 | 0.7451 | 0.7488 |
| `email_addcontact` | 5 | 0.9091 | 0.8000 | 0.8000 | 0.8000 |

### Paired prediction and confusion analysis

Predictions differed on **265** record IDs. LinearSVC corrected 144 records missed by logistic and lost 56 records logistic got right, for a net gain of 88. The unseen-text subgroup accounted for 85 of those net gains. Both models were correct on 1,572 records and wrong on 261. See the [paired outcomes, ID-level changes, and confusion deltas](../../results/INT-SVC-001/full/comparison.json).

- `qa_definition` → `general_quirky` fell from 11 to 2, `calendar_query` → `calendar_set` fell from 12 to 5, and `qa_factoid` → `general_quirky` fell from 15 to 10. The reciprocal `general_quirky` → `qa_factoid` confusion rose from 9 to 10, and `general_quirky` → `news_query` rose from 3 to 7. Precision on `general_quirky` improved while recall dropped.
- Record `24` changed from `general_quirky` to the correct `iot_hue_lightdim`. Record `806` regressed from the correct `audio_volume_down` to `audio_volume_mute`. Both utterances were absent verbatim from train. These are model disagreements against the official labels. They do not establish annotation errors. See the [overlap scores and top confusions](../../results/INT-SVC-001/full/error-analysis.json).

## Verification and reproduction

The full run started at 11:21:15 p.m. Eastern Time on November 22, 2026 from Git base `7562d834a0b965dd034d356fca9eb561a1712e88`, with the new experiment files in the working tree. It used CPython 3.12.4 on macOS arm64 with scikit-learn 1.5.2 and the existing [PR5 dependency versions](../../requirements-pr5.txt). No packages were installed. The [full manifest](../../results/INT-SVC-001/full/run.json) records provenance and file hashes, effective classifier parameters, times, and environment. The original test partition was skipped before modeling. File-level checksums cover the release archive and the complete locale file.

```bash
export PYTHONPATH=src
python3 -B -m unittest discover -s tests -p 'test_intent_svc.py' -v
python3 -B -m unittest discover -s tests
python3 -B -m azmassive.intent_svc --config configs/INT-SVC-001.json --mode smoke
python3 -B -m azmassive.intent_svc --config configs/INT-SVC-001.json --mode smoke --verify
python3 -B -m azmassive.intent_svc --config configs/INT-SVC-001.json --mode full
python3 -B -m azmassive.intent_svc --config configs/INT-SVC-001.json --mode full --verify
```

Use a fresh `--output` path to rerun training and pass that same path to its verifier. The smoke run trained on 180 records and evaluated 59 records using the PR5 prefix sampling per class, and passed before full training. All 38 repository tests passed, including four focused new tests. Both saved runs passed verification, including metric and confusion recomputation and matching of baseline train and dev IDs. A separate count-based calculation reproduced accuracy and macro-F1 from the raw predictions. No training or verification commands failed. [Smoke verification](../../results/INT-SVC-001/smoke/verification.json) · [Full verification](../../results/INT-SVC-001/full/verification.json) · [Independent counts](../../results/INT-SVC-001/count-verification.json)

## Decision and limitations

LinearSVC is the preferable **current dev baseline** for intent classification. It improves both primary metrics, including among utterances without exact training matches. This compares one fixed configuration of each classifier on one benchmark split. Repeated-fit variation, confidence intervals, probability calibration, and held-out performance were not measured. The next meaningful comparison is a predeclared feature ablation with LinearSVC fixed: current character TF-IDF versus character plus word TF-IDF, with particular attention to the remaining query/action and general-question confusions. Test remains held out during that comparison.
