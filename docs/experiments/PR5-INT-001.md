# PR5-INT-001: preliminary Azerbaijani intent classification

## Status and scope

Completed on **November 7, 2026, Eastern Time**. These are real, **intent-only development results**, not test, slot-filling, or joint NLU results. This record supplies experiment evidence for Progress Report V; it is not the 6–8-page report itself. The report requirements and November 8, 2026, 11:59 PM Eastern deadline were supplied by Bahruz in the task instruction.

The full run began at `2026-11-07T21:14:37.767382-05:00`. The base Git revision was `d38303b65b6279f5e52e08ae731d280ba4409fce`; implementation was uncommitted. Exact source-file hashes, selected record IDs, configuration, environment, timings, warnings, and output hashes are in [the run manifest](../../results/PR5-INT-001/full/run.json). HEAD alone does not identify this implementation. No commit, push, or pull request was made.

## Source verification and split policy

Source: the [official MASSIVE 1.0 release](https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz), Azerbaijani locale `az-AZ`, checked against the user-provided [AUD-001](../../data/metadata/AUD-001.json). The audit was preserved byte-for-byte.

| Artifact | Verified SHA-256 |
|---|---|
| AUD-001 JSON | `6c87b2b77d859f087341e72c728a26dca2c9efc505b899741caef813c8bbdcb8` |
| MASSIVE 1.0 archive | `7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8` |
| `1.0/data/az-AZ.jsonl` | `9a643dbf6e07e90ba8b1554bd95e9946b42037bd52550aac90f4e774ee58ed8c` |
| `1.0/LICENSE` | `c2e6ea015269147de02117ebdd91f30ef09831251f5345fa8365273b1db1d435` |

All hashes matched; the Azerbaijani file is 12,195,341 bytes. Only this file and the license were extracted. The archive and extracted files remain under ignored `data/raw/`. The dataset license is CC BY 4.0; preserve attribution and cite MASSIVE and its SLURP source when writing the report. See the [official project documentation](https://github.com/alexa/massive).

The loader validated core string fields, annotation container types, numeric-string unique IDs, locale, allowed partitions, intent/scenario consistency, train-label coverage, audited counts, NFC diagnostics, and exact duplicate diagnostics. This intent-focused validation does not implement slot extraction or repeat native-speaker annotation review.

| Partition | Records | Use |
|---|---:|---|
| Official train | 11,514 | Fit vocabulary, IDF, majority label, and logistic regression |
| Official dev | 2,033 | Predictions, metrics, and descriptive error analysis |
| Official test | 2,974 | Structural validation only; no training, tuning, prediction, or scoring |

There are 16,521 unique IDs, 60 training intents, and 18 scenarios. The development split contains 59 intents: `audio_volume_other` has zero support. No random re-splitting, deduplication, relabeling, data augmentation, or hyperparameter search was performed. Test records are not exposed by the loader's modeling interface.

Exact text diagnostics reproduce AUD-001: 576 duplicate-text groups overall, 317 spanning partitions, and 19 cross-partition groups containing multiple intents. There are **144 train/dev overlapping text groups**, including **12 with conflicting intents**, covering **162 development records**. Original partitions and records were retained. Exact overlap can affect apparent generalization; it does not by itself establish how an individual prediction was produced.

## Model selection and application

Logistic regression follows the methodology continuity supplied by Bahruz: TF-IDF + logistic regression for intents, CRF for slots, and a potential XLM-R joint model. Earlier Reports II–IV were not available in the repository and were not independently checked. Previous academic reports were not modified.

Character TF-IDF offers a small CPU baseline without pretrained checkpoints or language-specific tokenizers. Character fragments provide a hypothesis for handling subword/suffix variation, not an experimentally established morphology benefit. A regularized linear classifier is inexpensive, supports all training intents, and gives a useful reference before neural work. LinearSVC remains an unrun alternative. CRF and XLM-R components are not implemented.

The input is **`utt` only** and the target is **`intent`**. `annot_utt`, scenario, judgments, worker IDs, and other annotation metadata are excluded from features.

- TF-IDF: `analyzer="char"`, character n-grams 3–5, `lowercase=False`, `strip_accents=None`, `min_df=1`, `max_df=1.0`, no vocabulary cap, sublinear term frequency, smoothed IDF, L2 normalization, float64.
- Text: no external normalization, transliteration, stemming, spelling correction, or case folding. Azerbaijani characters and case are preserved. The scikit-learn character analyzer internally collapses repeated whitespace; it also creates n-grams across word boundaries.
- Logistic regression: L2 penalty, `C=1.0`, `solver="lbfgs"`, tolerance `1e-4`, maximum 1,000 iterations, intercept enabled, no class weighting. In scikit-learn 1.5.2 this solver uses multinomial loss for the 60-class problem. The run converged in **66 iterations**, with **80,622 features** and no captured warnings.
- Majority reference: the most frequent training label, **`calendar_set`**. Ties would be broken lexicographically.
- Seed: 42 for Python, NumPy, and the classifier parameter. LBFGS does not use that classifier random-state parameter; perfect cross-platform determinism is not claimed.
- CPU numerical thread pools limited to one using `threadpoolctl`; no GPU used.

Implementation references: [scikit-learn 1.5 logistic regression](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html) and [TF-IDF vectorizer](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). These document component behavior, not expected Azerbaijani performance.

## Preliminary quantitative results

| Model | Correct / dev records | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Majority class | 131 / 2,033 | 6.4437% | 0.00201787 |
| Character TF-IDF + logistic regression | 1,628 / 2,033 | **80.0787%** | **0.74611424** |

Macro-F1 is the unweighted mean across **all 60 training intents**, including zero F1 for the unsupported development class, with `zero_division=0`. It is reported on a 0–1 scale above; 0.74611424 equals 74.6114%. This explicit definition matters when comparing other reported MASSIVE scores. Accuracy uses all 2,033 official dev records.

Every class's precision, recall, F1, support, and the confusion matrix are in [metrics.json](../../results/PR5-INT-001/full/metrics.json). Predictions for both models are keyed by development record ID in [predictions.jsonl](../../results/PR5-INT-001/full/predictions.jsonl). Raw utterances are referenced by ID rather than copied into prediction artifacts. No trained-model serialization was saved; reproduction retrains from the verified inputs and configuration.

| Measured operation | Seconds |
|---|---:|
| Input hash verification and loading | 0.1965 |
| Majority fit and prediction | 0.0009 |
| TF-IDF + logistic regression fit | 24.8101 |
| TF-IDF transform + dev prediction | 0.0730 |
| Metrics and descriptive error analysis | 0.0892 |
| Total inside runner before final manifest write | 25.5460 |

The total includes setup and output writing, but excludes interpreter startup, dataset download, unit tests, and subsequent artifact verification. No peak-memory measurement was made.

The preceding smoke run used the first three training examples per class and first development example per available class in original file order: **180 train / 59 dev**. It converged without warnings (fit 0.2040 seconds). Its scores are pipeline diagnostics and must not be presented as preliminary full-data results.

## Error analysis and limitations

The model made **405 development errors**. The largest directed confusion counts were:

| Gold | Predicted | Count |
|---|---|---:|
| `qa_factoid` | `general_quirky` | 15 |
| `calendar_query` | `calendar_set` | 12 |
| `qa_definition` | `general_quirky` | 11 |
| `general_quirky` | `qa_factoid` | 9 |
| `music_query` | `play_music` | 9 |

These observations motivate review of general-question categories and query/action distinctions. They do not establish annotation mistakes or a linguistic cause. `general_quirky` has precision 0.3469 and recall 0.6476 on 105 examples. Three supported classes have zero F1 and only two examples each: `cooking_query`, `general_greet`, and `music_dislikeness`; their estimates are fragile. The zero F1 of `audio_volume_other` reflects absent gold development support, not measured recall on examples of that class.

For qualitative follow-up, the first 20 errors in original dev order are recorded by ID in [error-analysis.json](../../results/PR5-INT-001/full/error-analysis.json). They are not a random sample. Examples inspected include IDs `24` (`iot_hue_lightdim` → `general_quirky`), `167` (`general_greet` → `general_quirky`), and `237` (`alarm_query` → `alarm_set`). Their meanings and annotation decisions require native-speaker review before any correction or causal categorization.

| Diagnostic dev subgroup | Records | Correct | Accuracy | Macro-F1 over fixed 60 labels |
|---|---:|---:|---:|---:|
| Exact text also in train | 162 | 153 | 94.4444% | 0.64007317 |
| Exact text absent from train | 1,871 | 1,475 | 78.8348% | 0.72755761 |

These are descriptive subgroups of the original evaluation, not replacement splits or retrained models. Their class compositions differ, so subgroup macro-F1 values are not directly comparable as a causal overlap effect. The primary score remains the full official development score. Different IDs across partitions do not guarantee distinct text.

Limitations: one configuration and seed, no hyperparameter comparison or confidence interval, no independent test evaluation, no out-of-domain/native-user evaluation, and no robustness or annotation-quality study. This is a localized virtual-assistant benchmark experiment, not a new Azerbaijani resource or a deployed application. Slot filling, CRF training, XLM-R modeling, slot-level F1, and joint exact-match evaluation remain outstanding.

## Environment and exact execution commands

No dependency installation was needed. The existing executable was `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` (CPython **3.12.4**, macOS **arm64**). Required package versions are pinned in [requirements-pr5.txt](../../requirements-pr5.txt): NumPy 2.0.2, SciPy 1.14.1, scikit-learn 1.5.2, joblib 1.4.2, threadpoolctl 3.5.0. Tests use standard-library `unittest`; pytest is not required. The minimal Python 3.11 Conda environment was not rebuilt and `environment.yml` was not changed.

Hardware: Apple M1 MacBook Pro, eight CPU cores, eight GPU cores, 8 GB unified memory; approximately 32 GiB free disk at inspection. All experiment computation used CPU. No neural dependency, checkpoint, or accelerator setup was needed.

The following commands were executed from the repository root, in this order for acquisition, tests, smoke, full run, and verification. `python3` resolved to the executable above. The test and extraction commands were independent; tests completed before smoke.

```bash
mkdir -p data/raw
curl --fail --location --retry 2 --output data/raw/amazon-massive-dataset-1.0.tar.gz https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
PYTHONPATH=src python3 -B -m azmassive.dataset --audit data/metadata/AUD-001.json --archive data/raw/amazon-massive-dataset-1.0.tar.gz --output-dir data/raw/massive-1.0
PYTHONPATH=src python3 -B -m azmassive.experiment --config configs/PR5-INT-001.json --smoke
PYTHONPATH=src python3 -B -m azmassive.experiment --config configs/PR5-INT-001.json
PYTHONPATH=src python3 -B -m azmassive.verify --run results/PR5-INT-001/smoke --output results/PR5-INT-001/smoke-verification.json
PYTHONPATH=src python3 -B -m azmassive.verify --run results/PR5-INT-001/full --output results/PR5-INT-001/full-verification.json
```

The run manifests also record commands using the absolute interpreter path. The runners refuse to overwrite existing run directories. For another run, use `--output` with a new directory. The extraction command verifies the existing archive and selected members before any write, and accepts existing matching extracted files without rewriting them. Do not re-download over an existing archive without first deciding whether another acquisition is necessary.

All **16 automated tests passed**. Smoke and full artifact verification both passed: source/output/input hashes, record alignment, gold labels, both models' complete metrics, duplicate diagnostics, and error-analysis aggregates were checked. Verification recomputes metrics from saved predictions without retraining. A separate standard-library `Counter` calculation independently reproduced both accuracies and macro-F1 values to within `1e-12` (including the 60-label denominator). Final source-hash and unchanged-AUD checks passed. Results occupy 1,110,300 bytes; no large model artifact was produced. No experiment command failed. Prior academic reports and the user-provided audit were preserved.

Additional read-only checks included `git status --short`, `git rev-parse HEAD`, `git log -3 --oneline`, `system_profiler SPHardwareDataType SPDisplaysDataType -detailLevel mini`, `df -h .`, `git diff --check`, and `git check-ignore data/raw/amazon-massive-dataset-1.0.tar.gz data/raw/massive-1.0/az-AZ.jsonl`. Python inspection commands printed audit hashes/metadata, dependency versions, constructor parameters, and selected error examples; they did not train additional models.

The post-run verifier was added after training; its own source hash and the run-manifest hash are recorded in the two verification JSON files. The training-time hashes cover the actual loader, classifier, evaluation code, tests, requirements, and config used at run time.

## Next steps

1. Use these verified intent-only results, rationale, limitations, and source references in Progress Report V; format the report separately to the supplied requirements.
2. Review the confusing intent pairs and conflicting train/dev text groups with Azerbaijani-language judgment. Keep annotations and official splits unchanged unless a separately documented study is approved.
3. Implement and validate slot parsing/alignment, then a CRF baseline and slot-level evaluation. Add joint intent-and-slot exact match only after both components exist.
4. Consider LinearSVC or XLM-R as separately configured future experiments. Freeze any selection protocol before eventual test evaluation.
