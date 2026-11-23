# SLOT-CRF-002 — CRF optimization budget follow-up

Does increasing the CRFsuite L-BFGS cap from 100 to 300 iterations improve optimizer termination and official development slot-filling performance?

## Protocol and reproduction

The comparison uses the committed [SLOT-CRF-001 baseline](SLOT-CRF-001.md) and changes only `crf.parameters.max_iterations` from 100 to 300. The configuration's experiment ID and output directory also change to keep artifacts separate. The runner, effective settings other than the cap, Python seed of 42, and original JSONL order are shared. The optimizer has no seed setting. The baseline and follow-up use CRFsuite `crf1d`, L-BFGS, `c1=0.0`, and `c2=1.0`.

Both runs use official MASSIVE 1.0 `az-AZ` (CC BY 4.0), with 11,514 train and 2,033 dev utterances. The runner skips test records before preprocessing. Both runs use the [audited slot preparation](SLOT-PREP-001.md), including `word_punctuation` tokenization, strict alignment, and exact IOB2 reconstruction. The features and scoring are identical to the baseline. Exact spans require the same slot type and both character boundaries within the same record ID. The all-`O` reference and exact train-text overlap definitions are shared as well. The runner verifies hashes against AUD-001 and checks the observed train/dev sizes, token counts, and slot counts against SLOT-PREP-001.

The run uses the existing `.venv/slot-crf` environment with [pinned python-crfsuite 0.9.12](../../requirements-slot-crf.txt), CPython 3.12.4, macOS arm64, and an Apple M1 CPU. See SLOT-CRF-001 for environment setup, wheel provenance, and local dataset acquisition. The full [run manifest](../../results/SLOT-CRF-002/full/run.json) records the start time, Git base, source and artifact hashes, and optimizer settings. The baseline fit predates this experiment. Wall time and peak process RSS are single measurements from separate runs, with possible machine load variation.

Commands from the repository root:

```bash
export PYTHONPATH=src
.venv/slot-crf/bin/python -B -m unittest discover -s tests -p 'test_slot_crf.py' -v
python3 -B -m unittest discover -s tests -v
.venv/slot-crf/bin/python -B -m azmassive.slot_crf --config configs/SLOT-CRF-002.json --mode smoke
.venv/slot-crf/bin/python -B -m azmassive.slot_crf --config configs/SLOT-CRF-002.json --mode smoke --verify
.venv/slot-crf/bin/python -B -m azmassive.slot_crf --config configs/SLOT-CRF-002.json --mode full
.venv/slot-crf/bin/python -B -m azmassive.slot_crf --config configs/SLOT-CRF-002.json --mode full --verify
```

The runner refuses to overwrite saved results. For a new run, supply a fresh `--output` path and use the same path for verification. The transient CRF binary is deleted by the runner, while its hash is recorded. No training data or model binaries are included in the experiment outputs.

## Measured results

The full run started on November 22, 2026 at 11:01:08 p.m. Eastern Time from Git base `7562d834a0b965dd034d356fca9eb561a1712e88`. The smoke run used the same first 256 train and 64 dev records as SLOT-CRF-001. It met the stopping criterion at iteration 72 and reproduced the baseline smoke predictions, metrics, diagnostics, and transient model hash.

| Full run | SLOT-CRF-001 | SLOT-CRF-002 |
|---|---:|---:|
| Last L-BFGS iteration | 100 | 300 |
| Termination | Iteration cap | Iteration cap |
| CPU fit time | 119.701208 s | 363.853453 s |
| Dev feature extraction and prediction | 0.263374 s | 0.274512 s |
| Peak process RSS | 169.31250 MiB | 166.90625 MiB |
| Exact typed-span micro precision | 0.7798053528 | 0.8004881025 |
| Exact typed-span micro recall | 0.6432513798 | 0.6583040642 |
| Exact typed-span micro F1 | 0.7049766291 | **0.7224669604** |
| Utterance exact match | 1,359/2,033 (66.8470%) | **1,399/2,033 (68.8146%)** |
| True positives / false positives / false negatives | 1,282 / 362 / 711 | 1,312 / 327 / 681 |
| Invalid IOB2 transitions | 0 | 0 |

CRFsuite reported “L-BFGS terminated with the maximum number of iterations” in both full runs. The follow-up did **not** satisfy the optimizer's configured stopping criteria. It gained **1.7490 percentage points** of dev micro-F1 and **1.9675 points** of utterance exact match for an additional 244.1522 s of fit time (3.04× total). Peak process RSS stayed in a similar range. The all-`O` reference is unchanged, with zero span F1 and 647/2,033 exact matches. See [dev metrics and per-slot counts](../../results/SLOT-CRF-002/full/metrics.json).

| Dev subgroup | Records | Precision, 001 → 002 | Recall, 001 → 002 | Micro F1, 001 → 002 | Exact matches, 001 → 002 |
|---|---:|---:|---:|---:|---:|
| Exact text seen in train | 162 | 0.8878504673 → 0.9333333333 | 0.9047619048 → 0.9333333333 | 0.8962264151 → 0.9333333333 | 143 → 150 |
| Exact text unseen in train | 1,871 | 0.7722836695 → 0.7913950456 | 0.6287076271 → 0.6430084746 | 0.6931386861 → 0.7095265926 | 1,216 → 1,249 |

Both subgroups improved. Of the net 40 additional exact matches, 33 came from the unseen group. The subgroups differ in text and slot composition, so their differences are descriptive.

| Slot | Gold support | Precision, 001 → 002 | Recall, 001 → 002 | F1, 001 → 002 |
|---|---:|---:|---:|---:|
| `event_name` | 160 | 0.6788321168 → 0.6739130435 | 0.5812500000 → 0.5812500000 | 0.6262626263 → 0.6241610738 |
| `place_name` | 200 | 0.7714285714 → 0.8214285714 | 0.6750000000 → 0.6900000000 | 0.7200000000 → 0.7500000000 |
| `time` | 138 | 0.6379310345 → 0.6583333333 | 0.5362318841 → 0.5724637681 | 0.5826771654 → 0.6124031008 |

`event_name` retained 93 true positives and added one false positive. `place_name` gained three true positives and removed ten net false positives. `time` gained five true positives and removed one net false positive. All 55 per-slot comparisons are saved in [comparison.json](../../results/SLOT-CRF-002/comparison.json).

## Prediction and error changes

Raw BIO tags changed on **370 of 11,570 dev tokens** across **196 of 2,033 utterances**. The decoded span sets changed on those same 196 utterances. Eighty utterances became exact, 40 lost exact match, 1,319 were exact in both runs, and 594 had errors in both. The follow-up added 78 correct spans while losing 48 previously correct spans, and added 64 spurious spans while removing 99 previously spurious ones.

Dev utterances with errors fell from 674 to 634. Missed gold spans fell from 711 to 681, and spurious spans fell from 362 to 327. Missed spans with an exact-boundary wrong-type prediction fell from 54 to 47. Missed spans with a same-type overlapping wrong-boundary prediction fell from 151 to 147. These diagnostics can overlap. Record `113` gained an exact `player_setting` span after the predicted end moved from character 24 to 29. Record `242` lost its previously exact `place_name` span. See [ID-keyed predictions](../../results/SLOT-CRF-002/full/predictions.jsonl) and [error diagnostics](../../results/SLOT-CRF-002/full/diagnostics.json).

The comparison joins raw predictions by verified dev ID and decodes them with the existing scorer and original token offsets. Per utterance, it compares the two typed-span sets to each other and to gold. Added and removed predictions are the directed set differences, and exact-match transitions are computed from equality to gold in each run. Metric and error deltas are follow-up minus baseline. The comparison file includes input hashes, changed IDs, per-slot metrics, and example offsets.

## Verification and conclusion

All 7 focused CRF tests and all 34 repository tests passed. The [smoke verifier](../../results/SLOT-CRF-002/smoke/verification.json) and [full verifier](../../results/SLOT-CRF-002/full/verification.json) passed. The comparison also re-verified both baseline artifacts, checked that only the cap differed in the effective optimizer parameters, and matched the dataset, source, seed, and environment records between experiments. No fit or verification command failed.

The 300-iteration configuration improves **measured dev performance**, including on unseen exact text. The experiment establishes neither convergence nor held-out generalization. Two budgets and one full fit per budget give no measurement of gains beyond 300. The stronger observed dev score makes 300 the better reference for slot accuracy on this split, but the roughly threefold fit cost and unknown marginal gains do not justify another cap increase by default.

The recommended next experiment is a single predeclared feature ablation at the fixed 300-iteration budget, targeting the still frequent `event_name` omissions. Compare against SLOT-CRF-002 on the same dev split and keep test held out. This is proposed work.
