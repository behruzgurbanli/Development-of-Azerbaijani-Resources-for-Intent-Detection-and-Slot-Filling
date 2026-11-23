# JOINT-EVAL-001 — Joint intent and slot evaluation

On November 22, 2026 (Eastern Time), the primary system **INT-SVC-001 + SLOT-CRF-002** achieved **1,206/2,033 (59.321200%)** joint utterance exact match. The hybrid comparison achieved **1,211/2,033 (59.567142%)**. The five-record difference is descriptive. No statistical significance analysis was performed.

## Protocol

The [configuration](../../configs/JOINT-EVAL-001.json) pairs three saved intent outputs with two saved CRF outputs, scoring all six combinations. All inputs come from the official MASSIVE 1.0 `az-AZ` dev split (CC BY 4.0). The runner verifies the component artifacts against their own manifests and recomputes component metrics. It joins on unique official dev IDs and rejects missing, duplicate, foreign, or inconsistent records. Saved gold intents must agree with the audited source, and intent and slot source text must match for each ID.

Gold slots are reconstructed with the original MASSIVE annotation parser, `word_punctuation` tokenization, and half-open Python character offsets. The saved raw CRF tags are decoded with the established IOB2 policy: a well-formed orphan or mismatched `I-type` starts a span and is logged, and malformed tags, unknown types, or wrong tag counts are rejected. Neither CRF needed repairs on dev. Intent macro-F1 uses the fixed 60 training labels with `zero_division=0`. Slot micro-F1 uses exact type and character boundaries. Slot utterance exact match requires equality of the full sets, including when both are empty.

Joint exact match requires both the official intent and the **entire typed span set** to match for the same record. The overlap subsets use exact original `utt` strings in the 11,514 official train records. They are descriptive subsets of the same 2,033-record dev set. The runs were independently trained before this experiment. No model was fitted, tuned, or rescored on test here.

## Component scores

| Intent model | Correct / accuracy | Macro-F1 |
|---|---:|---:|
| PR5 logistic regression | 1,628/2,033 (80.078701%) | 0.7461142414 |
| Character LinearSVC | 1,716/2,033 (84.407280%) | 0.8228981629 |
| Hybrid LinearSVC | 1,720/2,033 (84.604033%) | 0.8243775119 |

| Slot model | Exact-span micro-F1 | Utterance slot exact match |
|---|---:|---:|
| CRF-001 (100-iteration cap) | 0.7049766291 | 1,359/2,033 (66.847024%) |
| CRF-002 (300-iteration cap) | 0.7224669604 | 1,399/2,033 (68.814560%) |

## Joint scores and breakdown

| Intent + slot | Joint exact match | Intent only | Slots only | Neither | Seen exact | Unseen exact |
|---|---:|---:|---:|---:|---:|---:|
| Logistic + CRF-001 | 1,129 (55.533694%) | 499 | 230 | 175 | 136/162 (83.950617%) | 993/1,871 (53.073223%) |
| Logistic + CRF-002 | 1,166 (57.353665%) | 462 | 233 | 172 | 143/162 (88.271605%) | 1,023/1,871 (54.676644%) |
| Character + CRF-001 | 1,171 (57.599606%) | 545 | 188 | 129 | 137/162 (84.567901%) | 1,034/1,871 (55.264564%) |
| **Character + CRF-002 (primary)** | **1,206 (59.321200%)** | **510** | **193** | **124** | **144/162 (88.888889%)** | **1,062/1,871 (56.761090%)** |
| Hybrid + CRF-001 | 1,176 (57.845548%) | 544 | 183 | 130 | 136/162 (83.950617%) | 1,040/1,871 (55.585249%) |
| Hybrid + CRF-002 (comparison) | 1,211 (59.567142%) | 509 | 188 | 125 | 143/162 (88.271605%) | 1,068/1,871 (57.081774%) |

For the two CRF-002 pairings, **1,188** records were jointly correct for both, **18** only for the primary, **23** only for the hybrid, and **804** for neither. The hybrid gained six exact matches on unseen text and lost one on seen text, for a net gain of five (**0.245942 percentage points** overall). The full counts by pair, intent, scenario and overlap, along with the IDs that changed joint status, are in [metrics.json](../../results/JOINT-EVAL-001/metrics.json).

## Error review

The primary system had 827 joint errors. Slot mismatches were more frequent than intent mismatches on this dev set. On 510 records intent was correct and slots differed, while 193 had exact slots and incorrect intent. A further 124 failed both. By raw error count, the largest gold-intent groups were `calendar_set` (68/131 errors), `general_quirky` (66/105), and `play_music` (51/123). The largest scenario groups were `calendar` (127/280), `play` (124/260), and `qa` (98/214). Raw counts reflect both performance and group size.

| Official dev ID | Primary outcome | Azerbaijani-language review prompt |
|---|---|---|
| `30` | Intent only | The model correctly predicts `iot_cleaning` and misses the annotated `house_place` span for *xalçamı* at `[49,56)`. Check the treatment of the inflected noun as a location. |
| `42` | Slots only | The gold and predicted slot sets are both empty. The primary predicts `calendar_query` for “mənə boş yaddaşım barədə məlumat ver,” whose gold intent is `general_quirky`. Review the boundary between the intent categories. |
| `26` | Neither | For “mənzili təmizlə,” the primary predicts `calendar_remove` instead of `iot_cleaning` and misses `house_place [0,7)`. Check intent and annotated span separately. |
| `147` | Primary only, paired | The slot set is exact. Gold and character-only intent are `play_music`, while hybrid predicts `music_likeness` for “mən caza qulaq asmağı xoşlayıram.” Review whether the wording expresses playback or preference under the guidelines. |
| `806` | Hybrid only, paired | Both slot sets are empty. The hybrid fixes `audio_volume_mute` to the gold `audio_volume_down`. Review the strength of the down-versus-mute cue in the original utterance. |

The examples are descriptive, and their official labels were retained. All rows have IDs, gold intent and spans, model intent predictions, missed and spurious slot offsets, and joint outcomes in [diagnostics.jsonl](../../results/JOINT-EVAL-001/diagnostics.jsonl). The diagnostic file contains no utterance text.

## Verification and recommendation

All 50 tests passed, including seven new focused tests. The evaluator matched all component metrics, re-read and verified all saved joint artifacts, and used an independent aggregation path to intersect the correct intent-ID and exact slot-ID sets. That cross-check recomputed **1,206** for the primary system and agreed with the row-wise totals for all six pairs. The expected count supplied with the task is absent from the scorer. [Saved verification](../../results/JOINT-EVAL-001/verification.json)

Executed from the repository root with the existing Python 3.12.4/scikit-learn 1.5.2 environment:

```bash
export PYTHONPATH=src
python3 -B -m unittest discover -s tests -p 'test_joint_eval.py' -v
python3 -B -m unittest discover -s tests
python3 -B -m azmassive.joint_eval --config configs/JOINT-EVAL-001.json
python3 -B -m azmassive.joint_eval --config configs/JOINT-EVAL-001.json --verify
```

Use a fresh `--output` directory to reproduce and use the same path with `--verify`. The [manifest](../../results/JOINT-EVAL-001/run.json) records Git base `7c79e3d4bc59e1ae4622b4184b8ed1de50adbe25` and the evaluator, inputs, and artifact hashes.

Retain character-only LinearSVC + CRF-002 as the primary reporting pair, and report the hybrid as a sensitivity comparison. CRF-002 improved joint dev point estimates for each intent model, but reached its iteration cap. The five-record hybrid gain does not establish an advantage for final selection. Freeze a selection protocol before any held-out scoring, and review calendar and playback span errors alongside the general-question intent confusions. Generalization outside this dev split and optimization convergence are unresolved.
