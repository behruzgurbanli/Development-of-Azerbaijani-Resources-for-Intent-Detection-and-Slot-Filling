# SLOT-PREP-001 — Azerbaijani slot preparation

On November 20, 2026 (Eastern Time), the preparation pipeline reconstructed all **13,547** official MASSIVE 1.0 `az-AZ` training and development records and verified an exact IOB2 round trip for all **13,195** annotated spans. The run measured preprocessing and alignment coverage. It did not train or evaluate a model.

The [configuration](../../configs/SLOT-PREP-001.json) and [audit result](../../results/SLOT-PREP-001/audit.json) record the commands, input and source hashes, Git base revision, environment, split counts, slot inventory by frequency, and failures by record ID. The base was `5b6a7ed9fd6174919a5b2b262a0ad5ead384e1f1` with the new experiment in the working tree. The run used Python 3.12.4 on macOS arm64 with Unicode database 15.0.0 and standard-library code. No random sampling or dependency installation was required. Local elapsed time was 0.61 seconds before writing the manifest.

## Source and alignment policy

- Used the [official MASSIVE 1.0 release](https://github.com/alexa/massive) under CC BY 4.0. The local **AUD-001**, archive, JSONL, and license SHA-256 checks all matched. The JSONL hash was `9a643dbf6e07e90ba8b1554bd95e9946b42037bd52550aac90f4e774ee58ed8c`. Raw files stay under ignored `data/raw/`. Test rows were skipped by partition before annotation parsing or tokenization.
- Inspected the source for its literal `[label : value]` syntax before implementation. The parser builds the original text left to right and derives offsets from the length reconstructed so far. It compares the result exactly against `utt` and rejects malformed or unmatched brackets, missing labels or values, invalid edge whitespace within a value, and reconstruction differences. It does not search for slot values in the utterance. Offsets are zero-based, half-open Python character indices, with original Unicode, capitalization, punctuation, and whitespace preserved.
- Compared whitespace chunks (`\S+`), Unicode word runs plus separate punctuation/symbols (`\w+|[^\w\s]`), and a variant that keeps internal ASCII hyphens and straight or curly apostrophes. All tied on exact coverage. Selected **word runs plus separate punctuation/symbols** so punctuation can be a distinct CRF feature, with no corpus-specific suffix or punctuation rules. The latter two patterns rely on Python's Unicode-aware [regular expression semantics](https://docs.python.org/3.12/library/re.html).
- Tokenization uses only `utt`. Every annotated span must start at a token start and end at a token end. Whitespace between tokens is recovered through the original offsets. IOB2 uses `B-type` for the first token, `I-type` for continuations, and `O` outside spans. Adjacent same-type slots each start with `B-type`. Decoding rejects invalid transitions and must reproduce the original ordered spans and labels. A span crossing a token boundary is reported with record ID, offsets, and reason. No span is expanded or dropped.

## Measured results

| Partition | Reconstructed / records | Slot spans | Multiword spans | Spans with Unicode punctuation | Records with no slots | Slot types |
|---|---:|---:|---:|---:|---:|---:|
| Train | 11,514 / 11,514 | 11,202 | 5,344 | 261 | 3,755 | 55 |
| Dev | 2,033 / 2,033 | 1,993 | 949 | 41 | 647 | 53 |
| Total | 13,547 / 13,547 | 13,195 | 6,293 | 302 | 4,402 | 55 |

| Tokenizer | Train tokens | Dev tokens | Train alignment | Dev alignment |
|---|---:|---:|---:|---:|
| Whitespace | 65,664 | 11,436 | 100% | 100% |
| Unicode word runs + punctuation | 66,528 | 11,570 | 100% | 100% |
| Keep internal hyphens/apostrophes | 66,504 | 11,562 | 100% | 100% |

The rates are based on all records in each partition, including slot-free records. All methods also aligned 100% of parsed spans. There were zero markup, token alignment, and IOB2 round-trip failures. No observed span boundary split an attached suffix or required punctuation to be detached for alignment. For example, record `1146` includes the full inflected form *tozsoranı* in its device span, and record `49` contains punctuation inside a time-zone slot. The synthetic tests cover the out-of-span suffix and punctuation cases separately. All 53 dev types are in train. `music_album` and `transport_descriptor` have zero dev support.

## Annotation decisions for human review

Exact duplicate utterances have differing slot annotations in **50 groups**. Of the **144** text groups shared between train and dev, **15** contain differing slot annotations. The remaining **35** disagreement groups occur within a single partition. The [audit result](../../results/SLOT-PREP-001/audit.json) lists every affected record ID, partition, intent and typed span offsets in `duplicate_text_review.groups`. These are differences worth reviewing. Some may depend on intent or annotation guidelines. The original labels and official splits were retained.

| Representative record IDs | Question for Azerbaijani annotation review |
|---|---|
| `115`, `1367` | Should the direction *aşağı* count as `change_amount` in this volume command? |
| `5752`, `2470` | Should the inflected reference to the lights be annotated as `device_type`? |
| `5971`, `1196` | Is the generic word for weather a `weather_descriptor` in this question? |
| `5854`, `5855` | Should the robot vacuum phrase be a `device_type` span? |
| `2452`, `1271` | Does *dünyada* specify a `place_name` in the news query? |
| `2383`, `6158` | Should the standalone inflected vacuum reference be a `device_type`? |
| `1610`, `2749`, `2971`, `4170` | Should the coffee machine phrase be included as `device_type` consistently? |
| `2525`, `2742` | The same span is labeled `food_type` and `order_type`. Which meaning applies? |
| `2563`, `4717` | Should the morning and clock time be separate spans or one `time` span? |
| `6778`, `6924`, `7907` | Is the generic reference to a meeting an `event_name`? |
| `7631`, `7633` | Should the noon expression be one span, or should it omit the intervening numeral and use two spans? |
| `11260`, `11300` | Is the navigation verb also a `player_setting` in the episode command? |
| `11358`, `11394` | Should the next-podcast phrase count as a `podcast_descriptor`? |
| `11373`, `11553` | Is the short phrase the title of a podcast in context? |
| `14248`, `14755` | Should the landmark in the factoid question be annotated as `place_name`? |

## Reproduction and next steps

```bash
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
PYTHONPATH=src python3 -B -m azmassive.slot_prep --config configs/SLOT-PREP-001.json
```

All 27 tests passed, including 11 new tests. The audit command refuses to overwrite an existing report. Repeat using `--output` with a fresh path. A repeat run matched all fields after excluding the command and run timestamps/duration. The existing PR5-INT-001 smoke and full verifiers passed without retraining.

Train/dev is structurally ready for a CRF baseline. Use the selected tokenizer and freeze strict typed-span evaluation before training. Report micro precision, recall and F1, per-type support and utterance-level span exact match. Keep zero-support types explicit. Run an all-`O` reference and a small CRF smoke test before the full run. Report overlap diagnostics for dev without changing the primary split. Review the annotation disagreements before proposing any separate curated dataset, and keep official test evaluation for a later experiment.
