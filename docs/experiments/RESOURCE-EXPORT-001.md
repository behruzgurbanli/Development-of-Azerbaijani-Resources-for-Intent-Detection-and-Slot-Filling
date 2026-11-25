# RESOURCE-EXPORT-001 · AzMASSIVE source annotation export v0.1.0

**2026-11-24, Eastern time.** Initial local export of the official MASSIVE 1.0 Azerbaijani training and development annotations. The scope is a technical milestone for GAP-004. The thesis contribution and release scope remain open under GAP-001. The implementation was verified in a working tree based on `c416afba8c3065f2827537562cf95e360d5e309a`. [Gap register](../thesis-gap-register.md)

## Package and schema

The five-file package is under ignored `artifacts/RESOURCE-EXPORT-001/full/`. The code and aggregate manifest can be shared independently of the local data.

| File | Content |
|---|---|
| `source.jsonl` | Immutable source snapshot in original locale-file order restricted to train/dev. Each record has `schema_version`, original string `id`, `partition`, `locale`, `scenario`, exact `utt`, and `source_annotation` with original `intent`, exact `annot_utt`, and ordered typed `spans` (`start`, `end`, `label`). |
| `review-proposals.jsonl` | Empty in v0.1.0. The reserved proposal schema keys entries by the exact source snapshot hash, original ID and partition, with reviewer, date and reason. A null proposal field leaves that field unchanged. An empty proposed span list represents a proposal to remove all slots. Future review should use a separate local layer against the sealed snapshot. |
| `annotation.schema.json` | JSON Schema Draft 2020-12. The root validates a source JSON record. The proposal definition is at `#/$defs/reviewProposal`. Offset bounds, order, source equality and package-level uniqueness are checked by the exporter and verifier. |
| `LICENSE` | Byte-verified upstream license text. |
| `manifest.json` | Release/member identification, attribution, configuration and implementation hashes, source checksum provenance, counts, and output SHA-256 values. |

The offset unit is a zero-based, half-open Unicode code point index into the original `utt`. No normalization, label edits or duplicate merging were applied. The retained fields and markup reconstruct the entire task annotation for each exported record. Auxiliary upstream worker identifiers, judgments and slot collection metadata are outside this projection. The manifest ties each original ID to `1.0/data/az-AZ.jsonl` in the audited release. [Schema](../../schemas/azmassive-annotation-0.1.0.schema.json) · [Configuration](../../configs/RESOURCE-EXPORT-001.json) · [Aggregate manifest](../../results/RESOURCE-EXPORT-001/manifest.json)

The local bundle carries the original MASSIVE license and attribution to Amazon MASSIVE contributors and SLURP as the source resource. The derived format adds span offsets and limits the included partitions and fields. The archive checksum comes from AUD-001. This run reverified the audit metadata, the extracted locale JSONL, the license and the slot audit by bytes. The mixed locale file was streamed for its SHA-256 and for leading partition markers. Non-eligible rows were excluded before JSON decoding, annotation parsing or export. The held-out projections and per-record held-out artifacts were never opened. [AUD-001](../../data/metadata/AUD-001.json) · [MASSIVE paper](https://aclanthology.org/2023.acl-long.235/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Verification results

The synthetic suite checked Unicode and repeated values, source order, empty slots, duplicate IDs and inconsistent split markers, malformed source markup, unknown labels, changed export data and the exclusion of an undecodable synthetic test payload. The real smoke package verified one official train record with two spans before the full run.

| Partition | Unique records | Exact text and markup round trips | Typed spans | Slot-free records |
|---|---:|---:|---:|---:|
| Train | 11,514 | 11,514 | 11,202 | 3,755 |
| Dev | 2,033 | 2,033 | 1,993 | 647 |
| Total | 13,547 | 13,547 | 13,195 | 4,402 |

Counts match AUD-001 and SLOT-PREP-001. The exported train IDs, text, intents and markup independently matched the previous train-only projection in order. All dev IDs, scenarios, intents and complete span lists matched the published joint-evaluation diagnostics. The readback verifier recomputed the source records, rebuilt their markup from the exported spans, and validated the package bytes against the deterministic manifest. A fresh full generation matched all five files byte-for-byte. The verifier rejected an altered proposal file in a disposable copy. There are zero review proposals. The aggregate manifest SHA-256 is `cd9ffe45001db991ffb390311627b6c6e5328ee460089ef08fcb80f06d5135ec`. [Prior slot audit](SLOT-PREP-001.md) · [Dev reference](../../results/JOINT-EVAL-001/diagnostics.jsonl)

All four focused tests and all 71 repository tests passed. Static freeze preparation verified 107 pins. Git ignore checks covered all five local package files. No training or scoring was performed.

### Publication verification

Reran the four focused tests and all 71 repository tests. Verified the 107 frozen pins, the public manifest and its recorded implementation and package hashes, and Git ignore rules for the local datasets, proposal files and review workbook. Compared the original full and repeat bundles byte-for-byte. In a disposable ignored directory, replayed export and verification on a train/dev-only input reconstructed from the sealed snapshot. All five files matched and the counts agreed with AUD-001 and SLOT-PREP-001. This replay reused the original audited source hashes. It did not reopen the mixed release or the held-out files.

## Reproduction

The exporter uses Python's standard library. Verified on CPython 3.12.4, Darwin arm64. Run from the repository root with the audited input paths in the configuration. New generations require a fresh directory. `verify` checks the full sealed initial package, including the empty proposal file.

```bash
PYTHONPATH=src python3 -B -m unittest discover -s tests -p 'test_resource_export.py' -v
PYTHONPATH=src python3 -B -m azmassive.resource_export smoke
PYTHONPATH=src python3 -B -m azmassive.resource_export export
PYTHONPATH=src python3 -B -m azmassive.resource_export verify
PYTHONPATH=src python3 -B -m azmassive.resource_export export --run-name repro
diff -qr artifacts/RESOURCE-EXPORT-001/full artifacts/RESOURCE-EXPORT-001/repro
PYTHONPATH=src:.venv/slot-crf/lib/python3.12/site-packages python3 -B -m unittest discover -s tests -q
PYTHONPATH=src python3 -B -m azmassive.eval_freeze prepare
```

## Remaining decisions

These checks establish fidelity to the upstream source. Semantic quality, conflicting annotation interpretations and a nonempty proposal workflow still require human review. The proposal schema reserves a format, while reviewer import, adjudication and validation of nonempty proposal files remain future work. Before treating this as the finalized thesis resource, record the agreed contribution and release scope, complete and validate semantic judgments, finish the annotation guidelines and data statement, and verify any approved release package from a clean checkout. No scope approval or human correction is recorded by this export.
