# Azerbaijani semantic review · draft guidelines v0.1

**Prepared 2026-11-23. Unreviewed draft.** Bahruz is the intended sole reviewer. All prepared judgment fields are blank. These instructions distinguish an upstream annotation, a reviewer interpretation and a proposed change. The official data and frozen scores remain the benchmark reference. [Gap register](thesis-gap-register.md)

## Sample and private sheets

From the repository root, run `PYTHONPATH=src python3 -B -m azmassive.annotation_review` with the original local release in the paths used by [SLOT-PREP-001](../configs/SLOT-PREP-001.json). The standard-library generator checks the source-audit, locale-file and slot-audit SHA-256 pins, filters the mixed release by partition before validating or parsing annotations, and reconstructs spans left to right. It accepts only a fresh output directory. Use a fresh checkout to reproduce once review has begun. [Generator](../src/azmassive/annotation_review.py)

| Private sheet under ignored `artifacts/THESIS-GAP-001/` | Train | Dev | Selection and intended inference |
|---|---:|---:|---|
| `stratified.csv` | 120 | 60 | Intent-stratified coverage. Two per training intent. One per available dev intent, plus a second `lists_query`. Dev has no `audio_volume_other`. Describe these 180 cases by split. |
| `targeted.csv` | 20 | 1 | Ten members of four disagreement groups, nine rare-slot examples and two boundary examples. Use to explain specific decisions. |

Within a split, visit supported intents round-robin until the quota is reached. Order intents and candidate IDs by ascending SHA-256 of UTF-8 `42|key`, breaking ties by the original key. Sample without replacement. The unit is an official record, so repeated text can occur. Equal-intent sampling changes the class mixture. Neither cohort estimates a population error rate.

Take the first four complete disagreement groups in ascending UTF-8 text-hash order whose IDs are all outside the stratified sample. For each slot with at most ten **training spans**, take the lowest-ranked eligible training ID outside the sample. Include the two prior audit boundary prompts (IDs 1146 and 49) if outside the sample. Merge reasons when one targeted ID qualifies twice. The four groups are a review starting point among 50 identified by the audit. Inspect all variants within a selected group before deciding. [Audit and inventory](../results/SLOT-PREP-001/audit.json)

The local `annotation-review.xlsx` presents the same two sheets for editing. Keep one working copy and preserve the generated CSVs as the blank snapshot. When importing CSV directly, use UTF-8 and set the columns to text to preserve literal strings and IDs. Keep completed sheets and any free-text examples private. The public record should contain the protocol version, cohort sizes, decision counts, limitations and approved non-identifying findings.

## Review procedure

1. Read the original utterance in Azerbaijani and state the requested action. Compare it with the source intent and scenario. Record plausible alternate readings where context is insufficient. A different model prediction alone provides no linguistic evidence.
2. For each annotated argument, compare its role, type and surface extent with the intended request. Check omitted and spurious arguments, repeated values, adjacent spans, multiword names, punctuation and the empty-slot case. Review identical text with different intents in its scenario context.
3. Use zero-based, half-open Python Unicode character offsets in the exact original string. Include whitespace internal to a span. Preserve the source spelling, case and diacritics. Inspect case suffixes and clitics in context. In the prior audit ID 1146 includes an inflection inside a device span, and ID 49 includes punctuation in a time-zone span. Both are prompts, not a general decision about Azerbaijani morphology. Record a reason for every proposed boundary or label change.
4. Enter `retain`, `propose_change` or `uncertain` separately for intent and slots. Record any proposed intent or **complete** replacement span list (including `[]` for no slots) and the linguistic reason. Use `unresolved_question` when the evidence does not decide. Sign/date only personally reviewed rows.
5. Pilot one disagreement group and a few stratified examples. Record policy changes with dated versions, then revisit any earlier rows affected. Summarize the two cohorts separately, counting uncertain and missing judgments explicitly. A sole review supports no inter-annotator agreement estimate. Keep official annotations immutable.

## Sheet schema

| Fields | Meaning |
|---|---|
| `reason_selected`, `partition`, `id` | Cohort rule, official split and original string record ID |
| `scenario`, `intent`, `utt`, `annot_utt`, `spans` | Source labels, original text and markup, ordered JSON array of objects with `start`, `end`, `label` |
| `intent_judgment`, `slot_judgment` | Independent reviewer decisions |
| `proposed_intent`, `proposed_spans` | Proposed interpretation against the source inventory, preserving offsets |
| `phenomenon`, `reason`, `unresolved_question` | Suggested descriptive codes include ambiguity, suffix boundary, missing/extra argument, type, spelling, duplicate-context or other. Document evidence and alternative readings. |
| `reviewer`, `review_date` | Human provenance and ISO date |

A completed review should verify each ID and all original fields against the pinned source, validate proposed bounds, ordering and types, and reconcile per-cohort counts before any derived release. This acceptance step awaits human input.
