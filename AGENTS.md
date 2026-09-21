# Master Thesis Repository Instructions

These instructions apply to every Codex task in this repository.

## Mission

Support a reproducible master's thesis in NLP, with particular interest in Azerbaijani language resources and methods. Optimize for correctness, transparency, reproducibility, and Bahruz's ability to understand and defend every result.

## Before changing anything

1. Inspect the relevant repository files, current Git status, and existing conventions.
2. Preserve user-authored and unrelated changes. Never discard or overwrite them.
3. State material assumptions. Ask only when a missing decision would significantly change the result.
4. Prefer the smallest coherent change that completes the requested task.

## Source of truth

- Repository code, configs, tracked results, and current files outrank remembered conversation details.
- Supervisor instructions and task requirements outrank these general defaults.
- Never invent completed work, citations, datasets, metrics, experiment outcomes, or supervisor feedback.
- Label proposed work, simulated examples, and unverified claims clearly.

## Repository structure

- Put reusable implementation in `src/` and exploratory work in `notebooks/`.
- Put tests in `tests/`, experiment configurations in `experiments/` or `configs/`, and generated summaries/figures in `results/`.
- Put process documentation and the research log in `docs/` when those paths exist.
- Do not commit private, licensed, or large raw datasets, credentials, tokens, model checkpoints, caches, or generated environments. Update `.gitignore` when needed.
- Avoid duplicating logic across notebooks and modules; refactor stable notebook code into tested modules.

## Reproducible experiments

Every experiment should record, where applicable:

- experiment ID and purpose;
- date/time and Git commit hash;
- dataset name, source, license, version, and split strategy;
- preprocessing and normalization choices;
- model and tokenizer names/versions;
- random seed(s);
- hyperparameters, hardware, and software environment;
- evaluation metrics and exact evaluation procedure;
- output paths, results, limitations, and failures.

Use configuration files rather than scattered hard-coded parameters. Fix random seeds where technically possible, but do not claim perfect determinism unless verified. Prevent train/validation/test leakage and document any exception.

## Coding and verification

- Write clear, modular Python with type hints where useful and concise docstrings for non-obvious behavior.
- Add or update tests for substantive logic and edge cases.
- Run the narrowest relevant checks first, then broader tests when practical.
- Report exactly what was run and what passed, failed, or could not be run.
- Do not silently change evaluation definitions or data splits to improve results.
- For long or expensive training, first validate the pipeline with a small smoke test unless the user explicitly requests the full run.

## Research records

After substantive work, update `docs/research-log.md` (or the repository's existing equivalent) with:

- what was attempted and completed;
- important decisions and their rationale;
- files/experiments affected;
- observed results and unresolved problems;
- next recommended step.

Keep entries factual and dated. Distinguish observations from interpretations.

## Git practice

- Keep changes focused and reviewable. Do not mix unrelated work in one commit.
- Codex is responsible for choosing a concise, specific Conventional Commit message from the actual diff. Do not ask Bahruz to provide or formulate the commit message.
- Use the most accurate type:
  - `feat:` for new functionality;
  - `fix:` for bug fixes;
  - `docs:` for documentation-only changes;
  - `test:` for tests;
  - `refactor:` for code restructuring without intended behavior changes;
  - `chore:` for tooling, setup, dependencies, or maintenance;
  - `data:` for dataset or annotation changes;
  - `eval:` for evaluation or experiment-related changes.
- Write the subject in imperative style and describe the concrete change, for example `feat: add Azerbaijani normalization pipeline` or `eval: add macro-F1 reporting`.
- Never use vague subjects such as `update`, `changes`, `setup`, `work`, or `fix stuff`.
- If a change contains multiple separable concerns, recommend separate commits and provide an appropriate message for each.
- Do not commit, push, rewrite history, delete branches, or open a pull request unless Bahruz explicitly requests that action.
- When a commit is authorized, inspect the final staged diff, exclude unrelated or sensitive files, choose the message according to these rules, and report the resulting commit hash.
- Before handoff, summarize changed files, validation results, and the recommended commit message when no commit was requested.

## Academic integrity and writing

- Help organize, explain, analyze, and edit, but do not misrepresent AI-generated text or code as independently produced work.
- Preserve traceability from claims to sources and from reported numbers to experiment artifacts.
- Use only verified citations; never fabricate bibliographic details.
- Flag copied text, licensing restrictions, privacy risks, possible bias, and ethical limitations.
- Write reports in Bahruz's voice only after his facts and interpretations are known; mark placeholders for personal reflection or missing evidence.

## Definition of done

A task is complete only when the requested change is implemented, relevant checks have been run or limitations disclosed, reproducibility records are updated when applicable, and the handoff explains what changed, what evidence supports it, and what remains unresolved.
