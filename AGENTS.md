# AGENTS.md

## Repository Expectations

This repository contains Flying Police, a drone security analysis prototype. Codex should treat
it as a Python application with deterministic alert rules, a LangChain agent
layer, local VLM processing, SQLite/Chroma storage, and Streamlit UI tooling.

## Project Map

- `main.py` is the batch pipeline entrypoint.
- `ui/app.py` is the Streamlit interface.
- `agent/` contains alert rules, LangChain tools, callbacks, and tracking logic.
- `pipeline/` owns frame/session orchestration.
- `vlm/` owns motion detection, lighting detection, and BLIP captions.
- `storage/` and `query/` own persistence and search.
- `tests/` contains unit, integration, and validation suites.
- `.agents/skills/` and `.codex/skills/` hold repo-local Codex skills.

## Codex Working Loop

Use the official Codex task shape for non-trivial work:

1. Identify the goal, relevant context, constraints, and done criteria.
2. Read the nearest code and tests before editing.
3. Prefer a short plan for multi-file changes.
4. Keep edits scoped to the requested behavior.
5. Run the narrowest useful verification before finishing.
6. Summarize changed files, checks run, and any remaining risk.

## Engineering Conventions

- Keep `config.py` as the single source of truth for environment variables,
  thresholds, model names, paths, and rule constants.
- Prefer typed Python interfaces and dataclasses where they make data flow
  clearer.
- Do not bypass deterministic rule decisions with LLM reasoning. Rules fire
  first; the agent may add narrative and context.
- Keep VLM work behind `vlm/` and avoid spreading model-specific logic into
  the agent, UI, or storage layers.
- Preserve ignored runtime data such as `.env`, `venv/`, `data/chroma/`,
  `data/events.db`, `data/sessions/`, and uploaded videos.
- Avoid committing large generated outputs unless they are intentional
  submission artifacts.

## Verification

Choose checks based on the changed surface:

- Pure logic: `pytest tests/unit -q`
- Pipeline behavior: `pytest tests/integration -q`
- Fixture integrity: `pytest tests/validation -q`
- Full local run: `python3 main.py`
- UI smoke test: `streamlit run ui/app.py`

If a check cannot run because dependencies, credentials, or sample videos are
missing, report that clearly and explain what was verified instead.

## Codex Skills And External Context

- Use the repo skill `loops-codex-community-hackathon-pune` for hackathon
  submission, Loops House project state, sponsor knowledge queries, or judging
  feedback.
- Use `doc-coauthoring` for substantial PRD, architecture, report, and proposal
  writing.
- Use project skills only when their descriptions match the task. Read the
  selected `SKILL.md` before acting.

## Review Checklist

Before finalizing Codex-generated changes, check:

- The diff matches the requested scope.
- Tests or manual checks cover the touched behavior.
- Secrets and local runtime files are not staged.
- README, architecture docs, or report text are updated when behavior changes.
- Git history tells a coherent implementation story when commits are requested.
