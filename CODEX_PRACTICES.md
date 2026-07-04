# Codex Practices For This Codebase

This file captures how Codex should be used on Flying Police so future work stays
fast, reviewable, and consistent with the architecture.

## Prompt Shape

For implementation tasks, give Codex four things:

- Goal: the behavior to add, fix, or explain.
- Context: files, commands, logs, screenshots, or docs that matter.
- Constraints: architecture boundaries, security requirements, or style rules.
- Done when: tests, manual checks, or observable behavior that prove completion.

Example:

```text
Goal: Add a new HIGH alert when a vehicle remains near the gate for more than
60 seconds.
Context: agent/alert_rules.py, agent/vehicle_tracker.py, tests/unit/test_alerts.py
Constraints: keep deterministic rules ahead of agent narration; constants live
in config.py.
Done when: unit tests cover normal, threshold-edge, and duplicate-alert cases.
```

## When To Plan First

Ask Codex to plan before coding when the change touches more than one subsystem,
such as VLM detection plus alerting plus storage. A good plan should name the
files to inspect, the expected data flow, the test surface, and the rollback
risk.

Small edits, isolated tests, docs tweaks, and obvious bug fixes can go straight
to implementation.

## Durable Guidance

`AGENTS.md` is the repo-level contract for Codex. Keep it short and practical:

- commands that actually work here
- architecture boundaries Codex should not cross
- files that should never be committed
- test expectations
- repeated mistakes worth preventing

When Codex repeats the same mistake twice, update `AGENTS.md` instead of
re-explaining the same instruction in every prompt.

## Validation Habits

Match verification to risk:

- rule logic changes need unit tests
- tracker or pipeline changes need integration tests
- fixture or sample-data changes need validation tests
- UI changes need a Streamlit smoke test
- report or README changes need a quick render/read-through check

Codex work is not complete just because files changed. It is complete when the
change is checked, summarized, and any unverified risk is called out.

## Skills And Hackathon Work

Use repo-local skills for repeated workflows:

- `doc-coauthoring` for PRDs, architecture notes, reports, and proposals
- `loops-codex-community-hackathon-pune` for Loops House project status,
  artifacts, sponsor knowledge, and evaluation prompts

Skills should stay focused. If a workflow grows scripts, examples, or external
resources, put them beside the skill instead of burying all detail in a prompt.

## Security And Data Boundaries

Codex should not stage local secrets, runtime databases, uploads, model caches,
or generated Chroma indexes. Use `.gitignore` as the first line of defense, then
check `git status --short` before every commit.

For commands that need network access, browser login, or protected file writes,
ask for scoped approval and explain why the command is needed.

## Source Basis

These practices are adapted for this repository from the official Codex manual
sections on best practices, prompting, `AGENTS.md`, and agent skills.
