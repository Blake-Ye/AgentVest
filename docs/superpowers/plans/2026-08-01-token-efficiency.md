# Token Efficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce repeated LLM context and nested retry token use while preserving evidence and delivery validation.

**Architecture:** Keep full typed state and artifacts, but serialize stage-specific compact prompt payloads. Let Flow own semantic repair and cap CrewAI's internal retries.

**Tech Stack:** Python 3.12, Pydantic 2, CrewAI 1.14.6, pytest, YAML.

## Global Constraints

- Do not change persisted evidence, review, report, or final-decision schemas.
- Keep the Flow-owned maximum of three targeted repair rounds.
- Add no dependency and do not upgrade CrewAI.

---

### Task 1: Compact Report Context

**Files:**
- Modify: `src/multi_agent/flows/market_review_flow.py`
- Test: `tests/test_flow_routing.py`

**Interfaces:**
- Consumes: `ReportGenerationContext`
- Produces: `_writer_context_json(context) -> str` and `_report_review_context_json(context) -> str`

- [x] Add failing tests asserting required control/evidence fields remain while irrelevant full-state fields are absent and payload bytes shrink.
- [x] Run the focused tests and confirm they fail because both serializers are missing.
- [x] Implement the two smallest JSON projection helpers and route Writer/Reviewer inputs through them.
- [x] Run the focused tests and confirm they pass.

### Task 2: Remove Nested Retries

**Files:**
- Modify: `src/multi_agent/crew.py`
- Test: `tests/test_crew_structure.py`

**Interfaces:**
- Consumes: existing Agent and Task constructors
- Produces: agents with `max_retry_limit=1`; Writer tasks use zero guardrail
  retries and Reviewer tasks retain one

- [x] Add a failing topology test for the retry values.
- [x] Run it and confirm current values `3` and `2` fail the assertions.
- [x] Change only the shared constructor values; keep Flow rerun budgets unchanged.
- [x] Run the focused test and confirm it passes.

### Task 3: Bound Intermediate Output

**Files:**
- Modify: `src/multi_agent/config/tasks.yaml`
- Test: `tests/test_task_prompts.py`

**Interfaces:**
- Consumes: existing task prompts
- Produces: explicit concise output limits for each LLM stage

- [x] Add a failing prompt test for the required limits.
- [x] Run it and confirm the limits are absent.
- [x] Add concise limits without removing evidence, source, or schema requirements.
- [x] Run prompt tests, related tests, and the complete suite.
- [x] Compare compact/full Apple context bytes, run `git diff --check`, commit, and push.
