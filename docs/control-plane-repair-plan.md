# AgentVest Control Plane Repair Implementation Plan

> **For agentic workers:** Execute inline with test-driven changes and verify every task independently.

**Goal:** Make canonical evidence and deterministic Python policy the only source of truth for review, rerun, report delivery, and metrics.

**Architecture:** Research agents produce artifacts and a typed `ResearchEvidenceBundle`; Python derives coverage, tool health, formal eligibility, and trust before an LLM reviewer is consulted. Review contracts carry bounded observations and stage-valid repair actions, while Flow owns routing and deterministic fallback after at most three executable repairs.

**Tech Stack:** Python 3.10-3.13, Pydantic v2, CrewAI 1.14.6, pytest.

## Global Constraints

- Do not add dependencies.
- Preserve the three terminal delivery states: `formal_report`, `evidence_limited_report`, `blocked_notice`.
- Only unresolved critical claims, policy violations, unsupported critical claims, or deterministic evidence conflicts may hard-block delivery.
- A repair action may run only when its target has the required capability in the current stage.
- After three unsuccessful executable repair rounds, deliver an evidence-limited report when allowed.

## Task 1: Harden Review Boundaries

**Files:** `src/multi_agent/core/review_contracts.py`, `src/multi_agent/tools/review_tools.py`, `tests/test_review_gate.py`, `tests/test_gate_snapshots.py`, `tests/test_crew_structure.py`

- [ ] Add failing tests for bounded ratios/counts, semantic delivery consistency, stage-valid repair targets, empty financial inputs, and unknown evidence references.
- [ ] Make reviewer tools consume a canonical evidence reference registry and compute gate fields from `FORMAL_GATE_REQUIRED_FIELDS` even when general fields are empty.
- [ ] Reject contradictory contracts in the task guardrail so CrewAI retries formatting/semantics instead of converting them into business blockers.
- [ ] Run focused tests and commit.

## Task 2: Make Gate and Report Review Deterministic

**Files:** `src/multi_agent/core/confidence_gate.py`, `src/multi_agent/flows/market_review_flow.py`, `src/multi_agent/config/tasks.yaml`, `tests/test_review_gate.py`, `tests/test_flow_routing.py`

- [ ] Add failing tests proving a report cannot pass with blocking reasons or repair actions.
- [ ] Build reviewer context from the persisted canonical bundle before reviewer-only execution.
- [ ] Keep report-writer instructions deferred, reject wrong-stage targets at the guardrail, and route only executable analysis repairs.
- [ ] Apply the same three-round report repair loop to every report mode.
- [ ] Run focused tests and commit.

## Task 3: Fix Evidence Health and Repair Capability

**Files:** `src/multi_agent/core/evidence.py`, `src/multi_agent/tools/investment_tools.py`, `src/multi_agent/crew.py`, `tests/test_tools.py`, `tests/test_review_gate.py`

- [ ] Add failing tests for repeated SEC candidate gaps and realistic multi-title Tavily results.
- [ ] Deduplicate SEC candidates and retain only actionable gaps.
- [ ] Evaluate Tavily health from sufficient confirmed core events rather than requiring every title to match another title exactly.
- [ ] Route SEC filing-content repairs only to an agent/tool that can fetch the requested filing URL; otherwise downgrade instead of retrying.
- [ ] Run focused tests and commit.

## Task 4: Unify Delivery Metrics and Remove Dead Paths

**Files:** `src/multi_agent/evaluation.py`, `src/multi_agent/recommendation.py`, `src/multi_agent/flows/market_review_flow.py`, `src/multi_agent/core/review_contracts.py`, `tests/test_evaluation.py`, `tests/test_flow_routing.py`

- [ ] Add failing tests for failed-run placeholders, blocked notices, and trust-score consistency.
- [ ] Derive report completeness and displayed trust from `FinalDecisionRecord`; failed runs without a decision cannot be complete or high trust.
- [ ] Delete the unused review normalizer and new-run-inaccessible prose/metrics gate fallbacks.
- [ ] Run the complete suite, replay the Apple artifacts through the typed gate, commit, and push.
