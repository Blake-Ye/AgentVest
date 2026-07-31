# AgentVest Report Audit And Directory Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the latest generated reports against the complete-report contract and leave one canonical, clean source tree without losing historical code or runtime artifacts.

**Architecture:** Preserve the dirty `main` workspace on a local backup branch, finish the isolated complete-report branch, then fast-forward `main` and remove the redundant worktree. Treat existing Apple, Tesla, and Microsoft outputs as legacy runtime data and archive them under ignored `var/archive/` rather than mixing them with future canonical runs.

**Tech Stack:** Git worktrees, Python 3.10+, pytest, CrewAI, JSON/Markdown report artifacts.

## Global Constraints

- Never discard uncommitted source changes; preserve them in `backup-pre-cleanup-20260730` first.
- Never commit `.env`, API keys, runtime logs, or generated reports.
- Existing Apple and Tesla reports are read-only audit evidence and must be archived, not rewritten.
- `main` must be the only remaining working source checkout after consolidation.

---

### Task 1: Audit Latest Reports

**Files:**
- Read: `var/runs/apple_inc__aapl/20260720_094741/*`
- Read: `var/runs/tesla_inc__tsla/20260720_111117/*`
- Read: `docs/superpowers/specs/2026-07-20-complete-report-pipeline-design.md`

**Interfaces:**
- Consumes: terminal decision, Markdown report, structured recommendation, structured report, and reviewer artifacts.
- Produces: a pass/fail judgment for each acceptance requirement and archive classification.

- [x] **Step 1: Verify terminal decision and delivery mode agree across artifacts.**
- [x] **Step 2: Verify seven canonical sections and source URLs exist.**
- [x] **Step 3: Verify structured summary, catalysts, risks, and validation fields are non-empty.**
- [x] **Step 4: Check numerical claims against values in the same report.**
- [x] **Step 5: Classify non-compliant runs as legacy evidence.**

### Task 2: Preserve Dirty Main Workspace

**Files:**
- Preserve: all tracked modifications and untracked source, test, plan, and debug-note files currently shown by `git status`.
- Exclude: `.env`, `.crewai_home/`, `.worktrees/`, and generated `var/` content.

**Interfaces:**
- Consumes: dirty `main` working tree at `88cc633`.
- Produces: local branch `backup-pre-cleanup-20260730` containing a recovery commit.

- [x] **Step 1: Create `backup-pre-cleanup-20260730` from current `main`.**
- [x] **Step 2: Stage tracked modifications plus untracked source/docs/tests only.**
- [x] **Step 3: Review the staged file list and confirm no secrets or runtime outputs are present.**
- [x] **Step 4: Commit the recovery snapshot (`dee1ef5`).**
- [x] **Step 5: Switch back to `main` and verify the source tree is restored to `88cc633`.**

### Task 3: Finish The Complete Report Branch

**Files:**
- Modify/commit existing changes: `src/multi_agent/evaluation.py`
- Modify/commit existing changes: `tests/conftest.py`
- Modify/commit existing changes: `tests/test_apple_pipeline.py`
- Modify/commit existing changes: `tests/test_evaluation.py`
- Modify: `src/multi_agent/core/formal_gate.py`
- Modify: `src/multi_agent/tools/official_sec.py`
- Modify: `src/multi_agent/tools/investment_tools.py`
- Test: `tests/test_review_gate.py`
- Add: `docs/superpowers/plans/2026-07-30-report-audit-directory-cleanup.md`
- Add: `docs/reports/legacy-report-audit-2026-07-30.md`

**Interfaces:**
- Consumes: four existing uncommitted hardening changes on `codex/complete-report-pipeline`.
- Produces: a clean branch whose commit can be fast-forwarded into `main`.

- [x] **Step 1: Add failing tests for duplicate required facts, cross-period facts, and quote currency mismatch.**
- [x] **Step 2: Run those tests and confirm each fails for the intended contract gap.**
- [x] **Step 3: Make Gate and semantic evaluation reject ambiguous, cross-period, or cross-currency evidence.**
- [x] **Step 4: Add a failing network-isolation test by blocking `urllib.request.urlopen` and raw sockets.**
- [x] **Step 5: Remove unconditional localhost debug HTTP calls and verify the network-isolated Apple test passes.**
- [x] **Step 6: Capture the actual in-memory Flow result in the Apple fixture rather than re-reading `final_decision.json`.**
- [x] **Step 7: Run `git diff --check` and focused Apple, Gate, and evaluation tests (`50 passed`).**
- [x] **Step 8: Run the full suite and investigate any reproducible order-dependent failure (`322 passed`).**
- [x] **Step 9: Review the exact staged diff and commit only after all tests pass.**

### Task 4: Consolidate To One Source Tree

**Files:**
- Update branch pointer: `main`
- Remove worktree: `.worktrees/complete-report-pipeline`

**Interfaces:**
- Consumes: clean `main` and clean `codex/complete-report-pipeline`.
- Produces: fast-forwarded `main` and no duplicate source checkout.

- [x] **Step 1: Fast-forward `main` to `codex/complete-report-pipeline`.**
- [x] **Step 2: Remove the clean complete-report worktree.**
- [x] **Step 3: Delete the now-merged local feature branch.**
- [x] **Step 4: Verify `git worktree list` shows only the project root.**

### Task 5: Separate Runtime And Debug Artifacts

**Files:**
- Modify: `.gitignore`
- Move locally: `var/runs/apple_inc__aapl` to `var/archive/legacy-pipeline-runs/`
- Move locally: `var/runs/tesla_inc__tsla` to `var/archive/legacy-pipeline-runs/`
- Move locally: `var/runs/microsoft_corporation__msft` to `var/archive/legacy-pipeline-runs/`
- Move locally: `.dbg/` and `var/logs/` to `var/archive/debug/`
- Create externally: `/private/tmp/AgentVest-runtime-pre-cleanup-20260730.tar.gz`
- Create externally: `/private/tmp/AgentVest-pre-cleanup-20260730.bundle`
- Add: `docs/reports/legacy-report-audit-2026-07-30.md`

**Interfaces:**
- Consumes: generated runtime files that currently appear in repository status.
- Produces: an empty canonical `var/runs/` ready for new pipeline outputs and a clearly labeled legacy archive.

- [x] **Step 1: Create checksummed external runtime and Git backups before moving files.**
- [x] **Step 2: Ignore `var/`, `.dbg/`, and common Python test caches.**
- [x] **Step 3: Move old run directories into the legacy archive without changing contents.**
- [x] **Step 4: Move debug outputs into the debug archive.**
- [x] **Step 5: Keep the audit result in versioned `docs/`; keep runtime data untracked and ignored.**

### Task 6: Final Verification

**Files:**
- Verify: repository root and full test suite.

**Interfaces:**
- Consumes: consolidated `main` and separated runtime artifacts.
- Produces: evidence that the source tree is clean and the canonical branch is testable.

- [x] **Step 1: Run focused report-contract and Apple pipeline tests (`49 passed`).**
- [x] **Step 2: Run the full pytest suite (`322 passed`).**
- [x] **Step 3: Run `git status --short --branch`.**
- [x] **Step 4: Run `git worktree list --porcelain`.**
- [x] **Step 5: Record compliance failures, archive locations, backup branch, and verification results.**
