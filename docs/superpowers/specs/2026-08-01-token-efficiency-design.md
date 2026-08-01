# Token Efficiency Design

## Goal

Reduce LLM input and retry token usage without weakening evidence validation,
delivery gates, or the existing three-round targeted repair policy.

## Design

- Keep the complete `ResearchEvidenceBundle` and `ReviewContract` in Flow state
  and persisted artifacts. Only compact the JSON projected into LLM prompts.
- Give the Writer formal financial facts, market snapshots, independently
  confirmed events, actionable gaps, allowed claim IDs, canonical sources, and
  the delivery decision fields it must obey.
- Give the Logic Reviewer the delivery decision boundary and canonical ID
  registry alongside `REPORT_DOCUMENT_JSON`; do not resend raw research data.
- Keep analysis-review prose available, but remove repeated event corroboration
  URL lists and duplicate raw source references from its canonical JSON input.
- Limit intermediate markdown length and use one provider retry. Structured
  guardrails perform no hidden retries because Flow owns up to three targeted
  repair rounds.

## Safety

- Persisted evidence and report schemas do not change.
- Claim/source validation remains deterministic in `ReportDocument`.
- Compact projections retain every field used to authorize a delivery state.
- Tests compare compact and full payload size and assert required evidence and
  control fields remain present.

## Verification

- Focused tests for prompt projections, Crew retry settings, and prompt limits.
- Full test suite.
- Report byte-size reduction from an existing Apple fixture; no paid LLM run is
  possible until the configured provider account has available balance.
