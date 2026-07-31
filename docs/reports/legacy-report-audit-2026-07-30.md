# Legacy Report Audit: Apple And Tesla

Audit date: 2026-07-30

## Scope

- Apple Inc. (AAPL), run `20260720_094741`
- Tesla, Inc. (TSLA), run `20260720_111117`
- Complete-report contract introduced by the complete report pipeline

The audit parsed all 14 JSON artifacts and inspected each Markdown review and report. The source artifacts were not rewritten.

## Verdict

| Run | Declared terminal state | Audit result | Classification |
| --- | --- | --- | --- |
| Apple | `passed` / `formal_report` | Fail | Legacy, not suitable for formal delivery |
| Tesla | `evidence_limited` / `evidence_limited_report` | Fail | Legacy, not suitable as a canonical limited report |

## Blocking Findings

1. Apple's financial analysis combines the six-month revenue value of USD 254.940 billion with FY2025 Services revenue of USD 109.158 billion, producing a false Services share of 42.81%. Same-period values imply about 23.92% for the six-month comparison or 26.23% for FY2025 (109.158 / 416.161).
2. Apple's structured recommendation has empty summary, catalysts, and risks, while its structured report has only four sections, zero validated sections, and zero citations. The terminal decision still marks the run as passed with a trust score of 100.
3. Apple's structured stance is `sell`, but the prose and logic reviewer explicitly avoid a buy/sell conclusion.
4. Tesla's terminal artifacts declare `evidence_limited_report`, while the Markdown report declares `formal_report`; its logic review recommends a rerun.
5. Tesla maps USD 92.614 billion of FY2025 sales-and-services revenue to `segment_revenue_services`, although the artifact itself notes that this is not a Services segment value.
6. Neither final Markdown report contains the seven exact canonical section headings or a direct source index. Both structured outputs report a citation count of zero.
7. Reviewer decisions and final delivery states are therefore not closed consistently across the artifact set.

## Disposition

These runs are retained only as regression evidence under the local ignored archive `var/archive/legacy-pipeline-runs/`. New reports must pass the typed delivery validator, the shared formal evidence gate, canonical section validation, direct URL citation checks, and terminal-state consistency checks before delivery.
