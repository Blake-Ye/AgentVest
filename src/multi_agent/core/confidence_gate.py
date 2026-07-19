from __future__ import annotations

from dataclasses import dataclass

from multi_agent.core.review_contracts import GateDecision, ReviewContract, ReviewToolSummary


@dataclass(frozen=True)
class ConfidenceGatePolicy:
    min_evidence_coverage_ratio: float = 0.60
    min_financial_coverage_score: float = 0.60
    min_trust_score: int = 60

    @classmethod
    def default(cls) -> "ConfidenceGatePolicy":
        return cls()

    def evaluate(self, summary: ReviewToolSummary | ReviewContract) -> GateDecision:
        if isinstance(summary, ReviewContract):
            summary = summary.to_review_tool_summary()

        blocking_reasons = list(summary.blocking_reasons)
        rerun_reasons: list[str] = []
        gate_financial_coverage_score = (
            summary.gate_financial_coverage_score
            if summary.gate_financial_coverage_score is not None
            else summary.financial_coverage_score
        )

        if summary.evidence_coverage_ratio < self.min_evidence_coverage_ratio:
            rerun_reasons.append("evidence_coverage_ratio<0.60")
        if summary.financial_coverage_score < self.min_financial_coverage_score:
            rerun_reasons.append("financial_coverage_score<0.60")
        if gate_financial_coverage_score < 1.0:
            rerun_reasons.append("gate_financial_coverage_incomplete")
        if summary.critical_conflict_count > 0:
            blocking_reasons.append("critical_conflict_count>0")
        if summary.market_policy_violations:
            blocking_reasons.append("market_policy_violations>0")
        if summary.unsupported_critical_claims:
            blocking_reasons.append("unsupported_critical_claims>0")

        trust_score = int(
            max(
                0,
                min(
                    100,
                    summary.evidence_coverage_ratio * 50
                    + summary.financial_coverage_score * 50,
                ),
            )
        )
        if trust_score < self.min_trust_score:
            rerun_reasons.append("trust_score<60")

        if blocking_reasons:
            final_decision = "blocked"
        elif rerun_reasons:
            blocking_reasons = rerun_reasons
            final_decision = "rerun"
        else:
            final_decision = "passed"

        passed = final_decision == "passed"
        return GateDecision(
            passed=passed,
            final_decision=final_decision,
            trust_score=trust_score,
            blocking_reasons=blocking_reasons,
        )
