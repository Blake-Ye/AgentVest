from __future__ import annotations

from dataclasses import dataclass

from multi_agent.core.evidence import ResearchEvidenceBundle
from multi_agent.core.formal_gate import FORMAL_GATE_REQUIRED_FIELDS, diagnose_formal_delivery
from multi_agent.core.review_contracts import (
    GateDecision,
    RepairAction,
    ReviewContract,
    ReviewToolSummary,
)


@dataclass(frozen=True)
class ConfidenceGatePolicy:
    min_evidence_coverage_ratio: float = 0.80
    min_financial_coverage_score: float = 0.80
    min_claim_binding_ratio: float = 0.75
    min_trust_score: int = 75

    @classmethod
    def default(cls) -> "ConfidenceGatePolicy":
        return cls()

    def evaluate(self, bundle: ResearchEvidenceBundle, contract: ReviewContract) -> GateDecision:
        """Evaluate only typed evidence and the strict reviewer contract for new runs."""
        diagnostics = diagnose_formal_delivery(bundle, FORMAL_GATE_REQUIRED_FIELDS)
        coverage = contract.coverage_summary
        blocking_reasons = [
            *diagnostics.blocking_reasons,
            *contract.blocking_reasons,
            *coverage.blocking_reasons,
            *coverage.market_policy_violations,
            *coverage.unsupported_critical_claims,
        ]
        if coverage.critical_conflict_count > 0:
            blocking_reasons.append("critical_conflict_count>0")
        if coverage.unresolved_critical_claim_count > 0:
            blocking_reasons.append("unresolved_critical_claims>0")
        if contract.delivery_eligibility.blocked_notice_required:
            blocking_reasons.append("delivery_blocked")

        repair_actions = [*contract.repair_actions, *diagnostics.repair_actions]
        trust_score = self.calculate_trust_score(bundle, contract)
        below_threshold = (
            coverage.evidence_coverage_ratio < self.min_evidence_coverage_ratio
            or coverage.financial_coverage_score < self.min_financial_coverage_score
            or coverage.claim_binding_ratio < self.min_claim_binding_ratio
            or trust_score < self.min_trust_score
        )
        if below_threshold:
            repair_actions.append(
                RepairAction(
                    target="data_quality_reviewer",
                    code="coverage_below_formal_threshold",
                    instruction="重新计算结构化覆盖率并补齐 claim-source 绑定。",
                )
            )

        if blocking_reasons:
            outcome = "blocked"
        elif repair_actions:
            outcome = "rerun"
        elif not contract.delivery_eligibility.formal_report_allowed:
            outcome = (
                "evidence_limited"
                if contract.delivery_eligibility.evidence_limited_report_allowed
                else "blocked"
            )
        else:
            outcome = "passed"
        return GateDecision(
            passed=outcome == "passed",
            final_decision=outcome,
            trust_score=trust_score,
            blocking_reasons=sorted(set(blocking_reasons)),
            repair_actions=_deduplicate_actions(repair_actions),
        )

    def calculate_trust_score(
        self, bundle: ResearchEvidenceBundle, contract: ReviewContract
    ) -> int:
        coverage = contract.coverage_summary
        base = (
            coverage.evidence_coverage_ratio * 35
            + coverage.financial_coverage_score * 35
            + coverage.claim_binding_ratio * 30
        )
        degraded_count = sum(item.status == "degraded" for item in bundle.tool_health)
        failed_count = sum(item.status == "failed" for item in bundle.tool_health)
        return int(max(0, min(100, base - degraded_count * 5 - failed_count * 20)))

    def evaluate_legacy(self, summary: ReviewToolSummary | ReviewContract) -> GateDecision:
        """Explicit compatibility path for historical summary-only artifacts."""
        if isinstance(summary, ReviewContract):
            summary = summary.to_review_tool_summary()
        blocking_reasons = list(summary.blocking_reasons)
        rerun_reasons: list[str] = []
        gate_financial_coverage_score = (
            summary.gate_financial_coverage_score
            if summary.gate_financial_coverage_score is not None
            else summary.financial_coverage_score
        )
        if summary.evidence_coverage_ratio < 0.60:
            rerun_reasons.append("evidence_coverage_ratio<0.60")
        if summary.financial_coverage_score < 0.60:
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
            max(0, min(100, summary.evidence_coverage_ratio * 50 + summary.financial_coverage_score * 50))
        )
        if trust_score < 60:
            rerun_reasons.append("trust_score<60")
        if blocking_reasons:
            outcome = "blocked"
        elif rerun_reasons:
            blocking_reasons = rerun_reasons
            outcome = "rerun"
        else:
            outcome = "passed"
        return GateDecision(
            passed=outcome == "passed",
            final_decision=outcome,
            trust_score=trust_score,
            blocking_reasons=blocking_reasons,
        )


def _deduplicate_actions(actions: list[RepairAction]) -> list[RepairAction]:
    unique: dict[tuple[object, ...], RepairAction] = {}
    for action in actions:
        key = (action.target, action.code, tuple(action.fields), tuple(action.sources), action.instruction)
        unique[key] = action
    return list(unique.values())
