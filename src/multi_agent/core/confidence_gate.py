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
        if contract.decision.gate_outcome == "block":
            blocking_reasons.append("reviewer_requested_block")

        repair_actions = [
            *contract.repair_actions,
            *diagnostics.repair_actions,
            *self._tool_health_actions(bundle, contract),
        ]
        if contract.decision.gate_outcome == "rerun" and not contract.repair_actions:
            rerun_reasons = contract.rerun_reasons or ["Reviewer requested evidence repair."]
            repair_actions.extend(
                RepairAction(
                    target="data_quality_reviewer",
                    code="reviewer_rerun_requested",
                    instruction=reason,
                )
                for reason in rerun_reasons
            )
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

    @staticmethod
    def _tool_health_actions(
        bundle: ResearchEvidenceBundle, contract: ReviewContract
    ) -> list[RepairAction]:
        raw_statuses = {item.tool_name: item.status for item in bundle.tool_health}
        raw_degraded = sorted(
            tool_name for tool_name, status in raw_statuses.items() if status == "degraded"
        )
        raw_failed = sorted(
            tool_name for tool_name, status in raw_statuses.items() if status == "failed"
        )
        reviewed_degraded = set(contract.tool_health_summary.degraded_tools)
        reviewed_failed = set(contract.tool_health_summary.failed_tools)
        reviewed_status = contract.tool_health_summary.overall_status
        actions: list[RepairAction] = []
        if raw_degraded:
            actions.append(
                RepairAction(
                    target="data_quality_reviewer",
                    code="degraded_tool_health",
                    sources=raw_degraded,
                    instruction="修复或复核降级工具的数据后重新审查。",
                )
            )
        critical_targets = {
            "sec_company_facts": "fundamental_analyst",
            "sec_filing": "fundamental_analyst",
            "quote": "market_validation_analyst",
        }
        for tool_name in raw_failed:
            target = critical_targets.get(tool_name, "data_quality_reviewer")
            actions.append(
                RepairAction(
                    target=target,
                    code="critical_tool_failed",
                    sources=[tool_name],
                    instruction="修复失败工具并补齐受影响证据后重新审查。",
                )
            )
        declared_problem_tools = reviewed_degraded | reviewed_failed
        raw_problem_tools = set(raw_degraded) | set(raw_failed)
        if not _tool_health_summary_is_consistent(
            status=reviewed_status,
            failed_tools=reviewed_failed,
            degraded_tools=reviewed_degraded,
        ):
            actions.append(
                RepairAction(
                    target="data_quality_reviewer",
                    code="tool_health_contract_inconsistent",
                    instruction="修复审查契约中 overall_status 与工具列表的矛盾后重新审查。",
                )
            )
        if declared_problem_tools != raw_problem_tools or (
            raw_problem_tools and contract.tool_health_summary.overall_status == "healthy"
        ):
            actions.append(
                RepairAction(
                    target="data_quality_reviewer",
                    code="tool_health_disagreement",
                    sources=sorted(declared_problem_tools | raw_problem_tools),
                    instruction="使审查契约中的工具健康状态与原始证据记录一致后重新审查。",
                )
            )
        return actions

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
        normalized = action.model_copy(
            update={
                "fields": sorted(set(action.fields)),
                "sources": sorted(set(action.sources)),
            }
        )
        key = (
            normalized.target,
            normalized.code,
            tuple(normalized.fields),
            tuple(normalized.sources),
            normalized.instruction,
        )
        unique[key] = normalized
    return [unique[key] for key in sorted(unique)]


def _tool_health_summary_is_consistent(
    *, status: object, failed_tools: set[str], degraded_tools: set[str]
) -> bool:
    if status == "healthy":
        return not failed_tools and not degraded_tools
    if status == "failed":
        return bool(failed_tools)
    if status == "degraded":
        return bool(degraded_tools or failed_tools)
    return False
