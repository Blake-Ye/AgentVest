from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel, ConfigDict, Field

from crewai.tools import BaseTool

from multi_agent.core.formal_gate import FORMAL_GATE_REQUIRED_FIELDS


def _is_missing_extracted_value(value: Any) -> bool:
    return value is None or value == "" or value is False


class ReviewClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(description="The normalized claim text under review.")
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="Evidence references supporting the claim.",
    )
    conflict_level: str | None = Field(
        default=None,
        description="Optional conflict severity for cross-source checks.",
    )


class EvidenceCoverageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[ReviewClaim] = Field(
        default_factory=list,
        description="Claim review items with explicit claim text and evidence references.",
    )


class EvidenceCoverageTool(BaseTool):
    name: str = "evidence_coverage_tool"
    description: str = "根据 claim 与 evidence_refs 计算证据覆盖率和无证据结论。"
    args_schema: Type[BaseModel] = EvidenceCoverageInput

    def _run(self, claims: list[ReviewClaim | dict[str, object]]) -> dict[str, object]:
        normalized_claims = [self._normalize_claim(item) for item in claims]
        total_claims = len(normalized_claims)
        unsupported_claims = [
            item.claim.strip()
            for item in normalized_claims
            if not self._normalize_refs(item.evidence_refs)
        ]
        covered_claims = total_claims - len(unsupported_claims)
        coverage_ratio = 0.0 if total_claims == 0 else covered_claims / total_claims
        return {
            "evidence_coverage_ratio": coverage_ratio,
            "unsupported_claims": unsupported_claims,
            "covered_claim_count": covered_claims,
            "total_claim_count": total_claims,
        }

    def _normalize_refs(self, raw_refs: object) -> list[str]:
        if not isinstance(raw_refs, list):
            return []
        return [str(item).strip() for item in raw_refs if str(item).strip()]

    def _normalize_claim(self, claim: ReviewClaim | dict[str, object]) -> ReviewClaim:
        if isinstance(claim, ReviewClaim):
            return claim
        return ReviewClaim.model_validate(claim)


class CrossSourceConsistencyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[ReviewClaim] = Field(
        default_factory=list,
        description="Claim review items with optional conflict metadata.",
    )


class CrossSourceConsistencyTool(BaseTool):
    name: str = "cross_source_consistency_tool"
    description: str = "汇总跨来源冲突，输出关键冲突数量。"
    args_schema: Type[BaseModel] = CrossSourceConsistencyInput

    def _run(self, claims: list[ReviewClaim | dict[str, object]]) -> dict[str, object]:
        normalized_claims = [self._normalize_claim(item) for item in claims]
        critical_conflicts = [
            item.claim.strip()
            for item in normalized_claims
            if (item.conflict_level or "").strip().lower() == "critical"
        ]
        return {
            "critical_conflict_count": len(critical_conflicts),
            "critical_conflicts": critical_conflicts,
        }

    def _normalize_claim(self, claim: ReviewClaim | dict[str, object]) -> ReviewClaim:
        if isinstance(claim, ReviewClaim):
            return claim
        return ReviewClaim.model_validate(claim)


class ToolInvocationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(description="Invoked tool name.")
    allowed: bool | None = Field(
        default=None,
        description="Whether the invocation complies with market tool policy.",
    )


class MarketToolPolicyAuditInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_invocations: list[ToolInvocationRecord] = Field(
        default_factory=list,
        description="Tool invocation records with explicit tool name and policy result.",
    )


class MarketToolPolicyAuditTool(BaseTool):
    name: str = "market_tool_policy_audit_tool"
    description: str = "检查工具调用是否违反市场/来源策略。"
    args_schema: Type[BaseModel] = MarketToolPolicyAuditInput

    def _run(
        self, tool_invocations: list[ToolInvocationRecord | dict[str, object]]
    ) -> dict[str, object]:
        normalized_invocations = [self._normalize_invocation(item) for item in tool_invocations]
        violations = [
            item.tool_name.strip() or "unknown_tool"
            for item in normalized_invocations
            if item.allowed is False
        ]
        return {
            "market_policy_violations": violations,
            "market_policy_violation_count": len(violations),
        }

    def _normalize_invocation(
        self, invocation: ToolInvocationRecord | dict[str, object]
    ) -> ToolInvocationRecord:
        if isinstance(invocation, ToolInvocationRecord):
            return invocation
        return ToolInvocationRecord.model_validate(invocation)


class FinancialFieldCompletenessInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_fields: list[str] = Field(
        default_factory=list,
        description="Financial fields that should be present in the analysis output.",
    )
    extracted_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted field map keyed by financial field name.",
    )
    gate_required_fields: list[str] = Field(
        default_factory=list,
        description="Minimum required fields for a formal report gate decision.",
    )


class FinancialFieldCompletenessTool(BaseTool):
    name: str = "financial_field_completeness_tool"
    description: str = "根据必需字段与已提取字段计算财务字段完整度。"
    args_schema: Type[BaseModel] = FinancialFieldCompletenessInput

    def _run(
        self,
        required_fields: list[str],
        extracted_fields: dict[str, Any],
        gate_required_fields: list[str] | None = None,
    ) -> dict[str, object]:
        if not required_fields:
            return {
                "financial_coverage_score": 1.0,
                "missing_fields": [],
                "required_field_count": 0,
                "gate_financial_coverage_score": 1.0,
                "gate_missing_fields": [],
            }

        missing_fields = [
            field_name
            for field_name in required_fields
            if field_name not in extracted_fields
            or _is_missing_extracted_value(extracted_fields[field_name])
        ]
        coverage_score = (len(required_fields) - len(missing_fields)) / len(required_fields)
        gate_required_fields = list(FORMAL_GATE_REQUIRED_FIELDS)
        gate_missing_fields = [
            field_name
            for field_name in gate_required_fields
            if field_name not in extracted_fields
            or _is_missing_extracted_value(extracted_fields[field_name])
        ]
        gate_coverage_score = (
            1.0
            if not gate_required_fields
            else (len(gate_required_fields) - len(gate_missing_fields)) / len(gate_required_fields)
        )
        return {
            "financial_coverage_score": coverage_score,
            "missing_fields": missing_fields,
            "required_field_count": len(required_fields),
            "gate_financial_coverage_score": gate_coverage_score,
            "gate_missing_fields": gate_missing_fields,
            "gate_required_field_count": len(gate_required_fields),
        }
