from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ReviewToolSummary(BaseModel):
    evidence_coverage_ratio: float
    financial_coverage_score: float
    gate_financial_coverage_score: float | None = None
    critical_conflict_count: int = 0
    market_policy_violations: list[str] = Field(default_factory=list)
    unsupported_critical_claims: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)


ReviewStage = Literal["analysis", "report", "analysis_review", "report_review"]
FailureCode = Literal[
    "coverage_gap",
    "critical_conflict",
    "policy_violation",
    "unsupported_claim",
    "tool_failure",
    "summary_missing",
    "report_inconsistency",
    "delivery_blocked",
]
FailurePrimaryClass = Literal[
    "none",
    "research_blocked",
    "pipeline_degraded",
    "coverage_gap",
    "critical_conflict",
    "policy_violation",
    "unsupported_claim",
    "tool_failure",
    "summary_missing",
    "report_inconsistency",
    "delivery_blocked",
]
FinalDeliveryState = Literal[
    "formal_report",
    "evidence_limited_report",
    "blocked_notice",
]


class DeliveryEligibility(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    formal_report_allowed: bool = False
    evidence_limited_report_allowed: bool = False
    blocked_notice_required: bool = False
    recommended_delivery_state: FinalDeliveryState | None = None


class FailureTaxonomy(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    primary_class: FailurePrimaryClass = Field(
        validation_alias=AliasChoices("primary_class", "primary_code")
    )
    secondary_causes: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("secondary_causes", "secondary_codes"),
    )

    @property
    def primary_code(self) -> str:
        return self.primary_class

    @property
    def secondary_codes(self) -> list[str]:
        return list(self.secondary_causes)


class CoverageSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    evidence_coverage_ratio: float = 0.0
    financial_coverage_score: float = 0.0
    gate_financial_coverage_score: float | None = None
    critical_conflict_count: int = 0
    claim_binding_ratio: float = 1.0
    unresolved_critical_claim_count: int = 0
    market_policy_violations: list[str] = Field(default_factory=list)
    unsupported_critical_claims: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)

    def to_review_tool_summary(self) -> ReviewToolSummary:
        return ReviewToolSummary(
            evidence_coverage_ratio=self.evidence_coverage_ratio,
            financial_coverage_score=self.financial_coverage_score,
            gate_financial_coverage_score=self.gate_financial_coverage_score,
            critical_conflict_count=self.critical_conflict_count,
            market_policy_violations=list(self.market_policy_violations),
            unsupported_critical_claims=list(self.unsupported_critical_claims),
            blocking_reasons=list(self.blocking_reasons),
        )


class ToolHealthSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    overall_status: str = "healthy"
    failed_tools: list[str] = Field(default_factory=list)
    degraded_tools: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    tool_status: list[dict[str, str]] = Field(default_factory=list)


class ReviewDecision(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    gate_outcome: Literal["pass", "rerun", "block"] = "rerun"
    decision_confidence: Literal["high", "medium", "low"] = "medium"


class ReviewSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    one_sentence_summary: str = ""
    operator_notes: str = ""


class ReviewContract(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    stage: ReviewStage = Field(validation_alias=AliasChoices("stage", "review_stage"))
    reviewer_name: str = ""
    decision: ReviewDecision = Field(default_factory=ReviewDecision)
    delivery_eligibility: DeliveryEligibility
    failure_taxonomy: FailureTaxonomy
    coverage_summary: CoverageSummary = Field(
        default_factory=CoverageSummary,
        validation_alias=AliasChoices("coverage_summary", "coverage"),
    )
    tool_health_summary: ToolHealthSummary = Field(
        default_factory=ToolHealthSummary,
        validation_alias=AliasChoices("tool_health_summary", "tool_health"),
    )
    blocking_reasons: list[str] = Field(default_factory=list)
    rerun_reasons: list[str] = Field(default_factory=list)
    allow_limited_delivery: bool = False
    review_summary: ReviewSummary = Field(default_factory=ReviewSummary)
    artifact_refs: list[dict[str, str]] = Field(default_factory=list)
    findings: list[dict[str, object]] = Field(default_factory=list)
    summary_text: str | None = None

    def to_review_tool_summary(self) -> ReviewToolSummary:
        return self.coverage_summary.to_review_tool_summary()


GateControlDecision = Literal["passed", "rerun", "blocked", "evidence_limited"]


class GateDecision(BaseModel):
    passed: bool
    final_decision: GateControlDecision
    trust_score: int
    blocking_reasons: list[str] = Field(default_factory=list)


class FinalDecisionRecord(BaseModel):
    final_decision: GateControlDecision
    final_delivery_state: FinalDeliveryState
    trust_score: int
    blocking_reasons: list[str] = Field(default_factory=list)
    analysis_review_contract: ReviewContract | None = None
    report_review_contract: ReviewContract | None = None
