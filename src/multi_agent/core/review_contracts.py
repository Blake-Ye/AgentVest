from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from multi_agent.core.evidence import EvidenceTarget


class ReviewToolSummary(BaseModel):
    """Legacy tool summary retained only for historical artifact reconstruction."""

    model_config = ConfigDict(extra="forbid")

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
GateControlDecision = Literal["passed", "rerun", "blocked", "evidence_limited"]
ToolHealthStatus = Literal["healthy", "degraded", "failed"]


class RepairAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: EvidenceTarget
    code: str
    fields: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    instruction: str = ""


class DeliveryEligibility(BaseModel):
    model_config = ConfigDict(extra="forbid")

    formal_report_allowed: bool = False
    evidence_limited_report_allowed: bool = False
    blocked_notice_required: bool = False
    recommended_delivery_state: FinalDeliveryState | None = None


class FailureTaxonomy(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

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
    model_config = ConfigDict(extra="forbid")

    evidence_coverage_ratio: float = 0.0
    financial_coverage_score: float = 0.0
    claim_binding_ratio: float = 1.0
    gate_financial_coverage_score: float | None = None
    critical_conflict_count: int = 0
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
    model_config = ConfigDict(extra="forbid")

    overall_status: ToolHealthStatus = "healthy"
    failed_tools: list[str] = Field(default_factory=list)
    degraded_tools: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    tool_status: list[dict[str, str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_overall_status(self) -> "ToolHealthSummary":
        if self.overall_status == "healthy" and (self.failed_tools or self.degraded_tools):
            raise ValueError("overall_status=healthy requires no failed_tools or degraded_tools")
        if self.overall_status == "failed" and not self.failed_tools:
            raise ValueError("overall_status=failed requires failed_tools")
        if self.overall_status == "degraded" and not (
            self.degraded_tools or self.failed_tools
        ):
            raise ValueError("overall_status=degraded requires degraded_tools or failed_tools")
        return self


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gate_outcome: Literal["pass", "rerun", "block"] = "rerun"
    decision_confidence: Literal["high", "medium", "low"] = "medium"


class ReviewSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    one_sentence_summary: str = ""
    operator_notes: str = ""


class ReviewContract(BaseModel):
    """Strict contract used by every new review artifact."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    stage: ReviewStage = Field(validation_alias=AliasChoices("stage", "review_stage"))
    reviewer_name: str = ""
    decision: ReviewDecision = Field(default_factory=ReviewDecision)
    delivery_eligibility: DeliveryEligibility
    failure_taxonomy: FailureTaxonomy
    coverage_summary: CoverageSummary = Field(default_factory=CoverageSummary)
    tool_health_summary: ToolHealthSummary = Field(default_factory=ToolHealthSummary)
    blocking_reasons: list[str] = Field(default_factory=list)
    rerun_reasons: list[str] = Field(default_factory=list)
    allow_limited_delivery: bool = False
    review_summary: ReviewSummary = Field(default_factory=ReviewSummary)
    artifact_refs: list[dict[str, str]] = Field(default_factory=list)
    findings: list[dict[str, object]] = Field(default_factory=list)
    repair_actions: list[RepairAction] = Field(default_factory=list)

    def to_review_tool_summary(self) -> ReviewToolSummary:
        return self.coverage_summary.to_review_tool_summary()


class GateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    final_decision: GateControlDecision
    trust_score: int
    blocking_reasons: list[str] = Field(default_factory=list)
    repair_actions: list[RepairAction] = Field(default_factory=list)


class FinalDecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_decision: GateControlDecision
    final_delivery_state: FinalDeliveryState
    trust_score: int
    blocking_reasons: list[str] = Field(default_factory=list)
    analysis_review_contract: ReviewContract | None = None
    report_review_contract: ReviewContract | None = None


def parse_legacy_review_contract(payload: dict[str, object]) -> ReviewContract:
    """Adapt historical, permissive review payloads before validating strictly."""

    raw = dict(payload)
    raw_decision = raw.get("decision")
    if isinstance(raw_decision, str):
        decision = {"gate_outcome": {"passed": "pass", "blocked": "block"}.get(raw_decision, raw_decision)}
    elif isinstance(raw_decision, dict):
        decision = dict(raw_decision)
    else:
        decision = {}
    decision["gate_outcome"] = {"passed": "pass", "blocked": "block"}.get(
        str(decision.get("gate_outcome", "rerun")).lower(),
        str(decision.get("gate_outcome", "rerun")).lower(),
    )
    decision.setdefault("decision_confidence", "medium")

    raw_delivery = raw.get("delivery_eligibility")
    delivery_source = raw_delivery if isinstance(raw_delivery, dict) else {}
    delivery = {
        key: delivery_source[key]
        for key in (
            "formal_report_allowed",
            "evidence_limited_report_allowed",
            "blocked_notice_required",
            "recommended_delivery_state",
        )
        if key in delivery_source
    }

    raw_taxonomy = raw.get("failure_taxonomy")
    taxonomy_source = raw_taxonomy if isinstance(raw_taxonomy, dict) else {}
    primary = taxonomy_source.get("primary_class", taxonomy_source.get("primary_code", "none"))
    taxonomy = {
        "primary_class": primary if primary in FailurePrimaryClass.__args__ else "none",
        "secondary_causes": taxonomy_source.get(
            "secondary_causes", taxonomy_source.get("secondary_codes", [])
        ),
    }

    raw_coverage = raw.get("coverage_summary", raw.get("coverage", {}))
    coverage_source = raw_coverage if isinstance(raw_coverage, dict) else {}
    coverage = {
        key: coverage_source[key]
        for key in (
            "evidence_coverage_ratio",
            "financial_coverage_score",
            "claim_binding_ratio",
            "gate_financial_coverage_score",
            "critical_conflict_count",
            "unresolved_critical_claim_count",
            "market_policy_violations",
            "unsupported_critical_claims",
            "blocking_reasons",
        )
        if key in coverage_source
    }

    raw_health = raw.get("tool_health_summary", raw.get("tool_health", {}))
    if isinstance(raw_health, dict) and "overall_status" not in raw_health:
        tool_status = [
            {"tool_name": str(name), "status": str(status)}
            for name, status in raw_health.items()
        ]
        failed = [item["tool_name"] for item in tool_status if item["status"] == "failed"]
        degraded = [item["tool_name"] for item in tool_status if item["status"] == "degraded"]
        health: dict[str, Any] = {
            "overall_status": "failed" if failed else "degraded" if degraded else "healthy",
            "failed_tools": failed,
            "degraded_tools": degraded,
            "tool_status": tool_status,
        }
    else:
        health_source = raw_health if isinstance(raw_health, dict) else {}
        health = {
            key: health_source[key]
            for key in ("overall_status", "failed_tools", "degraded_tools", "notes", "tool_status")
            if key in health_source
        }

    raw_summary = raw.get("review_summary", raw.get("summary_text", ""))
    summary = (
        raw_summary
        if isinstance(raw_summary, dict)
        else {"one_sentence_summary": str(raw_summary), "operator_notes": ""}
    )
    summary = {
        key: summary[key]
        for key in ("one_sentence_summary", "operator_notes")
        if key in summary
    }

    artifacts = raw.get("artifact_refs", [])
    if isinstance(artifacts, list):
        artifact_refs = [
            item if isinstance(item, dict) else {"artifact": str(item)}
            for item in artifacts
        ]
    else:
        artifact_refs = []

    return ReviewContract.model_validate(
        {
            "stage": raw.get("stage", raw.get("review_stage", "analysis_review")),
            "reviewer_name": str(raw.get("reviewer_name", "")),
            "decision": decision,
            "delivery_eligibility": delivery,
            "failure_taxonomy": taxonomy,
            "coverage_summary": coverage,
            "tool_health_summary": health,
            "blocking_reasons": raw.get("blocking_reasons", []),
            "rerun_reasons": raw.get("rerun_reasons", []),
            "allow_limited_delivery": bool(raw.get("allow_limited_delivery", False)),
            "review_summary": summary,
            "artifact_refs": artifact_refs,
            "findings": raw.get("findings", []),
            "repair_actions": raw.get("repair_actions", []),
        }
    )
