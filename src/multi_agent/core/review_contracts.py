from __future__ import annotations

import json
import re
from typing import Any, Literal, Tuple

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, model_validator

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

    evidence_coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    financial_coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    claim_binding_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    gate_financial_coverage_score: float | None = Field(default=None, ge=0.0, le=1.0)
    critical_conflict_count: int = Field(default=0, ge=0)
    unresolved_critical_claim_count: int = Field(default=0, ge=0)
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

    @model_validator(mode="after")
    def validate_repair_targets_for_stage(self) -> "ReviewContract":
        analysis_targets = {
            "market_validation_analyst",
            "event_guidance_analyst",
            "fundamental_analyst",
            "quant_valuation_analyst",
            "report_writing_analyst",
            "data_quality_reviewer",
        }
        report_targets = {"report_writing_analyst"}
        allowed = report_targets if self.stage in {"report", "report_review"} else analysis_targets
        invalid = sorted({action.target for action in self.repair_actions} - allowed)
        if invalid:
            raise ValueError(
                f"repair target is invalid for stage {self.stage}: {', '.join(invalid)}"
            )
        return self

    def to_review_tool_summary(self) -> ReviewToolSummary:
        return self.coverage_summary.to_review_tool_summary()


_MACHINE_READABLE_JSON = re.compile(
    r"PART A:\s*MACHINE_READABLE_JSON[\s\S]*?```(?:json)?\s*([\s\S]*?)\s*```",
    re.IGNORECASE,
)
_TOOL_HEALTH_KEYS = {
    "overall_status",
    "failed_tools",
    "degraded_tools",
    "notes",
    "tool_status",
}


def _tool_status(value: object) -> tuple[str, str]:
    if isinstance(value, dict):
        raw_status = value.get("status", value.get("overall_status", "degraded"))
        raw_note = value.get("notes", value.get("note", ""))
    else:
        raw_status, raw_note = value, ""
    status = str(raw_status).strip().lower()
    status = {
        "ok": "healthy",
        "pass": "healthy",
        "passed": "healthy",
        "success": "healthy",
        "warning": "degraded",
        "partial": "degraded",
        "error": "failed",
        "failure": "failed",
        "skipped": "not_used",
        "unused": "not_used",
    }.get(status, status)
    if status not in {"healthy", "degraded", "failed", "not_used"}:
        status = "degraded"
    note = (
        "; ".join(map(str, raw_note))
        if isinstance(raw_note, list)
        else str(raw_note).strip()
    )
    return status, note


def normalize_review_contract_payload(payload: dict[str, object]) -> dict[str, object]:
    """Canonicalize known reviewer aliases without weakening strict validation."""
    normalized = dict(payload)
    decision_reason: object | None = None
    raw_decision = normalized.get("decision")
    if isinstance(raw_decision, dict):
        decision = dict(raw_decision)
        for key in ("delivery_eligibility", "failure_taxonomy"):
            nested_value = decision.get(key)
            if key not in normalized and isinstance(nested_value, dict):
                normalized[key] = decision.pop(key)
            elif key in normalized and normalized[key] == nested_value:
                decision.pop(key)
        for alias in ("stage_decision", "outcome"):
            alias_value = decision.get(alias)
            canonical_outcome = {
                "pass": "pass",
                "passed": "pass",
                "formal_report_allowed": "pass",
                "rerun": "rerun",
                "block": "block",
                "blocked": "block",
            }.get(str(alias_value).strip().lower())
            if "gate_outcome" not in decision and canonical_outcome is not None:
                decision["gate_outcome"] = canonical_outcome
                decision.pop(alias)
            elif decision.get("gate_outcome") == canonical_outcome:
                decision.pop(alias, None)
        decision_reason = decision.pop("reason", None)
        normalized["decision"] = decision

    if normalized.get("delivery_eligibility") is True:
        normalized["delivery_eligibility"] = {
            "formal_report_allowed": True,
            "evidence_limited_report_allowed": True,
            "blocked_notice_required": False,
            "recommended_delivery_state": "formal_report",
        }

    if normalized.get("failure_taxonomy") == []:
        normalized["failure_taxonomy"] = {
            "primary_class": "none",
            "secondary_causes": [],
        }

    summary = normalized.get("review_summary")
    if isinstance(summary, str):
        normalized["review_summary"] = {
            "one_sentence_summary": summary,
            "operator_notes": "",
        }
    if decision_reason is not None and isinstance(normalized.get("review_summary"), dict):
        summary = dict(normalized["review_summary"])
        note = str(summary.get("operator_notes", "")).strip()
        reason = str(decision_reason).strip()
        summary["operator_notes"] = "; ".join(part for part in (note, reason) if part)
        normalized["review_summary"] = summary

    actions = normalized.get("repair_actions")
    if isinstance(actions, list):
        normalized_actions: list[object] = []
        for action in actions:
            if not isinstance(action, dict):
                normalized_actions.append(action)
                continue
            item = dict(action)
            description = item.pop("description", None)
            if not item.get("instruction") and description is not None:
                item["instruction"] = str(description)
            normalized_actions.append(item)
        normalized["repair_actions"] = normalized_actions

    artifact_refs = normalized.get("artifact_refs")
    if isinstance(artifact_refs, list):
        normalized["artifact_refs"] = [
            {
                str(key): value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                for key, value in artifact.items()
            }
            if isinstance(artifact, dict)
            else artifact
            for artifact in artifact_refs
        ]

    health = normalized.get("tool_health_summary")
    if isinstance(health, dict):
        named_tools = {
            str(name): value
            for name, value in health.items()
            if name not in _TOOL_HEALTH_KEYS and str(name).endswith("_tool")
        }
        if named_tools:
            canonical_health = dict(health)
            tool_status = (
                list(health.get("tool_status", []))
                if isinstance(health.get("tool_status"), list)
                else []
            )
            failed = (
                list(health.get("failed_tools", []))
                if isinstance(health.get("failed_tools"), list)
                else []
            )
            degraded = (
                list(health.get("degraded_tools", []))
                if isinstance(health.get("degraded_tools"), list)
                else []
            )
            notes = (
                list(health.get("notes", []))
                if isinstance(health.get("notes"), list)
                else []
            )
            for name, value in named_tools.items():
                status, note = _tool_status(value)
                tool_status.append({"tool_name": name, "status": status})
                if status == "failed":
                    failed.append(name)
                elif status == "degraded":
                    degraded.append(name)
                if note:
                    notes.append(f"{name}: {note}")
                canonical_health.pop(name, None)
            failed = list(dict.fromkeys(map(str, failed)))
            degraded = list(dict.fromkeys(map(str, degraded)))
            declared = str(health.get("overall_status", "healthy")).strip().lower()
            if declared == "failed" or failed:
                overall = "failed"
            elif declared == "degraded" or degraded:
                overall = "degraded"
            else:
                overall = "healthy"
            canonical_health.update(
                {
                    "overall_status": overall,
                    "failed_tools": failed,
                    "degraded_tools": degraded,
                    "notes": notes,
                    "tool_status": tool_status,
                }
            )
            normalized["tool_health_summary"] = canonical_health
    return normalized


def review_contract_from_payload(
    payload: dict[str, object], *, expected_stage: ReviewStage | None = None
) -> ReviewContract:
    contract = ReviewContract.model_validate(normalize_review_contract_payload(payload))
    if expected_stage is not None and contract.stage != expected_stage:
        raise ValueError(f"stage must be {expected_stage!r}, got {contract.stage!r}")
    return contract


def review_contract_from_text(
    review_text: str, *, expected_stage: ReviewStage | None = None
) -> ReviewContract | None:
    match = _MACHINE_READABLE_JSON.search(review_text)
    if match is None:
        return None
    payload = json.loads(match.group(1).strip())
    if not isinstance(payload, dict):
        raise ValueError("ReviewContract JSON must be an object")
    return review_contract_from_payload(payload, expected_stage=expected_stage)


def _validate_review_output(task_output: Any, expected_stage: ReviewStage) -> Tuple[bool, Any]:
    raw = str(getattr(task_output, "raw", "")).strip()
    match = _MACHINE_READABLE_JSON.search(raw)
    if match is None:
        return False, "ReviewContract validation failed: PART A JSON block is missing"
    try:
        payload = json.loads(match.group(1).strip())
        if not isinstance(payload, dict):
            raise ValueError("ReviewContract JSON must be an object")
        payload["stage"] = expected_stage
        contract = review_contract_from_payload(payload, expected_stage=expected_stage)
    except ValidationError as exc:
        errors = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()
        )
        return False, f"ReviewContract validation failed: {errors}"
    except ValueError as exc:
        return False, f"ReviewContract validation failed: {exc}"
    canonical = contract.model_dump_json(indent=2)
    return True, f"{raw[:match.start(1)]}{canonical}{raw[match.end(1):]}"


def validate_analysis_review_output(task_output: Any):
    return _validate_review_output(task_output, "analysis_review")


def validate_report_review_output(task_output: Any):
    return _validate_review_output(task_output, "report_review")


class GateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    final_decision: GateControlDecision
    trust_score: int
    blocking_reasons: list[str] = Field(default_factory=list)
    repair_actions: list[RepairAction] = Field(default_factory=list)


class FinalDecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_name: str = Field(min_length=1)
    company_ticker: str = Field(min_length=1)
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
