from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from multi_agent.core.evidence import ResearchEvidenceBundle
from multi_agent.core.review_contracts import ReviewContract

ReportMode = Literal["formal_report", "evidence_limited_report", "blocked_notice"]
ReportStance = Literal["buy", "hold", "sell", "watch", "blocked"]

REQUIRED_SECTION_KEYS = (
    "executive_summary",
    "business_overview",
    "recent_events",
    "financial_analysis",
    "key_risks",
    "investment_conclusion",
    "source_index",
)

SECTION_HEADINGS = {
    "executive_summary": "执行摘要",
    "business_overview": "公司与业务概览",
    "recent_events": "近期事件与催化剂",
    "financial_analysis": "财务分析与估值",
    "key_risks": "关键风险",
    "investment_conclusion": "投资结论",
    "source_index": "来源索引",
}

_STANCE_LABELS = {
    "buy": "买入",
    "hold": "持有",
    "sell": "卖出",
    "watch": "观察",
    "blocked": "阻断",
}
_MODE_TO_STATUS = {
    "formal_report": "passed",
    "evidence_limited_report": "evidence_limited",
    "blocked_notice": "blocked",
}

_FORMAL_CORE_SECTION_KEYS = (
    "executive_summary",
    "financial_analysis",
    "investment_conclusion",
)
_ACTIONABLE_RECOMMENDATION_PATTERN = re.compile(
    r"\b(?:strong\s+buy|buy|sell|overweight|underweight)\b"
    r"|(?:建议|维持)(?:买入|卖出|增持|减持|持有)"
    r"|(?:买入|卖出|增持|减持|强烈推荐|目标价|评级)",
    re.IGNORECASE,
)


class ReportGenerationContext(BaseModel):
    """Immutable post-gate inputs permitted to appear in a newly generated report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    company_name: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    report_mode: ReportMode
    evidence_bundle: ResearchEvidenceBundle
    analysis_review_contract: ReviewContract
    allowed_claim_ids: tuple[str, ...] = ()
    canonical_sources_json: tuple[str, ...] = ()
    revision_instructions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_locked_delivery_mode(self) -> "ReportGenerationContext":
        object.__setattr__(self, "evidence_bundle", self.evidence_bundle.model_copy(deep=True))
        object.__setattr__(self, "analysis_review_contract", self.analysis_review_contract.model_copy(deep=True))
        object.__setattr__(self, "allowed_claim_ids", tuple(self.allowed_claim_ids))
        object.__setattr__(self, "canonical_sources_json", tuple(self.canonical_sources_json))
        if len(self.allowed_claim_ids) != len(set(self.allowed_claim_ids)):
            raise ValueError("allowed_claim_ids must be unique")
        eligibility = self.analysis_review_contract.delivery_eligibility
        if self.report_mode == "formal_report" and not eligibility.formal_report_allowed:
            raise ValueError("formal_report context requires formal_report_allowed")
        if (
            self.report_mode == "evidence_limited_report"
            and not eligibility.evidence_limited_report_allowed
        ):
            raise ValueError(
                "evidence_limited_report context requires evidence_limited_report_allowed"
            )
        if self.report_mode == "blocked_notice" and not eligibility.blocked_notice_required:
            raise ValueError("blocked_notice context requires blocked_notice_required")
        return self

    def canonical_sources(self) -> tuple["SourceReference", ...]:
        return tuple(SourceReference.model_validate_json(item) for item in self.canonical_sources_json)


class ReportSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    heading: str = Field(min_length=1)
    content: str = ""
    claim_ids: list[str] = Field(default_factory=list)


class ReportClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    critical: bool = False
    source_ids: list[str] = Field(default_factory=list)


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    source_tag: str = Field(min_length=1)
    field_name: str | None = None


class ReportWriterSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading: str = Field(min_length=1)
    content: str = ""
    claim_ids: list[str] = Field(default_factory=list)


class ReportWriterSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    title: str | None = None
    url: str | None = None
    source_tag: str | None = None
    field_name: str | None = None


class ReportWriterPayload(BaseModel):
    """LLM-owned fields validated before Flow binds trusted control-plane data."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    stance: ReportStance
    executive_summary: str = ""
    catalysts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    claims: list[ReportClaim] = Field(default_factory=list)
    sources: list[ReportWriterSource] = Field(default_factory=list)
    sections: dict[str, ReportWriterSection]

    @model_validator(mode="after")
    def validate_section_keys(self) -> "ReportWriterPayload":
        actual_keys = tuple(self.sections)
        if set(actual_keys) != set(REQUIRED_SECTION_KEYS) or len(actual_keys) != len(
            REQUIRED_SECTION_KEYS
        ):
            raise ValueError("sections must contain exactly the required seven section keys")
        self.sections = {key: self.sections[key] for key in REQUIRED_SECTION_KEYS}
        return self


def validate_report_writer_output(task_output: Any):
    """Return actionable CrewAI retry feedback for malformed writer JSON."""
    raw = str(getattr(task_output, "raw", "")).strip()
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("writer JSON must be an object")
        validated = ReportWriterPayload.model_validate(payload)
    except json.JSONDecodeError as error:
        return False, f"ReportWriterPayload validation failed: invalid JSON: {error.msg}"
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors()
        )
        return False, f"ReportWriterPayload validation failed: {details}"
    except ValueError as error:
        return False, f"ReportWriterPayload validation failed: {error}"
    return True, validated.model_dump_json(indent=2, exclude_none=True)


class ReportDocument(BaseModel):
    """Canonical, validated source for every new report delivery projection."""

    model_config = ConfigDict(extra="forbid")

    company_name: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    report_mode: ReportMode
    title: str = Field(min_length=1)
    stance: ReportStance
    executive_summary: str = ""
    catalysts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    sections: dict[str, ReportSection]
    claims: list[ReportClaim] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    trust_score: int = Field(ge=0, le=100)
    allowed_claim_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_writer_payload(
        cls,
        *,
        context: ReportGenerationContext,
        writer_payload: dict[str, object],
        trust_score: int,
    ) -> "ReportDocument":
        """Bind untrusted writer data to the post-gate context before validation."""
        payload = dict(writer_payload)
        sections = payload.get("sections")
        if isinstance(sections, dict):
            payload["sections"] = {
                str(key): {**value, "key": str(key)} if isinstance(value, dict) else value
                for key, value in sections.items()
            }
        payload.update(
            {
                "company_name": context.company_name,
                "ticker": context.ticker,
                "report_mode": context.report_mode,
                "trust_score": trust_score,
                "allowed_claim_ids": list(context.allowed_claim_ids),
            }
        )
        canonical_sources = {source.source_id: source for source in context.canonical_sources()}
        raw_sources = payload.get("sources", [])
        if not isinstance(raw_sources, list):
            raise ValueError("writer sources must be a list")
        if not canonical_sources and raw_sources:
            raise ValueError("writer sources are not allowed without canonical sources")
        if canonical_sources:
            requested_ids = []
            for raw_source in raw_sources:
                if not isinstance(raw_source, dict):
                    raise ValueError("writer source must be an object")
                source_id = str(raw_source.get("source_id", ""))
                canonical = canonical_sources.get(source_id)
                if canonical is None:
                    raise ValueError(f"unknown canonical source: {source_id}")
                if any(
                    raw_source.get(key) not in (None, getattr(canonical, key))
                    for key in ("title", "url", "source_tag", "field_name")
                ):
                    raise ValueError(f"forged canonical source: {source_id}")
                requested_ids.append(source_id)
            payload["sources"] = [
                canonical_sources[source_id].model_dump()
                for source_id in sorted(set(requested_ids))
            ]
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_document_contract(self) -> "ReportDocument":
        actual_keys = tuple(self.sections)
        if set(actual_keys) != set(REQUIRED_SECTION_KEYS) or len(actual_keys) != len(
            REQUIRED_SECTION_KEYS
        ):
            raise ValueError("sections must contain exactly the required seven section keys")
        if len(self.allowed_claim_ids) != len(set(self.allowed_claim_ids)):
            raise ValueError("allowed_claim_ids must be unique")
        if self.report_mode == "formal_report" and (not self.claims or not self.sources):
            raise ValueError("formal_report requires source-bound critical claims and sources")

        self.sections = {key: self.sections[key] for key in REQUIRED_SECTION_KEYS}
        for key, section in self.sections.items():
            if section.key != key:
                raise ValueError(f"section key must match mapping key: {key}")
            if len(section.claim_ids) != len(set(section.claim_ids)):
                raise ValueError(f"section {key} has duplicate claim IDs")

        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim_id values must be unique")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique")
        known_claim_ids = set(claim_ids)
        known_source_ids = set(source_ids)

        for claim in self.claims:
            claim.source_ids = sorted(set(claim.source_ids))
            unknown_source_ids = set(claim.source_ids) - known_source_ids
            if unknown_source_ids:
                raise ValueError(
                    f"claim {claim.claim_id} references unknown source IDs: "
                    f"{sorted(unknown_source_ids)}"
                )
            if self.report_mode in {"formal_report", "evidence_limited_report"}:
                claim_label = "critical claim" if claim.critical else "claim"
                if not claim.source_ids:
                    raise ValueError(
                        f"{claim_label} {claim.claim_id} must bind at least one source"
                    )
                if claim.claim_id not in self.allowed_claim_ids:
                    raise ValueError(
                        f"{claim_label} {claim.claim_id} is not in allowed_claim_ids"
                    )

        for section in self.sections.values():
            unknown_claim_ids = set(section.claim_ids) - known_claim_ids
            if unknown_claim_ids:
                raise ValueError(
                    f"section {section.key} references unknown claim IDs: "
                    f"{sorted(unknown_claim_ids)}"
                )
        referenced_claim_ids = {
            claim_id for section in self.sections.values() for claim_id in section.claim_ids
        }
        for claim in self.claims:
            if claim.critical and claim.claim_id not in referenced_claim_ids:
                raise ValueError(
                    f"critical claim {claim.claim_id} must appear in a report section"
                )

        if self.report_mode == "formal_report":
            if self.stance not in {"buy", "hold", "sell"}:
                raise ValueError("formal_report requires buy, hold, or sell stance")
            empty_sections = [
                key for key, section in self.sections.items() if not section.content.strip()
            ]
            if empty_sections:
                raise ValueError(
                    "formal_report requires non-empty content for every section: "
                    f"{', '.join(empty_sections)}"
                )
            if not self.executive_summary.strip() or not self.catalysts or not self.risks:
                raise ValueError("formal_report requires summary, catalysts, and risks")
            critical_claim_ids = {claim.claim_id for claim in self.claims if claim.critical}
            for section_key in _FORMAL_CORE_SECTION_KEYS:
                if not critical_claim_ids.intersection(self.sections[section_key].claim_ids):
                    raise ValueError(
                        f"formal_report core section {section_key} requires a critical claim"
                    )
        elif self.report_mode == "evidence_limited_report":
            if self.stance != "watch":
                raise ValueError("evidence_limited_report requires watch stance")
            if "证据受限" not in self.executive_summary:
                raise ValueError("evidence_limited_report requires 证据受限 disclosure")
        else:
            if self.stance != "blocked":
                raise ValueError("blocked_notice requires blocked stance")
            if "阻断" not in self.executive_summary:
                raise ValueError("blocked_notice requires 阻断 disclosure")
        if self.report_mode in {"evidence_limited_report", "blocked_notice"}:
            conclusion_text = "\n".join(
                [
                    self.executive_summary,
                    *self.catalysts,
                    *self.risks,
                    *(
                        self.sections[key].content
                        for key in _FORMAL_CORE_SECTION_KEYS
                    ),
                ]
            )
            if _ACTIONABLE_RECOMMENDATION_PATTERN.search(conclusion_text):
                raise ValueError(
                    f"{self.report_mode} cannot contain actionable recommendation language"
                )
        return self


def render_markdown(document: ReportDocument) -> str:
    """Render the canonical document in stable section and source order."""
    claims = {claim.claim_id: claim for claim in document.claims}
    sources = {source.source_id: source for source in document.sources}
    lines = [f"# {document.title}", ""]
    for key in REQUIRED_SECTION_KEYS:
        section = document.sections[key]
        lines.extend((f"## {SECTION_HEADINGS[key]}", "", section.content.strip(), ""))
        if key != "source_index":
            for claim_id in section.claim_ids:
                claim = claims[claim_id]
                anchors = ", ".join(
                    f"[{source_id}]({sources[source_id].url})"
                    for source_id in sorted(claim.source_ids)
                )
                lines.extend((f"- [{claim.claim_id}] {claim.text}（来源：{anchors}）", ""))
        else:
            for source in sorted(document.sources, key=lambda item: item.source_id):
                field = f" | 字段：{source.field_name}" if source.field_name else ""
                lines.extend(
                    (
                        f"- [{source.source_id}]({source.url}) {source.title}"
                        f" | 来源：{source.source_tag}{field}",
                        "",
                    )
                )
    return "\n".join(lines).rstrip() + "\n"


def render_recommendation(document: ReportDocument) -> dict[str, object]:
    """Project recommendation JSON directly from the canonical document."""
    return {
        "company_name": document.company_name,
        "company_ticker": document.ticker,
        "report_mode": document.report_mode,
        "stance": document.stance,
        "stance_label": _stance_label_for_document(document),
        "status": _MODE_TO_STATUS[document.report_mode],
        "final_delivery_state": document.report_mode,
        "trust_score": document.trust_score,
        "summary": document.executive_summary,
        "catalysts": list(document.catalysts),
        "risks": list(document.risks),
        "source_ids": [source.source_id for source in sorted(document.sources, key=lambda item: item.source_id)],
    }


def render_structured_report(document: ReportDocument) -> dict[str, object]:
    """Project the full JSON report directly from the canonical document."""
    return {
        "company_name": document.company_name,
        "company_ticker": document.ticker,
        "title": document.title,
        "report_mode": document.report_mode,
        "stance": document.stance,
        "stance_label": _stance_label_for_document(document),
        "status": _MODE_TO_STATUS[document.report_mode],
        "final_decision": _MODE_TO_STATUS[document.report_mode],
        "final_delivery_state": document.report_mode,
        "trust_score": document.trust_score,
        "summary": document.executive_summary,
        "catalysts": list(document.catalysts),
        "risks": list(document.risks),
        "sections": {
            key: document.sections[key].content for key in REQUIRED_SECTION_KEYS
        },
        "claims": [
            claim.model_dump()
            for claim in sorted(document.claims, key=lambda item: item.claim_id)
        ],
        "sources": [
            source.model_dump()
            for source in sorted(document.sources, key=lambda item: item.source_id)
        ],
    }


def _stance_label_for_document(document: ReportDocument) -> str:
    if document.report_mode == "evidence_limited_report":
        return "证据受限"
    return _STANCE_LABELS[document.stance]
