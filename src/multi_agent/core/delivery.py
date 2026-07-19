"""Validated, single-truth delivery projections for newly generated reports."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from multi_agent.core.artifact_paths import RunArtifactPaths
from multi_agent.core.report_document import (
    REQUIRED_SECTION_KEYS,
    ReportDocument,
    render_markdown,
    render_recommendation,
    render_structured_report,
)
from multi_agent.core.review_contracts import FinalDecisionRecord


EXPECTED_DELIVERY_STATE = {
    "passed": "formal_report",
    "evidence_limited": "evidence_limited_report",
    "blocked": "blocked_notice",
}
_ALLOWED_STANCES = {
    "formal_report": {"buy", "hold", "sell"},
    "evidence_limited_report": {"watch"},
    "blocked_notice": {"blocked"},
}


class DeliveryValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    valid: bool
    errors: list[str] = Field(default_factory=list)


class DeliveryPackage(BaseModel):
    """Frozen set of projections derived from one canonical report document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: FinalDecisionRecord
    document: ReportDocument
    markdown: str
    recommendation: dict[str, object]
    structured_report: dict[str, object]

    @classmethod
    def from_document(
        cls, *, decision: FinalDecisionRecord, document: ReportDocument
    ) -> "DeliveryPackage":
        return cls(
            decision=decision.model_copy(deep=True),
            document=document.model_copy(deep=True),
            markdown=render_markdown(document),
            recommendation=render_recommendation(document),
            structured_report=render_structured_report(document),
        )


class DeliveryValidator:
    """Reject delivery drift before any terminal artifact is written."""

    def validate(
        self, *, decision: FinalDecisionRecord, document: ReportDocument
    ) -> DeliveryValidationResult:
        errors: list[str] = []
        expected_mode = EXPECTED_DELIVERY_STATE.get(decision.final_decision)
        if expected_mode is None:
            errors.append("final_decision_not_terminal")
        elif (
            decision.final_delivery_state != expected_mode
            or document.report_mode != expected_mode
        ):
            errors.append("report_mode_mismatch")
        if document.trust_score != decision.trust_score:
            errors.append("trust_score_mismatch")
        if expected_mode and document.stance not in _ALLOWED_STANCES[expected_mode]:
            errors.append("stance_mismatch")

        for key in REQUIRED_SECTION_KEYS:
            section = document.sections.get(key)
            if section is None or not section.content.strip():
                errors.append(f"section_empty:{key}")

        source_ids = {source.source_id for source in document.sources}
        section_claim_ids = {
            claim_id
            for section in document.sections.values()
            for claim_id in section.claim_ids
        }
        for claim in document.claims:
            if claim.critical and not set(claim.source_ids).intersection(source_ids):
                errors.append(f"source_unbound:{claim.claim_id}")
            if claim.critical and claim.claim_id not in section_claim_ids:
                errors.append(f"claim_unplaced:{claim.claim_id}")

        return DeliveryValidationResult(valid=not errors, errors=sorted(set(errors)))

    def validate_package(self, package: DeliveryPackage) -> DeliveryValidationResult:
        result = self.validate(decision=package.decision, document=package.document)
        errors = list(result.errors)
        expected_recommendation = render_recommendation(package.document)
        expected_structured_report = render_structured_report(package.document)
        expected_markdown = render_markdown(package.document)

        if package.markdown != expected_markdown:
            errors.append("markdown_projection_mismatch")
        if package.recommendation != expected_recommendation:
            errors.append("recommendation_projection_mismatch")
        if package.structured_report != expected_structured_report:
            errors.append("structured_report_projection_mismatch")

        sections = package.structured_report.get("sections")
        if not isinstance(sections, dict):
            errors.append("sections_missing")
        else:
            for key in REQUIRED_SECTION_KEYS:
                if not str(sections.get(key, "")).strip():
                    errors.append(f"section_empty:{key}")
        if package.recommendation.get("status") != package.decision.final_decision:
            errors.append("recommendation_decision_mismatch")
        if package.recommendation.get("final_delivery_state") != package.decision.final_delivery_state:
            errors.append("recommendation_delivery_state_mismatch")
        if package.recommendation.get("stance") != package.document.stance:
            errors.append("recommendation_stance_mismatch")
        if package.structured_report.get("final_decision") != package.decision.final_decision:
            errors.append("structured_report_decision_mismatch")
        if package.structured_report.get("final_delivery_state") != package.decision.final_delivery_state:
            errors.append("structured_report_delivery_state_mismatch")
        if package.structured_report.get("stance") != package.document.stance:
            errors.append("structured_report_stance_mismatch")
        return DeliveryValidationResult(valid=not errors, errors=sorted(set(errors)))


def write_delivery_package(*, paths: RunArtifactPaths, package: DeliveryPackage) -> None:
    """Validate then atomically replace all terminal delivery artifacts."""
    result = DeliveryValidator().validate_package(package)
    if not result.valid:
        raise ValueError(f"delivery package validation failed: {', '.join(result.errors)}")

    payloads = (
        (paths.report_document_path, _json_text(package.document.model_dump(mode="json"))),
        (paths.final_report_path, package.markdown),
        (paths.structured_recommendation_path, _json_text(package.recommendation)),
        (paths.structured_report_path, _json_text(package.structured_report)),
        (paths.final_decision_path, _json_text(package.decision.model_dump(mode="json"))),
    )
    for path, content in payloads:
        _atomic_write_text(path, content)


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            Path(temporary_name).unlink(missing_ok=True)
        except OSError:
            pass
        raise
