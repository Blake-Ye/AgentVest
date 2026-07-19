from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from multi_agent.core.artifact_paths import build_run_artifact_paths
from multi_agent.core.evidence import FinancialFact, ResearchEvidenceBundle
from multi_agent.core.report_document import ReportDocument, ReportGenerationContext
from multi_agent.core.review_contracts import (
    CoverageSummary,
    DeliveryEligibility,
    FailureTaxonomy,
    FinalDecisionRecord,
    ReviewContract,
)


def _formal_document() -> ReportDocument:
    evidence = ResearchEvidenceBundle(
        company_name="Apple Inc.",
        ticker="AAPL",
        financial_facts=[
            FinancialFact(
                field_name="revenue",
                value=416_161_000_000,
                unit="USD",
                period_start=date(2024, 9, 29),
                period_end=date(2025, 9, 27),
                fiscal_year=2025,
                fiscal_period="FY",
                form="10-K",
                accession="0000320193-25-000079",
                filed_at=date(2025, 10, 31),
                source_url="https://www.sec.gov/Archives/edgar/data/320193/aapl-20250927.htm",
                source_tag="sec_companyfacts",
            )
        ],
    )
    context = ReportGenerationContext(
        company_name="Apple Inc.",
        ticker="AAPL",
        report_mode="formal_report",
        evidence_bundle=evidence,
        analysis_review_contract=ReviewContract(
            stage="report",
            delivery_eligibility=DeliveryEligibility(formal_report_allowed=True),
            failure_taxonomy=FailureTaxonomy(primary_class="none"),
            coverage_summary=CoverageSummary(
                evidence_coverage_ratio=1.0,
                financial_coverage_score=1.0,
                claim_binding_ratio=1.0,
            ),
        ),
        allowed_claim_ids=["revenue"],
    )
    sections = {
        key: {
            "heading": key,
            "content": f"{key} content",
            "claim_ids": ["revenue"] if key in {"executive_summary", "financial_analysis", "investment_conclusion"} else [],
        }
        for key in (
            "executive_summary",
            "business_overview",
            "recent_events",
            "financial_analysis",
            "key_risks",
            "investment_conclusion",
            "source_index",
        )
    }
    return ReportDocument.from_writer_payload(
        context=context,
        trust_score=90,
        writer_payload={
            "title": "Apple Inc. Investment Research",
            "stance": "hold",
            "executive_summary": "Formal evidence supports continued monitoring.",
            "catalysts": ["Services growth"],
            "risks": ["Demand volatility"],
            "sections": sections,
            "claims": [
                {
                    "claim_id": "revenue",
                    "text": "FY2025 revenue was $416.2 billion.",
                    "critical": True,
                    "source_ids": ["sec-10k"],
                }
            ],
            "sources": [
                {
                    "source_id": "sec-10k",
                    "title": "Apple 2025 Form 10-K",
                    "url": "https://www.sec.gov/Archives/edgar/data/320193/aapl-20250927.htm",
                    "source_tag": "sec_filing",
                }
            ],
        },
    )


@pytest.fixture
def formal_document() -> ReportDocument:
    return _formal_document()


def _decision() -> FinalDecisionRecord:
    return FinalDecisionRecord(
        final_decision="passed",
        final_delivery_state="formal_report",
        trust_score=90,
    )


def test_validator_rejects_limited_body_with_passed_decision(formal_document: ReportDocument) -> None:
    from multi_agent.core.delivery import DeliveryValidator

    document = formal_document.model_copy(update={"report_mode": "evidence_limited_report"})

    result = DeliveryValidator().validate(decision=_decision(), document=document)

    assert result.valid is False
    assert "report_mode_mismatch" in result.errors


def test_validator_rejects_empty_structured_sections(formal_document: ReportDocument) -> None:
    from multi_agent.core.delivery import DeliveryPackage, DeliveryValidator

    package = DeliveryPackage.from_document(decision=_decision(), document=formal_document)
    structured = dict(package.structured_report)
    sections = dict(structured["sections"])
    sections["financial_analysis"] = ""
    structured["sections"] = sections
    package = package.model_copy(update={"structured_report": structured})

    result = DeliveryValidator().validate_package(package)

    assert result.valid is False
    assert "section_empty:financial_analysis" in result.errors


def test_atomic_writer_writes_only_a_valid_single_truth_package(
    tmp_path: Path, formal_document: ReportDocument
) -> None:
    from multi_agent.core.delivery import DeliveryPackage, write_delivery_package

    paths = build_run_artifact_paths(tmp_path / "run")
    package = DeliveryPackage.from_document(decision=_decision(), document=formal_document)

    write_delivery_package(paths=paths, package=package)

    assert json.loads(paths.final_decision_path.read_text(encoding="utf-8"))["final_decision"] == "passed"
    assert json.loads(paths.structured_recommendation_path.read_text(encoding="utf-8"))["stance"] == "hold"
    assert "## 财务分析与估值" in paths.final_report_path.read_text(encoding="utf-8")
