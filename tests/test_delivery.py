from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pytest

from multi_agent.core.artifact_paths import build_run_artifact_paths
from multi_agent.core.evidence import FinancialFact, ResearchEvidenceBundle
from multi_agent.core.report_document import ReportDocument, ReportGenerationContext, SourceReference
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
        canonical_sources_json=(
            SourceReference(
                source_id="sec-10k", title="Apple 2025 Form 10-K",
                url="https://www.sec.gov/Archives/edgar/data/320193/aapl-20250927.htm",
                source_tag="sec_filing",
            ).model_dump_json(),
        ),
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
        company_name="Apple Inc.",
        company_ticker="AAPL",
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
    package = DeliveryPackage(
        decision_json=package._decision_json,  # type: ignore[attr-defined]
        document_json=package._document_json,  # type: ignore[attr-defined]
        markdown=package.markdown,
        recommendation_json=package._recommendation_json,  # type: ignore[attr-defined]
        structured_report_json=json.dumps(structured),
    )

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


def _terminal_bytes(paths) -> dict[Path, bytes]:
    return {
        path: path.read_bytes()
        for path in (
            paths.report_document_path,
            paths.final_report_path,
            paths.structured_recommendation_path,
            paths.structured_report_path,
            paths.final_decision_path,
        )
    }


def test_delivery_commit_rolls_back_every_terminal_file_when_replacement_fails(
    tmp_path: Path, formal_document: ReportDocument
) -> None:
    from multi_agent.core.delivery import DeliveryPackage, write_delivery_package

    paths = build_run_artifact_paths(tmp_path / "run")
    previous = DeliveryPackage.from_document(decision=_decision(), document=formal_document)
    write_delivery_package(paths=paths, package=previous)
    before = _terminal_bytes(paths)
    next_document = formal_document.model_copy(update={"title": "Next generation"})
    next_package = DeliveryPackage.from_document(decision=_decision(), document=next_document)

    def fail_mid_commit(source: str | bytes | os.PathLike[str], target: str | bytes | os.PathLike[str]) -> None:
        if Path(target) == paths.structured_report_path and ".delivery-stage-" in Path(source).name:
            raise OSError("injected replacement failure")
        os.replace(source, target)

    with pytest.raises(OSError, match="injected replacement failure"):
        write_delivery_package(paths=paths, package=next_package, replace_file=fail_mid_commit)

    assert _terminal_bytes(paths) == before


def test_delivery_package_snapshots_cannot_be_mutated_after_validation(
    tmp_path: Path, formal_document: ReportDocument
) -> None:
    from multi_agent.core.delivery import DeliveryPackage, write_delivery_package

    paths = build_run_artifact_paths(tmp_path / "run")
    package = DeliveryPackage.from_document(decision=_decision(), document=formal_document)
    package.recommendation["status"] = "blocked"
    package.structured_report["sections"]["financial_analysis"] = "tampered"
    formal_document.sections["financial_analysis"].content = "caller mutation"

    write_delivery_package(paths=paths, package=package)

    recommendation = json.loads(paths.structured_recommendation_path.read_text(encoding="utf-8"))
    report = json.loads(paths.structured_report_path.read_text(encoding="utf-8"))
    assert recommendation["status"] == "passed"
    assert report["sections"]["financial_analysis"] == "financial_analysis content"


def test_watchlist_rebuild_recovers_interrupted_delivery_before_reading_terminal_artifacts(
    tmp_path: Path, formal_document: ReportDocument
) -> None:
    """Readers must restore the previous generation before consuming terminal truth."""
    from multi_agent import main
    from multi_agent.core.delivery import DeliveryPackage, write_delivery_package

    run_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
    paths = build_run_artifact_paths(run_dir)
    previous = DeliveryPackage.from_document(decision=_decision(), document=formal_document)
    write_delivery_package(paths=paths, package=previous)
    prior_generation = _terminal_bytes(paths)
    (run_dir / "latest_run_metrics.json").write_text(
        json.dumps({"company_name": "Apple Inc.", "company_ticker": "AAPL"}),
        encoding="utf-8",
    )

    replacement_document = formal_document.model_copy(update={"title": "Replacement generation"})
    replacement = DeliveryPackage.from_document(
        decision=_decision(), document=replacement_document
    )
    transaction_dir = run_dir / ".delivery-transactions" / "delivery-crash-simulation"
    transaction_dir.mkdir(parents=True)
    entries: list[dict[str, object]] = []
    for index, (target, payload) in enumerate(replacement.payloads(paths)):
        backup = target.parent / f".{target.name}.delivery-backup-crash-{index}"
        os.replace(target, backup)
        if index < 2:
            target.write_text(payload, encoding="utf-8")
        entries.append(
            {
                "target": str(target),
                "backup": str(backup),
                "stage": "",
                "original_exists": True,
            }
        )
    (transaction_dir / "manifest.json").write_text(
        json.dumps({"state": "prepared", "entries": entries}), encoding="utf-8"
    )

    rebuilt = main._rebuild_watchlist_from_artifacts(
        base_artifacts_dir=tmp_path / "artifacts",
        watchlist_path=tmp_path / "watchlist.json",
    )

    assert rebuilt == 1
    assert _terminal_bytes(paths) == prior_generation
    assert not (run_dir / ".delivery-transactions").exists()
