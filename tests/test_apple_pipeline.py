from __future__ import annotations

import json
from pathlib import Path

from multi_agent.core.delivery import DeliveryPackage, DeliveryValidator
from multi_agent.core.formal_gate import FORMAL_GATE_REQUIRED_FIELDS
from multi_agent.core.report_document import REQUIRED_SECTION_KEYS, ReportDocument
from multi_agent.core.review_contracts import FinalDecisionRecord, ReviewContract


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_apple_fixture_run_produces_complete_formal_delivery(tmp_path: Path, apple_pipeline) -> None:
    result, run_dir = apple_pipeline.run(tmp_path)

    expected_artifacts = {
        "00_market_validation.md",
        "01_market_intelligence.md",
        "02_filing_review.md",
        "03_financial_analysis.md",
        "04_investment_report.md",
        "05_runtime.txt",
        "06_structured_recommendation.json",
        "07_structured_report.json",
        "08_data_quality_review.md",
        "09_logic_compliance_review.md",
        "10_research_evidence.json",
        "11_report_document.json",
        "final_decision.json",
        "latest_run_metrics.json",
        "evaluation_summary.json",
        "README.md",
    }
    assert expected_artifacts <= {path.name for path in run_dir.iterdir()}

    decision = FinalDecisionRecord.model_validate(_load_json(run_dir / "final_decision.json"))
    recommendation = _load_json(run_dir / "06_structured_recommendation.json")
    structured_report = _load_json(run_dir / "07_structured_report.json")
    evidence = _load_json(run_dir / "10_research_evidence.json")
    document = ReportDocument.model_validate(_load_json(run_dir / "11_report_document.json"))
    metrics = _load_json(run_dir / "latest_run_metrics.json")

    assert result["status"] == "passed"
    assert decision.final_delivery_state == "formal_report"
    assert recommendation["summary"] and recommendation["catalysts"] and recommendation["risks"]
    assert tuple(structured_report["sections"]) == REQUIRED_SECTION_KEYS
    assert all(structured_report["sections"].values())
    assert set(document.sections) == set(REQUIRED_SECTION_KEYS)
    assert all(section.content for section in document.sections.values())
    assert all(
        fact["source_url"]
        and fact["period_end"]
        and fact["accession"]
        and fact["form"]
        and fact["source_tag"]
        for fact in evidence["financial_facts"]
        if fact["field_name"] in FORMAL_GATE_REQUIRED_FIELDS and fact["field_name"] != "stock_price"
    )
    assert ReviewContract.model_validate(result["analysis_review_contract"]).decision.gate_outcome == "pass"
    assert ReviewContract.model_validate(result["report_review_contract"]).decision.gate_outcome == "pass"

    package = DeliveryPackage(
        decision_json=(run_dir / "final_decision.json").read_text(encoding="utf-8"),
        document_json=(run_dir / "11_report_document.json").read_text(encoding="utf-8"),
        markdown=(run_dir / "04_investment_report.md").read_text(encoding="utf-8"),
        recommendation_json=(run_dir / "06_structured_recommendation.json").read_text(encoding="utf-8"),
        structured_report_json=(run_dir / "07_structured_report.json").read_text(encoding="utf-8"),
    )
    assert DeliveryValidator().validate_package(package).valid
    assert metrics["success"] is True
    assert all(
        metrics[name] is True
        for name in (
            "report_sections_complete",
            "structured_outputs_complete",
            "formal_fact_provenance_complete",
            "decision_projection_consistent",
            "delivery_validation_passed",
        )
    )
