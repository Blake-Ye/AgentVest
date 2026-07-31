from __future__ import annotations

import json
from math import isfinite
from pathlib import Path

from multi_agent.core.delivery import DeliveryPackage, DeliveryValidator
from multi_agent.core.evidence import FinancialFact, periods_are_compatible
from multi_agent.core.formal_gate import FORMAL_GATE_REQUIRED_FIELDS
from multi_agent.core.report_document import REQUIRED_SECTION_KEYS, SECTION_HEADINGS, ReportDocument
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
    assert decision.final_decision == "passed"
    assert decision.final_delivery_state == "formal_report"
    assert FinalDecisionRecord.model_validate(result["final_decision_record"]) == decision
    assert ReportDocument.model_validate(result["report_document"]) == document
    assert result["trust_score"] == decision.trust_score
    assert result["blocking_reasons"] == decision.blocking_reasons
    assert ReviewContract.model_validate(result["analysis_review_contract"]) == (
        decision.analysis_review_contract
    )
    assert ReviewContract.model_validate(result["report_review_contract"]) == (
        decision.report_review_contract
    )
    assert apple_pipeline.analysis_kickoffs == 1
    assert apple_pipeline.report_kickoffs == 1
    assert apple_pipeline.official_session is not None
    assert apple_pipeline.tavily_session is not None
    assert apple_pipeline.official_session.calls == [
        "https://www.sec.gov/files/company_tickers.json",
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        "https://data.sec.gov/submissions/CIK0000320193.json",
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
        "https://api.nasdaq.com/api/quote/AAPL/info?assetclass=stocks",
    ]
    assert apple_pipeline.tavily_session.calls == ["https://api.tavily.com/search"]

    for filename in (
        "00_market_validation.md",
        "01_market_intelligence.md",
        "02_filing_review.md",
        "03_financial_analysis.md",
        "08_data_quality_review.md",
        "09_logic_compliance_review.md",
    ):
        content = (run_dir / filename).read_text(encoding="utf-8")
        assert "<!-- PLACEHOLDER -->" not in content
        assert "Apple Inc." in content and "AAPL" in content
    assert "market_label=US" in (run_dir / "00_market_validation.md").read_text(encoding="utf-8")
    assert "https://news.example.com/apple-services" in (
        run_dir / "01_market_intelligence.md"
    ).read_text(encoding="utf-8")
    filing_review = (run_dir / "02_filing_review.md").read_text(encoding="utf-8")
    assert "0000320193-25-000079" in filing_review and "https://www.sec.gov/Archives/" in filing_review
    financial_analysis = (run_dir / "03_financial_analysis.md").read_text(encoding="utf-8")
    assert "416161000000 USD" in financial_analysis and "claim:revenue" in financial_analysis
    for path, stage in (
        (run_dir / "08_data_quality_review.md", "analysis_review"),
        (run_dir / "09_logic_compliance_review.md", "report_review"),
    ):
        contract_payload = json.loads(path.read_text(encoding="utf-8").split("```json\n", 1)[1].split("\n```", 1)[0])
        assert ReviewContract.model_validate(contract_payload).stage == stage

    assert recommendation["summary"] and recommendation["catalysts"] and recommendation["risks"]
    assert tuple(structured_report["sections"]) == REQUIRED_SECTION_KEYS
    assert all(structured_report["sections"].values())
    assert set(document.sections) == set(REQUIRED_SECTION_KEYS)
    assert all(section.content for section in document.sections.values())
    required_fields = set(FORMAL_GATE_REQUIRED_FIELDS)
    assert required_fields == {
        "revenue",
        "cash_and_equivalents",
        "total_debt",
        "diluted_shares",
        "stock_price",
        "segment_revenue_services",
    }
    facts_by_name = {
        fact["field_name"]: FinancialFact.model_validate(fact)
        for fact in evidence["financial_facts"]
    }
    required_financial_fields = required_fields - {"stock_price"}
    assert required_financial_fields <= set(facts_by_name)
    for field_name in required_financial_fields:
        fact = facts_by_name[field_name]
        assert isfinite(fact.value)
        assert fact.unit in {"USD", "shares"}
        assert fact.source_url and fact.source_url.startswith("https://")
        assert fact.accession == "0000320193-25-000079"
        assert fact.form == "10-K" and fact.filed_at is not None
        assert fact.fiscal_year == 2025 and fact.fiscal_period == "FY" and fact.period_end is not None
    assert periods_are_compatible(facts_by_name["revenue"], facts_by_name["diluted_shares"])
    assert periods_are_compatible(facts_by_name["revenue"], facts_by_name["total_debt"])
    assert periods_are_compatible(facts_by_name["revenue"], facts_by_name["segment_revenue_services"])
    assert facts_by_name["total_debt"].value == (
        facts_by_name["debt_current"].value + facts_by_name["debt_noncurrent"].value
    )
    assert "LongTermDebtCurrent" in (facts_by_name["total_debt"].taxonomy_concept or "")
    quote = evidence["market_snapshots"][0]
    assert isfinite(float(quote["price"])) and float(quote["price"]) > 0
    assert quote["currency"] == "USD" and quote["observed_at"] and quote["source_url"].startswith("https://")
    assert quote["diluted_shares_period_end"] == facts_by_name["diluted_shares"].period_end.isoformat()

    assert decision.analysis_review_contract is not None
    assert decision.report_review_contract is not None
    assert decision.analysis_review_contract.decision.gate_outcome == "pass"
    assert decision.report_review_contract.decision.gate_outcome == "pass"

    package = DeliveryPackage(
        decision_json=(run_dir / "final_decision.json").read_text(encoding="utf-8"),
        document_json=(run_dir / "11_report_document.json").read_text(encoding="utf-8"),
        markdown=(run_dir / "04_investment_report.md").read_text(encoding="utf-8"),
        recommendation_json=(run_dir / "06_structured_recommendation.json").read_text(encoding="utf-8"),
        structured_report_json=(run_dir / "07_structured_report.json").read_text(encoding="utf-8"),
    )
    assert DeliveryValidator().validate_package(package).valid
    markdown = (run_dir / "04_investment_report.md").read_text(encoding="utf-8")
    heading_positions = [markdown.index(f"## {SECTION_HEADINGS[key]}") for key in REQUIRED_SECTION_KEYS]
    assert heading_positions == sorted(heading_positions)
    assert "[claim:revenue]" in markdown
    for source in document.sources:
        assert f"[{source.source_id}]({source.url})" in markdown
        assert source.url in markdown
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
