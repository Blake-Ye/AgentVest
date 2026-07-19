import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent.core.confidence_gate import ConfidenceGatePolicy
from multi_agent.core.formal_gate import FORMAL_GATE_REQUIRED_FIELDS
from multi_agent.core.review_contracts import RepairAction, ReviewContract
from multi_agent.tools.investment_tools import build_research_evidence_bundle


@pytest.fixture
def apple_bundle():
    fixture_dir = Path(__file__).parent / "fixtures" / "apple"
    company_facts = json.loads((fixture_dir / "companyfacts.json").read_text(encoding="utf-8"))
    facts = company_facts["facts"]["us-gaap"]
    annual = {
        "end": "2025-09-27",
        "fy": 2025,
        "fp": "FY",
        "form": "10-K",
        "filed": "2025-10-31",
        "accn": "0000320193-25-000079",
    }

    def add_fact(concept: str, unit: str, value: float, *, start: str | None = None) -> None:
        entry = dict(annual, val=value)
        if start:
            entry["start"] = start
        facts[concept] = {"units": {unit: [entry]}}

    add_fact("CashAndCashEquivalentsAtCarryingValue", "USD", 35_900_000_000)
    add_fact("LongTermDebtCurrent", "USD", 10_000_000_000)
    add_fact("LongTermDebtNoncurrent", "USD", 90_000_000_000)
    add_fact(
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "shares",
        15_000_000_000,
        start="2024-09-29",
    )
    return build_research_evidence_bundle(
        company_name="Apple Inc.",
        ticker="AAPL",
        company_facts=company_facts,
        filing_html=(fixture_dir / "filing.html").read_text(encoding="utf-8"),
        filing_metadata={
            "source_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
            "accession": "0000320193-25-000079",
            "filed_at": "2025-10-31",
            "form": "10-K",
            "fiscal_year": 2025,
            "fiscal_period": "FY",
            "period_start": "2024-09-29",
            "period_end": "2025-09-27",
        },
        quote_payload=json.loads((fixture_dir / "quote.json").read_text(encoding="utf-8")),
        tavily_payloads=[json.loads((fixture_dir / "news.json").read_text(encoding="utf-8"))],
    )


def _contract(**overrides: object) -> ReviewContract:
    payload: dict[str, object] = {
        "stage": "analysis_review",
        "reviewer_name": "data_quality_reviewer",
        "decision": {"gate_outcome": "pass", "decision_confidence": "high"},
        "delivery_eligibility": {
            "formal_report_allowed": True,
            "evidence_limited_report_allowed": True,
            "blocked_notice_required": False,
        },
        "failure_taxonomy": {"primary_class": "none"},
        "coverage_summary": {
            "evidence_coverage_ratio": 1.0,
            "financial_coverage_score": 1.0,
            "claim_binding_ratio": 1.0,
        },
        "tool_health_summary": {"overall_status": "healthy"},
        "review_summary": {"one_sentence_summary": "Evidence is complete."},
    }
    payload.update(overrides)
    return ReviewContract.model_validate(payload)


def test_gate_blocks_cross_period_valuation(apple_bundle) -> None:
    mixed = apple_bundle.model_copy(deep=True)
    shares = mixed.require_fact("diluted_shares")
    mixed.financial_facts = [
        fact.model_copy(
            update={
                "fiscal_year": 2024,
                "period_end": "2024-09-28",
                "accession": "0000320193-24-000081",
            }
        )
        if fact is shares
        else fact
        for fact in mixed.financial_facts
    ]

    result = ConfidenceGatePolicy.default().evaluate(mixed, _contract())

    assert result.final_decision == "blocked"
    assert "valuation_period_mismatch" in result.blocking_reasons


def test_gate_requests_targeted_rerun_for_missing_services_fact(apple_bundle) -> None:
    result = ConfidenceGatePolicy.default().evaluate(
        apple_bundle.without_fact("segment_revenue_services"), _contract()
    )

    assert result.final_decision == "rerun"
    assert result.repair_actions == [
        RepairAction(
            target="fundamental_analyst",
            code="missing_financial_fact",
            fields=["segment_revenue_services"],
            instruction="补齐 SEC 期间、filing 标识和来源 URL 后重新审查。",
        )
    ]


def test_gate_does_not_pass_when_reviewer_disallows_formal(apple_bundle) -> None:
    limited = _contract(
        delivery_eligibility={
            "formal_report_allowed": False,
            "evidence_limited_report_allowed": True,
            "blocked_notice_required": False,
        }
    )

    result = ConfidenceGatePolicy.default().evaluate(apple_bundle, limited)

    assert result.final_decision == "evidence_limited"
    assert result.passed is False


def test_new_contract_rejects_legacy_or_unknown_fields() -> None:
    payload = _contract().model_dump()
    payload["legacy_rating"] = "pass"

    with pytest.raises(ValidationError, match="legacy_rating"):
        ReviewContract.model_validate(payload)


def test_gate_at_exact_formal_thresholds_passes_without_prose_override(apple_bundle) -> None:
    bundle = apple_bundle.model_copy(deep=True)
    bundle.tool_health = [
        item.model_copy(update={"status": "healthy"}) for item in bundle.tool_health
    ]
    threshold_contract = _contract(
        coverage_summary={
            "evidence_coverage_ratio": 0.80,
            "financial_coverage_score": 0.80,
            "claim_binding_ratio": 0.75,
        }
    )

    result = ConfidenceGatePolicy.default().evaluate(bundle, threshold_contract)

    assert result.final_decision == "passed"
    assert result.trust_score == 78


def test_degraded_tool_health_requires_repair_at_threshold_boundary(apple_bundle) -> None:
    bundle = apple_bundle.model_copy(deep=True)
    bundle.tool_health = [
        item.model_copy(update={"status": "healthy"}) for item in bundle.tool_health
    ]
    bundle.tool_health[0] = bundle.tool_health[0].model_copy(update={"status": "degraded"})
    threshold_contract = _contract(
        coverage_summary={
            "evidence_coverage_ratio": 0.80,
            "financial_coverage_score": 0.80,
            "claim_binding_ratio": 0.75,
        }
    )

    result = ConfidenceGatePolicy.default().evaluate(bundle, threshold_contract)

    assert result.final_decision == "rerun"
    assert result.trust_score == 73
    assert [action.code for action in result.repair_actions] == [
        "coverage_below_formal_threshold"
    ]


def test_gate_rejects_conflicting_formal_values_for_one_source_period(apple_bundle) -> None:
    bundle = apple_bundle.model_copy(deep=True)
    revenue = bundle.require_fact("revenue")
    bundle.financial_facts.append(revenue.model_copy(update={"value": revenue.value + 1}))

    result = ConfidenceGatePolicy.default().evaluate(bundle, _contract())

    assert result.final_decision == "blocked"
    assert "source_conflict" in result.blocking_reasons


def test_required_fields_remain_the_formal_gate_contract() -> None:
    assert "segment_revenue_services" in FORMAL_GATE_REQUIRED_FIELDS
