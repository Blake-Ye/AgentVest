import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent.core.formal_gate import FORMAL_GATE_REQUIRED_FIELDS
from multi_agent.tools.investment_tools import build_fcf_snapshot, build_market_snapshot
from multi_agent.tools.review_tools import FinancialFieldCompletenessTool
from multi_agent.core.review_contracts import parse_legacy_review_contract


def test_fcf_snapshot_uses_single_growth_rate_definition() -> None:
    snapshot = build_fcf_snapshot(
        fy2025_fcf=78.28,
        fy2026e_fcf=140.0,
        fy2026e_source_type="external_estimate",
    )

    assert round(snapshot["fy2026e_yoy_growth"], 1) == 78.8


def test_market_snapshot_requires_price_and_diluted_shares_for_market_cap() -> None:
    snapshot = build_market_snapshot(stock_price=None, diluted_shares=15_000)

    assert snapshot["market_cap"] is None
    assert snapshot["ready_for_formal_report"] is False


def test_financial_field_completeness_uses_canonical_gate_required_fields() -> None:
    tool = FinancialFieldCompletenessTool()
    extracted_fields = {field_name: 1 for field_name in FORMAL_GATE_REQUIRED_FIELDS}
    extracted_fields["cash_and_equivalents"] = 0
    extracted_fields["stock_price"] = False

    result = tool._run(
        required_fields=list(FORMAL_GATE_REQUIRED_FIELDS),
        extracted_fields=extracted_fields,
        gate_required_fields=["revenue"],
    )

    assert result["gate_required_field_count"] == len(FORMAL_GATE_REQUIRED_FIELDS)
    assert result["gate_financial_coverage_score"] == (
        len(FORMAL_GATE_REQUIRED_FIELDS) - 1
    ) / len(FORMAL_GATE_REQUIRED_FIELDS)
    assert result["gate_missing_fields"] == ["stock_price"]


def test_financial_field_completeness_cannot_return_full_gate_score_for_empty_input() -> None:
    result = FinancialFieldCompletenessTool()._run(
        required_fields=[],
        extracted_fields={},
    )

    assert result["gate_required_field_count"] == len(FORMAL_GATE_REQUIRED_FIELDS)
    assert result["gate_financial_coverage_score"] == 0.0
    assert result["gate_missing_fields"] == list(FORMAL_GATE_REQUIRED_FIELDS)


def test_legacy_review_contract_requires_explicit_adapter() -> None:
    contract = parse_legacy_review_contract(
        {
            "review_stage": "analysis",
            "decision": "passed",
            "coverage": {
                "evidence_coverage_ratio": 1.0,
                "financial_coverage_score": 1.0,
                "claim_binding_ratio": 1.0,
            },
            "tool_health": {"sec": "healthy"},
            "delivery_eligibility": {"formal_report_allowed": True},
            "failure_taxonomy": {"primary_code": "none"},
        }
    )

    assert contract.stage == "analysis"
    assert contract.decision.gate_outcome == "pass"
