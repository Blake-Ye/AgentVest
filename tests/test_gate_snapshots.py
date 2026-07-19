import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent.tools.investment_tools import build_fcf_snapshot, build_market_snapshot
from multi_agent.tools.review_tools import FinancialFieldCompletenessTool


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


def test_financial_field_completeness_supports_gate_required_fields() -> None:
    tool = FinancialFieldCompletenessTool()

    result = tool._run(
        required_fields=["Revenue", "CashAndEquivalents"],
        extracted_fields={"Revenue": 1},
        gate_required_fields=["Revenue", "CashAndEquivalents"],
    )

    assert result["gate_financial_coverage_score"] == 0.5
    assert result["gate_missing_fields"] == ["CashAndEquivalents"]
