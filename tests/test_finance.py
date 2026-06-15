from multi_agent.finance import CompanyFinancialSnapshot, compute_key_metrics


def test_compute_key_metrics_from_financial_snapshot() -> None:
    snapshot = CompanyFinancialSnapshot(
        revenue=1200.0,
        gross_profit=720.0,
        operating_income=300.0,
        net_income=180.0,
        current_assets=900.0,
        current_liabilities=450.0,
        total_assets=2400.0,
        total_liabilities=1400.0,
        operating_cash_flow=260.0,
        capital_expenditure=-80.0,
    )

    metrics = compute_key_metrics(snapshot)

    assert metrics["gross_margin"] == 0.6
    assert metrics["operating_margin"] == 0.25
    assert metrics["net_margin"] == 0.15
    assert metrics["current_ratio"] == 2.0
    assert metrics["debt_to_assets"] == 1400.0 / 2400.0
    assert metrics["free_cash_flow"] == 180.0
