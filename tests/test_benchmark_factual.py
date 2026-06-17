from multi_agent.benchmark_dataset import BenchmarkSample
from multi_agent.benchmark_factual import FactualChecker


def test_factual_checker_matches_numeric_and_text_facts() -> None:
    sample = BenchmarkSample(
        company_name="Apple Inc.",
        ticker="AAPL",
        expected_facts={
            "revenue": {"value": 391_000_000_000, "tolerance": 0.02},
            "net_income": {"value": 93_700_000_000, "tolerance": 0.02},
            "form_type": "10-K",
            "filing_year": 2024,
        },
        expected_key_points={"risks": [], "catalysts": [], "thesis": []},
        metadata={},
    )
    report_text = """
    # 投资备忘录

    Revenue: $391.0 billion
    Net income: $93.7 billion
    Form type: 10-K
    Filing year: 2024
    """

    result = FactualChecker().evaluate(sample=sample, report_text=report_text)

    assert result["factual_accuracy"] == 1.0
    assert result["fields_checked"] == 4
    assert result["checks"]["revenue"]["matched"] is True
    assert result["checks"]["net_income"]["actual_value"] == 93_700_000_000
    assert result["checks"]["form_type"]["actual_value"] == "10-K"
