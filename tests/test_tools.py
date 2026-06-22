import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent.evaluation import WorkflowEvaluation, activate_evaluation, clear_evaluation
from multi_agent.settings import InvestmentResearchSettings
from multi_agent.tools.investment_tools import (
    FileWriteTool,
    FinancialMetricsTool,
    SecCompanyFactsTool,
    SecFilingSearchTool,
)
from multi_agent.tools.tavily_search import TavilySearchTool


def build_settings() -> InvestmentResearchSettings:
    return InvestmentResearchSettings(
        fast_model="qwen-plus",
        deep_model="qwen-plus",
        review_model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        tavily_api_key="tvly-key",
        sec_api_email="analyst@example.com",
    )


def test_tool_metadata_can_remain_english() -> None:
    tavily_tool = TavilySearchTool(settings=build_settings(), service=None)
    filing_tool = SecFilingSearchTool(settings=build_settings(), service=None)

    assert tavily_tool.name == "Tavily Search Intelligence"
    assert "Search Tavily" in tavily_tool.description
    assert filing_tool.name == "SEC Filing Search"
    assert "official SEC endpoints" in filing_tool.description


def test_file_write_tool_persists_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifacts" / "note.md"
    tool = FileWriteTool()

    response = tool._run(
        file_path=str(artifact_path),
        content="# Analysis\n\nOperating margin improved.",
    )

    assert artifact_path.exists()
    assert "文件已保存到" in response
    assert "Operating margin improved." in artifact_path.read_text(encoding="utf-8")


class StubCompanyFactsService:
    def fetch_company_facts(self, _ticker: str) -> dict:
        return {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "val": 120.0,
                                    "end": "2024-12-31",
                                    "filed": "2025-02-01",
                                    "fy": 2024,
                                }
                            ]
                        }
                    },
                    "AssetsCurrent": {
                        "units": {
                            "USD": [
                                {
                                    "val": 80.0,
                                    "end": "2024-12-31",
                                    "filed": "2025-02-01",
                                    "fy": 2024,
                                }
                            ]
                        }
                    },
                    "LiabilitiesCurrent": {
                        "units": {
                            "USD": [
                                {
                                    "val": 40.0,
                                    "end": "2024-12-31",
                                    "filed": "2025-02-01",
                                    "fy": 2024,
                                }
                            ]
                        }
                    },
                }
            }
        }


def test_sec_company_facts_tool_records_financial_field_extraction_status(tmp_path: Path) -> None:
    settings = InvestmentResearchSettings(
        fast_model="qwen-plus",
        deep_model="qwen-plus",
        review_model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        tavily_api_key="tvly-key",
        sec_api_email="analyst@example.com",
        runs_dir=str(tmp_path / "artifacts"),
        final_report_path=str(tmp_path / "report.md"),
    )
    tool = SecCompanyFactsTool(settings=settings, service=StubCompanyFactsService())
    evaluation = WorkflowEvaluation(
        artifacts_dir=tmp_path / "artifacts",
        final_report_path=tmp_path / "report.md",
        expected_task_outputs={},
        company_name="Apple Inc.",
        company_ticker="AAPL",
    )
    evaluation.start()
    token = activate_evaluation(evaluation)

    try:
        tool._run("AAPL")
        latest_metrics = evaluation.finalize(success=True)
    finally:
        clear_evaluation(token)

    assert latest_metrics["financial_fields"]["revenue"]["extracted"] is True
    assert latest_metrics["financial_fields"]["gross_profit"]["extracted"] is False
    assert latest_metrics["financial_fields"]["current_assets"]["normalized_value"] == 80.0


class FailIfCalledSecService:
    def search_filings(
        self,
        _company_name: str,
        _ticker: str,
        _form_type: str,
        _limit: int,
    ) -> list[dict]:
        raise AssertionError("non-US market should not call SEC service")

    def fetch_company_facts(self, _ticker: str) -> dict:
        raise AssertionError("non-US market should not call SEC service")


def test_sec_company_facts_tool_blocks_non_us_market_before_service_call() -> None:
    settings = InvestmentResearchSettings(
        fast_model="qwen-plus",
        deep_model="qwen-plus",
        review_model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        tavily_api_key="tvly-key",
        sec_api_email="analyst@example.com",
        company_market_label="HK",
    )
    tool = SecCompanyFactsTool(settings=settings, service=FailIfCalledSecService())

    result = tool._run("0700.HK")

    assert "仅 US 市场可用" in result


def test_sec_filing_search_tool_blocks_non_us_market_before_service_call() -> None:
    settings = InvestmentResearchSettings(
        fast_model="qwen-plus",
        deep_model="qwen-plus",
        review_model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        tavily_api_key="tvly-key",
        sec_api_email="analyst@example.com",
        company_market_label="EU",
    )
    tool = SecFilingSearchTool(settings=settings, service=FailIfCalledSecService())

    result = tool._run(company_name="Shell plc", ticker="SHEL.L")

    assert "仅 US 市场可用" in result


def test_financial_metrics_tool_blocks_non_us_market_before_service_call() -> None:
    settings = InvestmentResearchSettings(
        fast_model="qwen-plus",
        deep_model="qwen-plus",
        review_model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        tavily_api_key="tvly-key",
        sec_api_email="analyst@example.com",
        company_market_label="HK",
    )
    tool = FinancialMetricsTool(settings=settings, service=FailIfCalledSecService())

    result = tool._run("0700.HK")

    assert "仅 US 市场可用" in result
