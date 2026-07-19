import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent.settings import InvestmentResearchSettings
from multi_agent.tools.investment_tools import (
    SecFilingSearchTool,
    _extract_latest_fact,
)


def test_extract_latest_fact_accepts_entries_with_none_fy() -> None:
    company_facts = {
        "facts": {
            "us-gaap": {
                "AssetsCurrent": {
                    "units": {
                        "USD": [
                            {"val": 10, "end": "2024-09-28", "filed": "2024-11-01", "fy": None},
                            {"val": 9, "end": "2024-09-28", "filed": "2024-11-01", "fy": 2023},
                        ]
                    }
                }
            }
        }
    }

    result = _extract_latest_fact(company_facts, ["AssetsCurrent"])

    assert result.extracted is True
    assert result.normalized_value == 9.0


def test_sec_filing_search_returns_structured_failure_message_on_request_error() -> None:
    class BrokenService:
        def search_filings(self, **_: object) -> list[dict[str, str]]:
            raise RuntimeError("ssl eof")

    settings = InvestmentResearchSettings(
        fast_model="qwen-v4-flash",
        deep_model="qwen-v4-pro",
        review_model="qwen-v4-review",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        tavily_api_key="tvly-key",
        sec_api_email="analyst@example.com",
    )
    tool = SecFilingSearchTool(
        settings=settings,
        service=BrokenService(),  # type: ignore[arg-type]
    )

    output = tool._run("Apple Inc.", "AAPL", "10-K", 3)

    assert "SEC 文件检索失败" in output
    assert "ssl eof" in output
