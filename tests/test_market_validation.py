import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent.core.market import build_tool_policy
from multi_agent.settings import InvestmentResearchSettings
from multi_agent.tools.market_validation import MarketValidationService, MarketValidationTool


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


class StubOfficialSecService:
    def __init__(self, known_tickers: dict[str, dict[str, str]] | None = None) -> None:
        self.known_tickers = known_tickers or {}

    def lookup_company_by_ticker(self, ticker: str):
        return self.known_tickers.get(ticker.upper())


def test_market_validation_marks_hk_suffix_as_hk_and_blocks_sec() -> None:
    tool = MarketValidationTool(
        service=MarketValidationService(
            settings=build_settings(),
            official_sec_service=StubOfficialSecService(),
        )
    )

    result = tool._run(company_name="Tencent Holdings", ticker="0700.HK", exchange="")

    assert result["market_label"] == "HK"
    assert result["resolution_status"] == "confirmed"
    assert result["tool_policy"]["sec_allowed"] is False
    assert result["requires_human_confirmation"] is False


def test_market_validation_confirms_us_when_exchange_is_nyse() -> None:
    tool = MarketValidationTool(
        service=MarketValidationService(
            settings=build_settings(),
            official_sec_service=StubOfficialSecService(),
        )
    )

    result = tool._run(company_name="Apple Inc.", ticker="AAPL", exchange="NYSE")

    assert result["market_label"] == "US"
    assert result["resolution_status"] == "confirmed"
    assert result["tool_policy"]["sec_allowed"] is True
    assert result["tool_policy"]["tavily_allowed"] is True


def test_market_validation_falls_back_to_unresolved_when_no_signal_exists() -> None:
    tool = MarketValidationTool(
        service=MarketValidationService(
            settings=build_settings(),
            official_sec_service=StubOfficialSecService(),
        )
    )

    result = tool._run(company_name="Completely Unknown", ticker="", exchange="")

    assert result["market_label"] == "UNRESOLVED"
    assert result["resolution_status"] == "unresolved"
    assert result["requires_human_confirmation"] is True
    assert "investment recommendation" in result["tool_policy"]["blocked_conclusions"]


def test_build_tool_policy_allows_sec_only_for_us() -> None:
    assert build_tool_policy("US").sec_allowed is True
    assert build_tool_policy("HK").sec_allowed is False
    assert build_tool_policy("EU").sec_allowed is False
    assert build_tool_policy("UNRESOLVED").sec_allowed is False
