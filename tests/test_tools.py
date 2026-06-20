import json
from pathlib import Path

import requests

from multi_agent.evaluation import WorkflowEvaluation, activate_evaluation, clear_evaluation
from multi_agent.settings import InvestmentResearchSettings
from multi_agent.market_profile import MarketIdentifierService, MarketScope
from multi_agent.tools.investment_tools import (
    GoogleSearchService,
    FileWriteTool,
    FatalAPIError,
    GoogleSearchTool,
    MarketProfileTool,
    OfficialDisclosureSearchTool,
    SecCompanyFactsTool,
    SecFilingSearchTool,
)


class StubGoogleSearchService:
    def search_company_news(self, company_name: str, query: str) -> list[dict[str, str]]:
        return [
            {
                "title": "Revenue acceleration",
                "link": "https://example.com/revenue",
                "snippet": f"{company_name} posted faster revenue growth.",
            }
        ]


def build_settings() -> InvestmentResearchSettings:
    return InvestmentResearchSettings(
        model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        search_provider="auto",
        serper_api_key="serper-key",
        serpapi_api_key="",
        sec_api_key="sec-key",
        sec_api_email="analyst@example.com",
    )


def test_google_search_tool_formats_search_results() -> None:
    tool = GoogleSearchTool(
        settings=build_settings(),
        service=StubGoogleSearchService(),
    )

    result = tool._run(company_name="NVIDIA", query="latest earnings")

    assert "Revenue acceleration" in result
    assert "https://example.com/revenue" in result
    assert "NVIDIA posted faster revenue growth." in result
    assert "链接：" in result
    assert "摘要：" in result


def test_tool_metadata_can_remain_english() -> None:
    google_tool = GoogleSearchTool(settings=build_settings(), service=StubGoogleSearchService())
    filing_tool = SecFilingSearchTool(settings=build_settings(), service=None)

    assert google_tool.name == "Google Search Intelligence"
    assert "Search Google" in google_tool.description
    assert filing_tool.name == "SEC Filing Search"
    assert "SEC API" in filing_tool.description


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


class FakeResponse:
    def __init__(self, status_code: int, payload: dict, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or str(payload)

    def json(self) -> dict:
        return self._payload


class RecordingSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def post(self, url: str, **kwargs):
        self.calls.append(("POST", url))
        return FakeResponse(status_code=403, payload={"message": "Unauthorized"}, text="Unauthorized")

    def get(self, url: str, **kwargs):
        self.calls.append(("GET", url))
        return FakeResponse(
            status_code=200,
            payload={
                "organic_results": [
                    {
                        "title": "Apple Earnings Beat Expectations",
                        "link": "https://example.com/apple-earnings",
                        "snippet": "Apple reported stronger-than-expected earnings.",
                    }
                ]
            },
        )


def test_google_search_service_falls_back_to_serpapi_on_auth_failure() -> None:
    settings = InvestmentResearchSettings(
        model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        search_provider="auto",
        serper_api_key="legacy-serpapi-key",
        serpapi_api_key="",
        sec_api_key="sec-key",
        sec_api_email="analyst@example.com",
    )
    session = RecordingSession()
    service = GoogleSearchService(settings=settings, session=session)

    results = service.search_company_news(company_name="Apple Inc.", query="latest earnings")

    assert results[0]["title"] == "Apple Earnings Beat Expectations"
    assert session.calls == [
        ("POST", "https://google.serper.dev/search"),
        ("GET", "https://serpapi.com/search.json"),
    ]


def test_google_search_service_records_api_failure_rate_in_evaluation(tmp_path: Path) -> None:
    settings = InvestmentResearchSettings(
        model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        search_provider="auto",
        serper_api_key="legacy-serpapi-key",
        serpapi_api_key="",
        sec_api_key="sec-key",
        sec_api_email="analyst@example.com",
        artifacts_dir=str(tmp_path / "artifacts"),
        final_report_path=str(tmp_path / "report.md"),
    )
    session = RecordingSession()
    service = GoogleSearchService(settings=settings, session=session)
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
        service.search_company_news(company_name="Apple Inc.", query="latest earnings")
        latest_metrics = evaluation.finalize(success=True)
    finally:
        clear_evaluation(token)

    assert latest_metrics["api_calls"]["total"] == 2
    assert latest_metrics["api_calls"]["failures"] == 1
    assert latest_metrics["api_calls"]["per_service"]["Google Search"]["failure_rate"] == 0.5


class StubCompanyFactsService:
    def fetch_company_facts(self, ticker: str) -> dict:
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
        model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        search_provider="auto",
        serper_api_key="serper-key",
        serpapi_api_key="",
        sec_api_key="sec-key",
        sec_api_email="analyst@example.com",
        artifacts_dir=str(tmp_path / "artifacts"),
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


def test_market_identifier_classifies_common_global_tickers() -> None:
    service = MarketIdentifierService(settings=build_settings())

    aapl = service.identify(company_name="Apple Inc.", ticker="AAPL")
    xiaomi = service.identify(company_name="Xiaomi Corporation", ticker="1810.HK")
    maotai = service.identify(company_name="Kweichow Moutai", ticker="600519.SH")
    asml = service.identify(company_name="ASML Holding", ticker="ASML.AS")

    assert aapl.market_scope == MarketScope.US_SEC.value
    assert aapl.sec_applicable is True
    assert xiaomi.market_scope == MarketScope.HKEX.value
    assert xiaomi.sec_applicable is False
    assert maotai.market_scope == MarketScope.CN_A_SHARE.value
    assert asml.market_scope == MarketScope.EU_LISTED.value


def test_sec_filing_search_tool_returns_degraded_payload_for_non_sec_ticker() -> None:
    tool = SecFilingSearchTool(settings=build_settings())

    payload = json.loads(tool._run(company_name="Xiaomi Corporation", ticker="1810.HK"))

    assert payload["status"] == "degraded"
    assert payload["error_type"] == "market_not_applicable"
    assert payload["source"] == "sec_provider"
    assert payload["details"]["market_scope"] == MarketScope.HKEX.value
    assert payload["details"]["sec_applicable"] is False


def test_sec_company_facts_tool_returns_degraded_payload_for_non_sec_ticker() -> None:
    tool = SecCompanyFactsTool(settings=build_settings())

    payload = json.loads(tool._run("600519.SH"))

    assert payload["status"] == "degraded"
    assert payload["source"] == "sec_provider"
    assert payload["details"]["market_scope"] == MarketScope.CN_A_SHARE.value
    assert payload["details"]["sec_applicable"] is False


def test_official_disclosure_tool_routes_non_sec_ticker_to_market_entrypoint() -> None:
    tool = OfficialDisclosureSearchTool(settings=build_settings())

    payload = json.loads(tool._run(company_name="Xiaomi Corporation", ticker="1810.HK"))

    assert payload["status"] == "success"
    assert payload["source"] == "hkex_provider"
    assert payload["data"]["issuer_profile"]["market_scope"] == MarketScope.HKEX.value
    assert payload["data"]["official_entrypoint"].startswith("https://www.hkexnews.hk")
    assert "Annual Report" in payload["data"]["priority_documents"]


def test_official_disclosure_tool_routes_cn_ticker_to_cninfo() -> None:
    tool = OfficialDisclosureSearchTool(settings=build_settings())

    payload = json.loads(tool._run(company_name="Kweichow Moutai", ticker="600519.SH"))

    assert payload["status"] == "success"
    assert payload["source"] == "cn_provider"
    assert payload["data"]["issuer_profile"]["market_scope"] == MarketScope.CN_A_SHARE.value
    assert "cninfo.com.cn" in payload["data"]["official_entrypoint"]
    assert payload["data"]["query_url"].startswith("http://www.cninfo.com.cn")


def test_official_disclosure_tool_routes_eu_ticker_to_exchange_entrypoint() -> None:
    tool = OfficialDisclosureSearchTool(settings=build_settings())

    payload = json.loads(tool._run(company_name="ASML Holding", ticker="ASML.AS"))

    assert payload["status"] == "success"
    assert payload["source"] == "eu_provider"
    assert payload["data"]["issuer_profile"]["market_scope"] == MarketScope.EU_LISTED.value
    assert payload["data"]["official_entrypoint"].startswith("https://live.euronext.com")


def test_official_disclosure_tool_routes_xetra_ticker_to_deutsche_boerse() -> None:
    tool = OfficialDisclosureSearchTool(settings=build_settings())

    payload = json.loads(tool._run(company_name="SAP SE", ticker="SAP.DE"))

    assert payload["status"] == "success"
    assert payload["source"] == "xetra_provider"
    assert payload["data"]["official_entrypoint"].startswith("https://www.boerse-frankfurt.de")


def test_official_disclosure_tool_routes_six_ticker_to_six_exchange() -> None:
    tool = OfficialDisclosureSearchTool(settings=build_settings())

    payload = json.loads(tool._run(company_name="Nestle SA", ticker="NESN.SW"))

    assert payload["status"] == "success"
    assert payload["source"] == "six_provider"
    assert payload["data"]["official_entrypoint"].startswith("https://www.six-group.com")


def test_market_profile_tool_returns_provider_capabilities() -> None:
    tool = MarketProfileTool(settings=build_settings())

    payload = json.loads(tool._run(company_name="Xiaomi Corporation", ticker="1810.HK"))

    assert payload["status"] == "success"
    assert payload["source"] == "market_profile"
    assert payload["data"]["issuer_profile"]["market_scope"] == MarketScope.HKEX.value
    assert payload["data"]["provider"]["source_name"] == "hkex_provider"
    assert payload["data"]["provider"]["supports_structured_facts"] is False
