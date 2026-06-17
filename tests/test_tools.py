from pathlib import Path
import json
from unittest.mock import patch

import requests
import pytest

from multi_agent.evaluation import WorkflowEvaluation, activate_evaluation, clear_evaluation
from multi_agent.settings import InvestmentResearchSettings
from multi_agent.tools.investment_tools import (
    GoogleSearchService,
    FileWriteTool,
    FatalAPIError,
    FinancialMetricsTool,
    GoogleSearchTool,
    SecApiService,
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


def test_google_search_tool_returns_structured_error_payload_on_nonfatal_failure() -> None:
    class FailingService:
        def search_company_news(self, company_name: str, query: str) -> list[dict[str, str]]:
            raise RuntimeError("timeout")

    tool = GoogleSearchTool(settings=build_settings(), service=FailingService())

    payload = json.loads(tool._run(company_name="Apple Inc.", query="latest earnings"))

    assert payload["success"] is False
    assert payload["data"] is None
    assert payload["error"]["service_name"] == "Google Search"
    assert payload["error"]["error_code"] == "google_search_failed"
    assert payload["error"]["message"] == "timeout"
    assert payload["error"]["retryable"] is True


def test_tool_metadata_can_remain_english() -> None:
    google_tool = GoogleSearchTool(settings=build_settings(), service=StubGoogleSearchService())
    filing_tool = SecFilingSearchTool(settings=build_settings(), service=None)

    assert google_tool.name == "Google Search Intelligence"
    assert "Search Google" in google_tool.description
    assert filing_tool.name == "SEC Filing Search"
    assert "SEC EDGAR" in filing_tool.description


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


class StubCompanyFactsServiceWithNullableFiscalYear:
    def fetch_company_facts(self, ticker: str) -> dict:
        return {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "val": 100.0,
                                    "end": "2024-12-31",
                                    "filed": "2025-02-01",
                                    "fy": None,
                                },
                                {
                                    "val": 120.0,
                                    "end": "2024-12-31",
                                    "filed": "2025-02-01",
                                    "fy": 2024,
                                },
                            ]
                        }
                    },
                    "GrossProfit": {
                        "units": {
                            "USD": [
                                {
                                    "val": 48.0,
                                    "end": "2024-12-31",
                                    "filed": "2025-02-01",
                                    "fy": None,
                                }
                            ]
                        }
                    },
                    "OperatingIncomeLoss": {
                        "units": {
                            "USD": [
                                {
                                    "val": 30.0,
                                    "end": "2024-12-31",
                                    "filed": "2025-02-01",
                                    "fy": None,
                                }
                            ]
                        }
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {
                                    "val": 24.0,
                                    "end": "2024-12-31",
                                    "filed": "2025-02-01",
                                    "fy": None,
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
                                    "fy": None,
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
                                    "fy": None,
                                }
                            ]
                        }
                    },
                    "Assets": {
                        "units": {
                            "USD": [
                                {
                                    "val": 200.0,
                                    "end": "2024-12-31",
                                    "filed": "2025-02-01",
                                    "fy": None,
                                }
                            ]
                        }
                    },
                    "Liabilities": {
                        "units": {
                            "USD": [
                                {
                                    "val": 90.0,
                                    "end": "2024-12-31",
                                    "filed": "2025-02-01",
                                    "fy": None,
                                }
                            ]
                        }
                    },
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {
                            "USD": [
                                {
                                    "val": 20.0,
                                    "end": "2024-12-31",
                                    "filed": "2025-02-01",
                                    "fy": None,
                                }
                            ]
                        }
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {
                            "USD": [
                                {
                                    "val": 5.0,
                                    "end": "2024-12-31",
                                    "filed": "2025-02-01",
                                    "fy": None,
                                }
                            ]
                        }
                    },
                }
            }
        }


def test_sec_company_facts_and_financial_metrics_handle_nullable_fiscal_year() -> None:
    settings = build_settings()
    service = StubCompanyFactsServiceWithNullableFiscalYear()

    facts_tool = SecCompanyFactsTool(settings=settings, service=service)
    metrics_tool = FinancialMetricsTool(settings=settings, service=service)

    facts_payload = json.loads(facts_tool._run("AAPL"))
    metrics_payload = json.loads(metrics_tool._run("AAPL"))

    assert facts_payload["revenue"] == 120.0
    assert metrics_payload["gross_margin"] == 0.4
    assert metrics_payload["current_ratio"] == 2.0


class SequencedSecSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def post(self, url: str, **kwargs):
        self.calls.append(url)
        if not self._responses:
            raise AssertionError(f"未为请求准备响应：{url}")
        return self._responses.pop(0)

    def get(self, url: str, **kwargs):
        self.calls.append(url)
        if not self._responses:
            raise AssertionError(f"未为请求准备响应：{url}")
        return self._responses.pop(0)


def test_sec_company_facts_retries_429_with_backoff_until_success() -> None:
    settings = build_settings()
    session = SequencedSecSession(
        [
            FakeResponse(
                status_code=200,
                payload={"0": {"ticker": "AAPL", "cik_str": 320193}},
            ),
            FakeResponse(status_code=429, payload={"message": "Too Many Requests"}, text="Too Many Requests"),
            FakeResponse(status_code=429, payload={"message": "Too Many Requests"}, text="Too Many Requests"),
            FakeResponse(
                status_code=200,
                payload={"facts": {"us-gaap": {"Revenues": {"units": {"USD": [{"val": 10.0}]}}}}},
            ),
        ]
    )
    service = SecApiService(settings=settings, session=session)
    sleep_calls: list[float] = []

    with patch("time.sleep", side_effect=lambda seconds: sleep_calls.append(seconds)):
        payload = service.fetch_company_facts("AAPL")

    assert payload["facts"]["us-gaap"]["Revenues"]["units"]["USD"][0]["val"] == 10.0
    assert session.calls == [
        service.SEC_TICKER_LOOKUP_URL,
        service.SEC_COMPANY_FACTS_URL.format(cik="0000320193"),
        service.SEC_COMPANY_FACTS_URL.format(cik="0000320193"),
        service.SEC_COMPANY_FACTS_URL.format(cik="0000320193"),
    ]
    assert len(sleep_calls) == 2
    assert sleep_calls[0] < sleep_calls[1]


def test_sec_filing_search_uses_sec_submissions_feed() -> None:
    settings = build_settings()
    session = SequencedSecSession(
        [
            FakeResponse(
                status_code=200,
                payload={"0": {"ticker": "AAPL", "cik_str": 320193}},
            ),
            FakeResponse(
                status_code=200,
                payload={
                    "filings": {
                        "recent": {
                            "form": ["8-K", "10-Q", "10-K"],
                            "filingDate": ["2025-02-10", "2025-01-31", "2024-11-01"],
                            "accessionNumber": [
                                "0000320193-25-000010",
                                "0000320193-25-000009",
                                "0000320193-24-000123",
                            ],
                            "primaryDocument": ["earnings.htm", "q1.htm", "annual.htm"],
                            "primaryDocDescription": [
                                "Current report",
                                "Quarterly report",
                                "Annual report",
                            ],
                        }
                    }
                },
            ),
        ]
    )
    service = SecApiService(settings=settings, session=session)

    filings = service.search_filings("Apple Inc.", "AAPL", "10-Q", 1)

    assert filings == [
        {
            "form_type": "10-Q",
            "filed_at": "2025-01-31",
            "filing_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000009/q1.htm",
            "filing_details": "Quarterly report",
            "accession_no": "0000320193-25-000009",
        }
    ]
    assert session.calls == [
        service.SEC_TICKER_LOOKUP_URL,
        service.SEC_SUBMISSIONS_URL.format(cik="0000320193"),
    ]
