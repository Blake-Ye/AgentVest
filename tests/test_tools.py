import sys
import json
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent.evaluation import WorkflowEvaluation, activate_evaluation, clear_evaluation
from multi_agent.settings import InvestmentResearchSettings
from multi_agent.tools.investment_tools import (
    FileWriteTool,
    FinancialMetricsTool,
    SecCompanyFactsTool,
    SecFilingSearchTool,
    build_research_evidence_bundle,
    _extract_latest_fact,
    _extract_services_revenue_from_filing_html,
)
from multi_agent.tools.tavily_search import TavilySearchTool
from multi_agent.tools.official_sec import FatalAPIError


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


@pytest.fixture
def apple_sources() -> dict[str, object]:
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
    return {
        "company_name": "Apple Inc.",
        "ticker": "AAPL",
        "company_facts": company_facts,
        "filing_html": (fixture_dir / "filing.html").read_text(encoding="utf-8"),
        "filing_metadata": {
            "source_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
            "accession": "0000320193-25-000079",
            "filed_at": "2025-10-31",
            "form": "10-K",
            "fiscal_year": 2025,
            "fiscal_period": "FY",
            "period_start": "2024-09-29",
            "period_end": "2025-09-27",
        },
        "quote_payload": json.loads((fixture_dir / "quote.json").read_text(encoding="utf-8")),
        "tavily_payloads": [json.loads((fixture_dir / "news.json").read_text(encoding="utf-8"))],
    }


def test_build_bundle_contains_all_formal_gate_facts(apple_sources: dict[str, object]) -> None:
    bundle = build_research_evidence_bundle(**apple_sources)

    assert {fact.field_name for fact in bundle.formal_facts()} >= {
        "revenue",
        "cash_and_equivalents",
        "total_debt",
        "diluted_shares",
        "segment_revenue_services",
    }
    quote = bundle.market_snapshots[0]
    assert quote.price > 0
    assert quote.observed_at is not None
    assert quote.source_url
    assert bundle.tool_status("sec_company_facts") == "healthy"
    assert bundle.tool_status("quote") == "healthy"
    assert bundle.tool_status("tavily") == "degraded"
    assert any(gap.code == "independent_event_sources_insufficient" for gap in bundle.gaps)


def test_tavily_events_require_distinct_valid_source_domains(apple_sources: dict[str, object]) -> None:
    sources = deepcopy(apple_sources)
    sources["tavily_payloads"] = [
        {
            "status": "ok",
            "artifact_ref": "https://api.tavily.com/search/request-1",
            "results": [
                {
                    "title": "Apple Services update",
                    "event_key": "apple-services-growth",
                    "url": "https://news.example.com/apple-services",
                    "source_type": "news",
                    "published_at": "2026-07-18T12:00:00+00:00",
                },
                {
                    "title": "Apple Services growth corroboration",
                    "event_key": "apple-services-growth",
                    "url": "https://wire.example.net/apple-valuation",
                    "source_type": "news",
                    "published_at": "2026-07-18T13:00:00+00:00",
                },
                {
                    "title": "Broken source",
                    "url": "not-a-url",
                    "source_type": "news",
                    "published_at": "2026-07-18T14:00:00+00:00",
                },
            ],
        }
    ]

    bundle = build_research_evidence_bundle(**sources)

    assert bundle.tool_status("tavily") == "degraded"
    assert len(bundle.events) == 2
    assert all(event.independently_confirmed for event in bundle.events)
    assert "https://api.tavily.com/search/request-1" in bundle.raw_artifact_refs
    assert all(event.source_url.startswith("https://") for event in bundle.events)
    assert any(gap.code == "independent_event_sources_insufficient" for gap in bundle.gaps)


def test_tavily_does_not_confirm_unrelated_events_across_domains(
    apple_sources: dict[str, object],
) -> None:
    sources = deepcopy(apple_sources)
    sources["tavily_payloads"] = [
        {
            "status": "ok",
            "results": [
                {
                    "title": "Apple launches a product",
                    "event_key": "apple-product-launch",
                    "url": "https://news.example.com/apple-product",
                    "source_type": "news",
                },
                {
                    "title": "Apple faces a lawsuit",
                    "event_key": "apple-lawsuit",
                    "url": "https://wire.example.net/apple-lawsuit",
                    "source_type": "news",
                },
            ],
        }
    ]

    bundle = build_research_evidence_bundle(**sources)

    assert bundle.tool_status("tavily") == "degraded"
    assert all(event.independently_confirmed is False for event in bundle.events)
    assert any(gap.code == "independent_event_sources_insufficient" for gap in bundle.gaps)


def test_tavily_subdomains_of_one_publisher_do_not_confirm_event(
    apple_sources: dict[str, object],
) -> None:
    sources = deepcopy(apple_sources)
    sources["tavily_payloads"] = [
        {
            "status": "ok",
            "results": [
                {
                    "title": "Apple Services update",
                    "event_key": "apple-services-growth",
                    "url": "https://www.publisher.co.uk/apple-services",
                    "source_type": "news",
                },
                {
                    "title": "Apple Services corroboration",
                    "event_key": "apple-services-growth",
                    "url": "https://news.publisher.co.uk/apple-services",
                    "source_type": "news",
                },
            ],
        }
    ]

    bundle = build_research_evidence_bundle(**sources)

    assert bundle.tool_status("tavily") == "degraded"
    assert all(event.independently_confirmed is False for event in bundle.events)


def test_tavily_degraded_reason_is_preserved_in_tool_health(apple_sources: dict[str, object]) -> None:
    sources = deepcopy(apple_sources)
    sources["tavily_payloads"] = [
        {"status": "degraded", "degraded_reason": "request budget exhausted", "results": []}
    ]

    bundle = build_research_evidence_bundle(**sources)

    tavily_health = [item for item in bundle.tool_health if item.tool_name == "tavily"]
    assert tavily_health[-1].message == "request budget exhausted"


def test_services_revenue_is_not_formal_without_actual_filing_metadata(
    apple_sources: dict[str, object],
) -> None:
    sources = deepcopy(apple_sources)
    sources.pop("filing_metadata", None)

    bundle = build_research_evidence_bundle(**sources)

    services = bundle.require_fact("segment_revenue_services")
    assert services.formal_eligible is False
    assert "accession_missing" in services.quality_flags
    assert any(gap.code == "services_revenue_provenance_incomplete" for gap in bundle.gaps)


def test_services_revenue_retains_actual_filing_metadata(apple_sources: dict[str, object]) -> None:
    sources = deepcopy(apple_sources)
    sources["filing_metadata"] = {
        "source_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
        "accession": "0000320193-25-000079",
        "filed_at": "2025-10-31",
        "form": "10-K",
        "fiscal_year": 2025,
        "fiscal_period": "FY",
        "period_start": "2024-09-29",
        "period_end": "2025-09-27",
    }

    bundle = build_research_evidence_bundle(**sources)

    services = bundle.require_fact("segment_revenue_services")
    assert services.formal_eligible is True
    assert services.source_url == sources["filing_metadata"]["source_url"]
    assert services.accession == "0000320193-25-000079"


def test_sec_company_facts_without_formal_facts_is_degraded(apple_sources: dict[str, object]) -> None:
    sources = deepcopy(apple_sources)
    for concept in sources["company_facts"]["facts"]["us-gaap"].values():
        for entries in concept["units"].values():
            for entry in entries:
                entry.pop("form", None)

    bundle = build_research_evidence_bundle(**sources)

    assert bundle.tool_status("sec_company_facts") == "degraded"
    assert any(gap.code == "sec_companyfacts_no_formal_facts" for gap in bundle.gaps)


def test_tavily_failure_is_recorded_as_degraded_not_hidden(
    apple_sources: dict[str, object],
) -> None:
    sources = deepcopy(apple_sources)
    sources["tavily_payloads"] = [{"status": "degraded", "results": []}]

    bundle = build_research_evidence_bundle(**sources)

    assert bundle.tool_status("tavily") == "degraded"
    assert any(gap.code == "independent_event_sources_insufficient" for gap in bundle.gaps)


def test_quote_failure_is_recorded_as_degraded_not_hidden(apple_sources: dict[str, object]) -> None:
    sources = deepcopy(apple_sources)
    sources["quote_payload"] = {"status": "degraded", "degraded_reason": "timeout"}

    bundle = build_research_evidence_bundle(**sources)

    assert bundle.tool_status("quote") == "degraded"
    assert any(gap.code == "market_quote_unavailable" for gap in bundle.gaps)


def test_legacy_financial_field_keeps_selected_sec_fact_provenance(
    apple_sources: dict[str, object],
) -> None:
    extraction = _extract_latest_fact(
        apple_sources["company_facts"],
        ["RevenueFromContractWithCustomerExcludingAssessedTax"],
    )

    assert extraction.fact is not None
    assert extraction.fact.accession == "0000320193-25-000079"
    assert extraction.as_dict()["fact"]["period_end"] == "2025-09-27"


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


class StubFormalFieldCompanyFactsService:
    def fetch_company_facts(self, _ticker: str) -> dict:
        return {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {"val": 391.0, "end": "2025-09-27", "filed": "2025-10-31", "fy": 2025}
                            ]
                        }
                    },
                    "CashAndCashEquivalentsAtCarryingValue": {
                        "units": {
                            "USD": [
                                {"val": 53.7, "end": "2025-09-27", "filed": "2025-10-31", "fy": 2025}
                            ]
                        }
                    },
                    "LongTermDebtCurrent": {
                        "units": {
                            "USD": [
                                {"val": 11.0, "end": "2025-09-27", "filed": "2025-10-31", "fy": 2025}
                            ]
                        }
                    },
                    "LongTermDebtNoncurrent": {
                        "units": {
                            "USD": [
                                {"val": 87.0, "end": "2025-09-27", "filed": "2025-10-31", "fy": 2025}
                            ]
                        }
                    },
                    "EntityCommonStockSharesOutstanding": {
                        "units": {
                            "shares": [
                                {
                                    "val": 14900000000,
                                    "end": "2025-09-27",
                                    "filed": "2025-10-31",
                                    "fy": 2025,
                                }
                            ]
                        }
                    },
                    "WeightedAverageNumberOfDilutedSharesOutstanding": {
                        "units": {
                            "shares": [
                                {
                                    "val": 14800000000,
                                    "start": "2024-09-29",
                                    "end": "2025-09-27",
                                    "filed": "2025-10-31",
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "accn": "0000320193-25-000079",
                                }
                            ]
                        }
                    },
                    "EarningsPerShareDiluted": {
                        "units": {
                            "USD/shares": [
                                {"val": 7.25, "end": "2025-09-27", "filed": "2025-10-31", "fy": 2025}
                            ]
                        }
                    },
                }
            }
        }


def test_sec_company_facts_tool_records_formal_gate_fields_from_company_facts(
    tmp_path: Path,
) -> None:
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
    tool = SecCompanyFactsTool(settings=settings, service=StubFormalFieldCompanyFactsService())
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

    assert latest_metrics["financial_fields"]["cash_and_equivalents"]["extracted"] is True
    assert latest_metrics["financial_fields"]["cash_and_equivalents"]["normalized_value"] == 53.7
    assert latest_metrics["financial_fields"]["total_debt"]["extracted"] is True
    assert latest_metrics["financial_fields"]["total_debt"]["normalized_value"] == 98.0
    assert latest_metrics["financial_fields"]["diluted_shares"]["extracted"] is True
    assert latest_metrics["financial_fields"]["shares_outstanding"]["normalized_value"] == 14900000000
    assert latest_metrics["financial_fields"]["diluted_shares"]["normalized_value"] == 14800000000
    assert latest_metrics["financial_fields"]["eps"]["extracted"] is True
    assert latest_metrics["financial_fields"]["eps"]["normalized_value"] == 7.25


class StubFormalReportInputsService(StubFormalFieldCompanyFactsService):
    def fetch_latest_annual_report_html(self, _ticker: str) -> str:
        return """
        <html>
          <body>
            <h2>Products and Services Performance</h2>
            <table>
              <tr><th>Category</th><th>Net sales</th></tr>
              <tr><td>Products</td><td>281,788</td></tr>
              <tr><td>Services</td><td>109,158</td></tr>
            </table>
          </body>
        </html>
        """

    def fetch_latest_annual_report(self, ticker: str) -> dict:
        return {
            "html": self.fetch_latest_annual_report_html(ticker),
            "source_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
            "accession": "0000320193-25-000079",
            "filed_at": "2025-10-31",
            "form": "10-K",
            "fiscal_year": 2025,
            "fiscal_period": "FY",
            "period_start": "2024-09-29",
            "period_end": "2025-09-27",
        }

    def fetch_market_quote(self, _ticker: str) -> dict:
        return {
            "data": {
                "primaryData": {
                    "lastSalePrice": "$333.74",
                    "lastTradeTimestamp": "Jul 19, 2026",
                }
            }
        }


class StubFallbackMarketQuoteService(StubFormalReportInputsService):
    def fetch_market_quote(self, _ticker: str) -> dict:
        return {
            "source": "stockanalysis_quote_page",
            "data": {
                "primaryData": {
                    "lastSalePrice": "$333.74",
                    "lastTradeTimestamp": "Jul 17, 2026, 4:00 PM EDT",
                }
            },
        }


class StubIndependentTavilyService:
    def search_company_news(self, **_kwargs: object) -> dict[str, object]:
        return {
            "status": "ok",
            "artifact_ref": "https://api.tavily.com/search/offline-request",
            "results": [
                {
                    "title": "Apple source one",
                    "event_key": "apple-services-growth",
                    "url": "https://news.example.com/apple-one",
                    "source_type": "news",
                    "published_at": "2026-07-18T12:00:00+00:00",
                    "snippet": "Offline fixture one.",
                },
                {
                    "title": "Apple source two",
                    "event_key": "apple-services-growth",
                    "url": "https://wire.example.net/apple-two",
                    "source_type": "news",
                    "published_at": "2026-07-18T13:00:00+00:00",
                    "snippet": "Offline fixture two.",
                },
            ],
        }


class FixtureEvidenceService:
    def __init__(self, sources: dict[str, object]) -> None:
        self._sources = sources

    def fetch_company_facts(self, _ticker: str) -> dict:
        return self._sources["company_facts"]

    def fetch_latest_annual_report(self, _ticker: str) -> dict:
        return {
            "html": self._sources["filing_html"],
            **self._sources["filing_metadata"],
        }

    def fetch_market_quote(self, _ticker: str) -> dict:
        return self._sources["quote_payload"]


def test_financial_metrics_output_uses_recorded_tavily_and_persists_evidence_artifact(
    tmp_path: Path,
    apple_sources: dict[str, object],
) -> None:
    evaluation = WorkflowEvaluation(
        artifacts_dir=tmp_path / "run",
        final_report_path=tmp_path / "run" / "04_investment_report.md",
        expected_task_outputs={},
        company_name="Apple Inc.",
        company_ticker="AAPL",
    )
    evaluation.start()
    token = activate_evaluation(evaluation)
    try:
        tavily = TavilySearchTool(settings=build_settings(), service=StubIndependentTavilyService())
        tavily._run(query="Apple", company_name="Apple Inc.")
        payload = json.loads(
            FinancialMetricsTool(
                settings=build_settings(),
                service=FixtureEvidenceService(apple_sources),
            )._run("AAPL")
        )
    finally:
        clear_evaluation(token)

    assert payload["evidence_bundle"]["tool_health"][-1]["tool_name"] == "tavily"
    assert payload["evidence_bundle"]["tool_health"][-1]["status"] == "healthy"
    evidence_path = tmp_path / "run" / "10_research_evidence.json"
    assert evidence_path.exists()
    persisted = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert persisted == payload["evidence_bundle"]
    assert "https://api.tavily.com/search/offline-request" in persisted["raw_artifact_refs"]


def test_runtime_path_establishes_services_period_from_same_accession_revenue(
    tmp_path: Path,
    apple_sources: dict[str, object],
) -> None:
    sources = deepcopy(apple_sources)
    filing_metadata = sources["filing_metadata"]
    for field in ("fiscal_year", "fiscal_period", "period_start", "period_end"):
        filing_metadata.pop(field)
    filing_metadata["report_date"] = "2025-09-27"
    evaluation = WorkflowEvaluation(
        artifacts_dir=tmp_path / "run",
        final_report_path=tmp_path / "run" / "04_investment_report.md",
        expected_task_outputs={},
        company_name="Apple Inc.",
        company_ticker="AAPL",
    )
    evaluation.start()
    token = activate_evaluation(evaluation)
    try:
        payload = json.loads(
            FinancialMetricsTool(
                settings=build_settings(),
                service=FixtureEvidenceService(sources),
            )._run("AAPL")
        )
    finally:
        clear_evaluation(token)

    services = next(
        fact
        for fact in payload["evidence_bundle"]["financial_facts"]
        if fact["field_name"] == "segment_revenue_services"
    )
    assert services["source_url"] == sources["filing_metadata"]["source_url"]
    assert services["accession"] == "0000320193-25-000079"
    assert services["period_start"] == "2024-09-29"
    assert services["period_end"] == "2025-09-27"
    assert services["quality_flags"] == []


def test_financial_metrics_records_missing_tavily_as_degraded() -> None:
    payload = json.loads(
        FinancialMetricsTool(settings=build_settings(), service=StubFormalReportInputsService())._run("AAPL")
    )

    tavily_health = [item for item in payload["evidence_bundle"]["tool_health"] if item["tool_name"] == "tavily"]
    assert tavily_health[-1]["status"] == "degraded"
    assert any(
        gap["code"] == "independent_event_sources_insufficient"
        for gap in payload["evidence_bundle"]["gaps"]
    )


class FatalFilingService(StubFormalReportInputsService):
    def fetch_latest_annual_report(self, _ticker: str) -> dict:
        raise FatalAPIError("SEC filing rejected", service_name="SEC Filing HTML")


def test_financial_metrics_reraises_fatal_filing_error() -> None:
    with pytest.raises(FatalAPIError, match="SEC filing rejected"):
        FinancialMetricsTool(settings=build_settings(), service=FatalFilingService())._run("AAPL")


def test_financial_metrics_tool_builds_market_and_segment_snapshots_for_formal_gate(
    tmp_path: Path,
) -> None:
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
    tool = FinancialMetricsTool(settings=settings, service=StubFormalReportInputsService())
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
        payload = json.loads(tool._run("AAPL"))
        latest_metrics = evaluation.finalize(success=True)
    finally:
        clear_evaluation(token)

    assert payload["market_snapshot"]["stock_price"] == 333.74
    assert payload["market_snapshot"]["diluted_shares"] == 14800000000
    assert payload["market_snapshot"]["ready_for_formal_report"] is False
    assert payload["segment_snapshot"]["services_revenue"] == 109158000000.0
    assert latest_metrics["financial_fields"]["stock_price"]["extracted"] is True
    assert latest_metrics["financial_fields"]["stock_price"]["normalized_value"] == 333.74
    assert latest_metrics["financial_fields"]["segment_revenue_services"]["extracted"] is True
    assert latest_metrics["financial_fields"]["segment_revenue_services"]["normalized_value"] == 109158000000.0


def test_financial_metrics_tool_exposes_formal_gate_fields_to_llm_output(
    tmp_path: Path,
) -> None:
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
    tool = FinancialMetricsTool(settings=settings, service=StubFormalReportInputsService())

    payload = json.loads(tool._run("AAPL"))

    assert payload["financial_fields"]["revenue"]["extracted"] is True
    assert payload["financial_fields"]["cash_and_equivalents"]["extracted"] is True
    assert payload["financial_fields"]["total_debt"]["extracted"] is True
    assert payload["financial_fields"]["diluted_shares"]["extracted"] is True
    assert payload["financial_fields"]["stock_price"]["extracted"] is True
    assert payload["financial_fields"]["segment_revenue_services"]["extracted"] is True
    assert {"revenue", "cash_and_equivalents", "total_debt"} <= set(
        payload["formal_gate_snapshot"]["missing_fields"]
    )
    assert payload["formal_gate_snapshot"]["ready_for_formal_report"] is False


def test_financial_metrics_tool_preserves_market_quote_source_provenance(
    tmp_path: Path,
) -> None:
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
    tool = FinancialMetricsTool(settings=settings, service=StubFallbackMarketQuoteService())
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
        payload = json.loads(tool._run("AAPL"))
        latest_metrics = evaluation.finalize(success=True)
    finally:
        clear_evaluation(token)

    assert payload["market_snapshot"]["source_refs"] == ["stockanalysis_quote_page"]
    assert latest_metrics["financial_fields"]["stock_price"]["source_tag"] == "stockanalysis_last_close_price"


def test_extract_services_revenue_from_inline_xbrl_products_services_table() -> None:
    filing_html = """
    <html>
      <body>
        <div>Products and Services Performance</div>
        <table>
          <tr>
            <td>Category</td>
            <td>2025</td>
            <td>Change</td>
            <td>2024</td>
          </tr>
          <tr>
            <td>
              <div>
                <span>Services </span>
                <span>(1)</span>
              </div>
            </td>
            <td><span>109,158&#160;</span></td>
            <td><span>14&#160;</span><span>%</span></td>
            <td><span>96,169&#160;</span></td>
          </tr>
        </table>
        <div>Geographic Segments Performance</div>
      </body>
    </html>
    """

    extracted = _extract_services_revenue_from_filing_html(filing_html)

    assert extracted.extracted is True
    assert extracted.normalized_value == 109158000000.0
    assert extracted.source_tag == "10k_products_services_table"


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
