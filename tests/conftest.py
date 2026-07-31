import json
import os
import socket
import urllib.request
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests


os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault(
    "CREWAI_STORAGE_DIR", str(Path(__file__).resolve().parents[1] / ".crewai-storage")
)


class UnexpectedNetworkRequest(BaseException):
    """Escape broad production Exception handlers during offline tests."""


@pytest.fixture(autouse=True)
def _block_unstubbed_network_and_crewai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must explicitly stub every external workflow boundary."""

    def _blocked_request(*_args: object, **_kwargs: object) -> object:
        raise UnexpectedNetworkRequest("unexpected network request in test")

    monkeypatch.setattr(requests.sessions.Session, "request", _blocked_request)
    monkeypatch.setattr(urllib.request, "urlopen", _blocked_request)
    monkeypatch.setattr(socket.socket, "connect", _blocked_request)
    monkeypatch.setattr(socket, "create_connection", _blocked_request)

    try:
        from crewai import Crew
        from crewai.events.utils import console_formatter
    except ImportError:
        return

    monkeypatch.setattr(
        console_formatter,
        "is_newer_version_available",
        lambda: (False, "test-version", None),
    )
    monkeypatch.setattr(
        console_formatter,
        "is_current_version_yanked",
        lambda: (False, ""),
    )

    def _blocked_kickoff(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unexpected CrewAI kickoff in test")

    monkeypatch.setattr(Crew, "kickoff", _blocked_kickoff)


@dataclass
class _FakeTaskOutput:
    name: str
    raw: str


class _FakeCrewOutput:
    """Minimal CrewOutput-shaped value emitted by the fixture execution adapter."""

    def __init__(
        self,
        task_outputs: list[_FakeTaskOutput],
        *,
        json_dict: dict[str, object] | None = None,
    ) -> None:
        self.tasks_output = task_outputs
        self.json_dict = json_dict or {}
        self.raw = ""


@dataclass
class _FixtureResponse:
    status_code: int
    payload: dict[str, object]
    text: str = ""

    def json(self) -> dict[str, object]:
        return self.payload


class _OfficialFixtureSession:
    """Recording SEC/Nasdaq session that exercises OfficialSecService URL parsing."""

    def __init__(self, payloads: dict[str, object]) -> None:
        self.payloads = payloads
        self.calls: list[str] = []

    def get(self, url: str, **_kwargs: object) -> _FixtureResponse:
        from multi_agent.tools.official_sec import OfficialSecService

        self.calls.append(url)
        if url == OfficialSecService.SEC_TICKERS_URL:
            return _FixtureResponse(
                200,
                {"0": {"ticker": "AAPL", "title": "Apple Inc.", "cik_str": 320193}},
            )
        if url == OfficialSecService.SEC_COMPANY_FACTS_URL.format(cik="0000320193"):
            return _FixtureResponse(200, self.payloads["company_facts"])
        if url == OfficialSecService.SEC_SUBMISSIONS_URL.format(cik="0000320193"):
            return _FixtureResponse(
                200,
                {
                    "filings": {
                        "recent": {
                            "form": ["10-K"],
                            "filingDate": ["2025-10-31"],
                            "reportDate": ["2025-09-27"],
                            "accessionNumber": ["0000320193-25-000079"],
                            "primaryDocument": ["aapl-20250927.htm"],
                            "primaryDocDescription": ["Apple 2025 Form 10-K"],
                        }
                    }
                },
            )
        if "/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm" in url:
            return _FixtureResponse(200, {}, str(self.payloads["filing_html"]))
        if url == OfficialSecService.NASDAQ_QUOTE_INFO_URL.format(ticker="AAPL"):
            return _FixtureResponse(200, self.payloads["quote_payload"])
        raise AssertionError(f"unexpected fixture SEC request: {url}")


class _TavilyFixtureSession:
    """Recording Tavily session returning two independent source domains."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def post(self, url: str, **_kwargs: object) -> _FixtureResponse:
        self.calls.append(url)
        if url != "https://api.tavily.com/search":
            raise AssertionError(f"unexpected fixture Tavily request: {url}")
        return _FixtureResponse(
            200,
            {
                "results": [
                    {
                        "title": "Apple Services Growth",
                        "url": "https://news.example.com/apple-services",
                        "type": "news",
                        "published_date": "2026-07-18T12:00:00+00:00",
                        "content": "Apple reported continued Services growth.",
                    },
                    {
                        "title": "Apple Services Growth",
                        "url": "https://wire.example.net/apple-services",
                        "type": "news",
                        "published_date": "2026-07-18T13:00:00+00:00",
                        "content": "A second independent publisher confirmed Services growth.",
                    },
                ]
            },
        )


def _append_annual_fact(
    facts: dict[str, object],
    concept: str,
    unit: str,
    value: float,
    *,
    start: str | None = None,
) -> None:
    entry: dict[str, object] = {
        "end": "2025-09-27",
        "val": value,
        "fy": 2025,
        "fp": "FY",
        "form": "10-K",
        "filed": "2025-10-31",
        "accn": "0000320193-25-000079",
    }
    if start:
        entry["start"] = start
    facts[concept] = {"units": {unit: [entry]}}


@pytest.fixture
def apple_fixture_payloads() -> dict[str, object]:
    """Production-shaped Apple tool responses, kept entirely on disk."""
    fixture_dir = Path(__file__).parent / "fixtures" / "apple"
    company_facts = json.loads((fixture_dir / "companyfacts.json").read_text(encoding="utf-8"))
    company_facts["entityName"] = "Apple Inc."
    facts = company_facts["facts"]["us-gaap"]
    _append_annual_fact(facts, "CashAndCashEquivalentsAtCarryingValue", "USD", 35_900_000_000)
    _append_annual_fact(facts, "LongTermDebtCurrent", "USD", 10_000_000_000)
    _append_annual_fact(facts, "LongTermDebtNoncurrent", "USD", 90_000_000_000)
    _append_annual_fact(
        facts,
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
    }


@pytest.fixture
def apple_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    apple_fixture_payloads: dict[str, object],
):
    """Run actual runtime Flow construction with fixture-backed execution adapters."""
    from multi_agent import main
    from multi_agent.core.report_document import REQUIRED_SECTION_KEYS, SECTION_HEADINGS
    from multi_agent.core.evidence import ResearchEvidenceBundle
    from multi_agent.core.review_contracts import ReviewContract
    from multi_agent.flows import market_review_flow
    from multi_agent.settings import InvestmentResearchSettings
    from multi_agent.tools.investment_tools import FinancialMetricsTool
    from multi_agent.tools.official_sec import OfficialSecService
    from multi_agent.tools.tavily_search import TavilySearchService, TavilySearchTool

    def review_contract(stage: str) -> ReviewContract:
        return ReviewContract.model_validate(
            {
                "stage": stage,
                "reviewer_name": "offline-reviewer",
                "decision": {"gate_outcome": "pass", "decision_confidence": "high"},
                "delivery_eligibility": {
                    "formal_report_allowed": True,
                    "evidence_limited_report_allowed": False,
                    "blocked_notice_required": False,
                    "recommended_delivery_state": "formal_report",
                },
                "failure_taxonomy": {"primary_class": "none", "secondary_causes": []},
                "coverage_summary": {
                    "evidence_coverage_ratio": 1.0,
                    "financial_coverage_score": 1.0,
                    "claim_binding_ratio": 1.0,
                },
                "tool_health_summary": {"overall_status": "healthy"},
                "review_summary": {"one_sentence_summary": "Offline fixture review passed."},
            }
        )

    def writer_payload(report_inputs: dict[str, str]) -> dict[str, object]:
        report_context = json.loads(report_inputs["REPORT_CONTEXT_JSON"])
        source_ids = [
            json.loads(source_json)["source_id"]
            for source_json in report_context["canonical_sources_json"]
        ]
        return {
            "title": "Apple Inc. (AAPL) 投资研究报告",
            "stance": "hold",
            "executive_summary": "已验证的 Apple 年报和市场数据支持维持持有观点。",
            "catalysts": ["服务业务收入保持增长。"],
            "risks": ["硬件需求和估值可能波动。"],
            "sections": {
                key: {
                    "heading": SECTION_HEADINGS[key],
                    "content": f"{SECTION_HEADINGS[key]}：离线 Apple 证据已完成严格核验。",
                    "claim_ids": ["claim:revenue"] if key in {
                        "executive_summary",
                        "financial_analysis",
                        "investment_conclusion",
                    } else [],
                }
                for key in REQUIRED_SECTION_KEYS
            },
            "claims": [
                {
                    "claim_id": "claim:revenue",
                    "text": "Apple FY2025 收入来自同期间 10-K 事实。",
                    "critical": True,
                    "source_ids": ["claim:revenue"],
                }
            ],
            "sources": [{"source_id": source_id} for source_id in source_ids],
        }

    class _ApplePipeline:
        official_session: _OfficialFixtureSession | None = None
        tavily_session: _TavilyFixtureSession | None = None
        analysis_kickoffs: int = 0
        report_kickoffs: int = 0

        def run(self, tmp_path: Path) -> tuple[dict[str, object], Path]:
            captured_flow_result: dict[str, object] | None = None
            runs_dir = tmp_path / "runs"
            settings = InvestmentResearchSettings(
                fast_model="offline-fast",
                deep_model="offline-deep",
                review_model="offline-review",
                company_resolver_model="offline-fast",
                openai_api_key="offline-key",
                openai_base_url="http://offline.invalid/v1",
                tavily_api_key="offline-tavily",
                sec_api_email="offline@example.com",
                runs_dir=str(runs_dir),
                watchlist_path=str(tmp_path / "watchlist.json"),
            )
            workflow_inputs = {
                "company_name": "Apple Inc.",
                "company_ticker": "AAPL",
                "company_market_label": "US",
                "market_resolution_status": "confirmed",
                "run_id": "20260720_120000",
                "local_filing_pdf_path": "",
                "local_filing_pdf_available": "no",
            }

            self.official_session = _OfficialFixtureSession(apple_fixture_payloads)
            self.tavily_session = _TavilyFixtureSession()
            official_service = OfficialSecService(settings, session=self.official_session)
            tavily_service = TavilySearchService(settings, session=self.tavily_session)
            artifact_dir: Path | None = None

            def write_task_artifact(filename: str, content: str) -> None:
                if artifact_dir is None:
                    raise AssertionError("fixture task output has no runtime artifacts directory")
                (artifact_dir / filename).write_text(content, encoding="utf-8")

            class _FixtureAnalysisCrew:
                def kickoff(_self, *, inputs: dict[str, str]) -> _FakeCrewOutput:
                    nonlocal artifact_dir
                    self.analysis_kickoffs += 1
                    artifact_dir = Path(os.environ["ARTIFACTS_DIR"])
                    tavily_payload = TavilySearchTool(
                        settings=settings,
                        service=tavily_service,
                    )._run(
                        query="Services growth",
                        topic="news",
                        market_label="US",
                        company_name="Apple Inc.",
                    )
                    metrics_payload = json.loads(
                        FinancialMetricsTool(settings=settings, service=official_service)._run("AAPL")
                    )
                    evidence_bundle = ResearchEvidenceBundle.model_validate(
                        metrics_payload["evidence_bundle"]
                    )
                    analysis_contract = review_contract("analysis_review")
                    contract_json = analysis_contract.model_dump_json(indent=2)
                    evidence_url = evidence_bundle.require_fact("revenue").source_url
                    write_task_artifact(
                        "00_market_validation.md",
                        "# 市场验证结果\n\nApple Inc.（AAPL）market_label=US；状态：confirmed。\n",
                    )
                    write_task_artifact(
                        "01_market_intelligence.md",
                        "# 市场情报简报\n\nApple Inc.（AAPL）Tavily 事件：Apple Services Growth；"
                        f"来源：{tavily_payload['results'][0]['url']}。\n",
                    )
                    write_task_artifact(
                        "02_filing_review.md",
                        "# 监管文件复核\n\nApple Inc.（AAPL）10-K；"
                        "Accession：0000320193-25-000079；"
                        f"来源：{evidence_url}。\n",
                    )
                    write_task_artifact(
                        "03_financial_analysis.md",
                        "# 财务分析结果\n\nApple Inc.（AAPL）收入：416161000000 USD；"
                        "稀释股数：15000000000 shares；"
                        "来源 ID：claim:revenue。\n",
                    )
                    write_task_artifact(
                        "08_data_quality_review.md",
                        "# 数据质量审查结果\n\nApple Inc.（AAPL）严格审查契约：\n\n"
                        f"```json\n{contract_json}\n```\n",
                    )
                    return _FakeCrewOutput(
                        [
                            _FakeTaskOutput("market_validation_task", "Apple Inc. AAPL US confirmed"),
                            _FakeTaskOutput("market_intelligence_task", json.dumps(tavily_payload)),
                            _FakeTaskOutput("filing_review_task", "Apple 10-K 0000320193-25-000079"),
                            _FakeTaskOutput("financial_analysis_task", json.dumps(metrics_payload)),
                            _FakeTaskOutput("data_quality_review_task", contract_json),
                        ],
                        json_dict={
                            "evidence_bundle": evidence_bundle.model_dump(mode="json"),
                            "analysis_review_contract": analysis_contract.model_dump(mode="json"),
                        },
                    )

            class _FixtureReportCrew:
                def kickoff(_self, *, inputs: dict[str, str]) -> _FakeCrewOutput:
                    self.report_kickoffs += 1
                    if artifact_dir is None:
                        raise AssertionError("analysis crew must run before report crew")
                    report_contract = review_contract("report_review")
                    report_contract_json = report_contract.model_dump_json(indent=2)
                    write_task_artifact(
                        "09_logic_compliance_review.md",
                        "# 逻辑与合规审查结果\n\nApple Inc.（AAPL）严格审查契约：\n\n"
                        f"```json\n{report_contract_json}\n```\n",
                    )
                    return _FakeCrewOutput(
                        [
                            _FakeTaskOutput(
                                "investment_report_task", json.dumps(writer_payload(inputs), ensure_ascii=False)
                            ),
                            _FakeTaskOutput("logic_compliance_review_task", report_contract_json),
                        ]
                    )

            class _FixtureExecutionAdapter:
                def configure_run(_self, **_kwargs: object) -> None:
                    return None

                def analysis_crew(_self) -> _FixtureAnalysisCrew:
                    return _FixtureAnalysisCrew()

                def report_crew(_self) -> _FixtureReportCrew:
                    return _FixtureReportCrew()

                def crew(_self) -> object:
                    raise AssertionError("new runs must execute the runtime Flow adapter")

            monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", lambda: settings)
            monkeypatch.setattr(main, "_workflow_inputs", lambda *_args: dict(workflow_inputs))
            monkeypatch.delenv("USE_FLOW_EXECUTION", raising=False)
            monkeypatch.setattr(market_review_flow, "MultiAgent", _FixtureExecutionAdapter)
            monkeypatch.setattr(
                main,
                "_build_parser",
                lambda: SimpleNamespace(
                    parse_args=lambda: Namespace(
                        company_name="Apple Inc.",
                        company_ticker="AAPL",
                        watchlist_list=False,
                        watchlist_rebuild=False,
                        save_to_watchlist=False,
                    )
                ),
            )

            original_finalize = main._finalize_successful_result

            def capture_flow_result(output_paths: object, result: object) -> str:
                nonlocal captured_flow_result
                if not isinstance(result, dict):
                    raise AssertionError("runtime Flow must return a structured result")
                captured_flow_result = dict(result)
                return original_finalize(output_paths, result)

            monkeypatch.setattr(main, "_finalize_successful_result", capture_flow_result)

            main.run()
            run_dir = runs_dir / "apple_inc__aapl" / "20260720_120000"
            if captured_flow_result is None:
                raise AssertionError("runtime Flow result was not captured")
            return captured_flow_result, run_dir

    return _ApplePipeline()
