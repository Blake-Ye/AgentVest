import os
import json
from argparse import Namespace
from copy import deepcopy
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


@pytest.fixture(autouse=True)
def _block_unstubbed_network_and_crewai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must explicitly stub every external workflow boundary."""

    def _blocked_request(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unexpected network request in test")

    monkeypatch.setattr(requests.sessions.Session, "request", _blocked_request)

    try:
        from crewai import Crew
    except ImportError:
        return

    def _blocked_kickoff(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unexpected CrewAI kickoff in test")

    monkeypatch.setattr(Crew, "kickoff", _blocked_kickoff)


@dataclass
class _FakeTaskOutput:
    name: str
    raw: str


class _FakeCrewOutput:
    """Minimal CrewOutput-shaped boundary object for an offline CLI run."""

    def __init__(self, flow_result: dict[str, object], task_outputs: list[_FakeTaskOutput]) -> None:
        self.tasks_output = task_outputs
        self.raw = ""
        for key, value in flow_result.items():
            setattr(self, key, value)


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
    news = json.loads((fixture_dir / "news.json").read_text(encoding="utf-8"))
    first_event = dict(news["results"][0])
    first_event.update({"event_key": "apple-services-growth", "url": "https://news.example.com/apple-services"})
    second_event = dict(first_event)
    second_event.update(
        {
            "title": "Apple Services growth independently confirmed",
            "url": "https://wire.example.net/apple-services",
        }
    )
    news["results"] = [first_event, second_event]
    news["artifact_ref"] = "https://api.tavily.com/search/apple-services"
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
        "tavily_payloads": [news],
    }


@pytest.fixture
def apple_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    apple_fixture_payloads: dict[str, object],
):
    """Run the typed Flow through the production CLI finalization boundary offline."""
    from multi_agent import main
    from multi_agent.core.report_document import REQUIRED_SECTION_KEYS, SECTION_HEADINGS
    from multi_agent.core.review_contracts import ReviewContract
    from multi_agent.flows.market_review_flow import MarketReviewFlow, MarketReviewFlowState
    from multi_agent.settings import InvestmentResearchSettings
    from multi_agent.tools.investment_tools import build_research_evidence_bundle

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

    def writer_payload() -> dict[str, object]:
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
            "sources": [{"source_id": "claim:revenue"}],
        }

    class _ApplePipeline:
        last_result: object | None = None

        def run(self, tmp_path: Path) -> tuple[dict[str, object], Path]:
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

            def kickoff(inputs: dict[str, str]) -> _FakeCrewOutput:
                bundle = build_research_evidence_bundle(**deepcopy(apple_fixture_payloads))
                analysis_contract = review_contract("analysis_review")
                flow = MarketReviewFlow(
                    analysis_executor=lambda _inputs: {
                        "evidence_bundle": bundle.model_dump(mode="json"),
                        "analysis_review_contract": analysis_contract.model_dump(mode="json"),
                    },
                    report_writer=lambda _inputs: writer_payload(),
                    report_reviewer=lambda _document: review_contract("report_review").model_dump(mode="json"),
                    initial_state=MarketReviewFlowState(
                        request_id=inputs["run_id"],
                        company_name=inputs["company_name"],
                        input_ticker=inputs["company_ticker"],
                        execution_mode="new",
                        artifacts_dir=inputs["artifacts_dir"],
                        final_report_path=inputs["final_report_path"],
                    ),
                )
                flow_result = flow.kickoff()
                self.last_result = flow_result
                task_outputs = [
                    _FakeTaskOutput(name, f"# {name}\n\nOffline Apple fixture output.")
                    for name in (
                        "market_validation_task",
                        "market_intelligence_task",
                        "filing_review_task",
                        "financial_analysis_task",
                        "investment_report_task",
                        "data_quality_review_task",
                        "logic_compliance_review_task",
                    )
                ]
                return _FakeCrewOutput(flow_result, task_outputs)

            monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", lambda: settings)
            monkeypatch.setattr(main, "_workflow_inputs", lambda *_args: dict(workflow_inputs))
            monkeypatch.setattr(main, "_ensure_run_id", lambda inputs: inputs)
            monkeypatch.setattr(main, "_kickoff_workflow", kickoff)
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

            main.run()
            run_dir = runs_dir / "apple_inc__aapl" / "20260720_120000"
            assert isinstance(self.last_result, dict)
            return self.last_result, run_dir

    return _ApplePipeline()
