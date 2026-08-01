import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent import runtime
from multi_agent.core.market import MarketValidationResult


def test_kickoff_workflow_uses_flow_when_enabled(monkeypatch) -> None:
    class StubFlow:
        def kickoff(self):
            return "flow"

    monkeypatch.setenv("USE_FLOW_EXECUTION", "1")
    monkeypatch.setattr(runtime, "build_flow", lambda inputs: StubFlow())
    monkeypatch.setattr(runtime, "build_crew", lambda: (_ for _ in ()).throw(AssertionError("crew should not run")))

    result = runtime.kickoff_workflow({"company_name": "Apple Inc.", "company_ticker": "AAPL"})

    assert result == "flow"


def test_kickoff_workflow_uses_flow_by_default(monkeypatch) -> None:
    class StubFlow:
        def kickoff(self):
            return "flow"

    monkeypatch.delenv("USE_FLOW_EXECUTION", raising=False)
    monkeypatch.setattr(runtime, "build_flow", lambda inputs: StubFlow())
    monkeypatch.setattr(runtime, "build_crew", lambda: (_ for _ in ()).throw(AssertionError("crew should not run")))

    result = runtime.kickoff_workflow({"company_name": "Apple Inc.", "company_ticker": "AAPL"})

    assert result == "flow"


def test_kickoff_workflow_uses_crew_when_flow_explicitly_disabled(monkeypatch) -> None:
    class StubCrew:
        def kickoff(self, *, inputs):
            assert inputs["company_name"] == "Apple Inc."
            return "crew"

    monkeypatch.setenv("USE_FLOW_EXECUTION", "0")
    monkeypatch.setattr(runtime, "build_flow", lambda inputs: (_ for _ in ()).throw(AssertionError("flow should not run")))
    monkeypatch.setattr(runtime, "build_crew", lambda: StubCrew())

    result = runtime.kickoff_workflow({"company_name": "Apple Inc.", "company_ticker": "AAPL"})

    assert result == "crew"


def test_build_flow_constructs_complete_market_validation_state() -> None:
    flow = runtime.build_flow(
        {
            "company_name": "Microsoft Corporation",
            "company_ticker": "MSFT",
            "company_market_label": "US",
            "market_resolution_status": "confirmed",
            "current_year": "2026",
            "local_filing_pdf_path": "/tmp/msft.pdf",
            "local_filing_pdf_available": "yes",
        }
    )

    assert flow.state.company_name == "Microsoft Corporation"
    assert flow.state.input_ticker == "MSFT"
    assert flow.state.current_year == "2026"
    assert flow.state.local_filing_pdf_path == "/tmp/msft.pdf"
    assert flow.state.local_filing_pdf_available == "yes"
    assert isinstance(flow.state.market_validation, MarketValidationResult)
    assert flow.state.market_validation.market_label == "US"
    assert flow.state.market_validation.resolution_status == "confirmed"
    assert flow.state.market_validation.confidence > 0
    assert flow.state.market_validation.tool_policy.sec_allowed is True
    assert flow.state.rerun_budget["analysis"] == 3
    assert flow.state.rerun_budget["report_writing_analyst"] == 3


def test_build_flow_caps_repair_rounds_at_three() -> None:
    flow = runtime.build_flow(
        {
            "company_name": "Apple Inc.",
            "company_ticker": "AAPL",
            "rerun_budget_json": '{"analysis": 99, "report_writing_analyst": 99}',
        }
    )

    assert flow.state.rerun_budget["analysis"] == 3
    assert flow.state.rerun_budget["report_writing_analyst"] == 3
