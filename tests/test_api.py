import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent import api


def test_execute_trigger_payload_delegates_to_main(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def stub_run_trigger_payload(payload: dict[str, object]) -> str:
        observed.update(payload)
        return "ok"

    monkeypatch.setattr(api, "run_trigger_payload", stub_run_trigger_payload)

    result = api.execute_trigger_payload({"company_name": "Apple Inc.", "company_ticker": "AAPL"})

    assert result == "ok"
    assert observed["company_name"] == "Apple Inc."
    assert observed["company_ticker"] == "AAPL"
