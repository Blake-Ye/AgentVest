import os
from pathlib import Path

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

    monkeypatch.setattr(requests.Session, "request", _blocked_request)

    try:
        from crewai import Crew
    except ImportError:
        return

    def _blocked_kickoff(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unexpected CrewAI kickoff in test")

    monkeypatch.setattr(Crew, "kickoff", _blocked_kickoff)
