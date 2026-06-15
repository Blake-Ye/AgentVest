import pytest

from multi_agent.tools.investment_tools import FatalAPIError, _raise_for_status_with_context


class FakeResponse:
    def __init__(self, status_code: int, text: str = "forbidden", url: str = "https://example.com") -> None:
        self.status_code = status_code
        self.text = text
        self.url = url


def test_raise_for_status_with_context_stops_on_403() -> None:
    response = FakeResponse(status_code=403, text="permission denied")

    with pytest.raises(FatalAPIError) as exc_info:
        _raise_for_status_with_context(response, service_name="Google Search")

    assert "403" in str(exc_info.value)
    assert "Google Search" in str(exc_info.value)
    assert "已终止" in str(exc_info.value)
