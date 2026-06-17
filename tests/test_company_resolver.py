import pytest

from multi_agent.resolver import CompanyResolution, CompanyResolver
from multi_agent.settings import InvestmentResearchSettings


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


class FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class RecordingSession:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str, **kwargs):
        self.calls.append(url)
        response = self.responses.get(url)
        if response is None:
            return FakeResponse(404, {"message": "not found"})
        return FakeResponse(200, response)


class FailingResolver:
    def __call__(self, company_name: str):
        raise RuntimeError("Error code: 400 - function.arguments must be JSON format")


def test_company_resolver_maps_business_unit_alias_to_parent_company() -> None:
    resolver = CompanyResolver(settings=build_settings(), session=RecordingSession({}), llm_resolver=None)

    resolution = resolver.resolve("索尼手机")

    assert resolution.normalized_name == "Sony Group Corporation"
    assert resolution.ticker == "SONY"
    assert resolution.entity_type == "business_unit"
    assert resolution.parent_company == "Sony Group Corporation"


def test_company_resolver_rejects_private_company_without_ticker() -> None:
    resolver = CompanyResolver(settings=build_settings(), session=RecordingSession({}), llm_resolver=None)

    with pytest.raises(ValueError) as exc_info:
        resolver.resolve("字节")

    assert "未上市公司" in str(exc_info.value)


def test_company_resolver_resolves_public_company_name_from_ticker_directory() -> None:
    url = "https://www.sec.gov/files/company_tickers.json"
    session = RecordingSession(
        {
            url: {
                "0": {
                    "title": "Alibaba Group Holding Ltd",
                    "ticker": "BABA",
                    "exchange": "NYSE",
                    "cik_str": 1577552,
                }
            }
        }
    )
    resolver = CompanyResolver(settings=build_settings(), session=session, llm_resolver=None)

    resolution = resolver.resolve("Alibaba Group")

    assert resolution.normalized_name == "Alibaba Group Holding Ltd"
    assert resolution.ticker == "BABA"
    assert session.calls == [url]


def test_company_resolver_uses_official_ticker_lookup_for_name_matching() -> None:
    ticker_lookup_url = "https://www.sec.gov/files/company_tickers.json"
    session = RecordingSession(
        {
            ticker_lookup_url: {
                "0": {
                    "ticker": "BABA",
                    "title": "Alibaba Group Holding Ltd",
                    "cik_str": 1577552,
                },
                "1": {
                    "ticker": "AAPL",
                    "title": "Apple Inc.",
                    "cik_str": 320193,
                },
            }
        }
    )
    resolver = CompanyResolver(settings=build_settings(), session=session, llm_resolver=None)

    resolution = resolver.resolve("Alibaba Group")

    assert resolution.normalized_name == "Alibaba Group Holding Ltd"
    assert resolution.ticker == "BABA"
    assert session.calls == [ticker_lookup_url]


def test_company_resolver_resolves_google_alias_to_alphabet() -> None:
    resolver = CompanyResolver(settings=build_settings(), session=RecordingSession({}), llm_resolver=None)

    resolution = resolver.resolve("Google")

    assert resolution.normalized_name == "Alphabet Inc."
    assert resolution.ticker in {"GOOGL", "GOOG"}
    assert resolution.entity_type == "public_company"


def test_company_resolver_resolves_apple_chinese_alias() -> None:
    resolver = CompanyResolver(settings=build_settings(), session=RecordingSession({}), llm_resolver=None)

    resolution = resolver.resolve("苹果")

    assert resolution.normalized_name == "Apple Inc."
    assert resolution.ticker == "AAPL"
    assert resolution.entity_type == "public_company"
    assert resolution.resolution_source == "alias"
    assert "命中“苹果” -> Apple Inc. / AAPL" in resolution.resolution_steps


def test_company_resolver_accepts_explicit_ticker_and_normalizes_company_name() -> None:
    url = "https://www.sec.gov/files/company_tickers.json"
    session = RecordingSession(
        {
            url: {
                "0": {
                    "title": "Apple Inc.",
                    "ticker": "AAPL",
                    "exchange": "NASDAQ",
                    "cik_str": 320193,
                }
            }
        }
    )
    resolver = CompanyResolver(settings=build_settings(), session=session, llm_resolver=None)

    resolution = resolver.resolve("apple", ticker="aapl")

    assert resolution == CompanyResolution(
        user_input="apple",
        normalized_name="Apple Inc.",
        ticker="AAPL",
        entity_type="public_company",
        parent_company="Apple Inc.",
        exchange="NASDAQ",
        confidence=1.0,
        resolution_source="explicit_ticker",
        resolution_steps=(
            "校验用户输入 ticker：AAPL",
            "官方目录确认 Apple Inc. / AAPL",
        ),
    )


def test_company_resolver_falls_back_to_user_facing_error_when_llm_fails() -> None:
    url = "https://www.sec.gov/files/company_tickers.json"
    resolver = CompanyResolver(
        settings=build_settings(),
        session=RecordingSession({url: {}}),
        llm_resolver=FailingResolver(),
    )

    with pytest.raises(ValueError) as exc_info:
        resolver.resolve("完全未知公司")

    assert "无法根据输入" in str(exc_info.value)
