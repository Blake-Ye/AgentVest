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
        sec_api_key="sec-key",
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


class FixedResolutionResolver:
    def __init__(self, resolution: CompanyResolution) -> None:
        self.resolution = resolution

    def __call__(self, company_name: str):
        return self.resolution


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


def test_company_resolver_uses_sec_mapping_api_for_public_company_name() -> None:
    url = "https://api.sec-api.io/mapping/name/Alibaba%20Group?token=sec-key"
    session = RecordingSession(
        {
            url: [
                {
                    "name": "Alibaba Group Holding Ltd",
                    "ticker": "BABA",
                    "exchange": "NYSE",
                    "isDelisted": False,
                }
            ]
        }
    )
    resolver = CompanyResolver(settings=build_settings(), session=session, llm_resolver=None)

    resolution = resolver.resolve("Alibaba Group")

    assert resolution.normalized_name == "Alibaba Group Holding Ltd"
    assert resolution.ticker == "BABA"
    assert session.calls == [url]


def test_company_resolver_accepts_explicit_ticker_and_normalizes_company_name() -> None:
    url = "https://api.sec-api.io/mapping/ticker/AAPL?token=sec-key"
    session = RecordingSession(
        {
            url: [
                {
                    "name": "Apple Inc.",
                    "ticker": "AAPL",
                    "exchange": "NASDAQ",
                    "isDelisted": False,
                }
            ]
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
    )


class UnexpectedNetworkSession:
    def get(self, url: str, **kwargs):
        raise AssertionError(f"不应访问外部映射 API：{url}")


def test_company_resolver_accepts_explicit_non_sec_ticker_without_sec_lookup() -> None:
    resolver = CompanyResolver(
        settings=build_settings(),
        session=UnexpectedNetworkSession(),
        llm_resolver=None,
        pair_validator=lambda company_name, ticker: True,
    )

    resolution = resolver.resolve("Xiaomi Corporation", ticker="1810.HK")

    assert resolution.normalized_name == "Xiaomi Corporation"
    assert resolution.ticker == "1810.HK"
    assert resolution.exchange == "Hong Kong Stock Exchange"
    assert resolution.entity_type == "public_company"


def test_company_resolver_rejects_unknown_market_ticker() -> None:
    resolver = CompanyResolver(
        settings=build_settings(),
        session=UnexpectedNetworkSession(),
        llm_resolver=None,
    )

    with pytest.raises(ValueError) as exc_info:
        resolver.resolve("Unknown Example", ticker="BAD.TICKER")

    assert "暂不支持" in str(exc_info.value)


def test_company_resolver_rejects_non_sec_ticker_when_llm_detects_conflicting_company() -> None:
    resolver = CompanyResolver(
        settings=build_settings(),
        session=UnexpectedNetworkSession(),
        llm_resolver=FixedResolutionResolver(
            CompanyResolution(
                user_input="Apple Inc.",
                normalized_name="Apple Inc.",
                ticker="AAPL",
                entity_type="public_company",
                parent_company="Apple Inc.",
                exchange="NASDAQ",
                confidence=0.99,
            )
        ),
        pair_validator=lambda company_name, ticker: False,
    )

    with pytest.raises(ValueError) as exc_info:
        resolver.resolve("Apple Inc.", ticker="1810.HK")

    assert "不一致" in str(exc_info.value)


def test_company_resolver_allows_known_dual_listed_company_ticker_pair() -> None:
    resolver = CompanyResolver(
        settings=build_settings(),
        session=UnexpectedNetworkSession(),
        llm_resolver=FixedResolutionResolver(
            CompanyResolution(
                user_input="阿里巴巴",
                normalized_name="Alibaba Group Holding Ltd",
                ticker="BABA",
                entity_type="public_company",
                parent_company="Alibaba Group Holding Ltd",
                exchange="NYSE",
                confidence=0.99,
            )
        ),
        pair_validator=lambda company_name, ticker: True,
    )

    resolution = resolver.resolve("阿里巴巴", ticker="9988.HK")

    assert resolution.normalized_name == "Alibaba Group Holding Ltd"
    assert resolution.ticker == "9988.HK"
    assert resolution.exchange == "Hong Kong Stock Exchange"


def test_company_resolver_rejects_non_sec_ticker_when_pair_validator_unavailable() -> None:
    resolver = CompanyResolver(
        settings=build_settings(),
        session=UnexpectedNetworkSession(),
        llm_resolver=None,
        pair_validator=lambda company_name, ticker: None,
    )

    with pytest.raises(ValueError) as exc_info:
        resolver.resolve("Apple Inc.", ticker="1810.HK")

    assert "暂时无法校验" in str(exc_info.value)


def test_company_resolver_ignores_empty_canonical_name_from_pair_validator() -> None:
    resolver = CompanyResolver(
        settings=build_settings(),
        session=UnexpectedNetworkSession(),
        llm_resolver=None,
        pair_validator=lambda company_name, ticker: (True, None),
    )

    resolution = resolver.resolve("Alibaba", ticker="9988.HK")

    assert resolution.normalized_name == "Alibaba"
    assert resolution.parent_company == "Alibaba"


def test_company_resolver_falls_back_to_user_facing_error_when_llm_fails() -> None:
    url = "https://api.sec-api.io/mapping/name/%E5%AE%8C%E5%85%A8%E6%9C%AA%E7%9F%A5%E5%85%AC%E5%8F%B8?token=sec-key"
    resolver = CompanyResolver(
        settings=build_settings(),
        session=RecordingSession({url: []}),
        llm_resolver=FailingResolver(),
    )

    with pytest.raises(ValueError) as exc_info:
        resolver.resolve("完全未知公司")

    assert "无法根据输入" in str(exc_info.value)
