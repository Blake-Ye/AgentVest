import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent.resolver import CompanyResolution, CompanyResolver, OpenAICompanyResolver
from multi_agent.settings import InvestmentResearchSettings


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
        self.last_post_kwargs: dict[str, object] | None = None

    def get(self, url: str, **kwargs):
        self.calls.append(url)
        response = self.responses.get(url)
        if response is None:
            return FakeResponse(404, {"message": "not found"})
        return FakeResponse(200, response)

    def post(self, url: str, **kwargs):
        self.calls.append(url)
        self.last_post_kwargs = kwargs
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


def test_company_resolver_maps_microsoft_chinese_alias_to_public_company() -> None:
    resolver = CompanyResolver(settings=build_settings(), session=RecordingSession({}), llm_resolver=None)

    resolution = resolver.resolve("微软")

    assert resolution.normalized_name == "Microsoft Corporation"
    assert resolution.ticker == "MSFT"
    assert resolution.entity_type == "public_company"
    assert resolution.parent_company == "Microsoft Corporation"
    assert resolution.exchange == "NASDAQ"


def test_company_resolver_rejects_private_company_without_ticker() -> None:
    resolver = CompanyResolver(settings=build_settings(), session=RecordingSession({}), llm_resolver=None)

    with pytest.raises(ValueError) as exc_info:
        resolver.resolve("字节")

    assert "未上市公司" in str(exc_info.value)


def test_company_resolver_uses_official_sec_ticker_directory_for_public_company_name() -> None:
    url = "https://www.sec.gov/files/company_tickers.json"
    session = RecordingSession(
        {
            url: {
                "0": {
                    "title": "Alibaba Group Holding Ltd",
                    "ticker": "BABA",
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


def test_company_resolver_accepts_explicit_ticker_and_normalizes_company_name() -> None:
    url = "https://www.sec.gov/files/company_tickers.json"
    session = RecordingSession(
        {
            url: {
                "0": {
                    "title": "Apple Inc.",
                    "ticker": "AAPL",
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
        exchange="",
        confidence=1.0,
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


def test_openai_company_resolver_prompt_guides_chinese_alias_resolution() -> None:
    session = RecordingSession(
        {
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions": {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"normalized_name":"Microsoft Corporation","ticker":"MSFT",'
                                '"entity_type":"public_company","parent_company":"Microsoft Corporation",'
                                '"exchange":"NASDAQ","confidence":0.96}'
                            )
                        }
                    }
                ]
            }
        }
    )
    resolver = OpenAICompanyResolver(settings=build_settings(), session=session)

    resolution = resolver.resolve("微软")

    system_prompt = session.last_post_kwargs["json"]["messages"][0]["content"]  # type: ignore[index]
    user_prompt = session.last_post_kwargs["json"]["messages"][1]["content"]  # type: ignore[index]

    assert resolution.ticker == "MSFT"
    assert "Chinese short names" in system_prompt
    assert "common aliases" in system_prompt
    assert "listed parent company" in system_prompt
    assert "微软" in user_prompt
