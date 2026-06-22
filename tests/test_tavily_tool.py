import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent.settings import InvestmentResearchSettings
from multi_agent.tools.tavily_search import TavilySearchOutput, TavilySearchService, TavilySearchTool


class StubTavilySearchService:
    def search_company_news(
        self,
        query: str,
        topic: str = "news",
        market_label: str = "",
        company_name: str = "",
    ) -> dict:
        return {
            "query": query,
            "topic": topic,
            "market_label": market_label,
            "company_name": company_name,
            "result_count": 1,
            "results": [
                {
                    "title": "Revenue acceleration",
                    "url": "https://example.com/revenue",
                    "source_type": "news",
                    "published_at": "2026-06-20T08:30:00Z",
                    "snippet": f"{company_name} posted faster revenue growth.",
                }
            ],
        }


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


def normalize_tool_output(payload: dict | str) -> dict:
    if isinstance(payload, str):
        return json.loads(payload)
    return payload


def test_tavily_search_tool_returns_structured_output() -> None:
    tool = TavilySearchTool(
        settings=build_settings(),
        service=StubTavilySearchService(),
    )

    result = normalize_tool_output(
        tool._run(
            query="latest earnings",
            topic="news",
            market_label="us_equity",
            company_name="NVIDIA",
        )
    )

    assert result["query"] == "latest earnings"
    assert result["topic"] == "news"
    assert result["market_label"] == "us_equity"
    assert result["company_name"] == "NVIDIA"
    assert result["result_count"] == 1
    assert result["results"][0]["title"] == "Revenue acceleration"
    assert result["results"][0]["url"] == "https://example.com/revenue"
    assert result["results"][0]["source_type"] == "news"
    assert result["results"][0]["published_at"] == "2026-06-20T08:30:00Z"
    assert result["results"][0]["snippet"] == "NVIDIA posted faster revenue growth."


class FakeResponse:
    def __init__(self, status_code: int, payload: dict, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or str(payload)

    def json(self) -> dict:
        return self._payload


class RecordingSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def post(self, url: str, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return FakeResponse(
            status_code=200,
            payload={
                "results": [
                    {
                        "title": "Apple Earnings Beat Expectations",
                        "url": "https://example.com/apple-earnings",
                        "content": "Apple reported stronger-than-expected earnings.",
                        "published_date": "2026-06-20T12:00:00Z",
                        "type": "news",
                    }
                ]
            },
        )


class ExplodingTavilySearchService:
    def search_company_news(self, **_kwargs) -> dict:
        raise RuntimeError("network timeout")


def test_tavily_search_service_calls_api_and_normalizes_results() -> None:
    settings = build_settings()
    session = RecordingSession()
    service = TavilySearchService(settings=settings, session=session)

    results = service.search_company_news(
        query="latest earnings",
        topic="news",
        market_label="us_equity",
        company_name="Apple Inc.",
    )

    assert results == TavilySearchOutput(
        query="latest earnings",
        topic="news",
        market_label="us_equity",
        company_name="Apple Inc.",
        result_count=1,
        results=[
            {
                "title": "Apple Earnings Beat Expectations",
                "url": "https://example.com/apple-earnings",
                "source_type": "news",
                "published_at": "2026-06-20T12:00:00Z",
                "snippet": "Apple reported stronger-than-expected earnings.",
            }
        ],
    ).model_dump()
    assert session.calls == [
        (
            "POST",
            "https://api.tavily.com/search",
            {
                "headers": {
                    "Authorization": "Bearer tvly-key",
                    "Content-Type": "application/json",
                },
                "json": {
                    "query": "Apple Inc. latest earnings",
                    "topic": "news",
                    "market_label": "us_equity",
                    "search_depth": "advanced",
                    "max_results": settings.max_search_results,
                },
                "timeout": settings.http_timeout_seconds,
            },
        )
    ]


def test_tavily_search_tool_returns_degraded_output_on_nonfatal_exception() -> None:
    tool = TavilySearchTool(
        settings=build_settings(),
        service=ExplodingTavilySearchService(),
    )

    result = normalize_tool_output(
        tool._run(
            query="latest earnings",
            topic="news",
            market_label="us_equity",
            company_name="Apple",
        )
    )

    assert result["query"] == "latest earnings"
    assert result["topic"] == "news"
    assert result["market_label"] == "us_equity"
    assert result["company_name"] == "Apple"
    assert result["result_count"] == 0
    assert result["results"] == []
    assert result["status"] == "degraded"
    assert "Tavily 搜索失败：network timeout" in result["degraded_reason"]
