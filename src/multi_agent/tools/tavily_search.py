from __future__ import annotations

from typing import Any, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from multi_agent.settings import InvestmentResearchSettings
from multi_agent.tools.official_sec import (
    FatalAPIError,
    _build_retry_session,
    _perform_request,
    _raise_for_status_with_context,
)


class TavilySearchInput(BaseModel):
    query: str = Field(..., description="The specific market or news query to search.")
    topic: str = Field(default="news", description="The Tavily topic to search, such as news.")
    market_label: str = Field(default="", description="Normalized market label for the target company.")
    company_name: str = Field(
        default="",
        description="Optional company name kept for backward compatibility and query expansion.",
    )


class TavilySearchResult(BaseModel):
    title: str
    url: str
    source_type: str
    published_at: str
    snippet: str


class TavilySearchOutput(BaseModel):
    query: str
    topic: str
    market_label: str
    company_name: str = ""
    result_count: int
    results: list[TavilySearchResult]
    status: str = "ok"
    degraded_reason: str = ""


class TavilySearchService:
    """Tavily news search client for market intelligence collection."""

    def __init__(
        self,
        settings: InvestmentResearchSettings,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings
        if session is None:
            # Tavily 搜索属于情报补充源，网络抖动时应快速降级而不是长时间阻塞整条 CLI。
            session = _build_retry_session(0)
        self.session = session

    def search_company_news(
        self,
        query: str,
        topic: str = "news",
        market_label: str = "",
        company_name: str = "",
    ) -> dict[str, Any]:
        api_key = self._require_api_key(self.settings.tavily_api_key)
        search_query = f"{company_name} {query}".strip()
        response = _perform_request(
            lambda: self.session.post(
                "https://api.tavily.com/search",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": search_query,
                    "topic": topic,
                    "market_label": market_label,
                    "search_depth": "advanced",
                    "max_results": self.settings.max_search_results,
                },
                timeout=self.settings.http_timeout_seconds,
            ),
            service_name="Tavily Search",
        )
        _raise_for_status_with_context(response, service_name="Tavily Search")
        results = self._normalize_search_results(response.json())
        return TavilySearchOutput(
            query=query,
            topic=topic,
            market_label=market_label,
            company_name=company_name,
            result_count=len(results),
            results=results,
        ).model_dump()

    def _require_api_key(self, api_key: str) -> str:
        if not api_key.strip():
            raise ValueError("Tavily 的 API Key 未配置。")
        return api_key.strip()

    def _normalize_search_results(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        normalized_results: list[dict[str, str]] = []
        for result in payload.get("results", [])[: self.settings.max_search_results]:
            normalized_results.append(
                {
                    "title": str(result.get("title", "")),
                    "url": str(result.get("url", "")),
                    "source_type": str(result.get("type", "")),
                    "published_at": str(result.get("published_date", "")),
                    "snippet": str(result.get("content", "")),
                }
            )
        return normalized_results


class TavilySearchTool(BaseTool):
    name: str = "Tavily Search Intelligence"
    description: str = (
        "Search Tavily for recent company news, competition signals, and market events."
    )
    args_schema: Type[BaseModel] = TavilySearchInput

    def __init__(
        self,
        settings: InvestmentResearchSettings | None = None,
        service: TavilySearchService | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._settings = settings or InvestmentResearchSettings.from_env()
        self._service = service or TavilySearchService(self._settings)

    def _run(
        self,
        query: str,
        topic: str = "news",
        market_label: str = "",
        company_name: str = "",
    ) -> dict[str, Any]:
        try:
            return self._service.search_company_news(
                query=query,
                topic=topic,
                market_label=market_label,
                company_name=company_name,
            )
        except FatalAPIError:
            raise
        except Exception as exc:  # pragma: no cover - network failure path
            return TavilySearchOutput(
                query=query,
                topic=topic,
                market_label=market_label,
                company_name=company_name,
                result_count=0,
                results=[],
                status="degraded",
                degraded_reason=f"Tavily 搜索失败：{exc}",
            ).model_dump()
