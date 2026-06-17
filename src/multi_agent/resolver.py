from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import requests

from multi_agent.settings import InvestmentResearchSettings
from multi_agent.tools.investment_tools import (
    _build_retry_session,
    _perform_request,
    _raise_for_status_with_context,
    _sec_user_agent,
)


_CORPORATE_SUFFIXES = (
    "inc",
    "inc.",
    "corp",
    "corp.",
    "corporation",
    "co",
    "co.",
    "company",
    "limited",
    "ltd",
    "ltd.",
    "holdings",
    "holding",
    "group",
    "plc",
    "ag",
    "sa",
    "nv",
    "llc",
)


@dataclass(frozen=True)
class CompanyResolution:
    user_input: str
    normalized_name: str
    ticker: str
    entity_type: str
    parent_company: str
    exchange: str = ""
    confidence: float = 1.0


class OpenAICompanyResolver:
    """使用轻量 LLM 兜底解析模糊公司输入。"""

    def __init__(
        self,
        settings: InvestmentResearchSettings,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings
        self.session = session or _build_retry_session(settings.max_http_retries)

    def resolve(self, company_name: str) -> CompanyResolution | None:
        try:
            response = _perform_request(
                lambda: self.session.post(
                    f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.settings.company_resolver_model,
                        "temperature": 0,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You normalize company-like user input for an investment research workflow. "
                                    "Return strict JSON with keys: normalized_name, ticker, entity_type, "
                                    "parent_company, exchange, confidence. "
                                    "entity_type must be one of public_company, private_company, business_unit, ambiguous. "
                                    "If the input is a business unit, map it to the listed parent company when clear. "
                                    "If the company is private or no reliable ticker exists, set ticker to an empty string."
                                ),
                            },
                            {
                                "role": "user",
                                "content": f"Normalize this company input: {company_name}",
                            },
                        ],
                        "response_format": {"type": "json_object"},
                    },
                    timeout=self.settings.http_timeout_seconds,
                ),
                service_name="Company Resolver LLM",
            )
            _raise_for_status_with_context(response, service_name="Company Resolver LLM")
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            normalized_name = str(parsed.get("normalized_name", "")).strip()
        except Exception:
            return None

        return CompanyResolution(
            user_input=company_name,
            normalized_name=normalized_name,
            ticker=str(parsed.get("ticker", "")).strip().upper(),
            entity_type=str(parsed.get("entity_type", "ambiguous")).strip() or "ambiguous",
            parent_company=str(parsed.get("parent_company", normalized_name)).strip() or normalized_name,
            exchange=str(parsed.get("exchange", "")).strip(),
            confidence=float(parsed.get("confidence", 0.6) or 0.6),
        )


class CompanyResolver:
    """先走别名和权威映射 API，再用轻量 LLM 兜底的公司解析器。"""

    SEC_TICKER_LOOKUP_URL = "https://www.sec.gov/files/company_tickers.json"

    _ALIASES: dict[str, CompanyResolution] = {
        "阿里": CompanyResolution(
            user_input="阿里",
            normalized_name="Alibaba Group Holding Ltd",
            ticker="BABA",
            entity_type="public_company",
            parent_company="Alibaba Group Holding Ltd",
            exchange="NYSE",
            confidence=0.99,
        ),
        "阿里巴巴": CompanyResolution(
            user_input="阿里巴巴",
            normalized_name="Alibaba Group Holding Ltd",
            ticker="BABA",
            entity_type="public_company",
            parent_company="Alibaba Group Holding Ltd",
            exchange="NYSE",
            confidence=0.99,
        ),
        "字节": CompanyResolution(
            user_input="字节",
            normalized_name="ByteDance Ltd.",
            ticker="",
            entity_type="private_company",
            parent_company="ByteDance Ltd.",
            exchange="",
            confidence=0.98,
        ),
        "字节跳动": CompanyResolution(
            user_input="字节跳动",
            normalized_name="ByteDance Ltd.",
            ticker="",
            entity_type="private_company",
            parent_company="ByteDance Ltd.",
            exchange="",
            confidence=0.98,
        ),
        "索尼手机": CompanyResolution(
            user_input="索尼手机",
            normalized_name="Sony Group Corporation",
            ticker="SONY",
            entity_type="business_unit",
            parent_company="Sony Group Corporation",
            exchange="NYSE",
            confidence=0.97,
        ),
        "Google": CompanyResolution(
            user_input="Google",
            normalized_name="Alphabet Inc.",
            ticker="GOOGL",
            entity_type="public_company",
            parent_company="Alphabet Inc.",
            exchange="NASDAQ",
            confidence=0.99,
        ),
        "谷歌": CompanyResolution(
            user_input="谷歌",
            normalized_name="Alphabet Inc.",
            ticker="GOOGL",
            entity_type="public_company",
            parent_company="Alphabet Inc.",
            exchange="NASDAQ",
            confidence=0.99,
        ),
    }

    def __init__(
        self,
        settings: InvestmentResearchSettings,
        session: requests.Session | None = None,
        llm_resolver: OpenAICompanyResolver | Callable[[str], CompanyResolution | None] | None = None,
    ) -> None:
        self.settings = settings
        self.session = session or _build_retry_session(settings.max_http_retries)
        if llm_resolver is None:
            self.llm_resolver: Callable[[str], CompanyResolution | None] = OpenAICompanyResolver(
                settings=settings,
                session=self.session,
            ).resolve
        elif callable(llm_resolver):
            self.llm_resolver = llm_resolver
        else:
            self.llm_resolver = llm_resolver.resolve

    def resolve(self, company_name: str, ticker: str = "") -> CompanyResolution:
        cleaned_name = company_name.strip()
        cleaned_ticker = ticker.strip().upper()
        if not cleaned_name:
            raise ValueError("公司名称不能为空。")

        resolution = None
        if cleaned_ticker:
            resolution = self._resolve_by_ticker(cleaned_ticker)
            if resolution is None:
                raise ValueError(f"无法识别 ticker：{cleaned_ticker}。")
        else:
            resolution = self._resolve_alias(cleaned_name)
            if resolution is None:
                resolution = self._resolve_by_name(cleaned_name)
            if resolution is None:
                try:
                    resolution = self.llm_resolver(cleaned_name)
                except Exception:
                    resolution = None
            if resolution is None:
                raise ValueError(
                    f"无法根据输入“{cleaned_name}”解析出上市公司，请提供更完整的公司名称或 ticker。"
                )

        finalized = self._ensure_supported_resolution(resolution)
        if cleaned_ticker and finalized.ticker != cleaned_ticker:
            raise ValueError(
                f"输入的 ticker {cleaned_ticker} 与解析结果 {finalized.normalized_name} / {finalized.ticker} 不一致，请确认。"
            )
        return CompanyResolution(
            user_input=cleaned_name,
            normalized_name=finalized.normalized_name,
            ticker=finalized.ticker,
            entity_type=finalized.entity_type,
            parent_company=finalized.parent_company,
            exchange=finalized.exchange,
            confidence=finalized.confidence,
        )

    def _resolve_alias(self, company_name: str) -> CompanyResolution | None:
        return self._ALIASES.get(company_name.strip())

    @lru_cache(maxsize=1)
    def _load_company_directory(self) -> list[dict[str, Any]]:
        response = _perform_request(
            lambda: self.session.get(
                self.SEC_TICKER_LOOKUP_URL,
                headers={"User-Agent": _sec_user_agent(self.settings)},
                timeout=self.settings.http_timeout_seconds,
            ),
            service_name="SEC Ticker Lookup",
        )
        _raise_for_status_with_context(response, service_name="SEC Ticker Lookup")
        payload = response.json()
        if isinstance(payload, dict):
            return [item for item in payload.values() if isinstance(item, dict)]
        return [item for item in payload if isinstance(item, dict)]

    def _resolve_by_ticker(self, ticker: str) -> CompanyResolution | None:
        for item in self._load_company_directory():
            if str(item.get("ticker", "")).upper() == ticker:
                return self._from_directory_result(ticker, item)
        return None

    def _resolve_by_name(self, company_name: str) -> CompanyResolution | None:
        directory = self._load_company_directory()
        if not directory:
            return None
        ranked_results = sorted(
            directory,
            key=lambda item: (
                bool(item.get("isDelisted", False)),
                self._name_distance(company_name, str(item.get("title") or item.get("name", ""))),
                len(str(item.get("title") or item.get("name", ""))),
            ),
        )
        best_match = ranked_results[0]
        if self._name_distance(company_name, str(best_match.get("title") or best_match.get("name", ""))) > 3:
            return None
        return self._from_directory_result(company_name, best_match)

    def _from_mapping_result(self, user_input: str, item: dict[str, Any]) -> CompanyResolution:
        normalized_name = str(item.get("name", "")).strip()
        return CompanyResolution(
            user_input=user_input,
            normalized_name=normalized_name,
            ticker=str(item.get("ticker", "")).strip().upper(),
            entity_type="public_company",
            parent_company=normalized_name,
            exchange=str(item.get("exchange", "")).strip(),
            confidence=1.0,
        )

    def _from_directory_result(self, user_input: str, item: dict[str, Any]) -> CompanyResolution:
        normalized_name = str(item.get("title") or item.get("name", "")).strip()
        return CompanyResolution(
            user_input=user_input,
            normalized_name=normalized_name,
            ticker=str(item.get("ticker", "")).strip().upper(),
            entity_type="public_company",
            parent_company=normalized_name,
            exchange=str(item.get("exchange", "")).strip(),
            confidence=1.0,
        )

    def _ensure_supported_resolution(self, resolution: CompanyResolution) -> CompanyResolution:
        if resolution.entity_type == "ambiguous":
            raise ValueError(
                f"输入“{resolution.user_input}”存在歧义，请补充更完整的公司名称、交易所或 ticker。"
            )
        if not resolution.ticker:
            raise ValueError(
                f"输入“{resolution.user_input}”被识别为未上市公司或无法交易的实体（{resolution.normalized_name}），当前工作流需要上市主体 ticker。"
            )
        return CompanyResolution(
            user_input=resolution.user_input,
            normalized_name=resolution.normalized_name,
            ticker=resolution.ticker.upper(),
            entity_type=resolution.entity_type,
            parent_company=resolution.parent_company or resolution.normalized_name,
            exchange=resolution.exchange,
            confidence=resolution.confidence,
        )

    def _name_distance(self, source: str, candidate: str) -> int:
        normalized_source = self._normalize_label(source)
        normalized_candidate = self._normalize_label(candidate)
        if normalized_source == normalized_candidate:
            return 0
        if normalized_source and normalized_source in normalized_candidate:
            return 1
        return abs(len(normalized_candidate) - len(normalized_source)) + 2

    def _normalize_label(self, value: str) -> str:
        lowered = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value.lower()).strip()
        parts = [part for part in lowered.split() if part not in _CORPORATE_SUFFIXES]
        return " ".join(parts)
