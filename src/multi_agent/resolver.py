from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import quote

import requests

from multi_agent.market_profile import MarketIdentifierService
from multi_agent.settings import InvestmentResearchSettings
from multi_agent.tools.investment_tools import (
    _build_retry_session,
    _perform_request,
    _raise_for_status_with_context,
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

    def validate_pair(self, company_name: str, ticker: str) -> tuple[bool, str] | None:
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
                                    "You validate whether a company name and a public ticker refer to the same public issuer. "
                                    "Treat dual listings, ADRs, and secondary listings of the same issuer as consistent. "
                                    "Return strict JSON with keys: is_consistent, canonical_name, confidence."
                                ),
                            },
                            {
                                "role": "user",
                                "content": f"company_name={company_name}\nticker={ticker}",
                            },
                        ],
                        "response_format": {"type": "json_object"},
                    },
                    timeout=self.settings.http_timeout_seconds,
                ),
                service_name="Company Pair Validator LLM",
            )
            _raise_for_status_with_context(response, service_name="Company Pair Validator LLM")
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except Exception:
            return None
        canonical_name = parsed.get("canonical_name")
        return bool(parsed.get("is_consistent")), str(canonical_name).strip() if canonical_name is not None else ""


class CompanyResolver:
    """先走别名和权威映射 API，再用轻量 LLM 兜底的公司解析器。"""

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
    }

    def __init__(
        self,
        settings: InvestmentResearchSettings,
        session: requests.Session | None = None,
        llm_resolver: OpenAICompanyResolver | Callable[[str], CompanyResolution | None] | None = None,
        pair_validator: Callable[[str, str], tuple[bool, str] | bool | None] | None = None,
    ) -> None:
        self.settings = settings
        self.session = session or _build_retry_session(settings.max_http_retries)
        default_llm_resolver = OpenAICompanyResolver(
            settings=settings,
            session=self.session,
        )
        if llm_resolver is None:
            self.llm_resolver: Callable[[str], CompanyResolution | None] = default_llm_resolver.resolve
        elif callable(llm_resolver):
            self.llm_resolver = llm_resolver
        else:
            self.llm_resolver = llm_resolver.resolve
        self.pair_validator = pair_validator or default_llm_resolver.validate_pair
        self.market_identifier = MarketIdentifierService(settings)

    def resolve(self, company_name: str, ticker: str = "") -> CompanyResolution:
        cleaned_name = company_name.strip()
        cleaned_ticker = ticker.strip().upper()
        if not cleaned_name:
            raise ValueError("公司名称不能为空。")

        resolution = None
        if cleaned_ticker:
            market_profile = self.market_identifier.identify(company_name=cleaned_name, ticker=cleaned_ticker)
            if market_profile.sec_applicable:
                resolution = self._resolve_by_ticker(cleaned_ticker)
            elif market_profile.market_scope != "unknown":
                canonical_name = self._validate_non_sec_ticker_alignment(cleaned_name, cleaned_ticker)
                normalized_name = canonical_name or cleaned_name or market_profile.canonical_ticker
                resolution = CompanyResolution(
                    user_input=cleaned_name,
                    normalized_name=normalized_name,
                    ticker=market_profile.canonical_ticker,
                    entity_type="public_company",
                    parent_company=normalized_name,
                    exchange=market_profile.exchange,
                    confidence=0.95,
                )
            else:
                raise ValueError(f"暂不支持识别 ticker：{cleaned_ticker}，请提供受支持市场的 ticker 或更完整的公司名称。")
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

    def _validate_non_sec_ticker_alignment(self, company_name: str, ticker: str) -> str:
        alias_resolution = self._resolve_alias(company_name)
        if alias_resolution is not None and alias_resolution.entity_type != "public_company":
            raise ValueError(
                f"输入的公司名称 {company_name} 不是可直接研究的上市公司，请确认后重试。"
            )

        validation_result = self._coerce_pair_validation_result(self.pair_validator(company_name, ticker))
        if validation_result is None:
            raise ValueError(
                f"暂时无法校验公司名称 {company_name} 与 ticker {ticker} 是否一致，请确认后重试。"
            )
        is_consistent, canonical_name = validation_result
        if not is_consistent:
            raise ValueError(
                f"输入的公司名称 {company_name} 与 ticker {ticker} 不一致，请确认后重试。"
            )
        if alias_resolution is not None and alias_resolution.normalized_name:
            return alias_resolution.normalized_name
        return canonical_name

    def _coerce_pair_validation_result(
        self,
        validation_result: tuple[bool, str] | bool | None,
    ) -> tuple[bool, str] | None:
        if validation_result is None:
            return None
        if isinstance(validation_result, tuple):
            is_consistent, canonical_name = validation_result
            return bool(is_consistent), str(canonical_name).strip() if canonical_name is not None else ""
        return bool(validation_result), ""

    def _resolve_alias(self, company_name: str) -> CompanyResolution | None:
        return self._ALIASES.get(company_name.strip())

    def _resolve_by_ticker(self, ticker: str) -> CompanyResolution | None:
        url = f"https://api.sec-api.io/mapping/ticker/{quote(ticker)}?token={self.settings.sec_api_key}"
        response = _perform_request(
            lambda: self.session.get(url, timeout=self.settings.http_timeout_seconds),
            service_name="SEC Mapping API",
        )
        _raise_for_status_with_context(response, service_name="SEC Mapping API")
        payload = response.json()
        if not payload:
            return None
        for item in payload:
            if str(item.get("ticker", "")).upper() == ticker:
                return self._from_mapping_result(ticker, item)
        return self._from_mapping_result(ticker, payload[0])

    def _resolve_by_name(self, company_name: str) -> CompanyResolution | None:
        url = f"https://api.sec-api.io/mapping/name/{quote(company_name)}?token={self.settings.sec_api_key}"
        response = _perform_request(
            lambda: self.session.get(url, timeout=self.settings.http_timeout_seconds),
            service_name="SEC Mapping API",
        )
        _raise_for_status_with_context(response, service_name="SEC Mapping API")
        payload = response.json()
        if not payload:
            return None
        ranked_results = sorted(
            payload,
            key=lambda item: (
                bool(item.get("isDelisted", False)),
                self._name_distance(company_name, str(item.get("name", ""))),
                len(str(item.get("name", ""))),
            ),
        )
        return self._from_mapping_result(company_name, ranked_results[0])

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
