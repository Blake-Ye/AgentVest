from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Type

import requests
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from crewai.tools import BaseTool

from multi_agent.evaluation import record_api_call, record_financial_fields
from multi_agent.finance import CompanyFinancialSnapshot, compute_key_metrics
from multi_agent.market_profile import IssuerProfile, MarketIdentifierService, MarketScope
from multi_agent.settings import InvestmentResearchSettings

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency guard
    PdfReader = None


class FatalAPIError(RuntimeError):
    """表示必须立即中止整个工作流的外部服务错误。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        service_name: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.service_name = service_name


def _safe_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _success_payload(data: dict[str, Any], *, source: str) -> str:
    return _safe_json_dumps({"status": "success", "source": source, "data": data})


def _degraded_payload(
    *,
    source: str,
    message: str,
    details: dict[str, Any],
) -> str:
    return _safe_json_dumps(
        {
            "status": "degraded",
            "source": source,
            "error_type": "market_not_applicable",
            "message": message,
            "retryable": False,
            "details": details,
        }
    )


def _build_retry_session(max_retries: int) -> requests.Session:
    # 将常见瞬时失败交给 requests 自动重试，减少偶发网络抖动带来的失败。
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _raise_for_status_with_context(response: Any, service_name: str) -> None:
    """把外部 HTTP 错误转换成更适合终端用户理解的中文异常。"""
    status_code = getattr(response, "status_code", 0)
    if 200 <= status_code < 400:
        return

    response_text = getattr(response, "text", "")
    if status_code in {401, 403}:
        raise FatalAPIError(
            f"{service_name} 返回 {status_code}，通常表示 API Key 无效、权限不足或额度受限，程序已终止。"
            ,
            status_code=status_code,
            service_name=service_name,
        )
    if status_code == 429:
        raise FatalAPIError(
            f"{service_name} 返回 429，请求过于频繁或额度已用尽，程序已终止。",
            status_code=status_code,
            service_name=service_name,
        )
    if 400 <= status_code < 500:
        raise FatalAPIError(
            f"{service_name} 返回 {status_code}，请求未被接受，程序已终止。响应内容：{response_text[:200]}",
            status_code=status_code,
            service_name=service_name,
        )

    raise RuntimeError(
        f"{service_name} 服务暂时不可用，状态码 {status_code}。响应内容：{response_text[:200]}"
    )


def _perform_request(request_callable: Any, *, service_name: str) -> Any:
    try:
        response = request_callable()
    except requests.RequestException:
        record_api_call(service_name=service_name, success=False, status_code=None)
        raise

    status_code = getattr(response, "status_code", None)
    record_api_call(
        service_name=service_name,
        success=bool(status_code is not None and 200 <= status_code < 400),
        status_code=status_code,
    )
    return response


class GoogleSearchService:
    """同时兼容 Serper 与 SerpApi 的 Google 搜索客户端。"""

    def __init__(
        self,
        settings: InvestmentResearchSettings,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings
        self.session = session or _build_retry_session(settings.max_http_retries)

    def search_company_news(self, company_name: str, query: str) -> list[dict[str, str]]:
        provider = self.settings.search_provider

        if provider == "serper":
            api_key = self._require_api_key(self.settings.serper_api_key, provider_name="Serper")
            return self._search_via_serper(api_key=api_key, company_name=company_name, query=query)

        if provider == "serpapi":
            api_key = self._resolve_serpapi_key()
            return self._search_via_serpapi(api_key=api_key, company_name=company_name, query=query)

        return self._search_with_auto_provider(company_name=company_name, query=query)

    def _search_with_auto_provider(self, company_name: str, query: str) -> list[dict[str, str]]:
        serper_key = self.settings.serper_api_key
        serpapi_key = self.settings.serpapi_api_key

        if serper_key:
            try:
                return self._search_via_serper(
                    api_key=serper_key,
                    company_name=company_name,
                    query=query,
                )
            except FatalAPIError as exc:
                # 用户经常把 SerpApi 的 key 填进 SERPER_API_KEY，这里在鉴权失败时自动回退。
                if exc.status_code not in {401, 403}:
                    raise
                fallback_key = serpapi_key or serper_key
                return self._search_via_serpapi(
                    api_key=fallback_key,
                    company_name=company_name,
                    query=query,
                )

        return self._search_via_serpapi(
            api_key=self._resolve_serpapi_key(),
            company_name=company_name,
            query=query,
        )

    def _resolve_serpapi_key(self) -> str:
        return self._require_api_key(
            self.settings.serpapi_api_key or self.settings.serper_api_key,
            provider_name="SerpApi",
        )

    def _require_api_key(self, api_key: str, provider_name: str) -> str:
        if not api_key.strip():
            raise ValueError(f"{provider_name} 的 API Key 未配置。")
        return api_key.strip()

    def _search_via_serper(self, api_key: str, company_name: str, query: str) -> list[dict[str, str]]:
        response = _perform_request(
            lambda: self.session.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "q": f"{company_name} {query}",
                    "gl": "us",
                    "hl": "en",
                    "num": self.settings.max_search_results,
                },
                timeout=self.settings.http_timeout_seconds,
            ),
            service_name="Google Search",
        )
        _raise_for_status_with_context(response, service_name="Google Search")
        return self._normalize_search_results(payload=response.json(), provider_name="serper")

    def _search_via_serpapi(self, api_key: str, company_name: str, query: str) -> list[dict[str, str]]:
        response = _perform_request(
            lambda: self.session.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": "google",
                    "q": f"{company_name} {query}",
                    "api_key": api_key,
                    "num": self.settings.max_search_results,
                    "hl": "en",
                    "gl": "us",
                },
                timeout=self.settings.http_timeout_seconds,
            ),
            service_name="Google Search",
        )
        _raise_for_status_with_context(response, service_name="Google Search")
        return self._normalize_search_results(payload=response.json(), provider_name="serpapi")

    def _normalize_search_results(
        self,
        payload: dict[str, Any],
        provider_name: str,
    ) -> list[dict[str, str]]:
        organic_results = payload.get("organic", []) if provider_name == "serper" else payload.get("organic_results", [])
        normalized_results: list[dict[str, str]] = []
        for result in organic_results[: self.settings.max_search_results]:
            normalized_results.append(
                {
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                }
            )
        return normalized_results


class SecApiService:
    """封装 SEC filing 检索与 SEC 官方 company facts 获取逻辑。"""

    SEC_TICKER_LOOKUP_URL = "https://www.sec.gov/files/company_tickers.json"
    SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    def __init__(
        self,
        settings: InvestmentResearchSettings,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings
        self.session = session or _build_retry_session(settings.max_http_retries)

    def search_filings(
        self,
        company_name: str,
        ticker: str,
        form_type: str,
        limit: int,
    ) -> list[dict[str, str]]:
        query = (
            f'ticker:{ticker.upper()} AND formType:"{form_type}" AND companyName:"{company_name}"'
        )
        response = _perform_request(
            lambda: self.session.post(
                "https://api.sec-api.io",
                headers={
                    "Authorization": self.settings.sec_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "from": "0",
                    "size": str(limit),
                    "sort": [{"filedAt": {"order": "desc"}}],
                },
                timeout=self.settings.http_timeout_seconds,
            ),
            service_name="SEC API",
        )
        _raise_for_status_with_context(response, service_name="SEC API")
        payload = response.json()
        normalized_results: list[dict[str, str]] = []
        for filing in payload.get("filings", [])[:limit]:
            normalized_results.append(
                {
                    "form_type": filing.get("formType", ""),
                    "filed_at": filing.get("filedAt", ""),
                    "filing_url": filing.get("linkToFilingDetails", "")
                    or filing.get("linkToHtml", "")
                    or filing.get("linkToTxt", ""),
                    "filing_details": filing.get("description", ""),
                    "accession_no": filing.get("accessionNo", ""),
                }
            )
        return normalized_results

    @lru_cache(maxsize=64)
    def fetch_company_facts(self, ticker: str) -> dict[str, Any]:
        cik = self._lookup_cik(ticker)
        response = _perform_request(
            lambda: self.session.get(
                self.SEC_COMPANY_FACTS_URL.format(cik=cik),
                headers={"User-Agent": f"multi-agent-investment-research {self.settings.sec_api_email}"},
                timeout=self.settings.http_timeout_seconds,
            ),
            service_name="SEC Company Facts",
        )
        _raise_for_status_with_context(response, service_name="SEC Company Facts")
        return response.json()

    @lru_cache(maxsize=64)
    def _lookup_cik(self, ticker: str) -> str:
        response = _perform_request(
            lambda: self.session.get(
                self.SEC_TICKER_LOOKUP_URL,
                headers={"User-Agent": f"multi-agent-investment-research {self.settings.sec_api_email}"},
                timeout=self.settings.http_timeout_seconds,
            ),
            service_name="SEC Ticker Lookup",
        )
        _raise_for_status_with_context(response, service_name="SEC Ticker Lookup")
        ticker_payload = response.json()
        normalized_ticker = ticker.upper()
        for company in ticker_payload.values():
            if company.get("ticker", "").upper() == normalized_ticker:
                return str(company["cik_str"]).zfill(10)
        raise ValueError(f"Unable to find CIK for ticker: {ticker}")


@dataclass(frozen=True)
class MarketProviderDescriptor:
    source_name: str
    market_scope: str
    official_entrypoint: str

    def query_hint(self, *, company_name: str, ticker: str) -> str:
        normalized_name = company_name.strip()
        normalized_ticker = ticker.strip().upper()
        if self.market_scope == MarketScope.HKEX.value:
            return f"{normalized_ticker} {normalized_name}".strip()
        if self.market_scope == MarketScope.CN_A_SHARE.value:
            return f"{normalized_ticker} {normalized_name}".strip()
        if self.market_scope == MarketScope.EU_LISTED.value:
            return f"{normalized_ticker} {normalized_name}".strip()
        return normalized_ticker or normalized_name


class SourceRouter:
    HKEX_DISCLOSURE_URL = "https://www.hkexnews.hk/search/titlesearch.xhtml"
    CN_DISCLOSURE_URL = "http://www.cninfo.com.cn/new/fulltextSearch"
    EU_DISCLOSURE_URL = "https://live.euronext.com/"

    def __init__(self, settings: InvestmentResearchSettings) -> None:
        self._identifier = MarketIdentifierService(settings)

    def identify_issuer(self, company_name: str, ticker: str) -> IssuerProfile:
        return self._identifier.identify(company_name=company_name, ticker=ticker)

    def provider_for_disclosures(self, profile: IssuerProfile) -> MarketProviderDescriptor:
        if profile.sec_applicable:
            return MarketProviderDescriptor(
                source_name="sec_provider",
                market_scope=profile.market_scope,
                official_entrypoint="https://www.sec.gov/edgar/search/",
            )
        if profile.market_scope == MarketScope.HKEX.value:
            return MarketProviderDescriptor(
                source_name="hkex_provider",
                market_scope=profile.market_scope,
                official_entrypoint=self.HKEX_DISCLOSURE_URL,
            )
        if profile.market_scope == MarketScope.CN_A_SHARE.value:
            return MarketProviderDescriptor(
                source_name="cn_provider",
                market_scope=profile.market_scope,
                official_entrypoint=self.CN_DISCLOSURE_URL,
            )
        if profile.market_scope == MarketScope.EU_LISTED.value:
            return MarketProviderDescriptor(
                source_name="eu_provider",
                market_scope=profile.market_scope,
                official_entrypoint=self.EU_DISCLOSURE_URL,
            )
        return MarketProviderDescriptor(
            source_name="generic_official_provider",
            market_scope=profile.market_scope,
            official_entrypoint="",
        )


@dataclass(frozen=True)
class FinancialFieldExtraction:
    value: float
    normalized_value: float
    extracted: bool
    source_tag: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "normalized_value": self.normalized_value,
            "extracted": self.extracted,
            "source_tag": self.source_tag,
        }


def _extract_latest_fact(company_facts: dict[str, Any], candidate_tags: list[str]) -> FinancialFieldExtraction:
    # 同一财务指标常常对应多个候选标签，这里按优先级挑选最新且可解析的值。
    facts = company_facts.get("facts", {}).get("us-gaap", {})
    for tag in candidate_tags:
        tag_payload = facts.get(tag)
        if not tag_payload:
            continue

        units = tag_payload.get("units", {})
        for unit_name in ("USD", "USD/shares", "shares"):
            entries = units.get(unit_name)
            if not entries:
                continue

            comparable_entries = [
                entry
                for entry in entries
                if entry.get("val") is not None
            ]
            if not comparable_entries:
                continue

            latest_entry = max(
                comparable_entries,
                key=lambda item: (
                    item.get("end", ""),
                    item.get("filed", ""),
                    item.get("fy", 0),
                ),
            )
            value = float(latest_entry.get("val", 0.0))
            return FinancialFieldExtraction(
                value=value,
                normalized_value=value,
                extracted=True,
                source_tag=tag,
            )
    return FinancialFieldExtraction(
        value=0.0,
        normalized_value=0.0,
        extracted=False,
        source_tag=None,
    )


def _build_financial_snapshot_with_metadata(
    company_facts: dict[str, Any],
) -> tuple[CompanyFinancialSnapshot, dict[str, dict[str, Any]]]:
    revenue = _extract_latest_fact(
        company_facts,
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "Revenues",
        ],
    )
    gross_profit = _extract_latest_fact(company_facts, ["GrossProfit"])
    operating_income = _extract_latest_fact(company_facts, ["OperatingIncomeLoss"])
    net_income = _extract_latest_fact(company_facts, ["NetIncomeLoss"])
    current_assets = _extract_latest_fact(company_facts, ["AssetsCurrent"])
    current_liabilities = _extract_latest_fact(company_facts, ["LiabilitiesCurrent"])
    total_assets = _extract_latest_fact(company_facts, ["Assets"])
    total_liabilities = _extract_latest_fact(company_facts, ["Liabilities"])
    operating_cash_flow = _extract_latest_fact(
        company_facts,
        ["NetCashProvidedByUsedInOperatingActivities"],
    )
    capital_expenditure = _extract_latest_fact(
        company_facts,
        [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "CapitalExpendituresIncurredButNotYetPaid",
        ],
    )
    normalized_capex = -abs(capital_expenditure.value) if capital_expenditure.extracted else 0.0
    capital_expenditure_metadata = FinancialFieldExtraction(
        value=capital_expenditure.value,
        normalized_value=normalized_capex,
        extracted=capital_expenditure.extracted,
        source_tag=capital_expenditure.source_tag,
    )

    snapshot = CompanyFinancialSnapshot(
        revenue=revenue.normalized_value,
        gross_profit=gross_profit.normalized_value,
        operating_income=operating_income.normalized_value,
        net_income=net_income.normalized_value,
        current_assets=current_assets.normalized_value,
        current_liabilities=current_liabilities.normalized_value,
        total_assets=total_assets.normalized_value,
        total_liabilities=total_liabilities.normalized_value,
        operating_cash_flow=operating_cash_flow.normalized_value,
        capital_expenditure=capital_expenditure_metadata.normalized_value,
    )
    metadata = {
        "revenue": revenue.as_dict(),
        "gross_profit": gross_profit.as_dict(),
        "operating_income": operating_income.as_dict(),
        "net_income": net_income.as_dict(),
        "current_assets": current_assets.as_dict(),
        "current_liabilities": current_liabilities.as_dict(),
        "total_assets": total_assets.as_dict(),
        "total_liabilities": total_liabilities.as_dict(),
        "operating_cash_flow": operating_cash_flow.as_dict(),
        "capital_expenditure": capital_expenditure_metadata.as_dict(),
    }
    return snapshot, metadata


def _build_financial_snapshot(company_facts: dict[str, Any]) -> CompanyFinancialSnapshot:
    snapshot, _ = _build_financial_snapshot_with_metadata(company_facts)
    return snapshot


class FileReadInput(BaseModel):
    file_path: str = Field(..., description="Absolute or relative file path to read.")


class FileReadTool(BaseTool):
    name: str = "Read Local Artifact"
    description: str = "Read the contents of a local markdown, text, html, or json artifact."
    args_schema: Type[BaseModel] = FileReadInput

    def _run(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            return f"错误：文件不存在 - {file_path}"
        return path.read_text(encoding="utf-8")


class FileWriteInput(BaseModel):
    file_path: str = Field(..., description="Absolute or relative file path to write.")
    content: str = Field(..., description="Text content to persist to disk.")


class FileWriteTool(BaseTool):
    name: str = "Write Local Artifact"
    description: str = "Persist intermediate analysis or final reports to a local file."
    args_schema: Type[BaseModel] = FileWriteInput

    def _run(self, file_path: str, content: str) -> str:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"文件已保存到 {path}"


class PDFExtractInput(BaseModel):
    file_path: str = Field(..., description="Absolute or relative path to a PDF file.")
    max_pages: int = Field(default=10, description="Maximum number of pages to extract.")


class PDFTextExtractTool(BaseTool):
    name: str = "Extract PDF Text"
    description: str = "Extract text from a local PDF filing for downstream financial analysis."
    args_schema: Type[BaseModel] = PDFExtractInput

    def _run(self, file_path: str, max_pages: int = 10) -> str:
        if PdfReader is None:
            return "错误：当前环境未安装 pypdf，无法提取 PDF 内容。"

        path = Path(file_path)
        if not path.exists():
            return f"错误：文件不存在 - {file_path}"

        reader = PdfReader(str(path))
        extracted_pages: list[str] = []
        for page in reader.pages[:max_pages]:
            extracted_pages.append(page.extract_text() or "")
        return "\n".join(extracted_pages).strip()


class GoogleSearchInput(BaseModel):
    company_name: str = Field(..., description="The company name to investigate.")
    query: str = Field(..., description="The specific market or news query to search.")


class GoogleSearchTool(BaseTool):
    name: str = "Google Search Intelligence"
    description: str = (
        "Search Google via Serper for recent company news, competition signals, and market events."
    )
    args_schema: Type[BaseModel] = GoogleSearchInput

    def __init__(
        self,
        settings: InvestmentResearchSettings | None = None,
        service: GoogleSearchService | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._settings = settings or InvestmentResearchSettings.from_env()
        self._service = service or GoogleSearchService(self._settings)

    def _run(self, company_name: str, query: str) -> str:
        try:
            results = self._service.search_company_news(company_name=company_name, query=query)
        except FatalAPIError:
            raise
        except Exception as exc:  # pragma: no cover - network failure path
            return f"Google 搜索失败：{exc}"

        if not results:
            return "未找到相关的 Google 搜索结果。"

        formatted_results = []
        for index, result in enumerate(results, start=1):
            formatted_results.append(
                "\n".join(
                    [
                        f"{index}. {result['title']}",
                        f"链接：{result['link']}",
                        f"摘要：{result['snippet']}",
                    ]
                )
            )
        return "\n\n".join(formatted_results)


class OfficialDisclosureSearchInput(BaseModel):
    company_name: str = Field(..., description="The company name to investigate.")
    ticker: str = Field(..., description="The public ticker symbol to route.")


class OfficialDisclosureSearchTool(BaseTool):
    name: str = "Official Disclosure Search"
    description: str = "Route a company to the correct official disclosure entrypoint for its market."
    args_schema: Type[BaseModel] = OfficialDisclosureSearchInput

    def __init__(
        self,
        settings: InvestmentResearchSettings | None = None,
        sec_service: SecApiService | None = None,
        router: SourceRouter | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._settings = settings or InvestmentResearchSettings.from_env()
        self._sec_service = sec_service or SecApiService(self._settings)
        self._router = router or SourceRouter(self._settings)

    def _run(self, company_name: str, ticker: str) -> str:
        profile = self._router.identify_issuer(company_name=company_name, ticker=ticker)
        provider = self._router.provider_for_disclosures(profile)

        if provider.source_name == "sec_provider":
            filings = self._sec_service.search_filings(
                company_name=company_name,
                ticker=profile.canonical_ticker,
                form_type="10-K",
                limit=3,
            )
            return _success_payload(
                {
                    "issuer_profile": profile.as_dict(),
                    "official_results": filings,
                },
                source="sec_provider",
            )

        return _success_payload(
            {
                "issuer_profile": profile.as_dict(),
                "official_entrypoint": provider.official_entrypoint,
                "query_hint": provider.query_hint(company_name=company_name, ticker=ticker),
            },
            source=provider.source_name,
        )


class SecFilingSearchInput(BaseModel):
    company_name: str = Field(..., description="Legal company name to search in SEC filings.")
    ticker: str = Field(..., description="Public ticker symbol, such as AAPL.")
    form_type: str = Field(default="10-K", description="SEC form type, such as 10-K or 10-Q.")
    limit: int = Field(default=3, description="Maximum number of filings to return.")


class SecFilingSearchTool(BaseTool):
    name: str = "SEC Filing Search"
    description: str = "Find the latest SEC filings for a public company using SEC API."
    args_schema: Type[BaseModel] = SecFilingSearchInput

    def __init__(
        self,
        settings: InvestmentResearchSettings | None = None,
        service: SecApiService | None = None,
        market_identifier: MarketIdentifierService | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._settings = settings or InvestmentResearchSettings.from_env()
        self._service = service or SecApiService(self._settings)
        self._market_identifier = market_identifier or MarketIdentifierService(self._settings)

    def _run(self, company_name: str, ticker: str, form_type: str = "10-K", limit: int = 3) -> str:
        profile = self._market_identifier.identify(company_name=company_name, ticker=ticker)
        if not profile.sec_applicable:
            return _degraded_payload(
                source="sec_provider",
                message=f"当前 ticker {profile.canonical_ticker} 属于非 SEC 市场，SEC 数据源不适用。",
                details=profile.as_dict(),
            )
        try:
            filings = self._service.search_filings(
                company_name=company_name,
                ticker=profile.canonical_ticker,
                form_type=form_type,
                limit=limit,
            )
        except FatalAPIError:
            raise
        except Exception as exc:  # pragma: no cover - network failure path
            return f"SEC 文件检索失败：{exc}"

        if not filings:
            return "未找到相关的 SEC 文件。"

        lines: list[str] = []
        for filing in filings:
            lines.extend(
                [
                    f"- 表单类型：{filing['form_type']}",
                    f"  提交时间：{filing['filed_at']}",
                    f"  链接：{filing['filing_url']}",
                    f"  Accession 编号：{filing['accession_no']}",
                ]
            )
        return "\n".join(lines)


class SecCompanyFactsInput(BaseModel):
    ticker: str = Field(..., description="Public ticker symbol, such as AAPL.")


class SecCompanyFactsTool(BaseTool):
    name: str = "SEC Company Facts"
    description: str = "Fetch structured company facts from the SEC XBRL company facts API."
    args_schema: Type[BaseModel] = SecCompanyFactsInput

    def __init__(
        self,
        settings: InvestmentResearchSettings | None = None,
        service: SecApiService | None = None,
        market_identifier: MarketIdentifierService | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._settings = settings or InvestmentResearchSettings.from_env()
        self._service = service or SecApiService(self._settings)
        self._market_identifier = market_identifier or MarketIdentifierService(self._settings)

    def _run(self, ticker: str) -> str:
        profile = self._market_identifier.identify(company_name="", ticker=ticker)
        if not profile.sec_applicable:
            return _degraded_payload(
                source="sec_provider",
                message=f"当前 ticker {profile.canonical_ticker} 属于非 SEC 市场，SEC 数据源不适用。",
                details=profile.as_dict(),
            )
        try:
            company_facts = self._service.fetch_company_facts(profile.canonical_ticker)
            snapshot, metadata = _build_financial_snapshot_with_metadata(company_facts)
        except FatalAPIError:
            raise
        except Exception as exc:  # pragma: no cover - network failure path
            return f"获取 SEC 公司财务事实失败：{exc}"

        record_financial_fields(metadata)
        return json.dumps(snapshot.__dict__, indent=2, ensure_ascii=False)


class FinancialMetricsInput(BaseModel):
    ticker: str = Field(..., description="Public ticker symbol, such as AAPL.")


class FinancialMetricsTool(BaseTool):
    name: str = "Financial Metrics Calculator"
    description: str = "Compute core investment ratios from SEC company facts."
    args_schema: Type[BaseModel] = FinancialMetricsInput

    def __init__(
        self,
        settings: InvestmentResearchSettings | None = None,
        service: SecApiService | None = None,
        market_identifier: MarketIdentifierService | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._settings = settings or InvestmentResearchSettings.from_env()
        self._service = service or SecApiService(self._settings)
        self._market_identifier = market_identifier or MarketIdentifierService(self._settings)

    def _run(self, ticker: str) -> str:
        profile = self._market_identifier.identify(company_name="", ticker=ticker)
        if not profile.sec_applicable:
            return _degraded_payload(
                source="sec_provider",
                message=f"当前 ticker {profile.canonical_ticker} 属于非 SEC 市场，SEC 数据源不适用。",
                details=profile.as_dict(),
            )
        try:
            company_facts = self._service.fetch_company_facts(profile.canonical_ticker)
            snapshot, metadata = _build_financial_snapshot_with_metadata(company_facts)
            metrics = compute_key_metrics(snapshot)
        except FatalAPIError:
            raise
        except Exception as exc:  # pragma: no cover - network failure path
            return f"计算财务指标失败：{exc}"

        record_financial_fields(metadata)
        return json.dumps(metrics, indent=2, ensure_ascii=False)
