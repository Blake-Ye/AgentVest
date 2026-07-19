from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
import urllib.request

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from multi_agent.evaluation import record_api_call
from multi_agent.settings import InvestmentResearchSettings


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


# #region debug-point A:debug-helper
def _debug_report(
    hypothesis_id: str,
    location: str,
    msg: str,
    data: dict[str, Any] | None = None,
) -> None:
    env_path = Path(".dbg/apple-filing-stall.env")
    debug_server_url = "http://127.0.0.1:7777/event"
    debug_session_id = "apple-filing-stall"
    try:
        env_content = env_path.read_text(encoding="utf-8")
        for line in env_content.splitlines():
            if line.startswith("DEBUG_SERVER_URL="):
                debug_server_url = line.split("=", 1)[1].strip() or debug_server_url
            elif line.startswith("DEBUG_SESSION_ID="):
                debug_session_id = line.split("=", 1)[1].strip() or debug_session_id
        payload = {
            "sessionId": debug_session_id,
            "runId": "pre",
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": msg,
            "data": data or {},
        }
        urllib.request.urlopen(
            urllib.request.Request(
                debug_server_url,
                data=json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.8,
        ).read()
    except Exception:
        pass


# #endregion


def _build_retry_session(max_retries: int) -> requests.Session:
    retry = Retry(
        total=max_retries,
        connect=0,
        read=0,
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
    status_code = getattr(response, "status_code", 0)
    if 200 <= status_code < 400:
        return

    response_text = getattr(response, "text", "")
    if status_code in {401, 403}:
        raise FatalAPIError(
            f"{service_name} 返回 {status_code}，通常表示权限不足、User-Agent 不合规或服务限制，程序已终止。",
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
    # #region debug-point A:sec-request-start
    _debug_report(
        "A",
        "official_sec.py:_perform_request:start",
        f"[DEBUG] SEC request start: {service_name}",
        {"service_name": service_name},
    )
    # #endregion
    try:
        response = request_callable()
    except requests.RequestException as exc:
        # #region debug-point C:sec-request-exception
        _debug_report(
            "C",
            "official_sec.py:_perform_request:exception",
            f"[DEBUG] SEC request exception: {service_name}",
            {"service_name": service_name, "error": repr(exc)},
        )
        # #endregion
        record_api_call(service_name=service_name, success=False, status_code=None)
        raise

    status_code = getattr(response, "status_code", None)
    # #region debug-point B:sec-request-finish
    _debug_report(
        "B",
        "official_sec.py:_perform_request:finish",
        f"[DEBUG] SEC request finish: {service_name}",
        {"service_name": service_name, "status_code": status_code},
    )
    # #endregion
    record_api_call(
        service_name=service_name,
        success=bool(status_code is not None and 200 <= status_code < 400),
        status_code=status_code,
    )
    return response


class OfficialSecService:
    """统一封装基于 SEC 官方端点的数据访问。"""

    SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    NASDAQ_QUOTE_INFO_URL = "https://api.nasdaq.com/api/quote/{ticker}/info?assetclass=stocks"
    STOCK_ANALYSIS_QUOTE_URL = "https://stockanalysis.com/stocks/{ticker}/"

    def __init__(
        self,
        settings: InvestmentResearchSettings,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings
        self.session = session or _build_retry_session(settings.max_http_retries)

    def search_companies_by_name(self, company_name: str, limit: int = 5) -> list[dict[str, Any]]:
        normalized_query = self._normalize_label(company_name)
        ranked_matches: list[tuple[int, dict[str, Any]]] = []
        for item in self._load_ticker_directory():
            title = str(item.get("title", "")).strip()
            if not title:
                continue

            normalized_title = self._normalize_label(title)
            if normalized_query and normalized_query not in normalized_title:
                continue
            ranked_matches.append((self._name_distance(normalized_query, normalized_title), item))

        ranked_matches.sort(key=lambda entry: (entry[0], len(str(entry[1].get("title", "")))))
        return [item for _, item in ranked_matches[:limit]]

    def lookup_company_by_ticker(self, ticker: str) -> dict[str, Any] | None:
        normalized_ticker = ticker.strip().upper()
        for item in self._load_ticker_directory():
            if str(item.get("ticker", "")).upper() == normalized_ticker:
                return item
        return None

    @lru_cache(maxsize=64)
    def lookup_cik(self, ticker: str) -> str:
        item = self.lookup_company_by_ticker(ticker)
        if item is None:
            raise ValueError(f"Unable to find CIK for ticker: {ticker}")
        return str(item["cik_str"]).zfill(10)

    @lru_cache(maxsize=64)
    def fetch_company_facts(self, ticker: str) -> dict[str, Any]:
        cik = self.lookup_cik(ticker)
        response = _perform_request(
            lambda: self.session.get(
                self.SEC_COMPANY_FACTS_URL.format(cik=cik),
                headers=self._sec_headers(),
                timeout=self.settings.http_timeout_seconds,
            ),
            service_name="SEC Company Facts",
        )
        _raise_for_status_with_context(response, service_name="SEC Company Facts")
        return response.json()

    def search_filings(
        self,
        company_name: str,
        ticker: str,
        form_type: str,
        limit: int,
    ) -> list[dict[str, str]]:
        del company_name
        cik = self.lookup_cik(ticker)
        response = _perform_request(
            lambda: self.session.get(
                self.SEC_SUBMISSIONS_URL.format(cik=cik),
                headers=self._sec_headers(),
                timeout=self.settings.http_timeout_seconds,
            ),
            service_name="SEC Submissions",
        )
        _raise_for_status_with_context(response, service_name="SEC Submissions")
        payload = response.json()
        recent_filings = payload.get("filings", {}).get("recent", {})

        forms = recent_filings.get("form", [])
        filed_dates = recent_filings.get("filingDate", [])
        accession_numbers = recent_filings.get("accessionNumber", [])
        primary_documents = recent_filings.get("primaryDocument", [])
        descriptions = recent_filings.get("primaryDocDescription", [])

        normalized_results: list[dict[str, str]] = []
        for index, form in enumerate(forms):
            if str(form).upper() != form_type.upper():
                continue

            accession_number = str(accession_numbers[index])
            accession_compact = accession_number.replace("-", "")
            cik_without_padding = str(int(cik))
            primary_document = str(primary_documents[index])
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_without_padding}/{accession_compact}/{primary_document}"
            )
            normalized_results.append(
                {
                    "form_type": str(form),
                    "filed_at": str(filed_dates[index]),
                    "filing_url": filing_url,
                    "filing_details": str(descriptions[index]),
                    "accession_no": accession_number,
                }
            )
            if len(normalized_results) >= limit:
                break

        return normalized_results

    def fetch_latest_annual_report(self, ticker: str) -> dict[str, str]:
        filings = self.search_filings(company_name="", ticker=ticker, form_type="10-K", limit=1)
        if not filings:
            return {}
        filing = filings[0]
        filing_url = filing["filing_url"]
        response = _perform_request(
            lambda: self.session.get(
                filing_url,
                headers=self._sec_headers(),
                timeout=self.settings.http_timeout_seconds,
            ),
            service_name="SEC Filing HTML",
        )
        _raise_for_status_with_context(response, service_name="SEC Filing HTML")
        return {
            "html": response.text,
            "source_url": filing_url,
            "accession": filing.get("accession_no", ""),
            "filed_at": filing.get("filed_at", ""),
            "form": filing.get("form_type", "10-K"),
        }

    def fetch_latest_annual_report_html(self, ticker: str) -> str:
        """Backward-compatible HTML-only projection for legacy callers."""
        return self.fetch_latest_annual_report(ticker).get("html", "")

    def fetch_market_quote(self, ticker: str) -> dict[str, Any]:
        try:
            response = _perform_request(
                lambda: self.session.get(
                    self.NASDAQ_QUOTE_INFO_URL.format(ticker=ticker.upper()),
                    headers={
                        "User-Agent": f"multi-agent-investment-research {self.settings.sec_api_email}",
                        "Accept": "application/json, text/plain, */*",
                        "Referer": "https://www.nasdaq.com/",
                    },
                    timeout=self.settings.http_timeout_seconds,
                ),
                service_name="Nasdaq Quote Info",
            )
        except requests.RequestException:
            return self._fetch_market_quote_fallback(ticker)
        if getattr(response, "status_code", 0) != 200:
            return self._fetch_market_quote_fallback(ticker)
        try:
            payload = response.json()
        except ValueError:
            return self._fetch_market_quote_fallback(ticker)
        if self._payload_has_last_sale_price(payload):
            return payload
        return self._fetch_market_quote_fallback(ticker)

    def _fetch_market_quote_fallback(self, ticker: str) -> dict[str, Any]:
        try:
            response = _perform_request(
                lambda: self.session.get(
                    self.STOCK_ANALYSIS_QUOTE_URL.format(ticker=ticker.lower()),
                    headers={
                        "User-Agent": f"multi-agent-investment-research {self.settings.sec_api_email}",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Referer": "https://stockanalysis.com/",
                    },
                    timeout=self.settings.http_timeout_seconds,
                ),
                service_name="StockAnalysis Quote Page",
            )
        except requests.RequestException:
            return {}
        if getattr(response, "status_code", 0) != 200:
            return {}
        return self._parse_stockanalysis_quote_payload(getattr(response, "text", ""))

    @staticmethod
    def _payload_has_last_sale_price(payload: dict[str, Any]) -> bool:
        primary_data = payload.get("data", {}).get("primaryData", {})
        return bool(str(primary_data.get("lastSalePrice", "")).strip())

    @staticmethod
    def _parse_stockanalysis_quote_payload(page_html: str) -> dict[str, Any]:
        compact_html = re.sub(r"\s+", " ", page_html)
        match = re.search(
            r'text-4xl[^"]*">([0-9]+(?:\.[0-9]+)?)</div>.*?At close:</span>\s*([^<]+)</div>',
            compact_html,
            re.IGNORECASE,
        )
        if match is None:
            return {}
        stock_price, timestamp = match.groups()
        return {
            "source": "stockanalysis_quote_page",
            "data": {
                "primaryData": {
                    "lastSalePrice": f"${stock_price}",
                    "lastTradeTimestamp": timestamp.strip(),
                }
            }
        }

    @lru_cache(maxsize=1)
    def _load_ticker_directory(self) -> tuple[dict[str, Any], ...]:
        try:
            response = _perform_request(
                lambda: self.session.get(
                    self.SEC_TICKERS_URL,
                    headers=self._sec_headers(),
                    timeout=self.settings.http_timeout_seconds,
                ),
                service_name="SEC Ticker Lookup",
            )
            _raise_for_status_with_context(response, service_name="SEC Ticker Lookup")
            payload = response.json()
            self._write_ticker_directory_cache(payload)
        except requests.RequestException:
            payload = self._read_ticker_directory_cache()
            if payload is None:
                raise
        companies: list[dict[str, Any]] = []
        for raw_item in payload.values():
            title = str(raw_item.get("title", "")).strip()
            companies.append(
                {
                    "title": title,
                    "name": title,
                    "ticker": str(raw_item.get("ticker", "")).strip().upper(),
                    "cik_str": raw_item.get("cik_str"),
                    "exchange": str(raw_item.get("exchange", "")).strip(),
                }
            )
        return tuple(companies)

    def _ticker_directory_cache_path(self) -> Path:
        return Path(self.settings.artifact_root) / "cache" / "sec" / "company_tickers.json"

    def _write_ticker_directory_cache(self, payload: dict[str, Any]) -> None:
        cache_path = self._ticker_directory_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _read_ticker_directory_cache(self) -> dict[str, Any] | None:
        cache_path = self._ticker_directory_cache_path()
        if not cache_path.exists():
            return None
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _sec_headers(self) -> dict[str, str]:
        return {"User-Agent": f"multi-agent-investment-research {self.settings.sec_api_email}"}

    def _name_distance(self, normalized_source: str, normalized_candidate: str) -> int:
        if normalized_source == normalized_candidate:
            return 0
        if normalized_source and normalized_source in normalized_candidate:
            return 1
        return abs(len(normalized_candidate) - len(normalized_source)) + 2

    def _normalize_label(self, value: str) -> str:
        lowered = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value.lower()).strip()
        return " ".join(part for part in lowered.split() if part)
