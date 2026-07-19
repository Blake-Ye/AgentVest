import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent.settings import InvestmentResearchSettings
from multi_agent.tools.official_sec import OfficialSecService, _build_retry_session


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
    def __init__(self, status_code: int, payload) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class RecordingSession:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, **kwargs):
        self.requests.append((url, kwargs))
        response = self.responses.get(url)
        if response is None:
            return FakeResponse(404, {"message": "not found"})
        if isinstance(response, Exception):
            raise response
        return FakeResponse(200, response)


def test_official_sec_service_matches_company_name_from_ticker_directory() -> None:
    session = RecordingSession(
        {
            "https://www.sec.gov/files/company_tickers.json": {
                "0": {"title": "Apple Inc.", "ticker": "AAPL", "cik_str": 320193},
                "1": {"title": "Alibaba Group Holding Ltd", "ticker": "BABA", "cik_str": 1577552},
            }
        }
    )
    service = OfficialSecService(settings=build_settings(), session=session)

    matches = service.search_companies_by_name("Alibaba Group")

    assert matches[0]["title"] == "Alibaba Group Holding Ltd"
    assert matches[0]["ticker"] == "BABA"
    assert session.requests[0][0] == "https://www.sec.gov/files/company_tickers.json"


def test_official_sec_service_builds_filings_from_official_submissions() -> None:
    session = RecordingSession(
        {
            "https://www.sec.gov/files/company_tickers.json": {
                "0": {"title": "Apple Inc.", "ticker": "AAPL", "cik_str": 320193}
            },
            "https://data.sec.gov/submissions/CIK0000320193.json": {
                "name": "Apple Inc.",
                "filings": {
                    "recent": {
                        "form": ["10-K", "8-K", "10-K"],
                        "filingDate": ["2025-11-01", "2025-10-15", "2024-11-02"],
                        "accessionNumber": [
                            "0000320193-25-000010",
                            "0000320193-25-000008",
                            "0000320193-24-000090",
                        ],
                        "primaryDocument": ["a10-k2025.htm", "a8-k2025.htm", "a10-k2024.htm"],
                        "primaryDocDescription": [
                            "Annual report",
                            "Current report",
                            "Annual report",
                        ],
                    }
                },
            },
        }
    )
    service = OfficialSecService(settings=build_settings(), session=session)

    filings = service.search_filings(
        company_name="Apple Inc.",
        ticker="AAPL",
        form_type="10-K",
        limit=2,
    )

    assert filings == [
        {
            "form_type": "10-K",
            "filed_at": "2025-11-01",
            "filing_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000010/a10-k2025.htm",
            "filing_details": "Annual report",
            "accession_no": "0000320193-25-000010",
        },
        {
            "form_type": "10-K",
            "filed_at": "2024-11-02",
            "filing_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000090/a10-k2024.htm",
            "filing_details": "Annual report",
            "accession_no": "0000320193-24-000090",
        },
    ]
    assert session.requests[1][0] == "https://data.sec.gov/submissions/CIK0000320193.json"
    assert session.requests[1][1]["headers"] == {
        "User-Agent": "multi-agent-investment-research analyst@example.com"
    }


def test_official_sec_service_fetches_company_facts_from_official_endpoint() -> None:
    company_facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [{"val": 1}]}}}}}
    session = RecordingSession(
        {
            "https://www.sec.gov/files/company_tickers.json": {
                "0": {"title": "Apple Inc.", "ticker": "AAPL", "cik_str": 320193}
            },
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json": company_facts,
        }
    )
    service = OfficialSecService(settings=build_settings(), session=session)

    result = service.fetch_company_facts("AAPL")

    assert result == company_facts
    assert session.requests[1][0] == "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
    assert session.requests[1][1]["headers"] == {
        "User-Agent": "multi-agent-investment-research analyst@example.com"
    }


def test_official_sec_service_returns_annual_filing_identity_with_html() -> None:
    filing_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000010/a10-k2025.htm"
    session = RecordingSession(
        {
            "https://www.sec.gov/files/company_tickers.json": {
                "0": {"title": "Apple Inc.", "ticker": "AAPL", "cik_str": 320193}
            },
            "https://data.sec.gov/submissions/CIK0000320193.json": {
                "filings": {
                    "recent": {
                        "form": ["10-K"],
                        "filingDate": ["2025-11-01"],
                        "accessionNumber": ["0000320193-25-000010"],
                        "primaryDocument": ["a10-k2025.htm"],
                        "primaryDocDescription": ["Annual report"],
                    }
                }
            },
            filing_url: "<html>annual filing</html>",
        }
    )
    service = OfficialSecService(settings=build_settings(), session=session)

    report = service.fetch_latest_annual_report("AAPL")

    assert report == {
        "html": "<html>annual filing</html>",
        "source_url": filing_url,
        "accession": "0000320193-25-000010",
        "filed_at": "2025-11-01",
        "form": "10-K",
    }


def test_official_sec_service_falls_back_to_cached_ticker_directory_when_request_fails(
    tmp_path: Path,
) -> None:
    ticker_url = "https://www.sec.gov/files/company_tickers.json"
    cached_payload = {
        "0": {"title": "Apple Inc.", "ticker": "AAPL", "cik_str": 320193},
    }
    settings = InvestmentResearchSettings(
        fast_model="qwen-plus",
        deep_model="qwen-plus",
        review_model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        tavily_api_key="tvly-key",
        sec_api_email="analyst@example.com",
        artifact_root=str(tmp_path / "artifacts"),
    )
    primed_service = OfficialSecService(
        settings=settings,
        session=RecordingSession({ticker_url: cached_payload}),
    )

    assert primed_service.lookup_company_by_ticker("AAPL") == {
        "title": "Apple Inc.",
        "name": "Apple Inc.",
        "ticker": "AAPL",
        "cik_str": 320193,
        "exchange": "",
    }

    fallback_service = OfficialSecService(
        settings=settings,
        session=RecordingSession(
            {
                ticker_url: requests.exceptions.SSLError("unexpected eof"),
            }
        ),
    )

    assert fallback_service.lookup_company_by_ticker("AAPL") == {
        "title": "Apple Inc.",
        "name": "Apple Inc.",
        "ticker": "AAPL",
        "cik_str": 320193,
        "exchange": "",
    }


def test_retry_session_fails_fast_on_connection_and_read_errors() -> None:
    session = _build_retry_session(3)

    adapter = session.get_adapter("https://")
    retry = adapter.max_retries

    assert retry.total == 3
    assert retry.connect == 0
    assert retry.read == 0


def test_official_sec_service_falls_back_to_stockanalysis_quote_when_nasdaq_times_out() -> None:
    session = RecordingSession(
        {
            "https://api.nasdaq.com/api/quote/AAPL/info?assetclass=stocks": requests.exceptions.ReadTimeout(
                "nasdaq timeout"
            ),
            "https://stockanalysis.com/stocks/aapl/": """
            <html>
              <body>
                <div class="mb-5 flex flex-row items-end space-x-2 xs:space-x-3 bp:space-x-5">
                  <div class="max-w-[50%]">
                    <div class="text-4xl font-bold transition-colors duration-300 block sm:inline">333.74</div>
                    <div class="font-semibold block text-lg xs:text-xl sm:inline sm:text-2xl text-green-vivid">+0.48 (0.14%)</div>
                    <div class="mt-0.5 text-xxs text-faded xs:text-tiny bp:text-sm">
                      <span class="block font-semibold sm:inline">At close:</span> Jul 17, 2026, 4:00 PM EDT
                    </div>
                  </div>
                </div>
              </body>
            </html>
            """,
        }
    )
    service = OfficialSecService(settings=build_settings(), session=session)

    payload = service.fetch_market_quote("AAPL")

    assert payload["source"] == "stockanalysis_quote_page"
    assert payload["data"]["primaryData"]["lastSalePrice"] == "$333.74"
    assert payload["data"]["primaryData"]["lastTradeTimestamp"] == "Jul 17, 2026, 4:00 PM EDT"
    assert session.requests[0][0] == "https://api.nasdaq.com/api/quote/AAPL/info?assetclass=stocks"
    assert session.requests[1][0] == "https://stockanalysis.com/stocks/aapl/"
