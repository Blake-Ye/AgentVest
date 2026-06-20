from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum

from multi_agent.settings import InvestmentResearchSettings


class MarketScope(str, Enum):
    US_SEC = "us_sec"
    HKEX = "hkex"
    CN_A_SHARE = "cn_a_share"
    EU_LISTED = "eu_listed"
    UNKNOWN = "unknown"


_EU_SUFFIX_EXCHANGE = {
    "AS": "Euronext Amsterdam",
    "BR": "Euronext Brussels",
    "DE": "Xetra",
    "L": "London Stock Exchange",
    "MC": "Bolsas y Mercados Espanoles",
    "MI": "Borsa Italiana",
    "PA": "Euronext Paris",
    "SW": "SIX Swiss Exchange",
}


@dataclass(frozen=True)
class IssuerProfile:
    company_name: str
    input_ticker: str
    canonical_ticker: str
    market_scope: str
    exchange: str
    country_hint: str
    sec_applicable: bool

    def as_dict(self) -> dict[str, str | bool]:
        return asdict(self)


class MarketIdentifierService:
    """规则优先的市场识别器，后续可在此处接入 flash 作为兜底判断。"""

    def __init__(self, settings: InvestmentResearchSettings) -> None:
        self.settings = settings

    def identify(self, company_name: str, ticker: str) -> IssuerProfile:
        normalized_ticker = ticker.strip().upper()

        if re.fullmatch(r"\d{4}\.HK", normalized_ticker):
            return IssuerProfile(
                company_name=company_name,
                input_ticker=ticker,
                canonical_ticker=normalized_ticker,
                market_scope=MarketScope.HKEX.value,
                exchange="Hong Kong Stock Exchange",
                country_hint="HK",
                sec_applicable=False,
            )

        if re.fullmatch(r"\d{6}\.(SH|SZ)", normalized_ticker):
            exchange = "Shanghai Stock Exchange" if normalized_ticker.endswith(".SH") else "Shenzhen Stock Exchange"
            return IssuerProfile(
                company_name=company_name,
                input_ticker=ticker,
                canonical_ticker=normalized_ticker,
                market_scope=MarketScope.CN_A_SHARE.value,
                exchange=exchange,
                country_hint="CN",
                sec_applicable=False,
            )

        eu_match = re.fullmatch(r"[A-Z0-9-]+\.(AS|BR|DE|L|MC|MI|PA|SW)", normalized_ticker)
        if eu_match:
            suffix = eu_match.group(1)
            return IssuerProfile(
                company_name=company_name,
                input_ticker=ticker,
                canonical_ticker=normalized_ticker,
                market_scope=MarketScope.EU_LISTED.value,
                exchange=_EU_SUFFIX_EXCHANGE.get(suffix, "European Exchange"),
                country_hint="EU",
                sec_applicable=False,
            )

        if re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", normalized_ticker) and "." not in normalized_ticker:
            return IssuerProfile(
                company_name=company_name,
                input_ticker=ticker,
                canonical_ticker=normalized_ticker,
                market_scope=MarketScope.US_SEC.value,
                exchange="US Listed / SEC Universe",
                country_hint="US",
                sec_applicable=True,
            )

        return IssuerProfile(
            company_name=company_name,
            input_ticker=ticker,
            canonical_ticker=normalized_ticker,
            market_scope=MarketScope.UNKNOWN.value,
            exchange="Unknown",
            country_hint="",
            sec_applicable=False,
        )
