from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel, Field

from crewai.tools import BaseTool

from multi_agent.core.market import MarketValidationResult, build_tool_policy
from multi_agent.settings import InvestmentResearchSettings
from multi_agent.tools.official_sec import OfficialSecService


_US_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "NYSE ARCA", "NASDAQGS", "NASDAQGM", "NASDAQCM"}
_HK_EXCHANGES = {"HKEX", "SEHK"}
_EU_EXCHANGES = {"LSE", "XETRA", "FWB", "EURONEXT", "EPA", "AMS", "SWX", "SIX"}
_EU_SUFFIXES = (".L", ".DE", ".PA", ".AS", ".SW", ".BR", ".MI", ".MC", ".ST", ".HE", ".CO")


class MarketValidationInput(BaseModel):
    company_name: str = Field(..., description="Company name to validate.")
    ticker: str = Field(default="", description="Ticker symbol if available.")
    exchange: str = Field(default="", description="Exchange code if available.")


class MarketValidationTool(BaseTool):
    name: str = "Market Validation"
    description: str = "Infer company market label and allowed tool policy before analysis."
    args_schema: Type[BaseModel] = MarketValidationInput

    def __init__(
        self,
        settings: InvestmentResearchSettings | None = None,
        service: "MarketValidationService" | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._settings = settings or InvestmentResearchSettings.from_env()
        self._service = service or MarketValidationService(settings=self._settings)

    def _run(self, company_name: str, ticker: str = "", exchange: str = "") -> dict[str, object]:
        result = self._service.validate(company_name=company_name, ticker=ticker, exchange=exchange)
        return result.as_dict()


class MarketValidationService:
    def __init__(
        self,
        settings: InvestmentResearchSettings,
        official_sec_service: OfficialSecService | None = None,
    ) -> None:
        self.settings = settings
        self.official_sec_service = official_sec_service or OfficialSecService(settings)

    def validate(
        self,
        *,
        company_name: str,
        ticker: str = "",
        exchange: str = "",
    ) -> MarketValidationResult:
        del company_name
        normalized_ticker = ticker.strip().upper()
        normalized_exchange = exchange.strip().upper()

        if normalized_exchange in _US_EXCHANGES:
            return self._result("US", 0.98, "confirmed", [f"exchange={normalized_exchange}"])
        if normalized_exchange in _HK_EXCHANGES:
            return self._result("HK", 0.98, "confirmed", [f"exchange={normalized_exchange}"])
        if normalized_exchange in _EU_EXCHANGES:
            return self._result("EU", 0.95, "confirmed", [f"exchange={normalized_exchange}"])

        if normalized_ticker.endswith(".HK"):
            return self._result("HK", 0.96, "confirmed", [f"ticker={normalized_ticker}", "suffix=.HK"])
        if normalized_ticker.endswith(_EU_SUFFIXES):
            return self._result("EU", 0.9, "confirmed", [f"ticker={normalized_ticker}", "suffix=EU"])

        if normalized_ticker:
            official_match = self.official_sec_service.lookup_company_by_ticker(normalized_ticker)
            if official_match is not None:
                return self._result(
                    "US",
                    0.92,
                    "confirmed",
                    [f"ticker={normalized_ticker}", "official_sec_match=true"],
                )

        return self._result("UNRESOLVED", 0.2, "unresolved", ["no_market_signal"])

    def _result(
        self,
        market_label: str,
        confidence: float,
        resolution_status: str,
        evidence: list[str],
    ) -> MarketValidationResult:
        return MarketValidationResult(
            market_label=market_label,  # type: ignore[arg-type]
            confidence=confidence,
            resolution_status=resolution_status,  # type: ignore[arg-type]
            evidence=evidence,
            requires_human_confirmation=market_label == "UNRESOLVED",
            tool_policy=build_tool_policy(market_label),  # type: ignore[arg-type]
        )
