from __future__ import annotations

import html as html_module
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel, Field

from crewai.tools import BaseTool

from multi_agent.core.formal_gate import FORMAL_GATE_REQUIRED_FIELDS
from multi_agent.evaluation import record_financial_fields
from multi_agent.finance import CompanyFinancialSnapshot, compute_key_metrics
from multi_agent.settings import InvestmentResearchSettings
from multi_agent.tools.official_sec import (
    FatalAPIError,
    OfficialSecService,
    _debug_report,
    _raise_for_status_with_context,
)

__all__ = [
    "FatalAPIError",
    "_raise_for_status_with_context",
]

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency guard
    PdfReader = None


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


def build_fcf_snapshot(
    *,
    fy2025_fcf: float | None,
    fy2026e_fcf: float | None,
    fy2026e_source_type: str,
    source_refs: list[str] | None = None,
) -> dict[str, Any]:
    yoy_growth = None
    if fy2025_fcf not in (None, 0) and fy2026e_fcf is not None:
        yoy_growth = ((fy2026e_fcf - fy2025_fcf) / fy2025_fcf) * 100
    return {
        "fy2025_fcf": fy2025_fcf,
        "fy2026e_fcf": fy2026e_fcf,
        "fy2026e_yoy_growth": yoy_growth,
        "fy2026e_source_type": fy2026e_source_type,
        "source_refs": source_refs or [],
    }


def build_market_snapshot(
    *,
    stock_price: float | None,
    diluted_shares: float | None,
    as_of_date: str | None = None,
    source_refs: list[str] | None = None,
) -> dict[str, Any]:
    market_cap = None
    if stock_price is not None and diluted_shares is not None:
        market_cap = stock_price * diluted_shares
    return {
        "stock_price": stock_price,
        "diluted_shares": diluted_shares,
        "market_cap": market_cap,
        "as_of_date": as_of_date,
        "source_refs": source_refs or [],
        "ready_for_formal_report": market_cap is not None,
    }


def build_formal_gate_snapshot(
    financial_fields: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    missing_fields = [
        field_name
        for field_name in FORMAL_GATE_REQUIRED_FIELDS
        if not financial_fields.get(field_name, {}).get("extracted")
    ]
    return {
        "required_fields": list(FORMAL_GATE_REQUIRED_FIELDS),
        "missing_fields": missing_fields,
        "ready_for_formal_report": not missing_fields,
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
                    item.get("end") or "",
                    item.get("filed") or "",
                    item.get("fy") or 0,
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


def _extract_services_revenue_from_filing_html(filing_html: str) -> FinancialFieldExtraction:
    compact_html = re.sub(r"\s+", " ", filing_html)
    lowered_html = compact_html.lower()
    section_start = lowered_html.find("products and services performance")
    if section_start >= 0:
        section_end = lowered_html.find("geographic segments performance", section_start)
        compact_html = compact_html[
            section_start : section_end if section_end >= 0 else section_start + 40000
        ]
    section_text = html_module.unescape(re.sub(r"<[^>]+>", " ", compact_html))
    section_text = re.sub(r"\s+", " ", section_text)
    match = re.search(r"Services\s*(?:\(\d+\))?\s*([0-9][0-9,]{3,})\b", section_text, re.IGNORECASE)
    if match is None:
        return FinancialFieldExtraction(
            value=0.0,
            normalized_value=0.0,
            extracted=False,
            source_tag=None,
        )
    value_millions = float(match.group(1).replace(",", ""))
    normalized_value = value_millions * 1_000_000
    return FinancialFieldExtraction(
        value=normalized_value,
        normalized_value=normalized_value,
        extracted=True,
        source_tag="10k_products_services_table",
    )


def _extract_stock_price_from_quote_payload(
    quote_payload: dict[str, Any],
) -> tuple[float | None, str | None, str | None, str | None]:
    primary_data = quote_payload.get("data", {}).get("primaryData", {})
    raw_price = str(primary_data.get("lastSalePrice", "")).strip()
    if not raw_price:
        return None, None, None, None
    numeric_price = re.sub(r"[^0-9.]+", "", raw_price)
    if not numeric_price:
        return None, None, None, None
    try:
        stock_price = float(numeric_price)
    except ValueError:
        return None, None, None, None
    as_of_date = str(primary_data.get("lastTradeTimestamp", "")).strip() or None
    source_ref = str(quote_payload.get("source", "")).strip() or "nasdaq_quote_info"
    source_tag = (
        "stockanalysis_last_close_price"
        if source_ref == "stockanalysis_quote_page"
        else "nasdaq_last_sale_price"
    )
    return stock_price, as_of_date, source_ref, source_tag


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
    cash_and_equivalents = _extract_latest_fact(
        company_facts,
        ["CashAndCashEquivalentsAtCarryingValue"],
    )
    debt_current = _extract_latest_fact(company_facts, ["LongTermDebtCurrent"])
    debt_noncurrent = _extract_latest_fact(company_facts, ["LongTermDebtNoncurrent"])
    total_debt_value = 0.0
    total_debt_tags: list[str] = []
    for debt_component in (debt_current, debt_noncurrent):
        if debt_component.extracted:
            total_debt_value += debt_component.normalized_value
            if debt_component.source_tag:
                total_debt_tags.append(debt_component.source_tag)
    total_debt = FinancialFieldExtraction(
        value=total_debt_value,
        normalized_value=total_debt_value,
        extracted=bool(total_debt_tags),
        source_tag="+".join(total_debt_tags) if total_debt_tags else None,
    )
    diluted_shares = _extract_latest_fact(
        company_facts,
        [
            "EntityCommonStockSharesOutstanding",
            "CommonStockSharesOutstanding",
        ],
    )
    eps = _extract_latest_fact(company_facts, ["EarningsPerShareDiluted"])
    segment_revenue_services = _extract_latest_fact(
        company_facts,
        ["SalesRevenueServicesGross"],
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
        "cash_and_equivalents": cash_and_equivalents.as_dict(),
        "total_debt": total_debt.as_dict(),
        "diluted_shares": diluted_shares.as_dict(),
        "eps": eps.as_dict(),
        "segment_revenue_services": segment_revenue_services.as_dict(),
    }
    return snapshot, metadata


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

class SecFilingSearchInput(BaseModel):
    company_name: str = Field(..., description="Legal company name to search in SEC filings.")
    ticker: str = Field(..., description="Public ticker symbol, such as AAPL.")
    form_type: str = Field(default="10-K", description="SEC form type, such as 10-K or 10-Q.")
    limit: int = Field(default=3, description="Maximum number of filings to return.")


class SecFilingSearchTool(BaseTool):
    name: str = "SEC Filing Search"
    description: str = "Find the latest SEC filings for a public company using official SEC endpoints."
    args_schema: Type[BaseModel] = SecFilingSearchInput

    def __init__(
        self,
        settings: InvestmentResearchSettings | None = None,
        service: OfficialSecService | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._settings = settings or InvestmentResearchSettings.from_env()
        self._service = service or OfficialSecService(self._settings)

    def _run(self, company_name: str, ticker: str, form_type: str = "10-K", limit: int = 3) -> str:
        if self._settings.company_market_label and self._settings.company_market_label != "US":
            return (
                f"当前市场 {self._settings.company_market_label} 仅允许使用对应市场数据源，"
                "SEC Filing Search 仅 US 市场可用。"
            )
        # #region debug-point A:filing-tool-start
        _debug_report(
            "A",
            "investment_tools.py:SecFilingSearchTool._run:start",
            "[DEBUG] Filing tool start",
            {"company_name": company_name, "ticker": ticker, "form_type": form_type, "limit": limit},
        )
        # #endregion
        try:
            filings = self._service.search_filings(
                company_name=company_name,
                ticker=ticker,
                form_type=form_type,
                limit=limit,
            )
        except FatalAPIError:
            raise
        except Exception as exc:  # pragma: no cover - network failure path
            # #region debug-point C:filing-tool-exception
            _debug_report(
                "C",
                "investment_tools.py:SecFilingSearchTool._run:exception",
                "[DEBUG] Filing tool exception",
                {"ticker": ticker, "form_type": form_type, "error": repr(exc)},
            )
            # #endregion
            return f"SEC 文件检索失败：{exc}"

        # #region debug-point B:filing-tool-finish
        _debug_report(
            "B",
            "investment_tools.py:SecFilingSearchTool._run:finish",
            "[DEBUG] Filing tool finish",
            {"ticker": ticker, "form_type": form_type, "result_count": len(filings)},
        )
        # #endregion
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
        service: OfficialSecService | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._settings = settings or InvestmentResearchSettings.from_env()
        self._service = service or OfficialSecService(self._settings)

    def _run(self, ticker: str) -> str:
        if self._settings.company_market_label and self._settings.company_market_label != "US":
            return (
                f"当前市场 {self._settings.company_market_label} 仅允许使用对应市场数据源，"
                "SEC Company Facts 仅 US 市场可用。"
            )
        # #region debug-point D:company-facts-tool-start
        _debug_report(
            "D",
            "investment_tools.py:SecCompanyFactsTool._run:start",
            "[DEBUG] Company facts tool start",
            {"ticker": ticker},
        )
        # #endregion
        try:
            company_facts = self._service.fetch_company_facts(ticker)
            snapshot, metadata = _build_financial_snapshot_with_metadata(company_facts)
        except FatalAPIError:
            raise
        except Exception as exc:  # pragma: no cover - network failure path
            # #region debug-point E:company-facts-tool-exception
            _debug_report(
                "E",
                "investment_tools.py:SecCompanyFactsTool._run:exception",
                "[DEBUG] Company facts tool exception",
                {"ticker": ticker, "error": repr(exc)},
            )
            # #endregion
            return f"获取 SEC 公司财务事实失败：{exc}"

        record_financial_fields(metadata)
        # #region debug-point D:company-facts-tool-finish
        _debug_report(
            "D",
            "investment_tools.py:SecCompanyFactsTool._run:finish",
            "[DEBUG] Company facts tool finish",
            {"ticker": ticker, "fields": sorted(metadata.keys())},
        )
        # #endregion
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
        service: OfficialSecService | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._settings = settings or InvestmentResearchSettings.from_env()
        self._service = service or OfficialSecService(self._settings)

    def _run(self, ticker: str) -> str:
        if self._settings.company_market_label and self._settings.company_market_label != "US":
            return (
                f"当前市场 {self._settings.company_market_label} 仅允许使用对应市场数据源，"
                "Financial Metrics Calculator 仅 US 市场可用。"
            )
        try:
            company_facts = self._service.fetch_company_facts(ticker)
            snapshot, metadata = _build_financial_snapshot_with_metadata(company_facts)
            metrics = compute_key_metrics(snapshot)
        except FatalAPIError:
            raise
        except Exception as exc:  # pragma: no cover - network failure path
            return f"计算财务指标失败：{exc}"

        segment_snapshot = {
            "services_revenue": None,
            "source_refs": [],
        }
        fetch_filing_html = getattr(self._service, "fetch_latest_annual_report_html", None)
        if callable(fetch_filing_html):
            try:
                filing_html = fetch_filing_html(ticker)
                services_revenue = _extract_services_revenue_from_filing_html(filing_html)
                if services_revenue.extracted:
                    metadata["segment_revenue_services"] = services_revenue.as_dict()
                    segment_snapshot = {
                        "services_revenue": services_revenue.normalized_value,
                        "source_refs": ["10-K Products and Services Performance"],
                    }
            except Exception:
                pass

        diluted_shares = metadata.get("diluted_shares", {}).get("normalized_value")
        if metadata.get("segment_revenue_services", {}).get("extracted"):
            if segment_snapshot["services_revenue"] is None:
                segment_snapshot["services_revenue"] = metadata["segment_revenue_services"]["normalized_value"]
                segment_snapshot["source_refs"] = [metadata["segment_revenue_services"].get("source_tag") or "company_facts"]

        stock_price = None
        as_of_date = None
        stock_price_source_ref = None
        stock_price_source_tag = None
        fetch_market_quote = getattr(self._service, "fetch_market_quote", None)
        if callable(fetch_market_quote):
            try:
                quote_payload = fetch_market_quote(ticker)
                (
                    stock_price,
                    as_of_date,
                    stock_price_source_ref,
                    stock_price_source_tag,
                ) = _extract_stock_price_from_quote_payload(quote_payload)
            except Exception:
                stock_price = None
                as_of_date = None
                stock_price_source_ref = None
                stock_price_source_tag = None
        metadata["stock_price"] = FinancialFieldExtraction(
            value=stock_price or 0.0,
            normalized_value=stock_price or 0.0,
            extracted=stock_price is not None,
            source_tag=stock_price_source_tag if stock_price is not None else None,
        ).as_dict()
        market_snapshot = build_market_snapshot(
            stock_price=stock_price,
            diluted_shares=float(diluted_shares) if diluted_shares not in (None, "") else None,
            as_of_date=as_of_date,
            source_refs=[stock_price_source_ref] if stock_price is not None and stock_price_source_ref else [],
        )
        formal_gate_snapshot = build_formal_gate_snapshot(metadata)

        record_financial_fields(metadata)
        return json.dumps(
            {
                "metrics": metrics,
                "financial_fields": metadata,
                "market_snapshot": market_snapshot,
                "segment_snapshot": segment_snapshot,
                "formal_gate_snapshot": formal_gate_snapshot,
            },
            indent=2,
            ensure_ascii=False,
        )
