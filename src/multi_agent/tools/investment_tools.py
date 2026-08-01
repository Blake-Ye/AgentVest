from __future__ import annotations

import html as html_module
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Type
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from crewai.tools import BaseTool

from multi_agent.core.formal_gate import FORMAL_GATE_REQUIRED_FIELDS
from multi_agent.core.evidence import (
    EvidenceGap,
    EvidenceNormalizer,
    EventEvidence,
    FinancialFact,
    MarketSnapshotEvidence,
    ResearchEvidenceBundle,
    ToolHealthRecord,
    periods_are_compatible,
)
from multi_agent.evaluation import (
    record_financial_fields,
    record_research_evidence,
    recorded_tavily_payloads,
)
from multi_agent.finance import CompanyFinancialSnapshot, compute_key_metrics
from multi_agent.settings import InvestmentResearchSettings
from multi_agent.tools.official_sec import (
    FatalAPIError,
    OfficialSecService,
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
    fact: FinancialFact | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "value": self.value,
            "normalized_value": self.normalized_value,
            "extracted": self.extracted,
            "source_tag": self.source_tag,
        }
        if self.fact is not None:
            payload["fact"] = self.fact.model_dump(mode="json")
        return payload


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


def _formal_gate_snapshot_from_evidence(bundle: ResearchEvidenceBundle) -> dict[str, Any]:
    formal_field_names = {fact.field_name for fact in bundle.formal_facts()}
    evidence_fields = {
        field_name: {"extracted": field_name in formal_field_names}
        for field_name in FORMAL_GATE_REQUIRED_FIELDS
    }
    evidence_fields["stock_price"] = {"extracted": bool(bundle.market_snapshots)}
    return build_formal_gate_snapshot(evidence_fields)


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

            comparable_entries = []
            for entry in entries:
                try:
                    value = float(entry.get("val"))
                except (TypeError, ValueError):
                    continue
                if isfinite(value):
                    comparable_entries.append(entry)
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
            accession = str(latest_entry.get("accn", "")).strip() or None
            cik = str(company_facts.get("cik", "")).strip()
            source_url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
                if cik.isdigit() and accession
                else None
            )
            return FinancialFieldExtraction(
                value=value,
                normalized_value=value,
                extracted=True,
                source_tag=tag,
                fact=FinancialFact(
                    field_name=tag,
                    value=value,
                    unit=unit_name,
                    period_start=_optional_date(latest_entry.get("start")),
                    period_end=_optional_date(latest_entry.get("end")),
                    fiscal_year=_optional_int(latest_entry.get("fy")),
                    fiscal_period=str(latest_entry.get("fp", "")).strip() or None,
                    form=str(latest_entry.get("form", "")).strip() or None,
                    accession=accession,
                    filed_at=_optional_date(latest_entry.get("filed")),
                    source_url=source_url,
                    source_tag="sec_companyfacts",
                    taxonomy_concept=tag,
                ),
            )
    return FinancialFieldExtraction(
        value=0.0,
        normalized_value=0.0,
        extracted=False,
        source_tag=None,
    )


def _optional_date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None and not isinstance(value, bool) else None
    except (TypeError, ValueError):
        return None


def _extract_services_revenue_from_filing_html(
    filing_html: str,
    *,
    form: str | None = None,
    period_start: date | None = None,
    period_end: date | None = None,
) -> FinancialFieldExtraction:
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
    match = re.search(
        r"Services\s*(?:\(\d+\))?\s*(?=[0-9])",
        section_text,
        re.IGNORECASE,
    )
    if match is None:
        return FinancialFieldExtraction(
            value=0.0,
            normalized_value=0.0,
            extracted=False,
            source_tag=None,
        )
    values = re.findall(r"\b[0-9][0-9,]{3,}\b", section_text[match.end() : match.end() + 300])
    if not values:
        return FinancialFieldExtraction(
            value=0.0,
            normalized_value=0.0,
            extracted=False,
            source_tag=None,
        )
    use_ytd_column = (
        form == "10-Q"
        and period_start is not None
        and period_end is not None
        and (period_end - period_start).days >= 120
        and len(values) >= 3
    )
    # ponytail: Apple 10-Q tables list current quarter before YTD; parse XBRL contexts if layouts vary.
    selected = values[2] if use_ytd_column else values[0]
    value_millions = float(selected.replace(",", ""))
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


def _quote_source_url(ticker: str, quote_payload: dict[str, Any]) -> str:
    explicit_url = str(quote_payload.get("source_url", "")).strip()
    if explicit_url:
        return explicit_url
    if str(quote_payload.get("source", "")).strip() == "stockanalysis_quote_page":
        return f"https://stockanalysis.com/stocks/{ticker.lower()}/"
    return f"https://api.nasdaq.com/api/quote/{ticker.upper()}/info?assetclass=stocks"


def _parse_observed_at(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        observed_at = datetime.fromisoformat(normalized)
    except ValueError:
        for pattern in ("%b %d, %Y", "%B %d, %Y"):
            try:
                observed_at = datetime.strptime(normalized.split(",", 2)[0] + ", " + normalized.split(",", 2)[1].strip(), pattern)
                break
            except (ValueError, IndexError):
                continue
        else:
            return None
    return observed_at if observed_at.tzinfo is not None else observed_at.replace(tzinfo=timezone.utc)


def _event_timestamp(value: object) -> datetime | None:
    return _parse_observed_at(str(value).strip()) if value else None


def _evidence_gap(
    *,
    code: str,
    target: str,
    message: str,
    fields: list[str] | None = None,
    sources: list[str] | None = None,
) -> EvidenceGap:
    return EvidenceGap(
        code=code,
        target=target,  # type: ignore[arg-type]
        fields=fields or [],
        sources=sources or [],
        message=message,
    )


def _add_total_debt(bundle: ResearchEvidenceBundle) -> None:
    components = [
        fact
        for fact in bundle.financial_facts
        if fact.field_name in {"debt_current", "debt_noncurrent"}
    ]
    if len(components) != 2:
        bundle.gaps.append(
            _evidence_gap(
                code="total_debt_incomplete",
                target="fundamental_analyst",
                fields=["total_debt"],
                message="Both current and non-current debt are required to construct total debt.",
            )
        )
        return
    if not periods_are_compatible(components[0], components[1]):
        bundle.gaps.append(
            _evidence_gap(
                code="total_debt_period_mismatch",
                target="fundamental_analyst",
                fields=["total_debt"],
                sources=[fact.source_url for fact in components if fact.source_url],
                message="Current and non-current debt facts do not share a compatible fiscal period.",
            )
        )
        return
    representative = components[0]
    bundle.financial_facts.append(
        representative.model_copy(
            update={
                "field_name": "total_debt",
                "value": sum(fact.value for fact in components),
                "source_tag": "sec_companyfacts_total_debt",
                "taxonomy_concept": "+".join(
                    fact.taxonomy_concept or "" for fact in components
                ),
            }
        )
    )


def _add_services_revenue_from_filing(
    bundle: ResearchEvidenceBundle,
    filing_html: str,
    filing_metadata: dict[str, object] | None,
) -> None:
    existing = [
        fact
        for fact in bundle.financial_facts
        if fact.field_name == "segment_revenue_services"
    ]
    revenue = next(
        (fact for fact in bundle.financial_facts if fact.field_name == "revenue"),
        None,
    )
    if existing and revenue is not None and any(
        fact.formal_eligible and periods_are_compatible(revenue, fact) for fact in existing
    ):
        return
    if existing:
        bundle.financial_facts = [
            fact
            for fact in bundle.financial_facts
            if fact.field_name != "segment_revenue_services"
        ]
        bundle.gaps.append(
            _evidence_gap(
                code="stale_services_company_fact",
                target="fundamental_analyst",
                fields=["segment_revenue_services"],
                sources=[fact.source_url for fact in existing if fact.source_url],
                message="Company Facts Services value is not compatible with the current revenue period; filing extraction is required.",
            )
        )
    metadata = dict(filing_metadata or {})
    _complete_filing_period_from_revenue(bundle, metadata)
    extracted = _extract_services_revenue_from_filing_html(
        filing_html,
        form=str(metadata.get("form", "")).strip() or None,
        period_start=_optional_date(metadata.get("period_start")),
        period_end=_optional_date(metadata.get("period_end")),
    )
    if not extracted.extracted:
        bundle.tool_health.append(
            ToolHealthRecord(
                tool_name="sec_filing_html",
                status="degraded",
                message="Annual filing HTML did not yield Services revenue with reusable provenance.",
            )
        )
        bundle.gaps.append(
            _evidence_gap(
                code="services_revenue_unavailable",
                target="fundamental_analyst",
                fields=["segment_revenue_services"],
                message="Services revenue could not be extracted from the annual filing HTML.",
            )
        )
        return
    services_fact = FinancialFact(
        field_name="segment_revenue_services",
        value=extracted.normalized_value,
        unit=str(metadata.get("unit", "USD")).strip() or "USD",
        period_start=_optional_date(metadata.get("period_start")),
        period_end=_optional_date(metadata.get("period_end")),
        fiscal_year=_optional_int(metadata.get("fiscal_year")),
        fiscal_period=str(metadata.get("fiscal_period", "")).strip() or None,
        form=str(metadata.get("form", "")).strip() or None,
        accession=str(metadata.get("accession", "")).strip() or None,
        filed_at=_optional_date(metadata.get("filed_at")),
        source_url=str(metadata.get("source_url", "")).strip() or None,
        source_tag=extracted.source_tag or "sec_filing_html",
        taxonomy_concept="ProductsAndServicesPerformance.Services",
    )
    bundle.financial_facts.append(services_fact)
    if not services_fact.formal_eligible:
        bundle.gaps.append(
            _evidence_gap(
                code="services_revenue_provenance_incomplete",
                target="fundamental_analyst",
                fields=["segment_revenue_services"],
                sources=[services_fact.source_url] if services_fact.source_url else [],
                message="Services revenue is auditable but lacks filing identity or period metadata required for formal delivery.",
            )
        )
    bundle.tool_health.append(ToolHealthRecord(tool_name="sec_filing_html", status="healthy"))


def _complete_filing_period_from_revenue(
    bundle: ResearchEvidenceBundle,
    filing_metadata: dict[str, object],
) -> None:
    """Fill a filing's period only when its own identity matches a formal SEC revenue fact."""
    accession = str(filing_metadata.get("accession", "")).strip()
    source_url = str(filing_metadata.get("source_url", "")).strip()
    report_date = _optional_date(filing_metadata.get("report_date"))
    if not accession or not source_url:
        return
    revenue = next(
        (
            fact
            for fact in bundle.financial_facts
            if fact.field_name == "revenue"
            and fact.formal_eligible
            and fact.accession == accession
            and fact.form == str(filing_metadata.get("form", "10-K")).strip()
        ),
        None,
    )
    if revenue is None:
        return
    if report_date is not None and revenue.period_end != report_date:
        bundle.gaps.append(
            _evidence_gap(
                code="filing_report_date_mismatch",
                target="fundamental_analyst",
                fields=["segment_revenue_services"],
                sources=[source_url, revenue.source_url] if revenue.source_url else [source_url],
                message="The annual filing report date does not match the same-accession formal revenue period.",
            )
        )
        return
    filing_metadata.setdefault("fiscal_year", revenue.fiscal_year)
    filing_metadata.setdefault("fiscal_period", revenue.fiscal_period)
    filing_metadata.setdefault("period_start", revenue.period_start.isoformat() if revenue.period_start else None)
    filing_metadata.setdefault("period_end", revenue.period_end.isoformat() if revenue.period_end else None)


def _add_quote_evidence(
    bundle: ResearchEvidenceBundle,
    ticker: str,
    quote_payload: dict[str, Any],
) -> None:
    price, observed_text, _source_ref, source_tag = _extract_stock_price_from_quote_payload(quote_payload)
    observed_at = _parse_observed_at(observed_text)
    if quote_payload.get("status") == "degraded" or price is None or observed_at is None:
        bundle.tool_health.append(
            ToolHealthRecord(
                tool_name="quote",
                status="degraded",
                message=str(quote_payload.get("degraded_reason", "Market quote is unavailable or undated.")),
            )
        )
        bundle.gaps.append(
            _evidence_gap(
                code="market_quote_unavailable",
                target="market_validation_analyst",
                fields=["stock_price"],
                message="Market quote is unavailable, invalid, or has no observable timestamp.",
            )
        )
        return
    diluted_shares = next(
        (
            fact
            for fact in bundle.financial_facts
            if fact.field_name == "diluted_shares" and fact.formal_eligible
        ),
        None,
    )
    bundle.market_snapshots.append(
        MarketSnapshotEvidence(
            price=price,
            currency=str(quote_payload.get("currency", "USD")),
            observed_at=observed_at,
            source_url=_quote_source_url(ticker, quote_payload),
            source_tag=source_tag or "market_quote",
            diluted_shares_period_end=(
                diluted_shares.period_end if diluted_shares is not None else None
            ),
        )
    )
    bundle.tool_health.append(ToolHealthRecord(tool_name="quote", status="healthy"))


def _add_tavily_events(
    bundle: ResearchEvidenceBundle,
    tavily_payloads: list[dict[str, object]],
) -> None:
    if not tavily_payloads:
        bundle.tool_health.append(
            ToolHealthRecord(
                tool_name="tavily",
                status="degraded",
                message="No Tavily payload was recorded for this run.",
            )
        )
        bundle.gaps.append(
            _evidence_gap(
                code="independent_event_sources_insufficient",
                target="event_guidance_analyst",
                fields=["events"],
                message="No Tavily evidence was recorded for independent event confirmation.",
            )
        )
        return
    degraded = False
    event_count = 0
    degraded_reasons: list[str] = []
    event_group_indexes: dict[str, list[int]] = {}
    event_group_domains: dict[str, set[str]] = {}
    event_group_urls: dict[str, list[str]] = {}
    for payload_index, payload in enumerate(tavily_payloads):
        status = str(payload.get("status", "ok")).lower()
        query = str(payload.get("query", "")).strip()
        results = payload.get("results", [])
        if status != "ok" or not isinstance(results, list):
            degraded = True
            reason = str(payload.get("degraded_reason", "")).strip()
            if reason:
                degraded_reasons.append(reason)
            continue
        artifact_ref = str(payload.get("artifact_ref", "")).strip()
        if _is_valid_http_url(artifact_ref):
            bundle.raw_artifact_refs.append(artifact_ref)
        for result_index, raw_result in enumerate(results):
            if not isinstance(raw_result, dict):
                degraded = True
                continue
            url = str(raw_result.get("url", "")).strip()
            title = str(raw_result.get("title", "")).strip()
            if not title or not _is_valid_http_url(url):
                degraded = True
                degraded_reasons.append("Tavily returned an invalid event source URL or title.")
                continue
            event_count += 1
            corroboration_key = _event_corroboration_key(raw_result, title, query=query)
            publisher_domain = _registrable_publisher_domain(url)
            bundle.raw_artifact_refs.append(url)
            event_group_indexes.setdefault(corroboration_key, []).append(len(bundle.events))
            event_group_domains.setdefault(corroboration_key, set()).add(publisher_domain)
            event_group_urls.setdefault(corroboration_key, []).append(url)
            bundle.events.append(
                EventEvidence(
                    event_id=f"tavily-{payload_index}-{result_index}",
                    title=title,
                    published_at=_event_timestamp(raw_result.get("published_at")),
                    source_url=url,
                    source_type=str(raw_result.get("source_type", "news")) or "news",
                    confidence=0.6,
                    corroboration_key=corroboration_key,
                    corroborating_source_urls=[url],
                )
            )
    unconfirmed_groups = {
        key for key, domains in event_group_domains.items() if len(domains) < 2
    }
    for corroboration_key, indexes in event_group_indexes.items():
        group_sources = list(dict.fromkeys(event_group_urls[corroboration_key]))
        confirmed = corroboration_key not in unconfirmed_groups
        for index in indexes:
            bundle.events[index] = bundle.events[index].model_copy(
                update={
                    "independently_confirmed": confirmed,
                    "corroborating_source_urls": group_sources,
                }
            )
    if degraded or unconfirmed_groups:
        message = "; ".join(dict.fromkeys(degraded_reasons))
        bundle.tool_health.append(
            ToolHealthRecord(tool_name="tavily", status="degraded", message=message)
        )
    else:
        bundle.tool_health.append(ToolHealthRecord(tool_name="tavily", status="healthy"))
    if degraded or unconfirmed_groups or event_count == 0:
        bundle.gaps.append(
            _evidence_gap(
                code="independent_event_sources_insufficient",
                target="event_guidance_analyst",
                fields=["events"],
                message="Tavily did not return sufficient healthy, independently sourced event evidence.",
            )
        )


def _is_valid_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _event_corroboration_key(
    raw_result: dict[str, object], title: str, *, query: str = ""
) -> str:
    explicit_key = str(
        raw_result.get("event_key", raw_result.get("corroboration_key", ""))
    ).strip()
    if explicit_key:
        return explicit_key.lower()
    normalized_query = re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()
    if normalized_query:
        return f"query:{normalized_query}"
    normalized_title = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    return normalized_title or "unclassified-event"


def _registrable_publisher_domain(url: str) -> str:
    host = (urlparse(url).hostname or "").lower().strip(".")
    labels = [label for label in host.split(".") if label]
    if len(labels) < 2:
        return host
    country_second_level = {"ac", "co", "com", "edu", "gov", "net", "org"}
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in country_second_level:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def build_research_evidence_bundle(
    *,
    company_name: str,
    ticker: str,
    company_facts: dict[str, object],
    filing_html: str,
    quote_payload: dict[str, object],
    tavily_payloads: list[dict[str, object]],
    filing_metadata: dict[str, object] | None = None,
) -> ResearchEvidenceBundle:
    """Normalize SEC, filing, quote, and news payloads into one auditable bundle."""
    bundle = EvidenceNormalizer().normalize_company_facts(company_name, ticker, company_facts)
    if bundle.formal_facts():
        bundle.tool_health.append(ToolHealthRecord(tool_name="sec_company_facts", status="healthy"))
    else:
        bundle.tool_health.append(
            ToolHealthRecord(
                tool_name="sec_company_facts",
                status="degraded",
                message="SEC Company Facts did not yield a formal-eligible financial claim.",
            )
        )
        bundle.gaps.append(
            _evidence_gap(
                code="sec_companyfacts_no_formal_facts",
                target="fundamental_analyst",
                message="SEC Company Facts contains no formal-eligible financial facts.",
            )
        )
    _add_total_debt(bundle)
    _add_services_revenue_from_filing(bundle, filing_html, filing_metadata)
    _add_quote_evidence(bundle, ticker, quote_payload)
    _add_tavily_events(bundle, tavily_payloads)
    return bundle


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
    shares_outstanding = _extract_latest_fact(
        company_facts,
        [
            "EntityCommonStockSharesOutstanding",
            "CommonStockSharesOutstanding",
        ],
    )
    diluted_shares = _extract_latest_fact(
        company_facts,
        [
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "WeightedAverageNumberOfSharesOutstandingDiluted",
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
        "shares_outstanding": shares_outstanding.as_dict(),
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


class SecFilingContentInput(BaseModel):
    filing_url: str = Field(..., description="Official SEC EDGAR filing document URL.")
    max_chars: int = Field(
        default=12_000,
        ge=100,
        le=20_000,
        description="Maximum number of cleaned filing-text characters to return.",
    )


class SecFilingContentTool(BaseTool):
    name: str = "SEC Filing Content"
    description: str = "Fetch and clean a specific official SEC EDGAR filing document."
    args_schema: Type[BaseModel] = SecFilingContentInput

    def __init__(
        self,
        settings: InvestmentResearchSettings | None = None,
        service: OfficialSecService | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._settings = settings or InvestmentResearchSettings.from_env()
        self._service = service or OfficialSecService(self._settings)

    def _run(self, filing_url: str, max_chars: int = 12_000) -> str:
        max_chars = min(max_chars, 20_000)
        if self._settings.company_market_label and self._settings.company_market_label != "US":
            return json.dumps(
                {"status": "unavailable", "reason": "SEC filings are US-market only."}
            )
        try:
            raw_html = self._service.fetch_filing_html(filing_url)
        except FatalAPIError:
            raise
        except Exception as exc:
            return json.dumps(
                {"status": "failed", "source_url": filing_url, "reason": str(exc)},
                ensure_ascii=False,
            )
        clean_html = re.sub(
            r"<(?:script|style|ix:header)\b[^>]*>[\s\S]*?</(?:script|style|ix:header)>",
            " ",
            raw_html,
            flags=re.IGNORECASE,
        )
        clean_html = re.sub(r"<!--[\s\S]*?-->", " ", clean_html)
        text = html_module.unescape(re.sub(r"<[^>]+>", " ", clean_html))
        text = re.sub(r"\s+", " ", text).strip()
        return json.dumps(
            {
                "status": "ok",
                "source_url": filing_url,
                "text": text[:max_chars],
                "truncated": len(text) > max_chars,
            },
            ensure_ascii=False,
        )


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
        try:
            company_facts = self._service.fetch_company_facts(ticker)
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
        filing_html = ""
        filing_metadata: dict[str, object] | None = None
        fetch_financial_report = getattr(
            self._service, "fetch_latest_financial_report", None
        )
        fetch_annual_report = getattr(self._service, "fetch_latest_annual_report", None)
        fetch_filing_html = getattr(self._service, "fetch_latest_annual_report_html", None)
        fetch_report = (
            fetch_financial_report
            if callable(fetch_financial_report)
            else fetch_annual_report
        )
        if callable(fetch_report):
            try:
                financial_report = fetch_report(ticker)
                filing_html = str(financial_report.get("html", ""))
                filing_metadata = {
                    key: value
                    for key, value in financial_report.items()
                    if key != "html"
                }
            except FatalAPIError:
                raise
            except Exception:
                filing_html = ""
                filing_metadata = None
        elif callable(fetch_filing_html):
            try:
                filing_html = fetch_filing_html(ticker)
            except FatalAPIError:
                raise
            except Exception:
                filing_html = ""

        diluted_shares = metadata.get("diluted_shares", {}).get("normalized_value")
        if metadata.get("segment_revenue_services", {}).get("extracted"):
            if segment_snapshot["services_revenue"] is None:
                segment_snapshot["services_revenue"] = metadata["segment_revenue_services"]["normalized_value"]
                segment_snapshot["source_refs"] = [metadata["segment_revenue_services"].get("source_tag") or "company_facts"]

        stock_price = None
        as_of_date = None
        stock_price_source_ref = None
        stock_price_source_tag = None
        quote_payload: dict[str, Any] = {"status": "degraded", "degraded_reason": "Quote service unavailable."}
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
            except FatalAPIError:
                raise
            except Exception:
                stock_price = None
                as_of_date = None
                stock_price_source_ref = None
                stock_price_source_tag = None
                quote_payload = {"status": "degraded", "degraded_reason": "Quote service request failed."}
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
        company_name = str(company_facts.get("entityName", "")).strip() or ticker.upper()
        evidence_bundle = build_research_evidence_bundle(
            company_name=company_name,
            ticker=ticker.upper(),
            company_facts=company_facts,
            filing_html=filing_html,
            quote_payload=quote_payload,
            tavily_payloads=recorded_tavily_payloads(),
            filing_metadata=filing_metadata,
        )
        services_fact = next(
            (
                fact
                for fact in evidence_bundle.financial_facts
                if fact.field_name == "segment_revenue_services"
            ),
            None,
        )
        if services_fact is not None:
            metadata["segment_revenue_services"] = FinancialFieldExtraction(
                value=services_fact.value,
                normalized_value=services_fact.value,
                extracted=True,
                source_tag=services_fact.source_tag,
            ).as_dict()
            segment_snapshot = {
                "services_revenue": services_fact.value,
                "source_refs": [services_fact.source_url or services_fact.source_tag or "SEC filing"],
            }
        formal_gate_snapshot = _formal_gate_snapshot_from_evidence(evidence_bundle)
        market_snapshot["ready_for_formal_report"] = formal_gate_snapshot[
            "ready_for_formal_report"
        ]

        record_financial_fields(metadata)
        record_research_evidence(evidence_bundle.model_dump(mode="json"))
        return json.dumps(
            {
                "metrics": metrics,
                "financial_fields": metadata,
                "market_snapshot": market_snapshot,
                "segment_snapshot": segment_snapshot,
                "formal_gate_snapshot": formal_gate_snapshot,
                "evidence_bundle": evidence_bundle.model_dump(mode="json"),
            },
            indent=2,
            ensure_ascii=False,
        )
