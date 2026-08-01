from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FinancialFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_name: str
    value: float = Field(allow_inf_nan=False)
    unit: str
    period_start: date | None = None
    period_end: date | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    form: str | None = None
    accession: str | None = None
    filed_at: date | None = None
    source_url: str | None = None
    source_tag: str
    taxonomy_concept: str | None = None
    quality_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def derive_quality_flags(self) -> "FinancialFact":
        required = (
            "fiscal_year",
            "fiscal_period",
            "period_end",
            "form",
            "accession",
            "filed_at",
            "source_url",
        )
        self.quality_flags = sorted(
            set(self.quality_flags)
            | {f"{name}_missing" for name in required if getattr(self, name) in (None, "")}
        )
        return self

    @property
    def formal_eligible(self) -> bool:
        return not self.quality_flags and bool(self.unit and self.source_tag)


EvidenceTarget = Literal[
    "market_validation_analyst",
    "event_guidance_analyst",
    "fundamental_analyst",
    "quant_valuation_analyst",
    "report_writing_analyst",
    "data_quality_reviewer",
    "logic_compliance_reviewer",
]


class MarketSnapshotEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    price: float
    currency: str
    observed_at: datetime
    source_url: str
    source_tag: str
    diluted_shares_period_end: date | None = None


class EventEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    title: str
    occurred_at: datetime | None = None
    published_at: datetime | None = None
    source_url: str
    source_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    independently_confirmed: bool = False
    corroboration_key: str = ""
    corroborating_source_urls: list[str] = Field(default_factory=list)


class ToolHealthRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    status: Literal["healthy", "degraded", "failed"]
    error_type: str = ""
    service_name: str = ""
    http_status: int | None = None
    message: str = ""


class EvidenceGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    target: EvidenceTarget
    fields: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    message: str


class ResearchEvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_name: str
    ticker: str
    market_label: str = "US"
    financial_facts: list[FinancialFact] = Field(default_factory=list)
    market_snapshots: list[MarketSnapshotEvidence] = Field(default_factory=list)
    events: list[EventEvidence] = Field(default_factory=list)
    tool_health: list[ToolHealthRecord] = Field(default_factory=list)
    gaps: list[EvidenceGap] = Field(default_factory=list)
    raw_artifact_refs: list[str] = Field(default_factory=list)

    def require_fact(self, field_name: str) -> FinancialFact:
        matches = [fact for fact in self.financial_facts if fact.field_name == field_name]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one fact for {field_name}, got {len(matches)}")
        return matches[0]

    def formal_facts(self) -> list[FinancialFact]:
        return [fact for fact in self.financial_facts if fact.formal_eligible]

    def tool_status(self, tool_name: str) -> str:
        matches = [item.status for item in self.tool_health if item.tool_name == tool_name]
        return matches[-1] if matches else "failed"

    def without_fact(self, field_name: str) -> "ResearchEvidenceBundle":
        return self.model_copy(
            update={
                "financial_facts": [
                    fact for fact in self.financial_facts if fact.field_name != field_name
                ]
            }
        )


def periods_are_compatible(left: FinancialFact, right: FinancialFact) -> bool:
    """Compare SEC period identity while allowing an instant to relate to a flow's end date."""
    if not _has_complete_period(left) or not _has_complete_period(right):
        return False
    if left.fiscal_year != right.fiscal_year:
        return False
    if left.fiscal_period != right.fiscal_period:
        return False
    if left.period_end != right.period_end:
        return False
    if (
        left.period_start is not None
        and right.period_start is not None
        and left.period_start != right.period_start
    ):
        return False
    return True


@dataclass(frozen=True)
class _CandidateFact:
    fact: FinancialFact
    frame: str


_FIELD_CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
    ),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss",),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capital_expenditure": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpendituresIncurredButNotYetPaid",
    ),
    "cash_and_equivalents": ("CashAndCashEquivalentsAtCarryingValue",),
    "debt_current": ("LongTermDebtCurrent",),
    "debt_noncurrent": ("LongTermDebtNoncurrent",),
    "shares_outstanding": (
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesOutstanding",
    ),
    "diluted_shares": (
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingDiluted",
    ),
    "eps": ("EarningsPerShareDiluted",),
    "segment_revenue_services": ("SalesRevenueServicesGross",),
}


class EvidenceNormalizer:
    """Turns raw SEC company facts into provenance-complete financial evidence."""

    def normalize_company_facts(
        self,
        company_name: str,
        ticker: str,
        payload: dict[str, object],
    ) -> ResearchEvidenceBundle:
        cik = _normalize_cik(payload.get("cik"))
        artifact_ref = _company_facts_url(cik) if cik else ""
        bundle = ResearchEvidenceBundle(
            company_name=company_name,
            ticker=ticker,
            raw_artifact_refs=[artifact_ref] if artifact_ref else [],
        )
        facts = _mapping(payload.get("facts"))
        us_gaap = _mapping(facts.get("us-gaap"))

        for field_name, concepts in _FIELD_CONCEPTS.items():
            candidates = self._candidates_for_field(
                field_name=field_name,
                concepts=concepts,
                us_gaap=us_gaap,
                cik=cik,
                bundle=bundle,
            )
            if not candidates:
                continue

            complete_candidates = [
                candidate for candidate in candidates if _has_complete_period(candidate.fact)
            ]
            incomplete_candidates = [
                candidate for candidate in candidates if not _has_complete_period(candidate.fact)
            ]
            for candidate in incomplete_candidates:
                bundle.gaps.append(
                    _period_gap(
                        code="incomplete_financial_period",
                        field_name=field_name,
                        candidate=candidate,
                        message=(
                            "SEC Company Facts candidate is missing fiscal year, fiscal period, "
                            "or period end and cannot be matched confidently."
                        ),
                    )
                )
            if complete_candidates:
                winner = _select_newest_compatible_candidate(complete_candidates)
            else:
                winner = max(candidates, key=_filing_sort_key)
                bundle.gaps.append(
                    _period_gap(
                        code="ambiguous_financial_period",
                        field_name=field_name,
                        candidate=winner,
                        message=(
                            "No SEC Company Facts candidate has fiscal year, fiscal period, and period end; "
                            "the selected fact is period-ambiguous."
                        ),
                    )
                )
            bundle.financial_facts.append(winner.fact)
        return bundle

    def _candidates_for_field(
        self,
        field_name: str,
        concepts: tuple[str, ...],
        us_gaap: Mapping[str, object],
        cik: str,
        bundle: ResearchEvidenceBundle,
    ) -> list[_CandidateFact]:
        candidates: list[_CandidateFact] = []
        for concept in concepts:
            concept_payload = _mapping(us_gaap.get(concept))
            units = _mapping(concept_payload.get("units"))
            for unit, raw_entries in units.items():
                if not isinstance(unit, str) or not isinstance(raw_entries, list):
                    continue
                for raw_entry in raw_entries:
                    entry = _mapping(raw_entry)
                    value = _as_float(entry.get("val"))
                    if value is None:
                        bundle.gaps.append(
                            EvidenceGap(
                                code="invalid_financial_fact",
                                target="fundamental_analyst",
                                fields=[field_name],
                                sources=[],
                                message="SEC Company Facts candidate has no numeric value.",
                            )
                        )
                        continue
                    accession = _as_text(entry.get("accn"))
                    candidates.append(
                        _CandidateFact(
                            fact=FinancialFact(
                                field_name=field_name,
                                value=value,
                                unit=unit,
                                period_start=_as_date(entry.get("start")),
                                period_end=_as_date(entry.get("end")),
                                fiscal_year=_as_int(entry.get("fy")),
                                fiscal_period=_as_text(entry.get("fp")),
                                form=_as_text(entry.get("form")),
                                accession=accession,
                                filed_at=_as_date(entry.get("filed")),
                                source_url=_filing_url(cik, accession),
                                source_tag="sec_companyfacts",
                                taxonomy_concept=concept,
                            ),
                            frame=_as_text(entry.get("frame")) or "",
                        )
                    )
        return candidates


def _select_newest_compatible_candidate(candidates: list[_CandidateFact]) -> _CandidateFact:
    newest_period = max(candidates, key=_period_sort_key)
    compatible = [
        candidate
        for candidate in candidates
        if periods_are_compatible(newest_period.fact, candidate.fact)
    ]
    return max(compatible, key=_filing_sort_key)


def _has_complete_period(fact: FinancialFact) -> bool:
    return (
        fact.fiscal_year is not None
        and bool(fact.fiscal_period)
        and fact.period_end is not None
    )


def _period_gap(
    code: str,
    field_name: str,
    candidate: _CandidateFact,
    message: str,
) -> EvidenceGap:
    return EvidenceGap(
        code=code,
        target="fundamental_analyst",
        fields=[field_name],
        sources=[candidate.fact.source_url] if candidate.fact.source_url else [],
        message=message,
    )


def _period_sort_key(candidate: _CandidateFact) -> tuple[int, int, date, int, str]:
    fact = candidate.fact
    return (
        fact.fiscal_year or 0,
        _fiscal_period_rank(fact.fiscal_period),
        fact.period_end or date.min,
        _flow_duration_days(fact),
        candidate.frame,
    )


def _fiscal_period_rank(fiscal_period: str | None) -> int:
    if fiscal_period == "FY":
        return 5
    if fiscal_period and len(fiscal_period) == 2 and fiscal_period[0] == "Q":
        try:
            return int(fiscal_period[1])
        except ValueError:
            return 0
    return 0


def _flow_duration_days(fact: FinancialFact) -> int:
    if fact.period_start is None or fact.period_end is None:
        return -1
    return (fact.period_end - fact.period_start).days


def _filing_sort_key(candidate: _CandidateFact) -> tuple[date, str, str]:
    fact = candidate.fact
    return fact.filed_at or date.min, fact.accession or "", candidate.frame


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _as_date(value: object) -> date | None:
    text = _as_text(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric_value = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return numeric_value if numeric_value is not None and isfinite(numeric_value) else None


def _normalize_cik(value: object) -> str:
    cik = _as_int(value)
    return str(cik) if cik is not None and cik >= 0 else ""


def _company_facts_url(cik: str) -> str:
    return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"


def _filing_url(cik: str, accession: str | None) -> str | None:
    if not cik or not accession:
        return None
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/"
