from __future__ import annotations

from collections import defaultdict
from datetime import date
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field

from multi_agent.core.evidence import (
    FinancialFact,
    MarketSnapshotEvidence,
    ResearchEvidenceBundle,
    periods_are_compatible,
)
from multi_agent.core.review_contracts import RepairAction

FORMAL_GATE_REQUIRED_FIELDS = [
    "revenue",
    "cash_and_equivalents",
    "total_debt",
    "diluted_shares",
    "stock_price",
    "segment_revenue_services",
]


class FormalDeliveryDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blocking_reasons: list[str] = Field(default_factory=list)
    repair_actions: list[RepairAction] = Field(default_factory=list)


def diagnose_formal_delivery(
    bundle: ResearchEvidenceBundle,
    required_fields: list[str],
) -> FormalDeliveryDiagnostics:
    """Return deterministic, actionable evidence diagnostics for formal delivery."""
    facts_by_name: dict[str, list[FinancialFact]] = defaultdict(list)
    for fact in bundle.financial_facts:
        facts_by_name[fact.field_name].append(fact)

    required_financial_fields = [
        field_name for field_name in required_fields if field_name != "stock_price"
    ]
    eligible_required_facts = {
        field_name: [
            fact for fact in facts_by_name.get(field_name, []) if fact.formal_eligible
        ]
        for field_name in required_financial_fields
    }
    missing_financial = [
        field_name
        for field_name, facts in eligible_required_facts.items()
        if not facts
    ]
    repair_actions: list[RepairAction] = []
    if missing_financial:
        repair_actions.append(
            RepairAction(
                target="fundamental_analyst",
                code="missing_financial_fact",
                fields=missing_financial,
                instruction="补齐 SEC 期间、filing 标识和来源 URL 后重新审查。",
            )
        )
    blockers: list[str] = []
    for field_name, facts in facts_by_name.items():
        eligible = [fact for fact in facts if fact.formal_eligible]
        values_by_period: dict[tuple[object, ...], set[float]] = defaultdict(set)
        for fact in eligible:
            period_identity = (
                fact.fiscal_year,
                fact.fiscal_period,
                fact.period_start,
                fact.period_end,
            )
            values_by_period[period_identity].add(fact.value)
        if any(len(values) > 1 for values in values_by_period.values()):
            blockers.append("source_conflict")

    if any(len(facts) > 1 for facts in eligible_required_facts.values()):
        blockers.append("ambiguous_financial_fact")
    canonical_facts = {
        field_name: facts[0]
        for field_name, facts in eligible_required_facts.items()
        if len(facts) == 1
    }
    for field_name, fact in canonical_facts.items():
        expected_unit = "shares" if field_name == "diluted_shares" else "USD"
        if fact.unit != expected_unit:
            blockers.append("invalid_financial_unit")
        if not _is_http_url(fact.source_url):
            blockers.append("invalid_financial_source")

    revenue = canonical_facts.get("revenue")
    diluted_shares = canonical_facts.get("diluted_shares")
    if revenue is not None:
        for field_name, fact in canonical_facts.items():
            if field_name != "revenue" and not periods_are_compatible(revenue, fact):
                blockers.append("valuation_period_mismatch")

    if "stock_price" in required_fields:
        if not bundle.market_snapshots:
            repair_actions.append(
                RepairAction(
                    target="market_validation_analyst",
                    code="missing_market_snapshot",
                    fields=["stock_price"],
                    instruction="补齐可审计的市场报价和观察时间后重新审查。",
                )
            )
        elif revenue is not None and diluted_shares is not None:
            valid_snapshots = [
                snapshot
                for snapshot in bundle.market_snapshots
                if _formal_snapshot_is_eligible(
                    snapshot,
                    currency=revenue.unit,
                    diluted_shares_period_end=diluted_shares.period_end,
                )
            ]
            if not valid_snapshots:
                if not any(
                    _market_snapshot_has_valid_quote(snapshot, currency=revenue.unit)
                    for snapshot in bundle.market_snapshots
                ):
                    blockers.append("invalid_market_snapshot")
                if not any(
                    snapshot.currency.strip().upper() == revenue.unit.strip().upper()
                    for snapshot in bundle.market_snapshots
                ):
                    blockers.append("market_currency_mismatch")
                for snapshot in bundle.market_snapshots:
                    if snapshot.diluted_shares_period_end is None:
                        repair_actions.append(
                            RepairAction(
                                target="market_validation_analyst",
                                code="market_denominator_period_unknown",
                                fields=["stock_price", "diluted_shares"],
                                sources=[snapshot.source_url],
                                instruction="补齐报价使用的稀释股数期间后重新审查。",
                            )
                        )
                    elif snapshot.diluted_shares_period_end != diluted_shares.period_end:
                        repair_actions.append(
                            RepairAction(
                                target="market_validation_analyst",
                                code="market_denominator_period_mismatch",
                                fields=["stock_price", "diluted_shares"],
                                sources=[snapshot.source_url],
                                instruction="将报价估值分母与正式稀释股数期间对齐后重新审查。",
                            )
                        )

    return FormalDeliveryDiagnostics(
        blocking_reasons=sorted(set(blockers)),
        repair_actions=repair_actions,
    )


def formal_delivery_evidence_complete(
    bundle: ResearchEvidenceBundle,
    required_fields: list[str],
) -> bool:
    diagnostics = diagnose_formal_delivery(bundle, required_fields)
    return not diagnostics.blocking_reasons and not diagnostics.repair_actions


def _formal_snapshot_is_eligible(
    snapshot: MarketSnapshotEvidence,
    *,
    currency: str,
    diluted_shares_period_end: date | None,
) -> bool:
    return (
        _market_snapshot_has_valid_quote(snapshot, currency=currency)
        and snapshot.diluted_shares_period_end == diluted_shares_period_end
    )


def _market_snapshot_has_valid_quote(
    snapshot: MarketSnapshotEvidence,
    *,
    currency: str,
) -> bool:
    return (
        isfinite(snapshot.price)
        and snapshot.price > 0
        and snapshot.currency.strip().upper() == currency.strip().upper()
        and _is_http_url(snapshot.source_url)
        and bool(snapshot.source_tag.strip())
    )


def _is_http_url(value: str | None) -> bool:
    return bool(value and value.startswith(("http://", "https://")))
