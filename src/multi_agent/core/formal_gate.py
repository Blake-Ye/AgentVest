from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from multi_agent.core.evidence import FinancialFact, ResearchEvidenceBundle, periods_are_compatible
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

    missing_financial = [
        field_name
        for field_name in required_fields
        if field_name != "stock_price"
        and not any(fact.formal_eligible for fact in facts_by_name[field_name])
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
    if "stock_price" in required_fields and not bundle.market_snapshots:
        repair_actions.append(
            RepairAction(
                target="market_validation_analyst",
                code="missing_market_snapshot",
                fields=["stock_price"],
                instruction="补齐可审计的市场报价和观察时间后重新审查。",
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

    revenue = _first_formal_fact(facts_by_name.get("revenue", []))
    diluted_shares = _first_formal_fact(facts_by_name.get("diluted_shares", []))
    if revenue is not None and diluted_shares is not None and not periods_are_compatible(
        revenue, diluted_shares
    ):
        blockers.append("valuation_period_mismatch")

    if "stock_price" in required_fields and diluted_shares is not None:
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


def _first_formal_fact(facts: list[FinancialFact]) -> FinancialFact | None:
    return next((fact for fact in facts if fact.formal_eligible), None)
