import json
import sys
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent.core.confidence_gate import ConfidenceGatePolicy
from multi_agent.core.formal_gate import FORMAL_GATE_REQUIRED_FIELDS
from multi_agent.core.review_contracts import RepairAction, ReviewContract, ToolHealthSummary
from multi_agent.evaluation import WorkflowEvaluation
from multi_agent.tools.investment_tools import build_research_evidence_bundle


@pytest.fixture
def apple_bundle():
    fixture_dir = Path(__file__).parent / "fixtures" / "apple"
    company_facts = json.loads((fixture_dir / "companyfacts.json").read_text(encoding="utf-8"))
    facts = company_facts["facts"]["us-gaap"]
    annual = {
        "end": "2025-09-27",
        "fy": 2025,
        "fp": "FY",
        "form": "10-K",
        "filed": "2025-10-31",
        "accn": "0000320193-25-000079",
    }

    def add_fact(concept: str, unit: str, value: float, *, start: str | None = None) -> None:
        entry = dict(annual, val=value)
        if start:
            entry["start"] = start
        facts[concept] = {"units": {unit: [entry]}}

    add_fact("CashAndCashEquivalentsAtCarryingValue", "USD", 35_900_000_000)
    add_fact("LongTermDebtCurrent", "USD", 10_000_000_000)
    add_fact("LongTermDebtNoncurrent", "USD", 90_000_000_000)
    add_fact(
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "shares",
        15_000_000_000,
        start="2024-09-29",
    )
    return build_research_evidence_bundle(
        company_name="Apple Inc.",
        ticker="AAPL",
        company_facts=company_facts,
        filing_html=(fixture_dir / "filing.html").read_text(encoding="utf-8"),
        filing_metadata={
            "source_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
            "accession": "0000320193-25-000079",
            "filed_at": "2025-10-31",
            "form": "10-K",
            "fiscal_year": 2025,
            "fiscal_period": "FY",
            "period_start": "2024-09-29",
            "period_end": "2025-09-27",
        },
        quote_payload=json.loads((fixture_dir / "quote.json").read_text(encoding="utf-8")),
        tavily_payloads=[json.loads((fixture_dir / "news.json").read_text(encoding="utf-8"))],
    )


def _contract(**overrides: object) -> ReviewContract:
    payload: dict[str, object] = {
        "stage": "analysis_review",
        "reviewer_name": "data_quality_reviewer",
        "decision": {"gate_outcome": "pass", "decision_confidence": "high"},
        "delivery_eligibility": {
            "formal_report_allowed": True,
            "evidence_limited_report_allowed": True,
            "blocked_notice_required": False,
        },
        "failure_taxonomy": {"primary_class": "none"},
        "coverage_summary": {
            "evidence_coverage_ratio": 1.0,
            "financial_coverage_score": 1.0,
            "claim_binding_ratio": 1.0,
        },
        "tool_health_summary": {"overall_status": "healthy"},
        "review_summary": {"one_sentence_summary": "Evidence is complete."},
    }
    payload.update(overrides)
    return ReviewContract.model_validate(payload)


def test_gate_requests_targeted_rerun_for_cross_period_valuation(apple_bundle) -> None:
    mixed = apple_bundle.model_copy(deep=True)
    shares = mixed.require_fact("diluted_shares")
    mixed.financial_facts = [
        fact.model_copy(
            update={
                "fiscal_year": 2024,
                "period_end": "2024-09-28",
                "accession": "0000320193-24-000081",
            }
        )
        if fact is shares
        else fact
        for fact in mixed.financial_facts
    ]

    result = ConfidenceGatePolicy.default().evaluate(mixed, _contract())

    assert result.final_decision == "rerun"
    assert result.blocking_reasons == []
    assert any(
        action.target == "fundamental_analyst"
        and action.code == "valuation_period_mismatch"
        and action.fields == ["diluted_shares"]
        for action in result.repair_actions
    )


def test_gate_requests_targeted_rerun_for_missing_services_fact(apple_bundle) -> None:
    result = ConfidenceGatePolicy.default().evaluate(
        _healthy_bundle(apple_bundle).without_fact("segment_revenue_services"), _contract()
    )

    assert result.final_decision == "rerun"
    assert result.repair_actions == [
        RepairAction(
            target="fundamental_analyst",
            code="missing_financial_fact",
            fields=["segment_revenue_services"],
            instruction="补齐 SEC 期间、filing 标识和来源 URL 后重新审查。",
        )
    ]


def test_gate_does_not_pass_when_reviewer_disallows_formal(apple_bundle) -> None:
    limited = _contract(
        delivery_eligibility={
            "formal_report_allowed": False,
            "evidence_limited_report_allowed": True,
            "blocked_notice_required": False,
        }
    )

    result = ConfidenceGatePolicy.default().evaluate(_healthy_bundle(apple_bundle), limited)

    assert result.final_decision == "evidence_limited"
    assert result.passed is False


def test_formal_only_blocker_requests_repair_without_blocking_limited_delivery(
    apple_bundle,
) -> None:
    bundle = _healthy_bundle(apple_bundle)
    bundle.market_snapshots = []
    contract = _contract(
        decision={"gate_outcome": "rerun", "decision_confidence": "high"},
        delivery_eligibility={
            "formal_report_allowed": False,
            "evidence_limited_report_allowed": True,
            "blocked_notice_required": False,
            "recommended_delivery_state": "evidence_limited_report",
        },
        failure_taxonomy={"primary_class": "pipeline_degraded"},
        blocking_reasons=["stock_price_unavailable"],
        repair_actions=[
            {
                "target": "quant_valuation_analyst",
                "code": "missing_market_snapshot",
                "fields": ["stock_price"],
                "instruction": "重新获取报价。",
            }
        ],
    )

    result = ConfidenceGatePolicy.default().evaluate(bundle, contract)

    assert result.final_decision == "rerun"
    assert "stock_price_unavailable" in result.blocking_reasons


def test_new_contract_rejects_legacy_or_unknown_fields() -> None:
    payload = _contract().model_dump()
    payload["legacy_rating"] = "pass"

    with pytest.raises(ValidationError, match="legacy_rating"):
        ReviewContract.model_validate(payload)


def test_gate_at_exact_formal_thresholds_passes_without_prose_override(apple_bundle) -> None:
    bundle = _healthy_bundle(apple_bundle)
    threshold_contract = _contract(
        coverage_summary={
            "evidence_coverage_ratio": 0.80,
            "financial_coverage_score": 0.80,
            "claim_binding_ratio": 0.75,
        }
    )

    result = ConfidenceGatePolicy.default().evaluate(bundle, threshold_contract)

    assert result.final_decision == "passed"
    assert result.trust_score == 78


def test_degraded_tool_health_requires_repair_at_threshold_boundary(apple_bundle) -> None:
    bundle = _healthy_bundle(apple_bundle)
    bundle.tool_health[0] = bundle.tool_health[0].model_copy(update={"status": "degraded"})
    threshold_contract = _contract(
        coverage_summary={
            "evidence_coverage_ratio": 0.80,
            "financial_coverage_score": 0.80,
            "claim_binding_ratio": 0.75,
        }
    )

    result = ConfidenceGatePolicy.default().evaluate(bundle, threshold_contract)

    assert result.final_decision == "rerun"
    assert result.trust_score == 73
    assert {action.code for action in result.repair_actions} == {
        "coverage_below_formal_threshold",
        "degraded_tool_health",
        "tool_health_disagreement",
    }


def test_gate_rejects_conflicting_formal_values_for_one_source_period(apple_bundle) -> None:
    bundle = apple_bundle.model_copy(deep=True)
    revenue = bundle.require_fact("revenue")
    bundle.financial_facts.append(revenue.model_copy(update={"value": revenue.value + 1}))

    result = ConfidenceGatePolicy.default().evaluate(bundle, _contract())

    assert result.final_decision == "blocked"
    assert "source_conflict" in result.blocking_reasons


@pytest.mark.parametrize(
    "field_name",
    ["cash_and_equivalents", "total_debt", "segment_revenue_services"],
)
def test_gate_blocks_cross_period_required_fact(apple_bundle, field_name: str) -> None:
    bundle = _healthy_bundle(apple_bundle)
    original = bundle.require_fact(field_name)
    bundle.financial_facts = [
        fact.model_copy(
            update={
                "fiscal_year": 2024,
                "period_end": date(2024, 9, 28),
                "accession": "0000320193-24-000081",
            }
        )
        if fact is original
        else fact
        for fact in bundle.financial_facts
    ]

    result = ConfidenceGatePolicy.default().evaluate(bundle, _contract())

    assert result.final_decision == "rerun"
    assert any(
        action.code == "valuation_period_mismatch" and action.fields == [field_name]
        for action in result.repair_actions
    )


def test_gate_blocks_duplicate_required_fact_even_when_values_match(apple_bundle) -> None:
    bundle = _healthy_bundle(apple_bundle)
    bundle.financial_facts.append(bundle.require_fact("revenue").model_copy(deep=True))

    result = ConfidenceGatePolicy.default().evaluate(bundle, _contract())

    assert result.final_decision == "blocked"
    assert "ambiguous_financial_fact" in result.blocking_reasons


def test_gate_blocks_quote_currency_mismatch(apple_bundle) -> None:
    bundle = _healthy_bundle(apple_bundle)
    bundle.market_snapshots[0] = bundle.market_snapshots[0].model_copy(
        update={"currency": "EUR"}
    )

    result = ConfidenceGatePolicy.default().evaluate(bundle, _contract())

    assert result.final_decision == "blocked"
    assert "market_currency_mismatch" in result.blocking_reasons


@pytest.mark.parametrize("invalid_case", ["negative_price", "invalid_source", "invalid_unit"])
def test_gate_and_evaluation_reject_the_same_invalid_formal_evidence(
    apple_bundle,
    invalid_case: str,
) -> None:
    bundle = _healthy_bundle(apple_bundle)
    if invalid_case == "negative_price":
        bundle.market_snapshots[0] = bundle.market_snapshots[0].model_copy(
            update={"price": -1.0}
        )
    elif invalid_case == "invalid_source":
        bundle.market_snapshots[0] = bundle.market_snapshots[0].model_copy(
            update={"source_url": "not-a-url"}
        )
    else:
        revenue = bundle.require_fact("revenue")
        bundle.financial_facts = [
            fact.model_copy(update={"unit": "EUR"}) if fact is revenue else fact
            for fact in bundle.financial_facts
        ]
        bundle.market_snapshots[0] = bundle.market_snapshots[0].model_copy(
            update={"currency": "EUR"}
        )

    result = ConfidenceGatePolicy.default().evaluate(bundle, _contract())

    assert result.final_decision == "blocked"
    assert WorkflowEvaluation._formal_fact_provenance_complete(bundle) is False


def test_gate_and_evaluation_accept_one_canonical_snapshot_among_stale_history(
    apple_bundle,
) -> None:
    bundle = _healthy_bundle(apple_bundle)
    valid_snapshot = bundle.market_snapshots[0]
    bundle.market_snapshots.append(
        valid_snapshot.model_copy(
            update={
                "currency": "EUR",
                "diluted_shares_period_end": date(2024, 9, 28),
            }
        )
    )

    result = ConfidenceGatePolicy.default().evaluate(bundle, _contract())

    assert result.final_decision == "passed"
    assert WorkflowEvaluation._formal_fact_provenance_complete(bundle) is True


def test_gate_and_evaluation_require_one_snapshot_to_meet_every_constraint(
    apple_bundle,
) -> None:
    bundle = _healthy_bundle(apple_bundle)
    valid_snapshot = bundle.market_snapshots[0]
    bundle.market_snapshots = [
        valid_snapshot.model_copy(update={"currency": "EUR"}),
        valid_snapshot.model_copy(update={"source_url": "not-a-url"}),
    ]

    result = ConfidenceGatePolicy.default().evaluate(bundle, _contract())

    assert result.final_decision == "blocked"
    assert "invalid_market_snapshot" in result.blocking_reasons
    assert WorkflowEvaluation._formal_fact_provenance_complete(bundle) is False


def test_required_fields_remain_the_formal_gate_contract() -> None:
    assert "segment_revenue_services" in FORMAL_GATE_REQUIRED_FIELDS


def _healthy_bundle(apple_bundle):
    bundle = apple_bundle.model_copy(deep=True)
    bundle.tool_health = [
        item.model_copy(update={"status": "healthy"}) for item in bundle.tool_health
    ]
    bundle.market_snapshots[0] = bundle.market_snapshots[0].model_copy(
        update={"diluted_shares_period_end": bundle.require_fact("diluted_shares").period_end}
    )
    return bundle


def test_explicit_reviewer_block_has_precedence_over_complete_evidence(apple_bundle) -> None:
    contract = _contract(decision={"gate_outcome": "block", "decision_confidence": "high"})

    result = ConfidenceGatePolicy.default().evaluate(_healthy_bundle(apple_bundle), contract)

    assert result.final_decision == "blocked"
    assert "reviewer_requested_block" in result.blocking_reasons


def test_resolved_critical_conflict_does_not_block_delivery(apple_bundle) -> None:
    contract = _contract(
        coverage_summary={
            "evidence_coverage_ratio": 1.0,
            "financial_coverage_score": 1.0,
            "claim_binding_ratio": 1.0,
            "critical_conflict_count": 1,
            "unresolved_critical_claim_count": 0,
        }
    )

    result = ConfidenceGatePolicy.default().evaluate(_healthy_bundle(apple_bundle), contract)

    assert result.final_decision == "passed"
    assert "critical_conflict_count>0" not in result.blocking_reasons


def test_explicit_reviewer_rerun_has_targeted_repair_action(apple_bundle) -> None:
    contract = _contract(
        decision={"gate_outcome": "rerun", "decision_confidence": "high"},
        rerun_reasons=["Reconcile the filing source."],
    )

    result = ConfidenceGatePolicy.default().evaluate(_healthy_bundle(apple_bundle), contract)

    assert result.final_decision == "rerun"
    assert result.repair_actions == [
        RepairAction(
            target="data_quality_reviewer",
            code="reviewer_rerun_requested",
            instruction="Reconcile the filing source.",
        )
    ]


def test_degraded_raw_tool_health_cannot_pass_even_when_reviewer_says_healthy(apple_bundle) -> None:
    bundle = _healthy_bundle(apple_bundle)
    bundle.tool_health[0] = bundle.tool_health[0].model_copy(update={"status": "degraded"})

    result = ConfidenceGatePolicy.default().evaluate(bundle, _contract())

    assert result.final_decision == "rerun"
    assert {action.code for action in result.repair_actions} == {
        "degraded_tool_health",
        "tool_health_disagreement",
    }
    degraded_action = next(
        action for action in result.repair_actions if action.code == "degraded_tool_health"
    )
    assert degraded_action.target == "fundamental_analyst"


def test_degraded_tavily_routes_repair_to_event_research(apple_bundle) -> None:
    bundle = _healthy_bundle(apple_bundle)
    tavily_index = next(
        index for index, item in enumerate(bundle.tool_health) if item.tool_name == "tavily"
    )
    bundle.tool_health[tavily_index] = bundle.tool_health[tavily_index].model_copy(
        update={"status": "degraded"}
    )

    result = ConfidenceGatePolicy.default().evaluate(bundle, _contract())

    assert any(
        action.target == "event_guidance_analyst"
        and action.code == "degraded_tool_health"
        and action.sources == ["tavily"]
        for action in result.repair_actions
    )


def test_failed_formal_tool_has_targeted_repair_action(apple_bundle) -> None:
    bundle = _healthy_bundle(apple_bundle)
    quote_index = next(
        index for index, item in enumerate(bundle.tool_health) if item.tool_name == "quote"
    )
    bundle.tool_health[quote_index] = bundle.tool_health[quote_index].model_copy(
        update={"status": "failed"}
    )

    result = ConfidenceGatePolicy.default().evaluate(bundle, _contract())

    assert result.final_decision == "rerun"
    assert any(action.code == "critical_tool_failed" for action in result.repair_actions)


def test_market_snapshot_requires_known_diluted_share_period(apple_bundle) -> None:
    bundle = _healthy_bundle(apple_bundle)
    bundle.market_snapshots[0] = bundle.market_snapshots[0].model_copy(
        update={"diluted_shares_period_end": None}
    )

    result = ConfidenceGatePolicy.default().evaluate(bundle, _contract())

    assert result.final_decision == "rerun"
    assert any(action.code == "market_denominator_period_unknown" for action in result.repair_actions)


def test_market_snapshot_rejects_mismatched_diluted_share_period(apple_bundle) -> None:
    bundle = _healthy_bundle(apple_bundle)
    bundle.market_snapshots[0] = bundle.market_snapshots[0].model_copy(
        update={"diluted_shares_period_end": date(2024, 9, 28)}
    )

    result = ConfidenceGatePolicy.default().evaluate(bundle, _contract())

    assert result.final_decision == "rerun"
    assert any(action.code == "market_denominator_period_mismatch" for action in result.repair_actions)


def test_later_market_quote_with_matching_diluted_share_period_is_allowed(apple_bundle) -> None:
    bundle = _healthy_bundle(apple_bundle)
    diluted_shares = bundle.require_fact("diluted_shares")
    bundle.market_snapshots[0] = bundle.market_snapshots[0].model_copy(
        update={"diluted_shares_period_end": diluted_shares.period_end}
    )

    result = ConfidenceGatePolicy.default().evaluate(bundle, _contract())

    assert result.final_decision == "passed"


def test_duplicate_repair_actions_canonicalize_field_and_source_order(apple_bundle) -> None:
    duplicate_actions = [
        {
            "target": "fundamental_analyst",
            "code": "repair",
            "fields": ["revenue", "cash", "revenue"],
            "sources": ["b", "a", "a"],
        },
        {
            "target": "fundamental_analyst",
            "code": "repair",
            "fields": ["cash", "revenue"],
            "sources": ["a", "b"],
        },
    ]
    contract = _contract(repair_actions=duplicate_actions)

    result = ConfidenceGatePolicy.default().evaluate(_healthy_bundle(apple_bundle), contract)

    assert result.final_decision == "rerun"
    assert result.repair_actions[0].fields == ["cash", "revenue"]
    assert result.repair_actions[0].sources == ["a", "b"]


@pytest.mark.parametrize(
    "tool_health_summary",
    [
        {"overall_status": "failed"},
        {"overall_status": "degraded"},
        {"overall_status": "healthy", "failed_tools": ["quote"]},
        {"overall_status": "healthy", "degraded_tools": ["tavily"]},
    ],
)
def test_contract_rejects_inconsistent_tool_health_summary(
    tool_health_summary: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="overall_status"):
        _contract(tool_health_summary=tool_health_summary)


def test_gate_reruns_for_nonhealthy_summary_that_bypasses_model_validation(apple_bundle) -> None:
    contract = _contract().model_copy(
        update={"tool_health_summary": ToolHealthSummary.model_construct(overall_status="failed")}
    )

    result = ConfidenceGatePolicy.default().evaluate(_healthy_bundle(apple_bundle), contract)

    assert result.final_decision == "rerun"
    assert [action.code for action in result.repair_actions] == [
        "tool_health_contract_inconsistent"
    ]
