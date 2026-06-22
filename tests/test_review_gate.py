import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent.core.confidence_gate import ConfidenceGatePolicy
from multi_agent.core.review_contracts import ReviewToolSummary


def _review_summary(**overrides: object) -> ReviewToolSummary:
    defaults = {
        "evidence_coverage_ratio": 0.95,
        "financial_coverage_score": 0.9,
        "critical_conflict_count": 0,
        "market_policy_violations": [],
        "unsupported_critical_claims": [],
        "blocking_reasons": [],
    }
    defaults.update(overrides)
    return ReviewToolSummary(**defaults)


def test_confidence_gate_blocks_when_evidence_coverage_below_threshold() -> None:
    summary = _review_summary(evidence_coverage_ratio=0.6)

    decision = ConfidenceGatePolicy.default().evaluate(summary)

    assert decision.passed is False
    assert decision.final_decision == "rerun"
    assert "evidence_coverage_ratio<0.80" in decision.blocking_reasons


def test_confidence_gate_preserves_high_scores_on_a_full_0_to_100_scale() -> None:
    summary = _review_summary(
        evidence_coverage_ratio=0.95,
        financial_coverage_score=0.95,
    )

    decision = ConfidenceGatePolicy.default().evaluate(summary)

    assert decision.passed is True
    assert decision.final_decision == "passed"
    assert decision.trust_score == 95
    assert decision.blocking_reasons == []


@pytest.mark.parametrize(
    ("override_field", "override_value", "expected_reason"),
    [
        ("market_policy_violations", ["price target without source"], "market_policy_violations>0"),
        (
            "unsupported_critical_claims",
            ["management guidance will improve next quarter"],
            "unsupported_critical_claims>0",
        ),
    ],
    ids=["market-policy-violations", "unsupported-critical-claims"],
)
def test_confidence_gate_blocks_when_explicit_list_based_violations_exist(
    override_field: str,
    override_value: list[str],
    expected_reason: str,
) -> None:
    summary = _review_summary(**{override_field: override_value})

    decision = ConfidenceGatePolicy.default().evaluate(summary)

    assert decision.passed is False
    assert decision.final_decision == "blocked"
    assert expected_reason in decision.blocking_reasons


def test_confidence_gate_passes_when_all_thresholds_are_met() -> None:
    summary = _review_summary(
        evidence_coverage_ratio=1.0,
        financial_coverage_score=1.0,
    )

    decision = ConfidenceGatePolicy.default().evaluate(summary)

    assert decision.passed is True
    assert decision.final_decision == "passed"
    assert decision.trust_score == 100
    assert decision.blocking_reasons == []


def test_confidence_gate_blocks_when_critical_conflicts_exist() -> None:
    summary = _review_summary(critical_conflict_count=1)

    decision = ConfidenceGatePolicy.default().evaluate(summary)

    assert decision.passed is False
    assert decision.final_decision == "blocked"
    assert "critical_conflict_count>0" in decision.blocking_reasons
