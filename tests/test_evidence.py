from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from multi_agent.core.evidence import (
    EvidenceNormalizer,
    FinancialFact,
    ResearchEvidenceBundle,
    periods_are_compatible,
)
from multi_agent.core.state import ResearchRunState


@pytest.fixture
def apple_companyfacts() -> dict[str, object]:
    fixture_path = Path(__file__).parent / "fixtures" / "apple" / "companyfacts.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_apple_revenue_fact_keeps_period_and_filing_provenance(apple_companyfacts):
    bundle = EvidenceNormalizer().normalize_company_facts(
        company_name="Apple Inc.", ticker="AAPL", payload=apple_companyfacts
    )

    revenue = bundle.require_fact("revenue")

    assert revenue.unit == "USD"
    assert revenue.period_end == date(2025, 9, 27)
    assert revenue.fiscal_year == 2025
    assert revenue.form == "10-K"
    assert revenue.accession == "0000320193-25-000079"
    assert revenue.source_url.startswith("https://www.sec.gov/Archives/")
    assert revenue.formal_eligible is True


def test_periodless_fact_is_not_formal_eligible():
    fact = FinancialFact(field_name="revenue", value=10, unit="USD", source_tag="legacy")

    assert fact.formal_eligible is False
    assert "period_end_missing" in fact.quality_flags


def test_financial_facts_from_different_fiscal_years_are_not_period_compatible():
    current_period = FinancialFact(
        field_name="revenue",
        value=100,
        unit="USD",
        period_start=date(2024, 9, 29),
        period_end=date(2025, 9, 27),
        fiscal_year=2025,
        fiscal_period="FY",
        form="10-K",
        accession="0000320193-25-000079",
        filed_at=date(2025, 10, 31),
        source_url="https://www.sec.gov/Archives/edgar/data/320193/filing.htm",
        source_tag="sec_companyfacts",
    )
    stale_period = current_period.model_copy(
        update={
            "field_name": "diluted_shares",
            "value": 15,
            "unit": "shares",
            "period_start": date(2022, 9, 25),
            "period_end": date(2023, 9, 30),
            "fiscal_year": 2023,
            "accession": "0000320193-23-000106",
            "filed_at": date(2023, 11, 3),
        }
    )

    assert periods_are_compatible(current_period, stale_period) is False


def test_mixed_instant_and_flow_facts_with_same_period_identity_are_compatible():
    revenue = FinancialFact(
        field_name="revenue",
        value=100,
        unit="USD",
        period_start=date(2024, 9, 29),
        period_end=date(2025, 9, 27),
        fiscal_year=2025,
        fiscal_period="FY",
        form="10-K",
        accession="0000320193-25-000079",
        filed_at=date(2025, 10, 31),
        source_url="https://www.sec.gov/Archives/edgar/data/320193/filing.htm",
        source_tag="sec_companyfacts",
    )
    shares = revenue.model_copy(
        update={
            "field_name": "diluted_shares",
            "value": 15,
            "unit": "shares",
            "period_start": None,
        }
    )

    assert periods_are_compatible(revenue, shares) is True


def test_same_end_annual_and_quarterly_facts_are_not_period_compatible():
    annual = FinancialFact(
        field_name="revenue",
        value=100,
        unit="USD",
        period_start=date(2024, 9, 29),
        period_end=date(2025, 9, 27),
        fiscal_year=2025,
        fiscal_period="FY",
        source_tag="sec_companyfacts",
    )
    quarterly = annual.model_copy(
        update={"period_start": date(2025, 6, 29), "fiscal_period": "Q4"}
    )

    assert periods_are_compatible(annual, quarterly) is False


def test_same_period_flows_with_different_durations_are_not_period_compatible():
    annual = FinancialFact(
        field_name="revenue",
        value=100,
        unit="USD",
        period_start=date(2024, 9, 29),
        period_end=date(2025, 9, 27),
        fiscal_year=2025,
        fiscal_period="FY",
        source_tag="sec_companyfacts",
    )
    shorter_duration = annual.model_copy(update={"period_start": date(2025, 1, 1)})

    assert periods_are_compatible(annual, shorter_duration) is False


def test_normalizer_prefers_annual_over_newer_quarterly_same_end_candidate(apple_companyfacts):
    entries = apple_companyfacts["facts"]["us-gaap"][
        "RevenueFromContractWithCustomerExcludingAssessedTax"
    ]["units"]["USD"]
    quarterly = dict(entries[-1])
    quarterly.update(
        {
            "start": "2025-06-29",
            "val": 99_000_000_000,
            "fp": "Q4",
            "filed": "2025-11-15",
        }
    )
    entries.append(quarterly)

    bundle = EvidenceNormalizer().normalize_company_facts(
        company_name="Apple Inc.", ticker="AAPL", payload=apple_companyfacts
    )

    revenue = bundle.require_fact("revenue")
    assert revenue.value == 416_161_000_000
    assert revenue.fiscal_period == "FY"


def test_financial_fact_with_incomplete_period_metadata_is_not_period_compatible():
    complete = FinancialFact(
        field_name="revenue",
        value=100,
        unit="USD",
        period_end=date(2025, 9, 27),
        fiscal_year=2025,
        source_tag="sec_companyfacts",
    )
    incomplete = complete.model_copy(update={"fiscal_year": None})

    assert periods_are_compatible(complete, incomplete) is False


def test_incomplete_period_candidate_does_not_displace_complete_evidence(apple_companyfacts):
    entries = apple_companyfacts["facts"]["us-gaap"][
        "RevenueFromContractWithCustomerExcludingAssessedTax"
    ]["units"]["USD"]
    incomplete = dict(entries[-1])
    incomplete.update({"val": 500_000_000_000, "fy": 2026, "filed": "2026-10-31"})
    incomplete.pop("end")
    entries.append(incomplete)

    bundle = EvidenceNormalizer().normalize_company_facts(
        company_name="Apple Inc.", ticker="AAPL", payload=apple_companyfacts
    )

    assert bundle.require_fact("revenue").value == 416_161_000_000
    assert any(
        gap.code == "incomplete_financial_period" and gap.fields == ["revenue"]
        for gap in bundle.gaps
    )


def test_ambiguous_only_evidence_emits_targeted_period_gap():
    payload = {
        "cik": 320193,
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "val": 100,
                                "form": "10-K",
                                "filed": "2025-10-31",
                                "accn": "0000320193-25-000079",
                            }
                        ]
                    }
                }
            }
        },
    }

    bundle = EvidenceNormalizer().normalize_company_facts(
        company_name="Apple Inc.", ticker="AAPL", payload=payload
    )

    assert bundle.require_fact("revenue").formal_eligible is False
    assert any(
        gap.code == "ambiguous_financial_period" and gap.fields == ["revenue"]
        for gap in bundle.gaps
    )


@pytest.mark.parametrize(
    ("missing_key", "missing_flag"),
    [("fy", "fiscal_year_missing"), ("fp", "fiscal_period_missing")],
)
def test_ambiguous_period_fact_is_auditable_but_not_formal(
    missing_key,
    missing_flag,
):
    entry = {
        "start": "2024-09-29",
        "end": "2025-09-27",
        "val": 416_161_000_000,
        "fy": 2025,
        "fp": "FY",
        "form": "10-K",
        "filed": "2025-10-31",
        "accn": "0000320193-25-000079",
    }
    entry.pop(missing_key)
    payload = {
        "cik": 320193,
        "facts": {"us-gaap": {"Revenues": {"units": {"USD": [entry]}}}},
    }

    bundle = EvidenceNormalizer().normalize_company_facts(
        company_name="Apple Inc.", ticker="AAPL", payload=payload
    )

    revenue = bundle.require_fact("revenue")
    assert revenue in bundle.financial_facts
    assert missing_flag in revenue.quality_flags
    assert revenue.formal_eligible is False
    assert bundle.formal_facts() == []
    assert any(
        gap.code == "ambiguous_financial_period" and gap.fields == ["revenue"]
        for gap in bundle.gaps
    )


def test_point_in_time_shares_are_exposed_as_shares_outstanding_not_diluted(
    apple_companyfacts,
):
    bundle = EvidenceNormalizer().normalize_company_facts(
        company_name="Apple Inc.", ticker="AAPL", payload=apple_companyfacts
    )

    assert bundle.require_fact("shares_outstanding").taxonomy_concept == (
        "EntityCommonStockSharesOutstanding"
    )
    with pytest.raises(ValueError, match="diluted_shares"):
        bundle.require_fact("diluted_shares")


def test_dimension_member_is_not_mapped_as_shares_outstanding(apple_companyfacts):
    us_gaap = apple_companyfacts["facts"]["us-gaap"]
    entity_shares = us_gaap.pop("EntityCommonStockSharesOutstanding")
    us_gaap["CommonStocksIncludingAdditionalPaidInCapitalMember"] = entity_shares

    bundle = EvidenceNormalizer().normalize_company_facts(
        company_name="Apple Inc.", ticker="AAPL", payload=apple_companyfacts
    )

    with pytest.raises(ValueError, match="shares_outstanding"):
        bundle.require_fact("shares_outstanding")


def test_weighted_average_diluted_share_concept_maps_to_diluted_shares(apple_companyfacts):
    us_gaap = apple_companyfacts["facts"]["us-gaap"]
    us_gaap["WeightedAverageNumberOfDilutedSharesOutstanding"] = {
        "units": {
            "shares": [
                {
                    "start": "2024-09-29",
                    "end": "2025-09-27",
                    "val": 15_000_000_000,
                    "fy": 2025,
                    "fp": "FY",
                    "form": "10-K",
                    "filed": "2025-10-31",
                    "accn": "0000320193-25-000079",
                    "frame": "CY2025",
                }
            ]
        }
    }

    bundle = EvidenceNormalizer().normalize_company_facts(
        company_name="Apple Inc.", ticker="AAPL", payload=apple_companyfacts
    )

    assert bundle.require_fact("diluted_shares").taxonomy_concept == (
        "WeightedAverageNumberOfDilutedSharesOutstanding"
    )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_financial_candidate_is_rejected_before_creating_claim(
    apple_companyfacts,
    value,
):
    entries = apple_companyfacts["facts"]["us-gaap"][
        "RevenueFromContractWithCustomerExcludingAssessedTax"
    ]["units"]["USD"]
    invalid = dict(entries[-1])
    invalid.update({"val": value, "filed": "2025-11-01"})
    entries.append(invalid)

    bundle = EvidenceNormalizer().normalize_company_facts(
        company_name="Apple Inc.", ticker="AAPL", payload=apple_companyfacts
    )

    assert bundle.require_fact("revenue").value == 416_161_000_000
    assert any(
        gap.code == "invalid_financial_fact" and gap.fields == ["revenue"]
        for gap in bundle.gaps
    )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_financial_fact_rejects_non_finite_claim_value(value):
    with pytest.raises(ValidationError):
        FinancialFact(
            field_name="revenue",
            value=value,
            unit="USD",
            source_tag="sec_companyfacts",
        )


def test_normalizer_selects_newest_filed_fact_in_the_newest_fiscal_period_and_records_rejections(
    apple_companyfacts,
):
    bundle = EvidenceNormalizer().normalize_company_facts(
        company_name="Apple Inc.", ticker="AAPL", payload=apple_companyfacts
    )

    revenue = bundle.require_fact("revenue")

    assert revenue.value == 416_161_000_000
    assert revenue.filed_at == date(2025, 10, 31)
    assert any(
        gap.code == "rejected_financial_fact" and gap.fields == ["revenue"]
        for gap in bundle.gaps
    )


def test_normalizer_records_every_rejected_duplicate_candidate(apple_companyfacts):
    entries = apple_companyfacts["facts"]["us-gaap"][
        "RevenueFromContractWithCustomerExcludingAssessedTax"
    ]["units"]["USD"]
    entries.append(dict(entries[-1]))

    bundle = EvidenceNormalizer().normalize_company_facts(
        company_name="Apple Inc.", ticker="AAPL", payload=apple_companyfacts
    )

    rejected_revenue_candidates = [
        gap
        for gap in bundle.gaps
        if gap.code == "rejected_financial_fact" and gap.fields == ["revenue"]
    ]
    assert len(rejected_revenue_candidates) == 3


def test_research_run_state_keeps_typed_evidence_bundle(apple_companyfacts):
    bundle = EvidenceNormalizer().normalize_company_facts(
        company_name="Apple Inc.", ticker="AAPL", payload=apple_companyfacts
    )

    state = ResearchRunState(
        request_id="request-1",
        company_name="Apple Inc.",
        evidence_bundle=bundle,
    )

    assert isinstance(state.evidence_bundle, ResearchEvidenceBundle)
    assert state.evidence_bundle.require_fact("revenue").accession == "0000320193-25-000079"
