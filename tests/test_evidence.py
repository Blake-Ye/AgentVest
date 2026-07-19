from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

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


def test_flow_and_instant_facts_with_same_fiscal_period_are_compatible():
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
