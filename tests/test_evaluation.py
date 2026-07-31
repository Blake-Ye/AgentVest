import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from multi_agent.core.evidence import FinancialFact, MarketSnapshotEvidence, ResearchEvidenceBundle
from multi_agent.evaluation import WorkflowEvaluation


class StepClock:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def __call__(self) -> float:
        if not self._values:
            raise AssertionError("测试时钟已没有可用时间点。")
        return self._values.pop(0)


def _expected_outputs(base_dir: Path) -> dict[str, Path]:
    artifacts_dir = base_dir / "artifacts"
    return {
        "market_intelligence_task": artifacts_dir / "01_market_intelligence.md",
        "filing_review_task": artifacts_dir / "02_filing_review.md",
        "financial_analysis_task": artifacts_dir / "03_financial_analysis.md",
    }


def test_workflow_evaluation_writes_latest_metrics_and_summary(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    expected_outputs = _expected_outputs(tmp_path)
    for path in expected_outputs.values():
        path.write_text("# artifact\n", encoding="utf-8")

    final_report = tmp_path / "report.md"
    final_report.write_text(
        "结论参考 https://example.com/a 和 https://example.com/b",
        encoding="utf-8",
    )

    evaluator = WorkflowEvaluation(
        artifacts_dir=artifacts_dir,
        final_report_path=final_report,
        expected_task_outputs=expected_outputs,
        company_name="Apple Inc.",
        company_ticker="AAPL",
        time_source=StepClock([0.0, 1.5, 3.0, 5.5, 7.0]),
    )
    evaluator.start()
    initial_metrics = json.loads((artifacts_dir / "latest_run_metrics.json").read_text(encoding="utf-8"))
    assert initial_metrics["status"] == "running"
    assert initial_metrics["success"] is False
    assert initial_metrics["report_generated"] is False
    assert initial_metrics["error_message"] is None
    evaluator.record_api_call(service_name="Google Search", success=True, status_code=200)
    evaluator.record_api_call(service_name="SEC API", success=False, status_code=403)
    evaluator.record_task_completion("market_intelligence_task")
    evaluator.record_task_completion("filing_review_task")
    evaluator.record_task_completion("financial_analysis_task")
    evaluator.record_financial_fields(
        {
            "revenue": {
                "value": 100.0,
                "normalized_value": 100.0,
                "extracted": True,
                "source_tag": "Revenues",
            },
            "current_assets": {
                "value": 0.0,
                "normalized_value": 0.0,
                "extracted": False,
                "source_tag": None,
            },
        }
    )

    latest_metrics = evaluator.finalize(success=True)

    assert latest_metrics["success"] is True
    assert latest_metrics["status"] == "completed"
    assert latest_metrics["total_runtime_seconds"] == 7.0
    assert latest_metrics["task_durations_seconds"] == {
        "market_intelligence_task": 1.5,
        "filing_review_task": 1.5,
        "financial_analysis_task": 2.5,
    }
    assert latest_metrics["api_calls"]["total"] == 2
    assert latest_metrics["api_calls"]["failures"] == 1
    assert latest_metrics["api_calls"]["failure_rate"] == 0.5
    assert latest_metrics["report_generated"] is True
    assert latest_metrics["report_complete"] is True
    assert latest_metrics["citation_count"] == 2
    assert latest_metrics["intermediate_artifacts_complete"] is True
    assert latest_metrics["financial_fields_success_rate"] == 0.5
    assert latest_metrics["financial_fields"]["revenue"]["extracted"] is True
    assert latest_metrics["financial_fields"]["current_assets"]["extracted"] is False

    latest_report_path = artifacts_dir / "latest_run_metrics.json"
    summary_path = artifacts_dir / "evaluation_summary.json"
    assert latest_report_path.exists()
    assert summary_path.exists()

    written_latest = json.loads(latest_report_path.read_text(encoding="utf-8"))
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written_latest["company_ticker"] == "AAPL"
    assert written_summary["total_runs"] == 1
    assert written_summary["successful_runs"] == 1
    assert written_summary["success_rate"] == 1.0
    assert written_summary["api_calls"]["total"] == 2
    assert written_summary["api_calls"]["failures"] == 1
    assert written_summary["api_calls"]["failure_rate"] == 0.5


def test_workflow_evaluation_accumulates_summary_across_success_and_failure_runs(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    expected_outputs = _expected_outputs(tmp_path)
    for path in expected_outputs.values():
        path.write_text("# artifact\n", encoding="utf-8")

    success_report = tmp_path / "success_report.md"
    success_report.write_text("参考 https://example.com/a", encoding="utf-8")
    success_run = WorkflowEvaluation(
        artifacts_dir=artifacts_dir,
        final_report_path=success_report,
        expected_task_outputs=expected_outputs,
        company_name="Apple Inc.",
        company_ticker="AAPL",
        time_source=StepClock([0.0, 1.0]),
    )
    success_run.start()
    success_run.record_api_call(service_name="Google Search", success=True, status_code=200)
    success_run.finalize(success=True)

    failure_report = tmp_path / "failure_report.md"
    failure_run = WorkflowEvaluation(
        artifacts_dir=artifacts_dir,
        final_report_path=failure_report,
        expected_task_outputs=expected_outputs,
        company_name="Alibaba Group Holding Ltd",
        company_ticker="BABA",
        time_source=StepClock([0.0, 2.0]),
    )
    failure_run.start()
    failure_run.record_api_call(service_name="SEC API", success=False, status_code=429)
    latest_metrics = failure_run.finalize(success=False, error_message="SEC API 返回 429")

    summary_path = artifacts_dir / "evaluation_summary.json"
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert latest_metrics["success"] is False
    assert latest_metrics["status"] == "failed"
    assert latest_metrics["error_message"] == "SEC API 返回 429"
    assert latest_metrics["report_generated"] is False
    assert latest_metrics["report_complete"] is False
    assert written_summary["total_runs"] == 2
    assert written_summary["successful_runs"] == 1
    assert written_summary["success_rate"] == 0.5
    assert written_summary["api_calls"]["total"] == 2
    assert written_summary["api_calls"]["failures"] == 1
    assert written_summary["api_calls"]["failure_rate"] == 0.5


def test_workflow_evaluation_start_writes_running_placeholder_metrics(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    final_report = tmp_path / "report.md"
    evaluator = WorkflowEvaluation(
        artifacts_dir=artifacts_dir,
        final_report_path=final_report,
        expected_task_outputs={},
        company_name="Apple Inc.",
        company_ticker="AAPL",
        time_source=StepClock([0.0]),
    )

    evaluator.start()

    latest_metrics = json.loads((artifacts_dir / "latest_run_metrics.json").read_text(encoding="utf-8"))
    assert latest_metrics["status"] == "running"
    assert latest_metrics["success"] is False
    assert latest_metrics["company_name"] == "Apple Inc."
    assert latest_metrics["company_ticker"] == "AAPL"
    assert latest_metrics["report_generated"] is False


def test_workflow_evaluation_remains_stable_across_repeated_runs(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    total_runs = 25
    expected_successes = 0
    expected_failures = 0
    for index in range(total_runs):
        report_path = tmp_path / f"report_{index}.md"
        report_path.write_text("参考 https://example.com/item", encoding="utf-8")
        evaluator = WorkflowEvaluation(
            artifacts_dir=artifacts_dir,
            final_report_path=report_path,
            expected_task_outputs={},
            company_name=f"Company {index}",
            company_ticker=f"T{index}",
            time_source=StepClock([0.0, 0.5]),
        )
        evaluator.start()
        is_success = index % 4 != 0
        api_success = index % 5 != 0
        expected_successes += 1 if is_success else 0
        expected_failures += 0 if api_success else 1
        evaluator.record_api_call(
            service_name="Synthetic API",
            success=api_success,
            status_code=200 if api_success else 500,
        )
        evaluator.finalize(success=is_success, error_message=None if is_success else "synthetic failure")

    written_summary = json.loads((artifacts_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert written_summary["total_runs"] == total_runs
    assert written_summary["successful_runs"] == expected_successes
    assert written_summary["success_rate"] == round(expected_successes / total_runs, 3)
    assert written_summary["api_calls"]["total"] == total_runs
    assert written_summary["api_calls"]["failures"] == expected_failures
    assert written_summary["api_calls"]["failure_rate"] == round(expected_failures / total_runs, 3)


def test_formal_delivery_success_requires_all_semantic_completion_metrics(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "final_decision.json").write_text(
        json.dumps(
            {
                "company_name": "Apple Inc.",
                "company_ticker": "AAPL",
                "final_decision": "passed",
                "final_delivery_state": "formal_report",
                "trust_score": 91,
            }
        ),
        encoding="utf-8",
    )
    report = artifacts_dir / "04_investment_report.md"
    report.write_text("incomplete formal report", encoding="utf-8")
    evaluator = WorkflowEvaluation(
        artifacts_dir=artifacts_dir,
        final_report_path=report,
        expected_task_outputs={},
        company_name="Apple Inc.",
        company_ticker="AAPL",
        time_source=StepClock([0.0, 1.0]),
    )

    evaluator.start()
    metrics = evaluator.finalize(success=True)

    assert metrics["success"] is False
    assert metrics["status"] == "failed"
    assert metrics["report_sections_complete"] is False
    assert metrics["delivery_validation_passed"] is False


def test_formal_fact_provenance_requires_auditable_stock_price_snapshot() -> None:
    common = {
        "unit": "USD",
        "period_end": date(2025, 9, 27),
        "fiscal_year": 2025,
        "fiscal_period": "FY",
        "form": "10-K",
        "accession": "0000320193-25-000079",
        "filed_at": date(2025, 10, 31),
        "source_url": "https://www.sec.gov/Archives/edgar/data/320193/aapl-20250927.htm",
        "source_tag": "sec_companyfacts",
    }
    bundle = ResearchEvidenceBundle(
        company_name="Apple Inc.",
        ticker="AAPL",
        financial_facts=[
            FinancialFact(
                field_name=field_name,
                value=1.0,
                unit="shares" if field_name == "diluted_shares" else "USD",
                **{key: value for key, value in common.items() if key != "unit"},
            )
            for field_name in (
                "revenue",
                "cash_and_equivalents",
                "total_debt",
                "diluted_shares",
                "segment_revenue_services",
            )
        ],
        market_snapshots=[
            MarketSnapshotEvidence(
                price=210.05,
                currency="USD",
                observed_at=datetime(2026, 7, 19, tzinfo=timezone.utc),
                source_url="https://api.nasdaq.com/api/quote/AAPL/info?assetclass=stocks",
                source_tag="nasdaq_quote_info",
                diluted_shares_period_end=date(2025, 9, 27),
            )
        ],
    )

    assert WorkflowEvaluation._formal_fact_provenance_complete(bundle) is True
    assert WorkflowEvaluation._formal_fact_provenance_complete(
        bundle.model_copy(update={"market_snapshots": []})
    ) is False


@pytest.mark.parametrize(
    "field_name",
    ["cash_and_equivalents", "total_debt", "segment_revenue_services"],
)
def test_formal_fact_provenance_rejects_cross_period_required_fact(field_name: str) -> None:
    bundle = _complete_formal_bundle()
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

    assert WorkflowEvaluation._formal_fact_provenance_complete(bundle) is False


def test_formal_fact_provenance_rejects_duplicate_required_fact() -> None:
    bundle = _complete_formal_bundle()
    bundle.financial_facts.append(bundle.require_fact("revenue").model_copy(deep=True))

    assert WorkflowEvaluation._formal_fact_provenance_complete(bundle) is False


def test_formal_fact_provenance_rejects_quote_currency_mismatch() -> None:
    bundle = _complete_formal_bundle()
    bundle.market_snapshots[0] = bundle.market_snapshots[0].model_copy(
        update={"currency": "EUR"}
    )

    assert WorkflowEvaluation._formal_fact_provenance_complete(bundle) is False


def _complete_formal_bundle() -> ResearchEvidenceBundle:
    common = {
        "period_end": date(2025, 9, 27),
        "fiscal_year": 2025,
        "fiscal_period": "FY",
        "form": "10-K",
        "accession": "0000320193-25-000079",
        "filed_at": date(2025, 10, 31),
        "source_url": "https://www.sec.gov/Archives/edgar/data/320193/aapl-20250927.htm",
        "source_tag": "sec_companyfacts",
    }
    return ResearchEvidenceBundle(
        company_name="Apple Inc.",
        ticker="AAPL",
        financial_facts=[
            FinancialFact(
                field_name=field_name,
                value=1.0,
                unit="shares" if field_name == "diluted_shares" else "USD",
                **common,
            )
            for field_name in (
                "revenue",
                "cash_and_equivalents",
                "total_debt",
                "diluted_shares",
                "segment_revenue_services",
            )
        ],
        market_snapshots=[
            MarketSnapshotEvidence(
                price=210.05,
                currency="USD",
                observed_at=datetime(2026, 7, 19, tzinfo=timezone.utc),
                source_url="https://api.nasdaq.com/api/quote/AAPL/info?assetclass=stocks",
                source_tag="nasdaq_quote_info",
                diluted_shares_period_end=date(2025, 9, 27),
            )
        ],
    )
