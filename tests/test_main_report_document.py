from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from multi_agent import main
from multi_agent.core.evidence import FinancialFact, ResearchEvidenceBundle
from multi_agent.core.report_document import (
    ReportGenerationContext,
    render_recommendation,
    render_structured_report,
)
from multi_agent.core.review_contracts import (
    CoverageSummary,
    DeliveryEligibility,
    FailureTaxonomy,
    ReviewContract,
)


def _context(report_mode: str = "formal_report") -> ReportGenerationContext:
    evidence = ResearchEvidenceBundle(
        company_name="Apple Inc.",
        ticker="AAPL",
        financial_facts=[
            FinancialFact(
                field_name="revenue",
                value=416_161_000_000,
                unit="USD",
                period_start=date(2024, 9, 29),
                period_end=date(2025, 9, 27),
                fiscal_year=2025,
                fiscal_period="FY",
                form="10-K",
                accession="0000320193-25-000079",
                filed_at=date(2025, 10, 31),
                source_url="https://www.sec.gov/Archives/edgar/data/320193/aapl-20250927.htm",
                source_tag="sec_companyfacts",
            )
        ],
    )
    eligibility = {
        "formal_report": DeliveryEligibility(formal_report_allowed=True),
        "evidence_limited_report": DeliveryEligibility(
            evidence_limited_report_allowed=True
        ),
        "blocked_notice": DeliveryEligibility(blocked_notice_required=True),
    }[report_mode]
    review = ReviewContract(
        stage="report",
        delivery_eligibility=eligibility,
        failure_taxonomy=FailureTaxonomy(primary_class="none"),
        coverage_summary=CoverageSummary(
            evidence_coverage_ratio=1.0,
            financial_coverage_score=1.0,
            claim_binding_ratio=1.0,
        ),
    )
    return ReportGenerationContext(
        company_name="Apple Inc.",
        ticker="AAPL",
        report_mode=report_mode,
        evidence_bundle=evidence,
        analysis_review_contract=review,
        allowed_claim_ids=["revenue-claim"],
    )


def _writer_payload(report_mode: str = "formal_report") -> dict[str, object]:
    stance, summary, conclusion = {
        "formal_report": (
            "hold",
            "正式证据支持继续跟踪 Apple 的服务业务。",
            "维持持有。",
        ),
        "evidence_limited_report": (
            "watch",
            "证据受限，继续观察。",
            "证据受限，待补证后复核。",
        ),
        "blocked_notice": (
            "blocked",
            "报告已阻断，不能形成投资结论。",
            "报告已阻断，不提供可执行投资建议。",
        ),
    }[report_mode]
    return {
        "title": "Apple Inc. 投资研究报告",
        "stance": stance,
        "executive_summary": summary,
        "catalysts": ["服务业务增长"],
        "risks": ["需求波动"],
        "sections": {
            "executive_summary": {"heading": "执行摘要", "content": "证据充足。", "claim_ids": ["revenue-claim"]},
            "business_overview": {"heading": "公司与业务概览", "content": "生态系统稳定。", "claim_ids": []},
            "recent_events": {"heading": "近期事件与催化剂", "content": "服务增长。", "claim_ids": []},
            "financial_analysis": {"heading": "财务分析与估值", "content": "收入保持增长。", "claim_ids": ["revenue-claim"]},
            "key_risks": {"heading": "关键风险", "content": "需求存在波动。", "claim_ids": []},
            "investment_conclusion": {"heading": "投资结论", "content": conclusion, "claim_ids": ["revenue-claim"]},
            "source_index": {"heading": "来源索引", "content": "SEC 年报。", "claim_ids": []},
        },
        "claims": [{"claim_id": "revenue-claim", "text": "FY2025 收入为 416.2 十亿美元。", "critical": True, "source_ids": ["sec-10k"]}],
        "sources": [{"source_id": "sec-10k", "title": "Apple 2025 Form 10-K", "url": "https://www.sec.gov/Archives/edgar/data/320193/aapl-20250927.htm", "source_tag": "sec_filing", "field_name": "revenue"}],
    }


def _output_paths(tmp_path: Path) -> main.RunOutputPaths:
    return main._build_run_output_paths(
        base_artifacts_dir=tmp_path,
        company_name="Apple Inc.",
        company_ticker="AAPL",
        run_time=datetime(2026, 7, 20, 10, 30, 45),
    )


def test_new_run_materializes_canonical_document_before_any_projection(tmp_path: Path) -> None:
    output_paths = _output_paths(tmp_path)
    output_paths.run_dir.mkdir(parents=True)
    result = {
        "report_context": _context().model_dump(mode="json"),
        "report_writer_payload": _writer_payload(),
        "trust_score": 91,
    }

    document = main._materialize_new_run_report_document(
        output_paths, result=result, final_status="passed"
    )

    assert document.company_name == "Apple Inc."
    assert output_paths.report_document_path.exists()
    assert "## 财务分析与估值" in output_paths.final_report_path.read_text(encoding="utf-8")
    assert json.loads(output_paths.report_document_path.read_text(encoding="utf-8"))["stance"] == "hold"


@pytest.mark.parametrize(
    ("report_mode", "final_status"),
    (
        ("formal_report", "passed"),
        ("evidence_limited_report", "evidence_limited"),
        ("blocked_notice", "blocked"),
    ),
)
def test_new_run_structured_outputs_equal_direct_typed_renderers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_mode: str,
    final_status: str,
) -> None:
    output_paths = _output_paths(tmp_path)
    output_paths.run_dir.mkdir(parents=True)
    result = {
        "report_context": _context(report_mode).model_dump(mode="json"),
        "report_writer_payload": _writer_payload(report_mode),
        "trust_score": 91,
    }
    document = main._materialize_new_run_report_document(
        output_paths, result=result, final_status=final_status
    )
    monkeypatch.setattr(
        main,
        "build_structured_recommendation",
        lambda **_: (_ for _ in ()).throw(AssertionError("legacy parser must not run")),
    )
    monkeypatch.setattr(
        main,
        "build_structured_report",
        lambda **_: (_ for _ in ()).throw(AssertionError("legacy parser must not run")),
    )

    recommendation = main._write_structured_outputs(
        output_paths,
        final_decision={
            "final_decision": final_status,
            "final_delivery_state": report_mode,
        },
        latest_metrics={},
        company_name="Apple Inc.",
        company_ticker="AAPL",
        watchlist_path=tmp_path / "watchlist.json",
        save_to_watchlist=False,
    )

    structured_report = json.loads(output_paths.structured_report_path.read_text(encoding="utf-8"))
    assert recommendation == render_recommendation(document)
    assert structured_report == render_structured_report(document)


def test_new_run_refuses_direct_document_or_missing_writer_payload(tmp_path: Path) -> None:
    output_paths = _output_paths(tmp_path)
    output_paths.run_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="report_context and report_writer_payload"):
        main._materialize_new_run_report_document(
            output_paths,
            result={"report_document": {"untrusted": True}, "trust_score": 91},
            final_status="passed",
        )


def test_success_finalization_never_overwrites_canonical_report_for_blocked_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_paths = _output_paths(tmp_path)
    output_paths.run_dir.mkdir(parents=True)
    monkeypatch.setattr(
        main,
        "_write_blocked_report",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not overwrite canonical report")),
    )
    monkeypatch.setattr(main, "_validate_successful_outputs", lambda *_args, **_kwargs: None)

    assert main._finalize_successful_result(output_paths, {"status": "blocked"}) == "blocked"
