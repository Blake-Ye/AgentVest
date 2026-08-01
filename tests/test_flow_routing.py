import json
import sys
from pathlib import Path
from typing import Callable

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent.core.confidence_gate import ConfidenceGatePolicy
from multi_agent.core.evidence import (
    FinancialFact,
    MarketSnapshotEvidence,
    ResearchEvidenceBundle,
    ToolHealthRecord,
)
from multi_agent.core.market import MarketValidationResult, build_tool_policy
from multi_agent.core.model_routing import ModelRouter
from multi_agent.core.review_contracts import (
    GateDecision,
    RepairAction,
    ReviewContract,
    review_contract_from_text,
    validate_analysis_review_output,
    validate_report_review_output,
)
from multi_agent.core.state import EvidenceItem, ResearchRunState
from multi_agent.evaluation import WorkflowEvaluation, activate_evaluation, clear_evaluation
from multi_agent.flows.market_review_flow import MarketReviewFlow, MarketReviewFlowState
from multi_agent.settings import InvestmentResearchSettings


def _typed_bundle() -> ResearchEvidenceBundle:
    """A self-contained formal-eligible bundle for Flow control-plane tests."""
    from datetime import date, datetime, timezone

    common = {
        "period_end": date(2025, 9, 27),
        "fiscal_year": 2025,
        "fiscal_period": "FY",
        "form": "10-K",
        "accession": "0000320193-25-000079",
        "filed_at": date(2025, 10, 31),
        "source_url": "https://www.sec.gov/Archives/edgar/data/320193/example.htm",
        "source_tag": "sec_companyfacts",
    }
    return ResearchEvidenceBundle(
        company_name="Apple Inc.",
        ticker="AAPL",
        financial_facts=[
            FinancialFact(
                field_name=name,
                value=float(index + 1),
                unit="shares" if name == "diluted_shares" else "USD",
                **common,
            )
            for index, name in enumerate(
                (
                    "revenue",
                    "cash_and_equivalents",
                    "total_debt",
                    "diluted_shares",
                    "segment_revenue_services",
                )
            )
        ],
        market_snapshots=[
            MarketSnapshotEvidence(
                price=200.0,
                currency="USD",
                observed_at=datetime(2025, 10, 31, tzinfo=timezone.utc),
                source_url="https://example.com/quote",
                source_tag="quote",
                diluted_shares_period_end=date(2025, 9, 27),
            )
        ],
        tool_health=[ToolHealthRecord(tool_name="sec_company_facts", status="healthy")],
    )


def _typed_contract(*, outcome: str = "pass", actions: list[dict[str, object]] | None = None) -> ReviewContract:
    return ReviewContract.model_validate(
        {
            "stage": "analysis_review",
            "reviewer_name": "data_quality_reviewer",
            "decision": {"gate_outcome": outcome, "decision_confidence": "high"},
            "delivery_eligibility": {
                "formal_report_allowed": outcome == "pass",
                "evidence_limited_report_allowed": False,
                "blocked_notice_required": outcome == "block",
                "recommended_delivery_state": "formal_report" if outcome == "pass" else "blocked_notice",
            },
            "failure_taxonomy": {"primary_class": "none", "secondary_causes": []},
            "coverage_summary": {
                "evidence_coverage_ratio": 1.0,
                "financial_coverage_score": 1.0,
                "claim_binding_ratio": 1.0,
            },
            "tool_health_summary": {"overall_status": "healthy"},
            "repair_actions": actions or [],
        }
    )


def _typed_writer_payload() -> dict[str, object]:
    from multi_agent.core.report_document import REQUIRED_SECTION_KEYS, SECTION_HEADINGS

    source = {
        "source_id": "claim:revenue",
        "title": "Apple Inc. 10-K",
        "url": "https://www.sec.gov/Archives/edgar/data/320193/example.htm",
        "source_tag": "sec_companyfacts",
        "field_name": "revenue",
    }
    claim = {
        "claim_id": "claim:revenue",
        "text": "收入事实来自 2025 年 10-K。",
        "critical": True,
        "source_ids": ["claim:revenue"],
    }
    return {
        "title": "Apple Inc. (AAPL) 投资备忘录",
        "stance": "hold",
        "executive_summary": "基于已验证的财务证据维持持有观点。",
        "catalysts": ["经验证的收入增长。"],
        "risks": ["市场价格可能波动。"],
        "sections": {
            key: {
                "heading": SECTION_HEADINGS[key],
                "content": f"{SECTION_HEADINGS[key]}的已验证内容。",
                "claim_ids": ["claim:revenue"] if key != "source_index" else [],
            }
            for key in REQUIRED_SECTION_KEYS
        },
        "claims": [claim],
        "sources": [source],
    }


def _typed_report_contract(*, outcome: str = "pass") -> ReviewContract:
    payload = _typed_contract(outcome=outcome).model_dump(mode="json")
    payload["stage"] = "report_review"
    payload["reviewer_name"] = "logic_compliance_reviewer"
    return ReviewContract.model_validate(payload)


class _StubTaskOutput:
    def __init__(self, name: str, raw: str) -> None:
        self.name = name
        self.raw = raw


def build_settings() -> InvestmentResearchSettings:
    return InvestmentResearchSettings(
        fast_model="qwen-v4-flash",
        deep_model="qwen-v4-pro",
        review_model="qwen-v4-review",
        company_resolver_model="qwen-v4-flash",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        tavily_api_key="tvly-key",
        sec_api_email="analyst@example.com",
    )


def test_review_contract_accepts_analysis_and_report_stage_payloads() -> None:
    analysis_contract = ReviewContract(
        stage="analysis",
        reviewer_name="data_quality_reviewer",
        delivery_eligibility={
            "formal_report_allowed": False,
            "evidence_limited_report_allowed": True,
            "blocked_notice_required": False,
        },
        failure_taxonomy={
            "primary_code": "coverage_gap",
            "secondary_codes": ["summary_missing"],
        },
        coverage_summary={
            "evidence_coverage_ratio": 1.0,
            "financial_coverage_score": 0.8,
            "gate_financial_coverage_score": 0.5,
        },
        tool_health_summary={
            "overall_status": "degraded",
            "failed_tools": [],
            "degraded_tools": ["financial_field_completeness_tool"],
        },
    )

    decision = ConfidenceGatePolicy.default().evaluate_legacy(analysis_contract)

    assert analysis_contract.stage == "analysis"
    assert decision.final_decision == "rerun"
    assert "gate_financial_coverage_incomplete" in decision.blocking_reasons

    report_contract = ReviewContract(
        stage="report",
        reviewer_name="logic_compliance_reviewer",
        delivery_eligibility={
            "formal_report_allowed": False,
            "evidence_limited_report_allowed": True,
            "blocked_notice_required": False,
        },
        failure_taxonomy={
            "primary_code": "report_inconsistency",
            "secondary_codes": ["delivery_blocked"],
        },
        coverage_summary={
            "evidence_coverage_ratio": 1.0,
            "financial_coverage_score": 1.0,
            "gate_financial_coverage_score": 1.0,
        },
        tool_health_summary={
            "overall_status": "failed",
            "failed_tools": ["consistency_checker"],
            "degraded_tools": [],
        },
    )

    assert report_contract.stage == "report"
    assert report_contract.failure_taxonomy.primary_code == "report_inconsistency"


def test_review_contract_rejects_unknown_failure_taxonomy() -> None:
    with pytest.raises(ValidationError, match="primary_code"):
        ReviewContract(
            stage="analysis",
            reviewer_name="data_quality_reviewer",
            delivery_eligibility={
                "formal_report_allowed": False,
                "evidence_limited_report_allowed": True,
                "blocked_notice_required": False,
            },
            failure_taxonomy={
                "primary_code": "hallucination",
                "secondary_codes": [],
            },
        )


def test_model_router_returns_expected_model_by_tier() -> None:
    router = ModelRouter(settings=build_settings())

    assert router.for_tier("fast") == "qwen-v4-flash"
    assert router.for_tier("deep") == "qwen-v4-pro"
    assert router.for_tier("review") == "qwen-v4-review"


def test_model_router_uses_default_tiers_and_allows_overrides() -> None:
    router = ModelRouter(settings=build_settings())

    assert router.default_tier_for_agent("event_guidance_analyst") == "fast"
    assert router.default_tier_for_agent("report_writing_analyst") == "deep"
    assert router.default_tier_for_agent("logic_compliance_reviewer") == "review"
    assert router.for_agent("event_guidance_analyst") == "qwen-v4-flash"
    assert (
        router.for_agent(
            "event_guidance_analyst",
            overrides={"event_guidance_analyst": "deep"},
        )
        == "qwen-v4-pro"
    )


def test_model_router_rejects_unknown_agent() -> None:
    router = ModelRouter(settings=build_settings())

    with pytest.raises(ValueError, match="Unknown agent for model routing"):
        router.for_agent("unknown_agent")


def test_model_router_rejects_unknown_override_agent() -> None:
    router = ModelRouter(settings=build_settings())

    with pytest.raises(ValueError, match="Unknown agent in model tier overrides"):
        router.for_agent(
            "event_guidance_analyst",
            overrides={"unknown_agent": "deep"},
        )


def test_model_router_rejects_unknown_override_tier() -> None:
    router = ModelRouter(settings=build_settings())

    with pytest.raises(ValueError, match="Unknown model tier override"):
        router.for_agent(
            "event_guidance_analyst",
            overrides={"event_guidance_analyst": "turbo"},
        )


def test_research_run_state_tracks_market_validation_and_evidence() -> None:
    state = ResearchRunState(
        request_id="run-001",
        company_name="Alibaba Group Holding Ltd",
        input_ticker="BABA",
        input_exchange="NYSE",
        market_validation=MarketValidationResult(
            market_label="US",
            confidence=0.99,
            resolution_status="confirmed",
            evidence=["exchange=NYSE"],
            requires_human_confirmation=False,
            tool_policy=build_tool_policy("US"),
        ),
        evidence_ledger=[
            EvidenceItem(
                source_id="news-1",
                source_type="tavily",
                title="Quarterly update",
                url="https://example.com/news",
                summary="Revenue growth re-accelerated.",
            )
        ],
    )

    dumped = state.model_dump()

    assert dumped["market_validation"]["market_label"] == "US"
    assert dumped["evidence_ledger"][0]["source_id"] == "news-1"
    assert dumped["evidence_ledger"][0]["source_type"] == "tavily"


def test_research_run_state_tracks_gate_and_rerun_budget() -> None:
    state = ResearchRunState(
        request_id="run-001",
        company_name="Microsoft Corporation",
        rerun_budget={"event_guidance": 1},
        gate_decision=GateDecision(
            passed=False,
            final_decision="blocked",
            trust_score=68,
            blocking_reasons=["evidence_coverage_ratio<0.80"],
        ),
        final_decision="blocked",
    )

    dumped = state.model_dump()

    assert dumped["gate_decision"]["final_decision"] == "blocked"
    assert dumped["rerun_budget"]["event_guidance"] == 1
    assert dumped["final_decision"] == "blocked"


def test_research_run_state_rejects_invalid_control_plane_values() -> None:
    with pytest.raises(ValidationError):
        ResearchRunState(
            request_id="run-001",
            company_name="Microsoft Corporation",
            model_tier_overrides={"event_guidance_analyst": "turbo"},
        )

    with pytest.raises(ValidationError):
        ResearchRunState(
            request_id="run-001",
            company_name="Microsoft Corporation",
            final_decision="approved",
        )


def test_market_review_flow_state_extends_research_run_state_control_plane() -> None:
    state = MarketReviewFlowState(
        request_id="run-001",
        company_name="Alibaba Group Holding Ltd",
        input_ticker="BABA",
        market_validation=MarketValidationResult(
            market_label="US",
            confidence=0.99,
            resolution_status="confirmed",
            evidence=["exchange=NYSE"],
            requires_human_confirmation=False,
            tool_policy=build_tool_policy("US"),
        ),
        rerun_budget={"analysis": 1},
        model_tier_overrides={"event_guidance_analyst": "deep"},
    )

    assert isinstance(state, ResearchRunState)
    dumped = state.model_dump()
    assert dumped["input_ticker"] == "BABA"
    assert dumped["market_validation"]["market_label"] == "US"
    assert dumped["rerun_budget"]["analysis"] == 1
    assert dumped["model_tier_overrides"]["event_guidance_analyst"] == "deep"


def test_market_review_flow_state_rejects_invalid_final_decision() -> None:
    with pytest.raises(ValidationError):
        MarketReviewFlowState(
            request_id="run-001",
            company_name="Alibaba Group Holding Ltd",
            final_decision="approved",
        )


def test_market_review_flow_routes_through_explicit_review_stages_on_pass() -> None:
    events: list[str] = []

    def analysis_executor(inputs: dict[str, object]) -> dict[str, str]:
        assert inputs["company_name"] == "Alibaba Group Holding Ltd"
        assert inputs["company_ticker"] == "BABA"
        assert inputs["company_market_label"] == "US"
        assert inputs["model_tier_overrides"] == {}
        events.append("run_analysis")
        return {"analysis_markdown": "analysis complete"}

    def analysis_gate(result: dict[str, str]) -> GateDecision:
        assert result["analysis_markdown"] == "analysis complete"
        events.append("apply_analysis_gate")
        return GateDecision(
            passed=True,
            final_decision="passed",
            trust_score=82,
        )

    def report_writer(result: dict[str, str]) -> str:
        assert result["analysis_markdown"] == "analysis complete"
        events.append("write_report")
        return "# investment report"

    def report_reviewer(report: str) -> GateDecision:
        assert report == "# investment report"
        events.append("review_report")
        return GateDecision(
            passed=True,
            final_decision="passed",
            trust_score=88,
        )

    state = MarketReviewFlowState(
        request_id="run-001",
        company_name="Alibaba Group Holding Ltd",
        input_ticker="BABA",
        market_validation=MarketValidationResult(
            market_label="US",
            confidence=0.99,
            resolution_status="confirmed",
            evidence=["exchange=NYSE"],
            requires_human_confirmation=False,
            tool_policy=build_tool_policy("US"),
        ),
    )
    flow = MarketReviewFlow(
        analysis_executor=analysis_executor,
        analysis_gate=analysis_gate,
        report_writer=report_writer,
        report_reviewer=report_reviewer,
        initial_state=state,
    )

    result = flow.kickoff()

    assert result == {
        "status": "passed",
        "trust_score": 88,
        "blocking_reasons": [],
        "analysis_result": {"analysis_markdown": "analysis complete"},
        "report_result": "# investment report",
    }
    assert flow.state.final_result == result
    assert flow.state.final_decision == "passed"
    assert flow.state.gate_decision is not None
    assert flow.state.gate_decision.final_decision == "passed"
    assert flow.state.input_ticker == "BABA"
    assert flow.state.stage_history == [
        "validate_market",
        "run_analysis",
        "review_analysis",
        "apply_analysis_gate",
        "analysis_passed",
        "write_report",
        "review_report",
        "finalize_delivery",
    ]
    assert events == [
        "run_analysis",
        "apply_analysis_gate",
        "write_report",
        "review_report",
    ]
    assert flow.state.market_validation is not None
    assert flow.state.market_validation.market_label == "US"


def test_flow_reruns_analysis_once_with_model_tier_overrides_before_blocking() -> None:
    attempts: list[dict[str, object]] = []

    def analysis_executor(inputs: dict[str, object]) -> dict[str, str]:
        attempts.append(dict(inputs))
        return {"analysis_markdown": f"analysis attempt {len(attempts)}"}

    def analysis_gate(_: dict[str, str]) -> GateDecision:
        return GateDecision(
            passed=False,
            final_decision="rerun",
            trust_score=68,
            blocking_reasons=["evidence_coverage_ratio<0.80"],
        )

    def report_writer(_: dict[str, str]) -> str:
        raise AssertionError("blocked path must not write report")

    def report_reviewer(_: str) -> GateDecision:
        raise AssertionError("blocked path must not review report")

    flow = MarketReviewFlow(
        analysis_executor=analysis_executor,
        analysis_gate=analysis_gate,
        report_writer=report_writer,
        report_reviewer=report_reviewer,
        initial_state=MarketReviewFlowState(
            request_id="run-002",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            market_validation=MarketValidationResult(
                market_label="US",
                confidence=0.99,
                resolution_status="confirmed",
                evidence=["exchange=NASDAQ"],
                requires_human_confirmation=False,
                tool_policy=build_tool_policy("US"),
            ),
            rerun_budget={"analysis": 1},
        ),
    )

    result = flow.kickoff()

    assert flow.state.final_decision == "blocked"
    assert flow.state.final_result == result
    assert len(attempts) == 2
    assert attempts[0]["model_tier_overrides"] == {}
    assert attempts[1]["model_tier_overrides"] == {
        "event_guidance_analyst": "deep",
        "fundamental_analyst": "deep",
        "quant_valuation_analyst": "deep",
    }
    assert flow.state.model_tier_overrides == {
        "event_guidance_analyst": "deep",
        "fundamental_analyst": "deep",
        "quant_valuation_analyst": "deep",
    }
    assert flow.state.rerun_budget["analysis"] == 0
    assert flow.state.stage_history == [
        "validate_market",
        "run_analysis",
        "review_analysis",
        "apply_analysis_gate",
        "analysis_needs_rerun",
        "rerun_analysis_if_needed",
        "review_analysis",
        "apply_analysis_gate",
        "analysis_blocked",
        "finalize_delivery",
    ]
    assert result == {
        "status": "blocked",
        "trust_score": 68,
        "blocking_reasons": ["evidence_coverage_ratio<0.80"],
        "analysis_result": {"analysis_markdown": "analysis attempt 2"},
        "report_result": None,
    }
    assert flow.state.gate_decision is not None
    assert flow.state.gate_decision.final_decision == "blocked"


def test_flow_routes_fixable_gate_result_to_rerun_then_passes() -> None:
    attempts: list[dict[str, object]] = []

    def analysis_executor(inputs: dict[str, object]) -> dict[str, str]:
        attempts.append(dict(inputs))
        return {"analysis_markdown": f"analysis attempt {len(attempts)}"}

    decisions = iter(
        [
            GateDecision(
                passed=False,
                final_decision="rerun",
                trust_score=70,
                blocking_reasons=["evidence_coverage_ratio<0.80"],
            ),
            GateDecision(
                passed=True,
                final_decision="passed",
                trust_score=88,
            ),
        ]
    )

    flow = MarketReviewFlow(
        analysis_executor=analysis_executor,
        analysis_gate=lambda _result: next(decisions),
        report_writer=lambda _result: "# investment report",
        report_reviewer=lambda _report: GateDecision(
            passed=True,
            final_decision="passed",
            trust_score=90,
        ),
        initial_state=MarketReviewFlowState(
            request_id="run-002b",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            rerun_budget={"analysis": 1},
        ),
    )

    result = flow.kickoff()

    assert len(attempts) == 2
    assert result["status"] == "passed"
    assert result["report_result"] == "# investment report"
    assert flow.state.rerun_budget["analysis"] == 0
    assert flow.state.stage_history == [
        "validate_market",
        "run_analysis",
        "review_analysis",
        "apply_analysis_gate",
        "analysis_needs_rerun",
        "rerun_analysis_if_needed",
        "review_analysis",
        "apply_analysis_gate",
        "analysis_passed",
        "write_report",
        "review_report",
        "finalize_delivery",
    ]


def test_flow_blocks_when_report_gate_fails() -> None:
    def analysis_executor(_: dict[str, object]) -> dict[str, str]:
        return {"analysis_markdown": "analysis complete"}

    def analysis_gate(_: dict[str, str]) -> GateDecision:
        return GateDecision(
            passed=True,
            final_decision="passed",
            trust_score=84,
        )

    def report_writer(_: dict[str, str]) -> str:
        return "# investment report"

    def report_reviewer(_: str) -> GateDecision:
        return GateDecision(
            passed=False,
            final_decision="blocked",
            trust_score=61,
            blocking_reasons=["missing_risk_disclosure"],
        )

    flow = MarketReviewFlow(
        analysis_executor=analysis_executor,
        analysis_gate=analysis_gate,
        report_writer=report_writer,
        report_reviewer=report_reviewer,
        initial_state=MarketReviewFlowState(
            request_id="run-003",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            market_validation=MarketValidationResult(
                market_label="US",
                confidence=0.99,
                resolution_status="confirmed",
                evidence=["exchange=NASDAQ"],
                requires_human_confirmation=False,
                tool_policy=build_tool_policy("US"),
            ),
        ),
    )

    result = flow.kickoff()

    assert result == {
        "status": "blocked",
        "trust_score": 61,
        "blocking_reasons": ["missing_risk_disclosure"],
        "analysis_result": {"analysis_markdown": "analysis complete"},
        "report_result": "# investment report",
    }
    assert flow.state.final_result == result
    assert flow.state.final_decision == "blocked"
    assert flow.state.stage_history == [
        "validate_market",
        "run_analysis",
        "review_analysis",
        "apply_analysis_gate",
        "analysis_passed",
        "write_report",
        "review_report",
        "finalize_delivery",
    ]
    assert flow.state.gate_decision is not None
    assert flow.state.gate_decision.final_decision == "blocked"


def test_flow_rejects_inconsistent_gate_decision_in_analysis_review() -> None:
    def analysis_executor(_: dict[str, object]) -> dict[str, str]:
        return {"analysis_markdown": "analysis complete"}

    def analysis_gate(_: dict[str, str]) -> GateDecision:
        return GateDecision(
            passed=True,
            final_decision="rerun",
            trust_score=42,
            blocking_reasons=["should_not_continue"],
        )

    flow = MarketReviewFlow(
        analysis_executor=analysis_executor,
        analysis_gate=analysis_gate,
        initial_state=MarketReviewFlowState(
            request_id="run-004",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            market_validation=MarketValidationResult(
                market_label="US",
                confidence=0.99,
                resolution_status="confirmed",
                evidence=["exchange=NASDAQ"],
                requires_human_confirmation=False,
                tool_policy=build_tool_policy("US"),
            ),
        ),
    )

    with pytest.raises(ValueError, match="Inconsistent gate decision"):
        flow.kickoff()


def test_default_flow_report_gate_blocks_when_materialized_report_is_marked_blocked(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "04_investment_report.md"
    report_path.write_text(
        "# Microsoft Corporation (MSFT) 投资备忘录\n\n"
        "**备忘录状态：** ⛔ **阻断——不得用于正式投资建议**\n",
        encoding="utf-8",
    )

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        analysis_gate=lambda _result: GateDecision(
            passed=True,
            final_decision="passed",
            trust_score=90,
        ),
        initial_state=MarketReviewFlowState(
            request_id="run-007",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            final_report_path=str(report_path),
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "blocked"
    assert flow.state.final_decision == "blocked"
    assert "report_marked_blocked" in result["blocking_reasons"]


def test_default_flow_report_gate_blocks_when_report_uses_blocked_status_heading(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "04_investment_report.md"
    report_path.write_text(
        "# Microsoft Corporation (MSFT) 投资备忘录\n\n"
        "## ⚠️ 分析状态：阻断（Blocked）\n",
        encoding="utf-8",
    )

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        analysis_gate=lambda _result: GateDecision(
            passed=True,
            final_decision="passed",
            trust_score=90,
        ),
        initial_state=MarketReviewFlowState(
            request_id="run-007b",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            final_report_path=str(report_path),
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "blocked"
    assert flow.state.final_decision == "blocked"
    assert "report_marked_blocked" in result["blocking_reasons"]


def test_default_flow_report_gate_blocks_when_report_uses_restricted_memo_status(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "04_investment_report.md"
    report_path.write_text(
        "# Microsoft Corporation (MSFT) 投资备忘录\n\n"
        "**状态**：⚠️ 受限版备忘录 — Analysis Gate 未完全通过\n",
        encoding="utf-8",
    )

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        analysis_gate=lambda _result: GateDecision(
            passed=True,
            final_decision="passed",
            trust_score=90,
        ),
        initial_state=MarketReviewFlowState(
            request_id="run-007c",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            final_report_path=str(report_path),
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "evidence_limited"
    assert flow.state.final_decision == "evidence_limited"
    assert result["blocking_reasons"] == []


def test_flow_routes_gate_coverage_incomplete_to_evidence_limited_when_rerun_budget_is_exhausted() -> None:
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        analysis_gate=lambda _result: GateDecision(
            passed=False,
            final_decision="rerun",
            trust_score=74,
            blocking_reasons=["gate_financial_coverage_incomplete"],
        ),
        report_writer=lambda _result: "# 投资备忘录\n\n**状态**：⚠️ 受限版备忘录 — Analysis Gate 未完全通过\n",
        report_reviewer=lambda report: GateDecision(
            passed=False,
            final_decision="evidence_limited" if "受限版备忘录" in report else "passed",
            trust_score=74,
        ),
        initial_state=MarketReviewFlowState(
            request_id="run-007d",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            rerun_budget={"analysis": 0},
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "evidence_limited"
    assert result["report_result"].startswith("# 投资备忘录")
    assert flow.state.final_decision == "evidence_limited"
    assert result["blocking_reasons"] == ["gate_financial_coverage_incomplete"]


def test_flow_does_not_coerce_mixed_rerun_reasons_to_evidence_limited_when_rerun_budget_is_exhausted() -> None:
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        analysis_gate=lambda _result: GateDecision(
            passed=False,
            final_decision="rerun",
            trust_score=58,
            blocking_reasons=[
                "gate_financial_coverage_incomplete",
                "trust_score<60",
            ],
        ),
        initial_state=MarketReviewFlowState(
            request_id="run-007e",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            rerun_budget={"analysis": 0},
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "blocked"
    assert result["report_result"] is None
    assert result["blocking_reasons"] == [
        "gate_financial_coverage_incomplete",
        "trust_score<60",
    ]


def test_default_flow_analysis_gate_blocks_when_materialized_review_is_marked_blocked(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "08_data_quality_review.md"
    review_path.write_text(
        "# Microsoft Corporation（MSFT）数据质量审查报告\n\n"
        "## 二、必须修复（阻断级，共 6 项）\n\n"
        "- M-1：关键证据链断裂\n",
        encoding="utf-8",
    )

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-008",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            artifacts_dir=str(tmp_path),
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "blocked"
    assert result["report_result"] is None
    assert flow.state.final_decision == "blocked"
    assert "analysis_review_marked_blocked" in result["blocking_reasons"]


def test_default_flow_analysis_gate_blocks_when_review_uses_blockers_heading(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "08_data_quality_review.md"
    review_path.write_text(
        "# Microsoft Corporation（MSFT）数据质量审查报告\n\n"
        "## 一、必须修复（Blockers）\n\n"
        "### 🔴 B-1：核心财务事实来源链断裂\n",
        encoding="utf-8",
    )

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-008b",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            artifacts_dir=str(tmp_path),
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "blocked"
    assert result["report_result"] is None
    assert "analysis_review_marked_blocked" in result["blocking_reasons"]


def test_default_flow_analysis_gate_blocks_when_review_uses_must_fix_and_blocking_risk(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "08_data_quality_review.md"
    review_path.write_text(
        "# Microsoft Corporation（MSFT）数据质量审查报告\n\n"
        "### 🔴 必须修复（Must Fix）\n\n"
        "| Gate 风险 | **阻断级**——估值结论无法放行 |\n",
        encoding="utf-8",
    )

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-008c",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            artifacts_dir=str(tmp_path),
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "blocked"
    assert result["report_result"] is None
    assert "analysis_review_marked_blocked" in result["blocking_reasons"]


def test_default_flow_analysis_gate_blocks_when_review_uses_gate_risk_language(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "08_data_quality_review.md"
    review_path.write_text(
        "# Microsoft Corporation（MSFT）数据质量审查报告\n\n"
        "## 二、🔴 必须修复（Gate 阻断风险）\n\n"
        "| Gate 风险 | ⚠️ **阻断**：核心估值输入不可靠 |\n",
        encoding="utf-8",
    )

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-009",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            artifacts_dir=str(tmp_path),
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "blocked"
    assert result["report_result"] is None
    assert "analysis_review_marked_blocked" in result["blocking_reasons"]


def test_default_flow_analysis_gate_prefers_markdown_summary_over_blocker_prose_when_both_exist(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "08_data_quality_review.md"
    review_path.write_text(
        "# Microsoft Corporation（MSFT）数据质量审查报告\n\n"
        "## 一、审查工具结构化结果摘要\n\n"
        "| 工具 | 关键结果 | 解读 |\n"
        "|---|---|---|\n"
        "| **evidence_coverage_tool** | 覆盖率 **1.00**（18/18 claims 均有引用），无 unsupported claims | 所有结论均有引用 |\n"
        "| **cross_source_consistency_tool** | **0** critical conflicts | 无关键冲突 |\n"
        "| **market_tool_policy_audit_tool** | **0** violations | 无越界 |\n"
        "| **financial_field_completeness_tool** | 整体覆盖率 **45.45%**；Gate 覆盖率 **76.92%** | 仍有关键字段缺口 |\n\n"
        "## 二、审查发现\n\n"
        "| Gate风险 | BLOCK - 核心估值输入不可靠 |\n",
        encoding="utf-8",
    )

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-009b",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            artifacts_dir=str(tmp_path),
        ),
    )

    decision = flow._default_analysis_gate({"analysis_markdown": "analysis complete"})

    assert decision.final_decision == "rerun"
    assert "gate_financial_coverage_incomplete" in decision.blocking_reasons
    assert "analysis_review_marked_blocked" not in decision.blocking_reasons


def test_default_flow_analysis_gate_blocks_when_review_uses_p0_language(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "08_data_quality_review.md"
    review_path.write_text(
        "# Microsoft Corporation（MSFT）数据质量审查报告\n\n"
        "## 🔴 必须修复（P0）\n\n"
        "### P0-1：核心财务基础缺失\n",
        encoding="utf-8",
    )

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-010",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            artifacts_dir=str(tmp_path),
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "blocked"
    assert result["report_result"] is None
    assert "analysis_review_marked_blocked" in result["blocking_reasons"]


def test_default_flow_analysis_gate_does_not_block_when_review_marks_issue_as_non_blocking(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "08_data_quality_review.md"
    review_path.write_text(
        "# Microsoft Corporation（MSFT）数据质量审查报告\n\n"
        "## 二、审查发现\n\n"
        "| Gate 风险 | 非阻断级：该问题仅影响补充说明 |\n",
        encoding="utf-8",
    )

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-010b",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            artifacts_dir=str(tmp_path),
        ),
    )

    decision = flow._default_analysis_gate({"analysis_markdown": "analysis complete"})

    assert decision.final_decision == "rerun"
    assert decision.blocking_reasons == ["analysis_review_summary_missing"]


def test_default_flow_analysis_gate_does_not_block_when_review_says_block_is_resolved(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "08_data_quality_review.md"
    review_path.write_text(
        "# Microsoft Corporation（MSFT）数据质量审查报告\n\n"
        "## 三、复核结论\n\n"
        "已解除阻断，当前问题降级为观察项。\n",
        encoding="utf-8",
    )

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-010c",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            artifacts_dir=str(tmp_path),
        ),
    )

    decision = flow._default_analysis_gate({"analysis_markdown": "analysis complete"})

    assert decision.final_decision == "rerun"
    assert decision.blocking_reasons == ["analysis_review_summary_missing"]


def test_default_flow_analysis_gate_still_blocks_when_resolved_notice_and_new_blocker_coexist(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "08_data_quality_review.md"
    review_path.write_text(
        "# Microsoft Corporation（MSFT）数据质量审查报告\n\n"
        "## 三、复核结论\n\n"
        "已解除阻断，旧问题降级为观察项。\n\n"
        "## 四、新增问题\n\n"
        "### 🔴 必须修复（Blockers）\n"
        "- B-1：新的核心估值输入仍然不可放行。\n",
        encoding="utf-8",
    )

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-010c2",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            artifacts_dir=str(tmp_path),
        ),
    )

    decision = flow._default_analysis_gate({"analysis_markdown": "analysis complete"})

    assert decision.final_decision == "blocked"
    assert decision.blocking_reasons == ["analysis_review_marked_blocked"]


def test_default_flow_analysis_gate_does_not_block_when_review_only_records_gate_risk(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "08_data_quality_review.md"
    review_path.write_text(
        "# Microsoft Corporation（MSFT）数据质量审查报告\n\n"
        "## 二、审查发现\n\n"
        "| Gate 风险 | 记录即可：无需阻断，仅保留备查 |\n",
        encoding="utf-8",
    )

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-010d",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            artifacts_dir=str(tmp_path),
        ),
    )

    decision = flow._default_analysis_gate({"analysis_markdown": "analysis complete"})

    assert decision.final_decision == "rerun"
    assert decision.blocking_reasons == ["analysis_review_summary_missing"]


def test_default_flow_analysis_gate_prefers_structured_review_summary_over_markdown_markers(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "08_data_quality_review.md"
    review_path.write_text(
        "# Microsoft Corporation（MSFT）数据质量审查报告\n\n"
        "## 二、必须修复（阻断级，共 6 项）\n\n"
        "- M-1：关键证据链断裂\n",
        encoding="utf-8",
    )

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {
            "analysis_markdown": "analysis complete",
            "analysis_review_summary": {
                "evidence_coverage_ratio": 1.0,
                "financial_coverage_score": 1.0,
                "critical_conflict_count": 0,
                "market_policy_violations": [],
                "unsupported_critical_claims": [],
                "blocking_reasons": [],
            },
        },
        report_writer=lambda _result: "# investment report",
        report_reviewer=lambda _report: GateDecision(
            passed=True,
            final_decision="passed",
            trust_score=90,
        ),
        initial_state=MarketReviewFlowState(
            request_id="run-011",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            artifacts_dir=str(tmp_path),
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "passed"
    assert result["report_result"] == "# investment report"


def test_default_analysis_gate_prefers_inline_review_summary_over_blocker_headings(
    tmp_path: Path,
) -> None:
    (tmp_path / "08_data_quality_review.md").write_text(
        "# Apple Inc.（AAPL）数据质量审查报告\n\n"
        "## 结构化工具输出摘要\n\n"
        "| 工具 | 关键输出 | 解读 |\n"
        "|:-----|:---------|:-----|\n"
        "| `evidence_coverage_tool` | **覆盖率 100%（25/25）**，无 unsupported claims | 引用完整 |\n"
        "| `cross_source_consistency_tool` | **critical_conflict_count = 0** | 无关键冲突 |\n"
        "| `market_tool_policy_audit_tool` | **违规数 = 0** | 无越界 |\n"
        "| `financial_field_completeness_tool` | **总覆盖率 40%（12/30）**；**Gate 覆盖率 28.6%（2/7）** | Gate 字段缺口明显 |\n\n"
        "## 一、必须修复（P0 — 阻断下游分析或产生系统性误导）\n\n"
        "### 🔴 ISSUE-001：Revenue 期间归属未确认\n",
        encoding="utf-8",
    )
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-011a",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            artifacts_dir=str(tmp_path),
        ),
    )

    decision = flow._default_analysis_gate({"analysis_markdown": "analysis complete"})

    assert decision.final_decision == "rerun"
    assert "gate_financial_coverage_incomplete" in decision.blocking_reasons
    assert "analysis_review_marked_blocked" not in decision.blocking_reasons


def test_default_analysis_gate_prefers_machine_readable_review_contract_over_blocker_headings(
    tmp_path: Path,
) -> None:
    (tmp_path / "08_data_quality_review.md").write_text(
        "# Apple Inc.（AAPL）数据质量审查报告\n\n"
        "PART A: MACHINE_READABLE_JSON\n"
        "```json\n"
        "{\n"
        '  "review_stage": "analysis_review",\n'
        '  "reviewer_name": "data_quality_reviewer",\n'
        '  "decision": {"gate_outcome": "rerun", "decision_confidence": "high"},\n'
        '  "delivery_eligibility": {\n'
        '    "formal_report_allowed": false,\n'
        '    "evidence_limited_report_allowed": true,\n'
        '    "blocked_notice_required": false,\n'
        '    "recommended_delivery_state": "evidence_limited_report"\n'
        "  },\n"
        '  "failure_taxonomy": {"primary_class": "pipeline_degraded", "secondary_causes": ["summary_missing"]},\n'
        '  "coverage": {\n'
        '    "evidence_coverage_ratio": 1.0,\n'
        '    "financial_coverage_score": 0.8,\n'
        '    "gate_financial_coverage_score": 0.5,\n'
        '    "claim_binding_ratio": 1.0,\n'
        '    "unresolved_critical_claim_count": 0\n'
        "  },\n"
        '  "tool_health": {"overall_status": "degraded", "tool_status": []},\n'
        '  "blocking_reasons": [],\n'
        '  "rerun_reasons": ["gate_financial_coverage_incomplete"],\n'
        '  "allow_limited_delivery": true,\n'
        '  "review_summary": {"one_sentence_summary": "formal 证据未闭合，只允许 limited", "operator_notes": ""},\n'
        '  "artifact_refs": [],\n'
        '  "findings": []\n'
        "}\n"
        "```\n\n"
        "PART B: HUMAN_READABLE_MARKDOWN\n"
        "## 一、必须修复（P0 — 阻断下游分析或产生系统性误导）\n\n"
        "### 🔴 ISSUE-001：Revenue 期间归属未确认\n",
        encoding="utf-8",
    )
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-011aa",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            artifacts_dir=str(tmp_path),
        ),
    )

    decision = flow._default_analysis_gate({"analysis_markdown": "analysis complete"})

    assert decision.final_decision == "blocked"
    assert "invalid_review_contract" in decision.blocking_reasons
    assert [action.code for action in decision.repair_actions] == ["invalid_review_contract"]


def test_default_analysis_gate_uses_typed_state_evidence_for_strict_contract() -> None:
    strict_contract = {
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
        "review_summary": {"one_sentence_summary": "Strict contract."},
    }
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-strict-evidence",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            evidence_bundle=ResearchEvidenceBundle(company_name="Apple Inc.", ticker="AAPL"),
        ),
    )

    decision = flow._default_analysis_gate(
        {
            "tasks_output": [
                _StubTaskOutput(
                    "data_quality_review_task",
                    "PART A: MACHINE_READABLE_JSON\n```json\n"
                    + json.dumps(strict_contract)
                    + "\n```",
                )
            ]
        }
    )

    assert decision.final_decision == "rerun"
    assert [action.code for action in decision.repair_actions] == [
        "missing_financial_fact",
        "missing_market_snapshot",
    ]


def test_default_analysis_gate_loads_evidence_bundle_artifact_for_strict_contract(
    tmp_path: Path,
) -> None:
    (tmp_path / "10_research_evidence.json").write_text(
        ResearchEvidenceBundle(company_name="Apple Inc.", ticker="AAPL").model_dump_json(),
        encoding="utf-8",
    )
    strict_contract = {
        "stage": "analysis_review",
        "reviewer_name": "data_quality_reviewer",
        "decision": {"gate_outcome": "pass", "decision_confidence": "high"},
        "delivery_eligibility": {"formal_report_allowed": True},
        "failure_taxonomy": {"primary_class": "none"},
        "coverage_summary": {
            "evidence_coverage_ratio": 1.0,
            "financial_coverage_score": 1.0,
            "claim_binding_ratio": 1.0,
        },
        "tool_health_summary": {"overall_status": "healthy"},
    }
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-artifact-evidence",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            artifacts_dir=str(tmp_path),
        ),
    )

    decision = flow._default_analysis_gate(
        {
            "tasks_output": [
                _StubTaskOutput(
                    "data_quality_review_task",
                    "PART A: MACHINE_READABLE_JSON\n```json\n"
                    + json.dumps(strict_contract)
                    + "\n```",
                )
            ]
        }
    )

    assert flow.state.evidence_bundle is not None
    assert decision.final_decision == "rerun"
    assert any(action.code == "missing_financial_fact" for action in decision.repair_actions)


def test_evidence_backed_flow_rejects_legacy_summary_without_strict_contract() -> None:
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-no-strict-contract",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            evidence_bundle=ResearchEvidenceBundle(company_name="Apple Inc.", ticker="AAPL"),
        ),
    )

    decision = flow._default_analysis_gate(
        {
            "analysis_review_summary": {
                "evidence_coverage_ratio": 1.0,
                "financial_coverage_score": 1.0,
                "gate_financial_coverage_score": 1.0,
            }
        }
    )

    assert decision.final_decision == "rerun"
    assert decision.blocking_reasons == ["review_contract_missing"]


def test_default_analysis_gate_accepts_current_reviewer_contract_shape_and_locks_limited_delivery(
    tmp_path: Path,
) -> None:
    (tmp_path / "08_data_quality_review.md").write_text(
        "---\n\n"
        "# PART A: MACHINE_READABLE_JSON\n\n"
        "```json\n"
        "{\n"
        '  "decision": {\n'
        '    "formal_report_allowed": false,\n'
        '    "evidence_limited_report_allowed": true,\n'
        '    "blocked_notice_required": false,\n'
        '    "recommended_delivery_state": "evidence_limited",\n'
        '    "primary_class": "pipeline_degraded"\n'
        "  },\n"
        '  "delivery_eligibility": {\n'
        '    "formal_report_allowed": false,\n'
        '    "evidence_limited_report_allowed": true,\n'
        '    "blocked_notice_required": false,\n'
        '    "evidence_limited_rationale": "SEC-verified fundamentals provide a solid directional foundation."\n'
        "  },\n"
        '  "failure_taxonomy": {\n'
        '    "primary_class": "pipeline_degraded",\n'
        '    "research_blocked": false,\n'
        '    "pipeline_degraded": true,\n'
        '    "degradation_cause": "financial_data_incomplete_for_formal_valuation"\n'
        "  },\n"
        '  "coverage": {\n'
        '    "evidence_coverage_ratio": 1.0,\n'
        '    "financial_coverage_score": 1.0,\n'
        '    "gate_financial_coverage_score": 1.0,\n'
        '    "critical_conflict_count": 0,\n'
        '    "market_policy_violation_count": 0,\n'
        '    "unsupported_critical_claim_count": 0\n'
        "  },\n"
        '  "tool_health": {\n'
        '    "evidence_coverage_tool": "healthy",\n'
        '    "cross_source_consistency_tool": "healthy",\n'
        '    "market_tool_policy_audit_tool": "healthy",\n'
        '    "financial_field_completeness_tool": "healthy"\n'
        "  },\n"
        '  "blocking_reasons": [],\n'
        '  "rerun_reasons": [\n'
        '    {"priority": "P0", "action": "Ingest stock price", "unlocks": "formal_valuation"}\n'
        "  ],\n"
        '  "allow_limited_delivery": true,\n'
        '  "review_summary": "formal 不放行，但 evidence_limited 可交付。",\n'
        '  "artifact_refs": [\n'
        '    "00_market_validation.md",\n'
        '    "01_market_intelligence.md",\n'
        '    "02_filing_review.md",\n'
        '    "03_financial_analysis.md"\n'
        "  ],\n"
        '  "findings": []\n'
        "}\n"
        "```\n\n"
        "# PART B: HUMAN_READABLE_MARKDOWN\n\n"
        "正式报告不放行，但允许 evidence_limited 交付。\n",
        encoding="utf-8",
    )
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-011ab",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            artifacts_dir=str(tmp_path),
        ),
    )

    decision = flow._default_analysis_gate({"analysis_markdown": "analysis complete"})

    assert decision.final_decision == "blocked"
    assert "invalid_review_contract" in decision.blocking_reasons


def test_default_analysis_gate_prefers_latest_metrics_over_markdown_fallback(
    tmp_path: Path,
) -> None:
    (tmp_path / "08_data_quality_review.md").write_text(
        "# Microsoft Corporation（MSFT）数据质量审查报告\n\n"
        "## 二、审查发现\n\n"
        "| Gate风险 | Block: 核心估值输入不可靠 |\n",
        encoding="utf-8",
    )
    (tmp_path / "latest_run_metrics.json").write_text(
        '{"financial_fields":{"revenue":{"extracted":true,"normalized_value":1},"cash_and_equivalents":{"extracted":true,"normalized_value":1}},"financial_fields_success_rate":0.8}',
        encoding="utf-8",
    )
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-011b",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            artifacts_dir=str(tmp_path),
        ),
    )

    decision = flow._default_analysis_gate({"analysis_markdown": "analysis complete"})

    assert decision.final_decision == "rerun"
    assert "analysis_review_marked_blocked" not in decision.blocking_reasons
    assert "gate_financial_coverage_incomplete" in decision.blocking_reasons


def test_default_analysis_gate_reruns_when_summary_and_markers_are_both_missing(
    tmp_path: Path,
) -> None:
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-012",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            artifacts_dir=str(tmp_path),
        ),
    )

    decision = flow._default_analysis_gate({"analysis_markdown": "analysis complete"})

    assert decision.passed is False
    assert decision.final_decision == "rerun"
    assert decision.trust_score == 0
    assert "analysis_review_summary_missing" in decision.blocking_reasons


def test_default_analysis_gate_builds_summary_from_latest_metrics_when_structured_summary_missing(
    tmp_path: Path,
) -> None:
    (tmp_path / "latest_run_metrics.json").write_text(
        '{"financial_fields":{"revenue":{"extracted":true,"normalized_value":1},"cash_and_equivalents":{"extracted":true,"normalized_value":1}},"financial_fields_success_rate":0.8}',
        encoding="utf-8",
    )
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-013",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            artifacts_dir=str(tmp_path),
        ),
    )

    decision = flow._default_analysis_gate({"analysis_markdown": "analysis complete"})

    assert decision.final_decision == "rerun"
    assert "evidence_coverage_ratio<0.60" in decision.blocking_reasons
    assert "gate_financial_coverage_incomplete" in decision.blocking_reasons
    assert "trust_score<60" in decision.blocking_reasons
    assert decision.trust_score == 40


def test_default_analysis_gate_does_not_pass_from_latest_metrics_without_review_signals(
    tmp_path: Path,
) -> None:
    (tmp_path / "latest_run_metrics.json").write_text(
        '{"financial_fields":{"revenue":{"extracted":true,"normalized_value":1},"cash_and_equivalents":{"extracted":true,"normalized_value":1},"total_debt":{"extracted":true,"normalized_value":1},"diluted_shares":{"extracted":true,"normalized_value":1}},"financial_fields_success_rate":1.0}',
        encoding="utf-8",
    )
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-013b",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            artifacts_dir=str(tmp_path),
        ),
    )

    decision = flow._default_analysis_gate({"analysis_markdown": "analysis complete"})

    assert decision.passed is False
    assert decision.final_decision == "rerun"
    assert "evidence_coverage_ratio<0.60" in decision.blocking_reasons


def test_default_analysis_gate_uses_six_field_formal_contract_from_latest_metrics(
    tmp_path: Path,
) -> None:
    (tmp_path / "latest_run_metrics.json").write_text(
        json.dumps(
            {
                "financial_fields": {
                    "revenue": {"extracted": True, "normalized_value": 1},
                    "cash_and_equivalents": {"extracted": True, "normalized_value": 1},
                    "total_debt": {"extracted": True, "normalized_value": 1},
                    "diluted_shares": {"extracted": True, "normalized_value": 1},
                },
                "financial_fields_success_rate": 1.0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-013c",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            artifacts_dir=str(tmp_path),
        ),
    )

    decision = flow._default_analysis_gate({"analysis_markdown": "analysis complete"})

    assert decision.passed is False
    assert decision.final_decision == "rerun"
    assert "gate_financial_coverage_incomplete" in decision.blocking_reasons


def test_default_analysis_gate_builds_summary_from_materialized_review_when_structured_summary_missing(
    tmp_path: Path,
) -> None:
    (tmp_path / "08_data_quality_review.md").write_text(
        "# Microsoft Corporation（MSFT）数据质量审查报告\n\n"
        "## 一、审查工具结构化结果摘要\n\n"
        "| 工具 | 关键结果 | 解读 |\n"
        "|---|---|---|\n"
        "| **evidence_coverage_tool** | 覆盖率 **1.00**（18/18 claims 均有引用），无 unsupported claims | 所有结论均有引用 |\n"
        "| **cross_source_consistency_tool** | **0** critical conflicts | 无关键冲突 |\n"
        "| **market_tool_policy_audit_tool** | **0** violations | 无越界 |\n"
        "| **financial_field_completeness_tool** | 整体覆盖率 **45.45%**；Gate 覆盖率 **76.92%** | 仍有关键字段缺口 |\n",
        encoding="utf-8",
    )
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-014",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            artifacts_dir=str(tmp_path),
        ),
    )

    decision = flow._default_analysis_gate({"analysis_markdown": "analysis complete"})

    assert decision.final_decision == "rerun"
    assert "gate_financial_coverage_incomplete" in decision.blocking_reasons
    assert "analysis_review_summary_missing" not in decision.blocking_reasons
    assert decision.trust_score == 72


def test_default_flow_routes_tasks_output_review_to_evidence_limited_when_rerun_budget_is_exhausted(
) -> None:
    review_text = (
        "# Microsoft Corporation（MSFT）数据质量审查报告\n\n"
        "## 一、审查工具结构化结果摘要\n\n"
        "| 工具 | 关键结果 | 解读 |\n"
        "|---|---|---|\n"
        "| **evidence_coverage_tool** | 覆盖率 **1.00**（18/18 claims 均有引用），无 unsupported claims | 所有结论均有引用 |\n"
        "| **cross_source_consistency_tool** | **0** critical conflicts | 无关键冲突 |\n"
        "| **market_tool_policy_audit_tool** | **0** violations | 无越界 |\n"
        "| **financial_field_completeness_tool** | 整体覆盖率 **80.00%**；Gate 覆盖率 **76.92%** | 仅剩 Gate 关键字段缺口 |\n"
    )

    class StubTaskOutput:
        def __init__(self, name: str, raw: str) -> None:
            self.name = name
            self.raw = raw

    class StubCrewResult:
        def __init__(self, raw: str) -> None:
            self.tasks_output = [StubTaskOutput("data_quality_review_task", raw)]

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: StubCrewResult(review_text),
        report_writer=lambda _result: "# 投资备忘录\n\n**状态**：⚠️ 受限版备忘录 — Analysis Gate 未完全通过\n",
        initial_state=MarketReviewFlowState(
            request_id="run-016",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            rerun_budget={"analysis": 0},
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "evidence_limited"
    assert result["report_result"].startswith("# 投资备忘录")
    assert flow.state.final_decision == "evidence_limited"
    assert result["blocking_reasons"] == ["gate_financial_coverage_incomplete"]


def test_default_analysis_gate_builds_summary_from_tasks_output_when_structured_summary_missing(
) -> None:
    review_text = (
        "# Microsoft Corporation（MSFT）数据质量审查报告\n\n"
        "## 一、审查工具结构化结果摘要\n\n"
        "| 工具 | 关键结果 | 解读 |\n"
        "|---|---|---|\n"
        "| **evidence_coverage_tool** | 覆盖率 **1.00**（18/18 claims 均有引用），无 unsupported claims | 所有结论均有引用 |\n"
        "| **cross_source_consistency_tool** | **0** critical conflicts | 无关键冲突 |\n"
        "| **market_tool_policy_audit_tool** | **0** violations | 无越界 |\n"
        "| **financial_field_completeness_tool** | 整体覆盖率 **45.45%**；Gate 覆盖率 **76.92%** | 仍有关键字段缺口 |\n"
    )

    class StubTaskOutput:
        def __init__(self, name: str, raw: str) -> None:
            self.name = name
            self.raw = raw

    class StubCrewResult:
        def __init__(self, raw: str) -> None:
            self.tasks_output = [StubTaskOutput("data_quality_review_task", raw)]

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-015",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
        ),
    )

    decision = flow._default_analysis_gate(StubCrewResult(review_text))

    assert decision.final_decision == "rerun"
    assert "gate_financial_coverage_incomplete" in decision.blocking_reasons
    assert "analysis_review_summary_missing" not in decision.blocking_reasons
    assert decision.trust_score == 72


def test_default_analysis_gate_prefers_current_tasks_output_summary_over_stale_materialized_blocked_review(
    tmp_path: Path,
) -> None:
    (tmp_path / "08_data_quality_review.md").write_text(
        "# Microsoft Corporation（MSFT）数据质量审查报告\n\n"
        "## 二、审查发现\n\n"
        "| Gate 风险 | Block: 这是旧的阻断结果 |\n",
        encoding="utf-8",
    )
    review_text = (
        "# Microsoft Corporation（MSFT）数据质量审查报告\n\n"
        "## 一、审查工具结构化结果摘要\n\n"
        "| 工具 | 关键结果 | 解读 |\n"
        "|---|---|---|\n"
        "| **evidence_coverage_tool** | 覆盖率 **1.00**（18/18 claims 均有引用），无 unsupported claims | 所有结论均有引用 |\n"
        "| **cross_source_consistency_tool** | **0** critical conflicts | 无关键冲突 |\n"
        "| **market_tool_policy_audit_tool** | **0** violations | 无越界 |\n"
        "| **financial_field_completeness_tool** | 整体覆盖率 **45.45%**；Gate 覆盖率 **76.92%** | 当前仅为覆盖率不足 |\n"
    )

    class StubTaskOutput:
        def __init__(self, name: str, raw: str) -> None:
            self.name = name
            self.raw = raw

    class StubCrewResult:
        def __init__(self, raw: str) -> None:
            self.tasks_output = [StubTaskOutput("data_quality_review_task", raw)]

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-017",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            artifacts_dir=str(tmp_path),
        ),
    )

    decision = flow._default_analysis_gate(StubCrewResult(review_text))

    assert decision.final_decision == "rerun"
    assert "gate_financial_coverage_incomplete" in decision.blocking_reasons
    assert decision.blocking_reasons != ["analysis_review_marked_blocked"]


def test_default_analysis_gate_falls_back_to_materialized_review_when_current_tasks_output_is_unparseable(
    tmp_path: Path,
) -> None:
    (tmp_path / "08_data_quality_review.md").write_text(
        "# Microsoft Corporation（MSFT）数据质量审查报告\n\n"
        "## 一、审查工具结构化结果摘要\n\n"
        "| 工具 | 关键结果 | 解读 |\n"
        "|---|---|---|\n"
        "| **evidence_coverage_tool** | 覆盖率 **1.00**（18/18 claims 均有引用），无 unsupported claims | 所有结论均有引用 |\n"
        "| **cross_source_consistency_tool** | **0** critical conflicts | 无关键冲突 |\n"
        "| **market_tool_policy_audit_tool** | **0** violations | 无越界 |\n"
        "| **financial_field_completeness_tool** | 整体覆盖率 **45.45%**；Gate 覆盖率 **76.92%** | 当前仅为覆盖率不足 |\n",
        encoding="utf-8",
    )

    class StubTaskOutput:
        def __init__(self, name: str, raw: str) -> None:
            self.name = name
            self.raw = raw

    class StubCrewResult:
        def __init__(self) -> None:
            self.tasks_output = [
                StubTaskOutput("data_quality_review_task", "# 数据质量审查\n\n当前 reviewer 输出损坏，无法解析。")
            ]

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        initial_state=MarketReviewFlowState(
            request_id="run-017b",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            artifacts_dir=str(tmp_path),
        ),
    )

    decision = flow._default_analysis_gate(StubCrewResult())

    assert decision.final_decision == "rerun"
    assert "gate_financial_coverage_incomplete" in decision.blocking_reasons
    assert "analysis_review_summary_missing" not in decision.blocking_reasons


def test_analysis_evidence_limited_terminal_status_is_not_overridden_by_passed_report_gate() -> None:
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        analysis_gate=lambda _result: GateDecision(
            passed=False,
            final_decision="rerun",
            trust_score=74,
            blocking_reasons=["gate_financial_coverage_incomplete"],
        ),
        report_writer=lambda _result: "# 投资备忘录\n\n没有显式受限标记\n",
        report_reviewer=lambda _report: GateDecision(
            passed=True,
            final_decision="passed",
            trust_score=90,
        ),
        initial_state=MarketReviewFlowState(
            request_id="run-018",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            rerun_budget={"analysis": 0},
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "evidence_limited"
    assert flow.state.final_decision == "evidence_limited"
    assert result["blocking_reasons"] == ["gate_financial_coverage_incomplete"]


def test_analysis_evidence_limited_merges_report_reasons_and_uses_conservative_trust_score() -> None:
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        analysis_gate=lambda _result: GateDecision(
            passed=False,
            final_decision="rerun",
            trust_score=74,
            blocking_reasons=["gate_financial_coverage_incomplete"],
        ),
        report_writer=lambda _result: "# 投资备忘录\n\n**状态**：⚠️ 受限版备忘录 — Analysis Gate 未完全通过\n",
        report_reviewer=lambda _report: GateDecision(
            passed=False,
            final_decision="evidence_limited",
            trust_score=52,
            blocking_reasons=["report_needs_human_review"],
        ),
        initial_state=MarketReviewFlowState(
            request_id="run-018b",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            rerun_budget={"analysis": 0},
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "evidence_limited"
    assert result["trust_score"] == 52
    assert result["blocking_reasons"] == [
        "gate_financial_coverage_incomplete",
        "report_needs_human_review",
    ]


def test_analysis_evidence_limited_terminal_status_becomes_blocked_when_report_gate_is_blocked() -> None:
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        analysis_gate=lambda _result: GateDecision(
            passed=False,
            final_decision="rerun",
            trust_score=74,
            blocking_reasons=["gate_financial_coverage_incomplete"],
        ),
        report_writer=lambda _result: "# 投资备忘录\n\nAnalysis Gate 未完全通过\n",
        report_reviewer=lambda _report: GateDecision(
            passed=False,
            final_decision="blocked",
            trust_score=0,
            blocking_reasons=["report_marked_blocked"],
        ),
        initial_state=MarketReviewFlowState(
            request_id="run-019",
            company_name="Microsoft Corporation",
            input_ticker="MSFT",
            rerun_budget={"analysis": 0},
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "blocked"
    assert flow.state.final_decision == "blocked"
    assert result["blocking_reasons"] == ["report_marked_blocked"]


def test_flow_blocks_when_logic_review_supports_limited_delivery_but_report_text_still_looks_formal(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "04_investment_report.md"
    report_path.write_text(
        "# Apple Inc.（AAPL）投资备忘录\n\n"
        "## 一、执行摘要\n\n"
        "当前为受限交付，估值结论仍受关键字段缺失约束。\n",
        encoding="utf-8",
    )
    (tmp_path / "09_logic_compliance_review.md").write_text(
        "# 逻辑与合规审查报告\n\n"
        "审查结论：有条件通过——存在若干必须修复项。\n\n"
        "核心判断：最终备忘录的整体质量已达到“受限版”的预期标准，"
        "上述修复项均属局部问题，不损坏报告的核心逻辑和整体可靠性。\n",
        encoding="utf-8",
    )

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        analysis_gate=lambda _result: GateDecision(
            passed=False,
            final_decision="rerun",
            trust_score=58,
            blocking_reasons=["analysis_review_summary_missing"],
        ),
        initial_state=MarketReviewFlowState(
            request_id="run-020",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            artifacts_dir=str(tmp_path),
            final_report_path=str(report_path),
            rerun_budget={"analysis": 0},
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "blocked"
    assert result["report_result"] == report_path.read_text(encoding="utf-8").strip()
    assert flow.state.final_decision == "blocked"
    assert result["blocking_reasons"] == ["report_mode_mismatch"]


def test_flow_does_not_use_stale_materialized_report_to_fallback_analysis_review_summary_missing(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "04_investment_report.md"
    report_path.write_text(
        "# 受限版投资备忘录\n\n"
        "**状态**：⚠️ 受限版备忘录 — Analysis Gate 未完全通过\n",
        encoding="utf-8",
    )
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        analysis_gate=lambda _result: GateDecision(
            passed=False,
            final_decision="rerun",
            trust_score=58,
            blocking_reasons=["analysis_review_summary_missing"],
        ),
        initial_state=MarketReviewFlowState(
            request_id="run-021",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            artifacts_dir=str(tmp_path),
            final_report_path=str(report_path),
            rerun_budget={"analysis": 0},
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "blocked"
    assert flow.state.final_decision == "blocked"
    assert result["blocking_reasons"] == ["analysis_review_summary_missing"]


def test_default_flow_report_gate_blocks_when_report_mode_mismatches_analysis_limited_decision() -> None:
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        analysis_gate=lambda _result: GateDecision(
            passed=False,
            final_decision="rerun",
            trust_score=74,
            blocking_reasons=["gate_financial_coverage_incomplete"],
        ),
        report_writer=lambda _result: "# 投资备忘录\n\n## 执行摘要\n\n这是被错误写成正式版的文本。\n",
        initial_state=MarketReviewFlowState(
            request_id="run-022b",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            rerun_budget={"analysis": 0},
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "blocked"
    assert result["blocking_reasons"] == ["report_mode_mismatch"]


def test_flow_does_not_fallback_to_evidence_limited_when_logic_review_contains_blocker_language(
    tmp_path: Path,
) -> None:
    (tmp_path / "09_logic_compliance_review.md").write_text(
        "# 逻辑与合规审查报告\n\n"
        "审查结论：有条件通过——达到受限版的预期标准。\n\n"
        "## 必须修复（阻断级）\n"
        "- B1：当前版本仍应视为 Blocker，修复前不建议放行。\n",
        encoding="utf-8",
    )
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"analysis_markdown": "analysis complete"},
        analysis_gate=lambda _result: GateDecision(
            passed=False,
            final_decision="rerun",
            trust_score=58,
            blocking_reasons=["analysis_review_summary_missing"],
        ),
        initial_state=MarketReviewFlowState(
            request_id="run-022",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            artifacts_dir=str(tmp_path),
            rerun_budget={"analysis": 0},
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "blocked"
    assert flow.state.final_decision == "blocked"
    assert result["blocking_reasons"] == ["analysis_review_summary_missing"]


def test_typed_flow_passes_only_from_bundle_contract_and_document() -> None:
    bundle = _typed_bundle()
    analysis_contract = _typed_contract()
    report_contract = _typed_report_contract()
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {
            "evidence_bundle": bundle.model_dump(mode="json"),
            "analysis_review_contract": analysis_contract.model_dump(mode="json"),
        },
        report_writer=lambda context: _typed_writer_payload(),
        report_reviewer=lambda _document: report_contract.model_dump(mode="json"),
        initial_state=MarketReviewFlowState(
            request_id="typed-pass",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            evidence_bundle=bundle,
            execution_mode="new",
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "passed"
    assert result["final_decision_record"]["final_delivery_state"] == "formal_report"
    assert result["report_document"]["report_mode"] == "formal_report"
    assert result["analysis_review_contract"]["stage"] == "analysis_review"
    assert result["report_review_contract"]["stage"] == "report_review"


def test_typed_flow_rejects_prose_pass_without_strict_contract() -> None:
    bundle = _typed_bundle()
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {"review_text": "审查通过"},
        initial_state=MarketReviewFlowState(
            request_id="typed-no-contract",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            evidence_bundle=bundle,
            execution_mode="new",
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "blocked"
    assert "analysis_review_contract_missing" in result["blocking_reasons"]


def test_typed_flow_reruns_only_repair_target_with_override_and_budget() -> None:
    bundle = _typed_bundle()
    rerun_contract = _typed_contract(
        outcome="rerun",
        actions=[
            {
                "target": "quant_valuation_analyst",
                "code": "quote_missing",
                "instruction": "补齐报价与估值计算。",
            }
        ],
    )
    attempts: list[dict[str, object]] = []

    def executor(inputs: dict[str, object]) -> dict[str, object]:
        attempts.append(dict(inputs))
        contract = _typed_contract() if len(attempts) == 2 else rerun_contract
        return {
            "evidence_bundle": bundle.model_dump(mode="json"),
            "analysis_review_contract": contract.model_dump(mode="json"),
        }

    flow = MarketReviewFlow(
        analysis_executor=executor,
        report_writer=lambda _context: _typed_writer_payload(),
        report_reviewer=lambda _document: _typed_report_contract().model_dump(mode="json"),
        initial_state=MarketReviewFlowState(
            request_id="typed-targeted-rerun",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            evidence_bundle=bundle,
            execution_mode="new",
            rerun_budget={"analysis": 1},
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "passed"
    assert attempts[1]["rerun_targets"] == ["quant_valuation_analyst"]
    assert attempts[1]["model_tier_overrides"] == {"quant_valuation_analyst": "deep"}
    assert flow.state.rerun_budget["analysis"] == 0
    assert result["repair_actions"][0]["target"] == "quant_valuation_analyst"


def test_typed_flow_retries_repairable_block_before_terminal_delivery() -> None:
    bundle = _typed_bundle()
    blocked_contract = _typed_contract(
        outcome="block",
        actions=[{
            "target": "fundamental_analyst",
            "code": "filing_period_conflict",
            "fields": ["segment_revenue_services"],
            "instruction": "重新提取同一期间的 SEC 字段。",
        }],
    )
    blocked_contract.blocking_reasons = ["filing_period_conflict"]
    attempts: list[dict[str, object]] = []

    def executor(inputs: dict[str, object]) -> dict[str, object]:
        attempts.append(dict(inputs))
        contract = _typed_contract() if len(attempts) == 2 else blocked_contract
        return {
            "evidence_bundle": bundle.model_dump(mode="json"),
            "analysis_review_contract": contract.model_dump(mode="json"),
        }

    result = MarketReviewFlow(
        analysis_executor=executor,
        report_writer=lambda _context: _typed_writer_payload(),
        report_reviewer=lambda _document: _typed_report_contract().model_dump(mode="json"),
        initial_state=MarketReviewFlowState(
            request_id="repairable-block",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            evidence_bundle=bundle,
            execution_mode="new",
            rerun_budget={"fundamental_analyst": 1},
        ),
    ).kickoff()

    assert result["status"] == "passed"
    assert attempts[1]["rerun_targets"] == ["fundamental_analyst"]


def test_typed_flow_runs_three_targeted_repairs_then_continues_limited_delivery() -> None:
    bundle = _typed_bundle()
    contract_payload = _typed_contract(
        outcome="rerun",
        actions=[{
            "target": "fundamental_analyst",
            "code": "filing_gap",
            "fields": ["segment_revenue_services"],
            "instruction": "重新调用 SEC 工具补齐 Services 收入。",
        }],
    ).model_dump(mode="json")
    contract_payload["delivery_eligibility"] = {
        "formal_report_allowed": False,
        "evidence_limited_report_allowed": True,
        "blocked_notice_required": False,
        "recommended_delivery_state": "evidence_limited_report",
    }
    contract_payload["failure_taxonomy"] = {
        "primary_class": "coverage_gap",
        "secondary_causes": ["filing_gap"],
    }
    contract_payload["rerun_reasons"] = ["filing_gap"]
    contract_payload["allow_limited_delivery"] = True
    rerun_contract = ReviewContract.model_validate(contract_payload)
    attempts: list[dict[str, object]] = []
    writer_contexts: list[dict[str, object]] = []

    def executor(inputs: dict[str, object]) -> dict[str, object]:
        attempts.append(dict(inputs))
        return {
            "evidence_bundle": bundle.model_dump(mode="json"),
            "analysis_review_contract": rerun_contract.model_dump(mode="json"),
        }

    def writer(inputs: dict[str, object]) -> dict[str, object]:
        writer_contexts.append(json.loads(str(inputs["REPORT_CONTEXT_JSON"])))
        payload = _typed_writer_payload()
        payload["stance"] = "watch"
        payload["executive_summary"] = "证据受限：Services 收入仍未完成同期间验证。"
        return payload

    flow = MarketReviewFlow(
        analysis_executor=executor,
        report_writer=writer,
        report_reviewer=lambda _document: _typed_report_contract().model_dump(mode="json"),
        initial_state=MarketReviewFlowState(
            request_id="three-repair-rounds",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            evidence_bundle=bundle,
            execution_mode="new",
            rerun_budget={"analysis": 3},
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "evidence_limited"
    assert len(attempts) == 4
    assert [attempt["rerun_targets"] for attempt in attempts[1:]] == [
        ["fundamental_analyst"],
        ["fundamental_analyst"],
        ["fundamental_analyst"],
    ]
    assert flow.state.rerun_budget["analysis"] == 0
    assert writer_contexts[0]["report_mode"] == "evidence_limited_report"
    assert "rerun_budget_exhausted:analysis" in result["blocking_reasons"]


def test_analysis_review_guardrail_returns_content_validation_errors() -> None:
    class Output:
        raw = (
            "PART A: MACHINE_READABLE_JSON\n"
            "```json\n"
            '{"stage":"data_quality_review","decision":{"gate_outcome":"conditional_pass"}}\n'
            "```"
        )

    accepted, feedback = validate_analysis_review_output(Output())

    assert accepted is False
    assert "stage" not in str(feedback)
    assert "decision.gate_outcome" in str(feedback)


def test_analysis_review_guardrail_normalizes_known_reviewer_schema_drift() -> None:
    payload = _typed_contract(
        outcome="rerun",
        actions=[{
            "target": "fundamental_analyst",
            "code": "filing_gap",
            "instruction": "重新提取 SEC 字段。",
        }],
    ).model_dump(mode="json")
    payload["tool_health_summary"] = {
        "evidence_coverage_tool": {"status": "healthy", "notes": "coverage complete"},
        "cross_source_consistency_tool": {"status": "degraded", "notes": "one conflict"},
        "market_tool_policy_audit_tool": "healthy",
        "financial_field_completeness_tool": "failed",
    }
    payload["review_summary"] = "存在可定向修复的数据缺口。"
    payload["repair_actions"][0]["description"] = payload["repair_actions"][0].pop("instruction")

    class Output:
        raw = (
            "PART A: MACHINE_READABLE_JSON\n```json\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n```\n"
            "PART B: HUMAN_READABLE_MARKDOWN\n需要定向修复。"
        )

    accepted, normalized_raw = validate_analysis_review_output(Output())
    contract = review_contract_from_text(str(normalized_raw), expected_stage="analysis_review")

    assert accepted is True
    assert contract is not None
    assert contract.tool_health_summary.overall_status == "failed"
    assert contract.tool_health_summary.failed_tools == ["financial_field_completeness_tool"]
    assert contract.tool_health_summary.degraded_tools == ["cross_source_consistency_tool"]
    assert contract.review_summary.one_sentence_summary == "存在可定向修复的数据缺口。"
    assert contract.repair_actions[0].instruction == "重新提取 SEC 字段。"
    assert '"description"' not in str(normalized_raw)


@pytest.mark.parametrize(
    ("validator", "model_stage", "expected_stage"),
    (
        (validate_analysis_review_output, "report_review", "analysis_review"),
        (validate_report_review_output, "analysis_review", "report_review"),
    ),
)
def test_review_guardrail_owns_the_task_stage(
    validator: Callable[[object], tuple[bool, object]],
    model_stage: str,
    expected_stage: str,
) -> None:
    payload = _typed_contract().model_dump(mode="json")
    payload["stage"] = model_stage

    class Output:
        raw = (
            "PART A: MACHINE_READABLE_JSON\n```json\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n```\n"
            "PART B: HUMAN_READABLE_MARKDOWN\n审查完成。"
        )

    accepted, normalized_raw = validator(Output())
    contract = review_contract_from_text(str(normalized_raw), expected_stage=expected_stage)

    assert accepted is True
    assert contract is not None
    assert contract.stage == expected_stage


def test_typed_flow_blocks_after_analysis_repair_budget_is_exhausted() -> None:
    bundle = _typed_bundle()
    contract = _typed_contract(
        outcome="rerun",
        actions=[
            {
                "target": "quant_valuation_analyst",
                "code": "quote_missing",
                "instruction": "补齐报价。",
            }
        ],
    )
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {
            "evidence_bundle": bundle.model_dump(mode="json"),
            "analysis_review_contract": contract.model_dump(mode="json"),
        },
        initial_state=MarketReviewFlowState(
            request_id="typed-budget-exhausted",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            evidence_bundle=bundle,
            execution_mode="new",
            rerun_budget={"analysis": 0},
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "blocked"
    assert "rerun_budget_exhausted:analysis" in result["blocking_reasons"]


@pytest.mark.parametrize("payload", [{}, {"title": "bad"}])
def test_typed_flow_blocks_malformed_writer_payload(payload: dict[str, object]) -> None:
    bundle = _typed_bundle()
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {
            "evidence_bundle": bundle.model_dump(mode="json"),
            "analysis_review_contract": _typed_contract().model_dump(mode="json"),
        },
        report_writer=lambda _context: payload,
        initial_state=MarketReviewFlowState(
            request_id="typed-malformed-writer",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            evidence_bundle=bundle,
            execution_mode="new",
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "blocked"
    assert "writer_payload_invalid" in result["blocking_reasons"]


def test_typed_flow_blocks_malformed_or_rejecting_logic_review() -> None:
    bundle = _typed_bundle()
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {
            "evidence_bundle": bundle.model_dump(mode="json"),
            "analysis_review_contract": _typed_contract().model_dump(mode="json"),
        },
        report_writer=lambda _context: _typed_writer_payload(),
        report_reviewer=lambda _document: _typed_report_contract(outcome="block").model_dump(mode="json"),
        initial_state=MarketReviewFlowState(
            request_id="typed-logic-block",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            evidence_bundle=bundle,
            execution_mode="new",
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "blocked"
    assert "logic_reviewer_requested_block" in result["blocking_reasons"]


def test_typed_report_gate_does_not_ignore_blockers_on_pass_contract() -> None:
    bundle = _typed_bundle()
    review = _typed_report_contract().model_dump(mode="json")
    review["blocking_reasons"] = ["claim_binding_incomplete"]
    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {
            "evidence_bundle": bundle.model_dump(mode="json"),
            "analysis_review_contract": _typed_contract().model_dump(mode="json"),
        },
        report_writer=lambda _context: _typed_writer_payload(),
        report_reviewer=lambda _document: review,
        initial_state=MarketReviewFlowState(
            request_id="typed-pass-with-blocker",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            evidence_bundle=bundle,
            execution_mode="new",
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "blocked"
    assert "claim_binding_incomplete" in result["blocking_reasons"]


def test_typed_report_rework_passes_strict_feedback_to_second_writer() -> None:
    bundle = _typed_bundle()
    writer_inputs: list[dict[str, object]] = []
    reviewed_documents: list[object] = []

    rerun_review = _typed_report_contract(outcome="rerun").model_dump(mode="json")
    rerun_review["rerun_reasons"] = ["补充收入来源绑定。"]
    rerun_review["repair_actions"] = [{
        "target": "report_writing_analyst", "code": "bind_source",
        "instruction": "将收入 claim 绑定 SEC 来源。",
    }]
    reviews = iter([rerun_review, _typed_report_contract().model_dump(mode="json")])

    def writer(inputs: dict[str, object]) -> dict[str, object]:
        writer_inputs.append(inputs)
        payload = _typed_writer_payload()
        payload["executive_summary"] = (
            "已根据逻辑审查反馈重新绑定收入来源。"
            if len(writer_inputs) == 2
            else "基于已验证的财务证据维持持有观点。"
        )
        return payload

    def reviewer(document: object) -> dict[str, object]:
        reviewed_documents.append(document)
        return next(reviews)

    result = MarketReviewFlow(
        analysis_executor=lambda _inputs: {
            "evidence_bundle": bundle.model_dump(mode="json"),
            "analysis_review_contract": _typed_contract().model_dump(mode="json"),
        },
        report_writer=writer,
        report_reviewer=reviewer,
        initial_state=MarketReviewFlowState(
            request_id="typed-rework", company_name="Apple Inc.", input_ticker="AAPL",
            evidence_bundle=bundle, execution_mode="new",
            rerun_budget={"report_writing_analyst": 1},
        ),
    ).kickoff()

    assert result["status"] == "passed"
    assert len(writer_inputs) == 2
    first = json.loads(str(writer_inputs[0]["REPORT_CONTEXT_JSON"]))
    second = json.loads(str(writer_inputs[1]["REPORT_CONTEXT_JSON"]))
    assert first["revision_instructions"] == []
    assert "补充收入来源绑定。" in second["revision_instructions"]
    assert "将收入 claim 绑定 SEC 来源。" in second["revision_instructions"]
    assert len(reviewed_documents) == 2
    assert reviewed_documents[0].executive_summary != reviewed_documents[1].executive_summary
    assert reviewed_documents[1].executive_summary == "已根据逻辑审查反馈重新绑定收入来源。"


def test_blocked_notice_report_uses_the_same_repair_loop() -> None:
    bundle = _typed_bundle()
    writer_calls = 0
    reviews = iter([
        _typed_report_contract(outcome="rerun").model_dump(mode="json"),
        _typed_report_contract().model_dump(mode="json"),
    ])

    def writer(_inputs: dict[str, object]) -> dict[str, object]:
        nonlocal writer_calls
        writer_calls += 1
        payload = _typed_writer_payload()
        payload.update({
            "stance": "blocked",
            "executive_summary": "当前证据存在阻断，不能形成投资建议。",
            "catalysts": [],
            "risks": ["关键证据仍待补齐。"],
        })
        return payload

    result = MarketReviewFlow(
        analysis_executor=lambda _inputs: {
            "evidence_bundle": bundle.model_dump(mode="json"),
            "analysis_review_contract": _typed_contract(outcome="block").model_dump(mode="json"),
        },
        report_writer=writer,
        report_reviewer=lambda _document: next(reviews),
        initial_state=MarketReviewFlowState(
            request_id="typed-blocked-rework",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            evidence_bundle=bundle,
            execution_mode="new",
            rerun_budget={"report_writing_analyst": 1},
        ),
    ).kickoff()

    assert result["status"] == "blocked"
    assert writer_calls == 2


def test_analysis_review_defers_report_writer_action_to_report_generation() -> None:
    bundle = _typed_bundle()
    contract = _typed_contract(
        actions=[{
            "target": "report_writing_analyst",
            "code": "add_evidence_header",
            "instruction": "在事件章节增加证据等级说明。",
        }]
    )
    writer_contexts: list[dict[str, object]] = []
    executor_calls = 0

    def executor(_inputs: dict[str, object]) -> dict[str, object]:
        nonlocal executor_calls
        executor_calls += 1
        return {
            "evidence_bundle": bundle.model_dump(mode="json"),
            "analysis_review_contract": contract.model_dump(mode="json"),
        }

    def writer(inputs: dict[str, object]) -> dict[str, object]:
        writer_contexts.append(json.loads(str(inputs["REPORT_CONTEXT_JSON"])))
        return _typed_writer_payload()

    result = MarketReviewFlow(
        analysis_executor=executor,
        report_writer=writer,
        report_reviewer=lambda _document: _typed_report_contract().model_dump(mode="json"),
        initial_state=MarketReviewFlowState(
            request_id="defer-writer-repair",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            execution_mode="new",
        ),
    ).kickoff()

    assert result["status"] == "passed"
    assert executor_calls == 1
    assert writer_contexts[0]["revision_instructions"] == [
        "在事件章节增加证据等级说明。"
    ]


def test_default_report_crew_receives_identity_template_inputs() -> None:
    captured_inputs: dict[str, object] = {}

    class FakeReportCrew:
        def kickoff(self, *, inputs: dict[str, object]) -> object:
            captured_inputs.update(inputs)
            return type("Result", (), {
                "tasks_output": [
                    _StubTaskOutput(
                        "investment_report_task",
                        json.dumps(_typed_writer_payload(), ensure_ascii=False),
                    )
                ]
            })()

    class FakeCrewFactory:
        def configure_run(self, **_kwargs: object) -> None:
            pass

        def report_crew(self) -> FakeReportCrew:
            return FakeReportCrew()

    flow = MarketReviewFlow(
        crew_factory=FakeCrewFactory(),
        initial_state=MarketReviewFlowState(
            request_id="report-inputs",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            execution_mode="new",
            evidence_bundle=_typed_bundle(),
            analysis_review_contract=_typed_contract(),
            analysis_gate_decision=GateDecision(
                passed=True,
                final_decision="passed",
                trust_score=90,
            ),
        ),
    )

    flow._typed_writer_document()

    assert captured_inputs["company_name"] == "Apple Inc."
    assert captured_inputs["company_ticker"] == "AAPL"
    assert captured_inputs["run_id"] == "report-inputs"
    assert "REPORT_CONTEXT_JSON" in captured_inputs


@pytest.mark.parametrize(
    "target",
    (
        "market_validation_analyst",
        "event_guidance_analyst",
        "fundamental_analyst",
        "quant_valuation_analyst",
    ),
)
def test_evidence_producer_rerun_reloads_fresh_bundle_for_next_gate(
    target: str, tmp_path: Path
) -> None:
    initial_bundle = _typed_bundle()
    fresh_facts = [
        fact.model_copy(update={"value": fact.value + 100.0})
        for fact in initial_bundle.financial_facts
    ]
    fresh_bundle = initial_bundle.model_copy(update={"financial_facts": fresh_facts})
    rerun_contract = _typed_contract(
        outcome="rerun",
        actions=[{
            "target": target,
            "code": "refresh_evidence",
            "instruction": "重新生成 canonical evidence。",
        }],
    )
    attempts: list[dict[str, object]] = []
    evidence_path = tmp_path / "10_research_evidence.json"
    review_path = tmp_path / "08_data_quality_review.json"

    def executor(inputs: dict[str, object]) -> dict[str, object]:
        attempts.append(dict(inputs))
        if len(attempts) == 1:
            return {
                "evidence_bundle": initial_bundle.model_dump(mode="json"),
                "analysis_review_contract": rerun_contract.model_dump(mode="json"),
            }
        assert not evidence_path.exists()
        assert not review_path.exists()
        return {
            "evidence_bundle": fresh_bundle.model_dump(mode="json"),
            "analysis_review_contract": _typed_contract().model_dump(mode="json"),
        }

    flow = MarketReviewFlow(
        analysis_executor=executor,
        report_writer=lambda _context: _typed_writer_payload(),
        report_reviewer=lambda _document: _typed_report_contract().model_dump(mode="json"),
        initial_state=MarketReviewFlowState(
            request_id=f"fresh-{target}", company_name="Apple Inc.", input_ticker="AAPL",
            execution_mode="new", artifacts_dir=str(tmp_path), rerun_budget={target: 1},
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "passed"
    assert attempts[1]["rerun_targets"] == [target]
    assert flow.state.evidence_bundle == fresh_bundle
    assert ResearchEvidenceBundle.model_validate_json(evidence_path.read_text(encoding="utf-8")) == fresh_bundle


def test_real_crew_reviews_the_bundle_created_by_the_producer_phase(tmp_path: Path) -> None:
    bundle = _typed_bundle()
    evidence_path = tmp_path / "10_research_evidence.json"
    reviewer_inputs: list[dict[str, object]] = []

    class FakeCrew:
        def __init__(self, phase: str) -> None:
            self.phase = phase

        def kickoff(self, *, inputs: dict[str, object]) -> object:
            if self.phase == "producer":
                evidence_path.write_text(bundle.model_dump_json(), encoding="utf-8")
            else:
                reviewer_inputs.append(dict(inputs))
            return object()

    class FakeCrewFactory:
        def configure_run(self, **_kwargs: object) -> None:
            pass

        def analysis_crew(self) -> FakeCrew:
            return FakeCrew("producer")

        def analysis_review_crew(self) -> FakeCrew:
            return FakeCrew("reviewer")

    flow = MarketReviewFlow(
        crew_factory=FakeCrewFactory(),
        initial_state=MarketReviewFlowState(
            request_id="split-review-context",
            company_name="Apple Inc.",
            input_ticker="AAPL",
            execution_mode="new",
            artifacts_dir=str(tmp_path),
        ),
    )

    flow._execute_existing_crew({"rerun_targets": [], "review_evidence_context_json": "{}"})

    context = json.loads(str(reviewer_inputs[0]["review_evidence_context_json"]))
    assert context["ticker"] == "AAPL"
    assert context["financial_facts"]
    assert "claim:revenue" in context["valid_evidence_refs"]


def test_event_rerun_reloads_fresh_canonical_bundle_from_crewai_artifacts(
    tmp_path: Path,
) -> None:
    initial_bundle = _typed_bundle()
    fresh_bundle = initial_bundle.model_copy(update={
        "financial_facts": [
            fact.model_copy(update={"value": fact.value + 200.0})
            for fact in initial_bundle.financial_facts
        ],
    })
    rerun_contract = _typed_contract(
        outcome="rerun",
        actions=[{
            "target": "event_guidance_analyst",
            "code": "refresh_event_evidence",
            "instruction": "使用当前 Tavily 事件输出重建证据。",
        }],
    )
    evidence_path = tmp_path / "10_research_evidence.json"
    review_path = tmp_path / "08_data_quality_review.json"
    targeted_inputs: list[dict[str, object]] = []

    class FakeCrewOutput:
        json_dict = None
        pydantic = None
        raw = "CrewOutput without inline evidence bundle"
        tasks_output = [_StubTaskOutput("data_quality_review_task", "fresh review artifact")]

    class FakeCrew:
        def __init__(self, *, targeted: bool) -> None:
            self.targeted = targeted

        def kickoff(self, *, inputs: dict[str, object]) -> FakeCrewOutput:
            if not self.targeted:
                evidence_path.write_text(initial_bundle.model_dump_json(), encoding="utf-8")
                review_path.write_text(rerun_contract.model_dump_json(), encoding="utf-8")
                return FakeCrewOutput()
            targeted_inputs.append(dict(inputs))
            assert inputs["rerun_targets"] == ["event_guidance_analyst"]
            assert not evidence_path.exists()
            assert not review_path.exists()
            evidence_path.write_text(fresh_bundle.model_dump_json(), encoding="utf-8")
            review_path.write_text(_typed_contract().model_dump_json(), encoding="utf-8")
            return FakeCrewOutput()

    class FakeCrewFactory:
        def configure_run(self, **_kwargs: object) -> None:
            pass

        def analysis_crew(self) -> FakeCrew:
            return FakeCrew(targeted=False)

        def targeted_analysis_crew(self, targets: list[str]) -> FakeCrew:
            assert targets == ["event_guidance_analyst"]
            return FakeCrew(targeted=True)

    flow = MarketReviewFlow(
        crew_factory=FakeCrewFactory(),
        report_writer=lambda _context: _typed_writer_payload(),
        report_reviewer=lambda _document: _typed_report_contract().model_dump(mode="json"),
        initial_state=MarketReviewFlowState(
            request_id="artifact-event-rerun", company_name="Apple Inc.", input_ticker="AAPL",
            execution_mode="new", artifacts_dir=str(tmp_path),
            rerun_budget={"event_guidance_analyst": 1},
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "passed"
    assert targeted_inputs
    assert flow.state.evidence_bundle == fresh_bundle
    assert ResearchEvidenceBundle.model_validate_json(evidence_path.read_text(encoding="utf-8")) == fresh_bundle


def test_data_quality_rerun_preserves_evidence_and_rebuilds_only_review_contract(
    tmp_path: Path,
) -> None:
    bundle = _typed_bundle()
    rerun_contract = _typed_contract(
        outcome="rerun",
        actions=[{
            "target": "data_quality_reviewer",
            "code": "recheck_contract",
            "instruction": "基于当前证据重建审查契约。",
        }],
    )
    attempts: list[dict[str, object]] = []
    evidence_path = tmp_path / "10_research_evidence.json"
    review_path = tmp_path / "08_data_quality_review.json"

    def executor(inputs: dict[str, object]) -> dict[str, object]:
        attempts.append(dict(inputs))
        if len(attempts) == 1:
            return {
                "evidence_bundle": bundle.model_dump(mode="json"),
                "analysis_review_contract": rerun_contract.model_dump(mode="json"),
            }
        assert evidence_path.exists()
        assert not review_path.exists()
        assert ResearchEvidenceBundle.model_validate_json(evidence_path.read_text(encoding="utf-8")) == bundle
        return {"analysis_review_contract": _typed_contract().model_dump(mode="json")}

    flow = MarketReviewFlow(
        analysis_executor=executor,
        report_writer=lambda _context: _typed_writer_payload(),
        report_reviewer=lambda _document: _typed_report_contract().model_dump(mode="json"),
        initial_state=MarketReviewFlowState(
            request_id="review-only", company_name="Apple Inc.", input_ticker="AAPL",
            execution_mode="new", artifacts_dir=str(tmp_path),
            rerun_budget={"data_quality_reviewer": 1},
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "passed"
    assert attempts[1]["rerun_targets"] == ["data_quality_reviewer"]
    review_context = json.loads(str(attempts[1]["review_evidence_context_json"]))
    assert review_context["company_name"] == "Apple Inc."
    assert review_context["ticker"] == "AAPL"
    assert review_context["financial_facts"]
    assert all(item["source_url"] for item in review_context["financial_facts"])
    assert flow.state.evidence_bundle == bundle
    assert flow.state.analysis_review_contract == _typed_contract()


def test_reviewer_only_context_keeps_materialized_analysis_artifacts(tmp_path: Path) -> None:
    bundle = _typed_bundle()
    for filename, content in {
        "00_market_validation.md": "market validation",
        "01_market_intelligence.md": "event evidence",
        "02_filing_review.md": "FY2026 data not extracted",
        "03_financial_analysis.md": "FY2026 H1 data extracted",
    }.items():
        (tmp_path / filename).write_text(content, encoding="utf-8")
    flow = MarketReviewFlow(
        initial_state=MarketReviewFlowState(
            company_name="Apple Inc.",
            input_ticker="AAPL",
            execution_mode="new",
            artifacts_dir=str(tmp_path),
            evidence_bundle=bundle,
        )
    )

    context = json.loads(flow._review_evidence_context_json())

    assert context["analysis_artifacts"]["02_filing_review.md"] == "FY2026 data not extracted"
    assert context["analysis_artifacts"]["03_financial_analysis.md"] == "FY2026 H1 data extracted"


def test_analysis_contract_rejects_non_analysis_target_after_reviewer_retry() -> None:
    bundle = _typed_bundle()
    contract = _typed_contract(outcome="rerun").model_dump(mode="json")
    contract["repair_actions"] = [{
        "target": "logic_compliance_reviewer",
        "code": "wrong_phase",
        "instruction": "不应由分析 rerun 执行。",
    }]
    executor_calls = 0

    def executor(_inputs: dict[str, object]) -> dict[str, object]:
        nonlocal executor_calls
        executor_calls += 1
        return {
            "evidence_bundle": bundle.model_dump(mode="json"),
            "analysis_review_contract": contract,
        }

    result = MarketReviewFlow(
        analysis_executor=executor,
        initial_state=MarketReviewFlowState(
            request_id="invalid-review-phase", company_name="Apple Inc.", input_ticker="AAPL",
            execution_mode="new", rerun_budget={"analysis": 1},
        ),
    ).kickoff()

    assert result["status"] == "blocked"
    assert "invalid_review_contract" in result["blocking_reasons"]
    assert executor_calls == 2


def test_analysis_rerun_rejects_unknown_repair_target_before_executor() -> None:
    flow = MarketReviewFlow(
        initial_state=MarketReviewFlowState(
            request_id="unknown-target", company_name="Apple Inc.", input_ticker="AAPL",
            execution_mode="new", rerun_budget={"unrecognized_agent": 1},
        ),
    )
    gate = GateDecision(
        passed=False,
        final_decision="rerun",
        trust_score=0,
        repair_actions=[RepairAction.model_construct(
            target="unrecognized_agent",
            code="unknown_target",
            instruction="拒绝未知 agent。",
        )],
    )

    route = flow._route_after_analysis_gate(gate)

    assert route == "analysis_blocked"
    assert "unsupported_repair_target:unrecognized_agent" in flow.state.analysis_gate_decision.blocking_reasons


@pytest.mark.parametrize(
    ("target", "attempt_prefix"),
    (
        ("event_guidance_analyst", "tavily-attempt-"),
        ("quant_valuation_analyst", "tavily-snapshot-"),
    ),
)
def test_evidence_rerun_selects_tavily_attempt_before_financial_executor(
    target: str, attempt_prefix: str, tmp_path: Path
) -> None:
    evaluation = WorkflowEvaluation(
        artifacts_dir=tmp_path,
        final_report_path=tmp_path / "report.md",
        expected_task_outputs={},
        company_name="Apple Inc.",
        company_ticker="AAPL",
    )
    evaluation.start()
    evaluation.record_tavily_payload({"results": [{"url": "https://prior.example.com/event"}]})
    prior_attempt = evaluation.current_tavily_attempt_id()
    selected_attempts: list[str | None] = []
    token = activate_evaluation(evaluation)
    try:
        flow = MarketReviewFlow(
            analysis_executor=lambda _inputs: selected_attempts.append(
                evaluation.current_tavily_attempt_id()
            ) or {},
            initial_state=MarketReviewFlowState(
                request_id=f"tavily-{target}", company_name="Apple Inc.", input_ticker="AAPL",
                execution_mode="new", rerun_budget={target: 1},
            ),
        )
        flow._active_rerun_targets = [target]
        flow.rerun_analysis_if_needed()
    finally:
        clear_evaluation(token)

    assert selected_attempts[0] is not None
    assert selected_attempts[0].startswith(attempt_prefix)
    assert selected_attempts[0] != prior_attempt
