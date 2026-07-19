import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent.core.confidence_gate import ConfidenceGatePolicy
from multi_agent.core.evidence import ResearchEvidenceBundle
from multi_agent.core.market import MarketValidationResult, build_tool_policy
from multi_agent.core.model_routing import ModelRouter
from multi_agent.core.review_contracts import GateDecision, ReviewContract
from multi_agent.core.state import EvidenceItem, ResearchRunState
from multi_agent.flows.market_review_flow import MarketReviewFlow, MarketReviewFlowState
from multi_agent.settings import InvestmentResearchSettings


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
