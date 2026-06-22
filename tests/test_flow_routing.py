import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent.core.market import MarketValidationResult, build_tool_policy
from multi_agent.core.model_routing import ModelRouter
from multi_agent.core.review_contracts import GateDecision
from multi_agent.core.state import EvidenceItem, ResearchRunState
from multi_agent.flows.market_review_flow import MarketReviewFlow, MarketReviewFlowState
from multi_agent.settings import InvestmentResearchSettings


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

    assert result["status"] == "blocked"
    assert flow.state.final_decision == "blocked"
    assert "report_marked_blocked" in result["blocking_reasons"]


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
