from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from crewai.flow.flow import Flow, listen, router, start

from multi_agent.core.confidence_gate import ConfidenceGatePolicy
from multi_agent.core.market import MarketValidationResult
from multi_agent.core.review_contracts import GateDecision, ReviewToolSummary
from multi_agent.core.state import ResearchRunState
from multi_agent.crew import MultiAgent

AnalysisExecutor = Callable[[dict[str, Any]], Any]
GateEvaluator = Callable[[Any], GateDecision]


class MarketReviewFlowState(ResearchRunState):
    request_id: str = ""
    company_name: str = ""
    current_year: str = ""
    artifacts_dir: str = ""
    final_report_path: str = ""
    local_filing_pdf_path: str = ""
    local_filing_pdf_available: str = "no"
    current_stage: str = ""
    stage_history: list[str] = Field(default_factory=list)
    analysis_result: Any = None
    analysis_gate_decision: GateDecision | None = None
    report_result: Any = None
    report_gate_decision: GateDecision | None = None
    blocking_reasons: list[str] = Field(default_factory=list)
    final_result: dict[str, Any] = Field(default_factory=dict)


class MarketReviewFlow(Flow[MarketReviewFlowState]):
    ANALYSIS_RERUN_KEY = "analysis"
    ANALYSIS_RERUN_MODEL_OVERRIDES = {
        "event_guidance_analyst": "deep",
        "fundamental_analyst": "deep",
        "quant_valuation_analyst": "deep",
    }

    def __init__(
        self,
        crew_factory: Any | None = None,
        analysis_executor: AnalysisExecutor | None = None,
        analysis_gate: GateEvaluator | None = None,
        report_writer: Callable[[Any], Any] | None = None,
        report_reviewer: GateEvaluator | None = None,
        initial_state: MarketReviewFlowState | None = None,
        **data: Any,
    ) -> None:
        # ponytail: 先关闭 Flow 级 tracing，避免本地最小迁移阶段引入无关的遥测噪音；
        # 如果后续要做正式可观测性接入，再在 runtime 层统一打开。
        data.setdefault("tracing", False)
        super().__init__(**data)
        self._crew_factory = crew_factory or MultiAgent()
        self._analysis_executor = analysis_executor or self._execute_existing_crew
        self._analysis_gate = analysis_gate or self._default_analysis_gate
        self._report_writer = report_writer or self._load_materialized_report
        self._report_reviewer = report_reviewer or self._default_report_gate
        if initial_state is not None:
            snapshot = initial_state.model_copy(deep=True)
            for key in type(snapshot).model_fields:
                setattr(self.state, key, getattr(snapshot, key))

    def _record_stage(self, stage_name: str) -> None:
        self.state.current_stage = stage_name
        self.state.stage_history.append(stage_name)

    def _execute_existing_crew(self, inputs: dict[str, Any]) -> Any:
        return self._crew_factory.crew().kickoff(inputs=inputs)

    @staticmethod
    def _allow_stage_to_continue(_: Any) -> GateDecision:
        return GateDecision(
            passed=True,
            final_decision="passed",
            trust_score=100,
        )

    @staticmethod
    def _pass_through_report(analysis_result: Any) -> Any:
        return analysis_result

    def _materialized_report_content(self) -> str:
        report_path = self.state.final_report_path.strip()
        if not report_path:
            return ""
        path = Path(report_path)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def _load_materialized_report(self, analysis_result: Any) -> Any:
        report_content = self._materialized_report_content()
        if report_content:
            return report_content
        return self._pass_through_report(analysis_result)

    def _materialized_analysis_review_content(self) -> str:
        artifacts_dir = self.state.artifacts_dir.strip()
        if not artifacts_dir:
            return ""
        review_path = Path(artifacts_dir) / "08_data_quality_review.md"
        if not review_path.exists():
            return ""
        return review_path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _structured_review_summary(result: Any, key: str) -> ReviewToolSummary | None:
        if not isinstance(result, dict):
            return None
        raw_summary = result.get(key)
        if not isinstance(raw_summary, dict):
            return None
        try:
            return ReviewToolSummary.model_validate(raw_summary)
        except Exception:
            return None

    @staticmethod
    def _contains_blocked_review_marker(review_text: str) -> bool:
        blocked_markers = (
            "必须修复（阻断级",
            "阻断级，共",
            "必须修复（Must Fix）",
            "必须修复（Blockers）",
            "Blockers",
            "必须修复（Gate 阻断风险）",
            "Gate 阻断风险",
            "⚠️ **阻断**",
            "阻断级",
            "必须修复（P0）",
            "必须修复（P0",
            "P0-",
            "### 🔴 B-",
        )
        return any(marker in review_text for marker in blocked_markers)

    def _default_analysis_gate(self, analysis_result: Any) -> GateDecision:
        summary = self._structured_review_summary(analysis_result, "analysis_review_summary")
        if summary is not None:
            return ConfidenceGatePolicy.default().evaluate(summary)
        review_text = self._materialized_analysis_review_content()
        if self._contains_blocked_review_marker(review_text):
            return GateDecision(
                passed=False,
                final_decision="blocked",
                trust_score=0,
                blocking_reasons=["analysis_review_marked_blocked"],
            )
        return self._allow_stage_to_continue(analysis_result)

    def _default_report_gate(self, report: Any) -> GateDecision:
        report_text = str(report).strip()
        if not report_text:
            report_text = self._materialized_report_content()
        blocked_markers = (
            "阻断——不得用于正式投资建议",
            "投资备忘录（已阻断）",
            "Analysis Gate 未通过",
            "分析状态：阻断（Blocked）",
            "受限版备忘录",
            "Analysis Gate 未完全通过",
            "本报告不构成正式投资建议",
        )
        if any(marker in report_text for marker in blocked_markers):
            return GateDecision(
                passed=False,
                final_decision="blocked",
                trust_score=0,
                blocking_reasons=["report_marked_blocked"],
            )
        return self._allow_stage_to_continue(report)

    def _market_validation(self) -> MarketValidationResult | None:
        market_validation = self.state.market_validation
        if market_validation is None:
            return None
        if isinstance(market_validation, dict):
            market_validation = MarketValidationResult.model_validate(market_validation)
            self.state.market_validation = market_validation
        return market_validation

    @staticmethod
    def _ensure_gate_consistency(gate: GateDecision, stage_name: str) -> None:
        expected_passed = gate.final_decision == "passed"
        if gate.passed != expected_passed:
            raise ValueError(
                f"Inconsistent gate decision at {stage_name}: "
                f"passed={gate.passed}, final_decision={gate.final_decision}"
            )

    @staticmethod
    def _coerce_terminal_gate(gate: GateDecision) -> GateDecision:
        if gate.final_decision != "rerun":
            return gate
        return GateDecision(
            passed=False,
            final_decision="blocked",
            trust_score=gate.trust_score,
            blocking_reasons=list(gate.blocking_reasons),
        )

    def _analysis_inputs(self) -> dict[str, Any]:
        market_validation = self._market_validation()
        return {
            "company_name": self.state.company_name,
            "company_ticker": self.state.input_ticker,
            "company_market_label": (
                market_validation.market_label if market_validation is not None else ""
            ),
            "market_resolution_status": (
                market_validation.resolution_status
                if market_validation is not None
                else ""
            ),
            "current_year": self.state.current_year,
            "local_filing_pdf_path": self.state.local_filing_pdf_path,
            "local_filing_pdf_available": self.state.local_filing_pdf_available,
            "model_tier_overrides": dict(self.state.model_tier_overrides),
        }

    def _review_analysis(self, analysis_result: Any) -> Any:
        self._record_stage("review_analysis")
        return analysis_result

    def _apply_analysis_gate(self, analysis_result: Any) -> GateDecision:
        self._record_stage("apply_analysis_gate")
        gate = self._analysis_gate(analysis_result)
        self._ensure_gate_consistency(gate, "apply_analysis_gate")
        self.state.analysis_gate_decision = gate
        self.state.gate_decision = gate
        return gate

    def _route_after_analysis_gate(self, gate: GateDecision) -> str:
        if gate.final_decision == "passed":
            self._record_stage("analysis_passed")
            return "analysis_passed"
        if gate.final_decision == "rerun" and self.state.rerun_budget.get(self.ANALYSIS_RERUN_KEY, 0) > 0:
            self._record_stage("analysis_needs_rerun")
            return "analysis_needs_rerun"
        self._record_stage("analysis_blocked")
        return "analysis_blocked"

    def _build_final_result(self, gate: GateDecision) -> dict[str, Any]:
        gate = self._coerce_terminal_gate(gate)
        self._ensure_gate_consistency(gate, "finalize_delivery")
        return {
            "status": gate.final_decision,
            "trust_score": gate.trust_score,
            "blocking_reasons": list(gate.blocking_reasons),
            "analysis_result": self.state.analysis_result,
            "report_result": self.state.report_result,
        }

    def _finalize(self, gate: GateDecision) -> dict[str, Any]:
        gate = self._coerce_terminal_gate(gate)
        self._record_stage("finalize_delivery")
        self._ensure_gate_consistency(gate, "finalize_delivery")
        self.state.blocking_reasons = list(gate.blocking_reasons)
        self.state.gate_decision = gate
        self.state.final_decision = gate.final_decision
        result = self._build_final_result(gate)
        self.state.final_result = result
        return result

    @start()
    def validate_market(self) -> dict[str, Any]:
        self._record_stage("validate_market")
        return self._analysis_inputs()

    @listen(validate_market)
    def run_analysis(self, inputs: dict[str, Any]) -> Any:
        self._record_stage("run_analysis")
        result = self._analysis_executor(inputs)
        self.state.analysis_result = result
        return result

    @listen(run_analysis)
    def review_analysis(self, analysis_result: Any) -> Any:
        return self._review_analysis(analysis_result)

    @listen(review_analysis)
    def apply_analysis_gate(self, analysis_result: Any) -> GateDecision:
        return self._apply_analysis_gate(analysis_result)

    @router(apply_analysis_gate)
    def route_after_analysis_gate(self, gate: GateDecision) -> str:
        return self._route_after_analysis_gate(gate)

    @listen("analysis_needs_rerun")
    def rerun_analysis_if_needed(self) -> Any:
        self._record_stage("rerun_analysis_if_needed")
        remaining_budget = self.state.rerun_budget.get(self.ANALYSIS_RERUN_KEY, 0)
        self.state.rerun_budget[self.ANALYSIS_RERUN_KEY] = max(remaining_budget - 1, 0)
        self.state.model_tier_overrides = dict(self.ANALYSIS_RERUN_MODEL_OVERRIDES)
        result = self._analysis_executor(self._analysis_inputs())
        self.state.analysis_result = result
        return result

    @listen(rerun_analysis_if_needed)
    def review_rerun_analysis(self, analysis_result: Any) -> Any:
        return self._review_analysis(analysis_result)

    @listen(review_rerun_analysis)
    def apply_rerun_analysis_gate(self, analysis_result: Any) -> GateDecision:
        return self._apply_analysis_gate(analysis_result)

    @router(apply_rerun_analysis_gate)
    def route_after_rerun_analysis_gate(self, gate: GateDecision) -> str:
        return self._route_after_analysis_gate(gate)

    @listen("analysis_passed")
    def write_report(self) -> Any:
        self._record_stage("write_report")
        report = self._report_writer(self.state.analysis_result)
        self.state.report_result = report
        return report

    @listen(write_report)
    def review_report(self, report: Any) -> GateDecision:
        self._record_stage("review_report")
        gate = self._report_reviewer(report)
        self._ensure_gate_consistency(gate, "review_report")
        self.state.report_gate_decision = gate
        self.state.gate_decision = gate
        return gate

    @listen("analysis_blocked")
    def finalize_blocked_delivery(self) -> dict[str, Any]:
        gate = self.state.analysis_gate_decision or self.state.gate_decision or GateDecision(
            passed=False,
            final_decision="blocked",
            trust_score=0,
            blocking_reasons=["analysis_gate_missing"],
        )
        return self._finalize(gate)

    @listen(review_report)
    def finalize_delivery(self, report_gate: GateDecision) -> Any:
        self.state.report_gate_decision = report_gate
        return self._finalize(report_gate)
