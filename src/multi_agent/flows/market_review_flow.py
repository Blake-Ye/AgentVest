from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import json
import re

from pydantic import Field

from crewai.flow.flow import Flow, listen, router, start

from multi_agent.core.confidence_gate import ConfidenceGatePolicy
from multi_agent.core.formal_gate import FORMAL_GATE_REQUIRED_FIELDS
from multi_agent.core.market import MarketValidationResult
from multi_agent.core.review_contracts import GateDecision, ReviewContract, ReviewToolSummary
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

    def _materialized_logic_review_content(self) -> str:
        artifacts_dir = self.state.artifacts_dir.strip()
        if not artifacts_dir:
            return ""
        review_path = Path(artifacts_dir) / "09_logic_compliance_review.md"
        if not review_path.exists():
            return ""
        return review_path.read_text(encoding="utf-8").strip()

    def _latest_metrics_payload(self) -> dict[str, Any]:
        artifacts_dir = self.state.artifacts_dir.strip()
        if not artifacts_dir:
            return {}
        metrics_path = Path(artifacts_dir) / "latest_run_metrics.json"
        if not metrics_path.exists():
            return {}
        try:
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _result_value(result: Any, key: str) -> Any:
        if isinstance(result, dict):
            return result.get(key)
        return getattr(result, key, None)

    @classmethod
    def _structured_review_summary(cls, result: Any, key: str) -> ReviewToolSummary | None:
        raw_summary = cls._result_value(result, key)
        if not isinstance(raw_summary, dict):
            return None
        try:
            return ReviewToolSummary.model_validate(raw_summary)
        except Exception:
            return None

    @classmethod
    def _normalized_delivery_state(cls, value: Any) -> str | None:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return None
        if normalized in {"formal_report", "formal"}:
            return "formal_report"
        if normalized in {"evidence_limited", "evidence_limited_report", "limited"}:
            return "evidence_limited_report"
        if normalized in {"blocked_notice", "blocked", "block"}:
            return "blocked_notice"
        return None

    @classmethod
    def _normalize_machine_readable_review_contract_payload(
        cls, payload: dict[str, Any]
    ) -> dict[str, Any]:
        normalized = dict(payload)

        if "stage" not in normalized and "review_stage" not in normalized:
            if any(key in normalized for key in ("report_mode_detected", "mode_consistent", "report_overreach")):
                normalized["review_stage"] = "report"
            else:
                normalized["review_stage"] = "analysis_review"

        raw_decision = normalized.get("decision")
        raw_delivery = normalized.get("delivery_eligibility")
        if not isinstance(raw_delivery, dict):
            raw_delivery = {}
        else:
            formal_delivery = raw_delivery.get("formal_report")
            limited_delivery = raw_delivery.get("evidence_limited_report")
            blocked_delivery = raw_delivery.get("blocked_notice")
            if any(
                isinstance(item, dict)
                for item in (formal_delivery, limited_delivery, blocked_delivery)
            ):
                flattened_delivery: dict[str, Any] = {}
                if isinstance(formal_delivery, dict):
                    flattened_delivery["formal_report_allowed"] = bool(
                        formal_delivery.get("allowed")
                    )
                    reason = str(formal_delivery.get("reason", "")).strip()
                    if reason:
                        flattened_delivery["formal_block_reasons"] = [reason]
                if isinstance(limited_delivery, dict):
                    flattened_delivery["evidence_limited_report_allowed"] = bool(
                        limited_delivery.get("allowed")
                    )
                    reason = str(limited_delivery.get("reason", "")).strip()
                    if reason:
                        flattened_delivery["evidence_limited_rationale"] = reason
                if isinstance(blocked_delivery, dict):
                    flattened_delivery["blocked_notice_required"] = bool(
                        blocked_delivery.get("required")
                    )
                    reason = str(blocked_delivery.get("reason", "")).strip()
                    if reason:
                        flattened_delivery["blocked_notice_reason"] = reason
                raw_delivery = flattened_delivery

        if isinstance(raw_decision, dict):
            if not raw_delivery:
                raw_delivery = {
                    key: raw_decision[key]
                    for key in (
                        "formal_report_allowed",
                        "evidence_limited_report_allowed",
                        "blocked_notice_required",
                        "recommended_delivery_state",
                    )
                    if key in raw_decision
                }
            if "primary_class" in raw_decision and "failure_taxonomy" not in normalized:
                normalized["failure_taxonomy"] = {"primary_class": raw_decision["primary_class"]}
            gate_outcome = raw_decision.get("gate_outcome")
        elif isinstance(raw_decision, str):
            gate_outcome = raw_decision
        else:
            gate_outcome = None

        raw_failure_taxonomy = normalized.get("failure_taxonomy")
        if isinstance(raw_failure_taxonomy, dict) and not any(
            key in raw_failure_taxonomy for key in ("primary_class", "primary_code")
        ):
            secondary_causes = [
                key
                for key, value in raw_failure_taxonomy.items()
                if isinstance(value, bool) and value and key != "research_blocked"
            ]
            if raw_failure_taxonomy.get("research_blocked") is True:
                primary_class = "research_blocked"
            elif any(
                raw_failure_taxonomy.get(key) is True
                for key in ("tool_failure", "financial_data_incomplete")
            ):
                primary_class = "pipeline_degraded"
            elif raw_failure_taxonomy.get("critical_conflict") is True:
                primary_class = "critical_conflict"
            elif raw_failure_taxonomy.get("market_policy_violation") is True:
                primary_class = "policy_violation"
            elif raw_failure_taxonomy.get("unsupported_critical_claim") is True:
                primary_class = "unsupported_claim"
            else:
                primary_class = "none"
            normalized["failure_taxonomy"] = {
                "primary_class": primary_class,
                "secondary_causes": secondary_causes,
            }

        recommended_delivery_state = cls._normalized_delivery_state(
            raw_delivery.get("recommended_delivery_state")
        )
        if recommended_delivery_state is None:
            recommended_delivery_state = cls._normalized_delivery_state(
                raw_decision.get("recommended_delivery_state") if isinstance(raw_decision, dict) else None
            )
        if recommended_delivery_state is not None:
            raw_delivery["recommended_delivery_state"] = recommended_delivery_state
        if raw_delivery:
            normalized["delivery_eligibility"] = raw_delivery

        normalized_gate_outcome = str(gate_outcome or "").strip().lower()
        if normalized_gate_outcome in {"pass", "passed"}:
            normalized["decision"] = {"gate_outcome": "pass", "decision_confidence": "medium"}
        elif normalized_gate_outcome in {"block", "blocked"}:
            normalized["decision"] = {"gate_outcome": "block", "decision_confidence": "medium"}
        elif normalized_gate_outcome == "rerun":
            normalized["decision"] = {"gate_outcome": "rerun", "decision_confidence": "medium"}
        else:
            normalized["decision"] = {"gate_outcome": "rerun", "decision_confidence": "medium"}

        raw_review_summary = normalized.get("review_summary")
        if isinstance(raw_review_summary, str):
            normalized["review_summary"] = {
                "one_sentence_summary": raw_review_summary,
                "operator_notes": "",
            }

        artifact_refs = normalized.get("artifact_refs")
        if isinstance(artifact_refs, list) and artifact_refs and all(
            isinstance(item, str) for item in artifact_refs
        ):
            normalized["artifact_refs"] = [{"artifact": item} for item in artifact_refs]

        rerun_reasons = normalized.get("rerun_reasons")
        if isinstance(rerun_reasons, list) and rerun_reasons:
            normalized["rerun_reasons"] = [
                item.get("action", str(item)).strip() if isinstance(item, dict) else str(item).strip()
                for item in rerun_reasons
                if str(item).strip()
            ]

        tool_health = normalized.get("tool_health")
        if isinstance(tool_health, dict) and "overall_status" not in tool_health:
            failed_tools = sorted(
                tool_name
                for tool_name, status in tool_health.items()
                if str(status).strip().lower() == "failed"
            )
            degraded_tools = sorted(
                tool_name
                for tool_name, status in tool_health.items()
                if str(status).strip().lower() == "degraded"
            )
            overall_status = "failed" if failed_tools else "degraded" if degraded_tools else "healthy"
            normalized["tool_health"] = {
                "overall_status": overall_status,
                "failed_tools": failed_tools,
                "degraded_tools": degraded_tools,
                "tool_status": [
                    {"tool_name": str(tool_name), "status": str(status)}
                    for tool_name, status in tool_health.items()
                ],
            }

        return normalized

    @classmethod
    def _machine_readable_review_contract_from_text(cls, review_text: str) -> ReviewContract | None:
        if not review_text.strip():
            return None
        match = re.search(
            r"PART A:\s*MACHINE_READABLE_JSON[\s\S]*?```(?:json)?\s*([\s\S]*?)\s*```",
            review_text,
            re.IGNORECASE,
        )
        if match is None:
            return None
        raw_contract = match.group(1).strip()
        if not raw_contract:
            return None
        try:
            return ReviewContract.model_validate_json(raw_contract)
        except Exception:
            try:
                normalized_payload = cls._normalize_machine_readable_review_contract_payload(
                    json.loads(raw_contract)
                )
                return ReviewContract.model_validate(normalized_payload)
            except Exception:
                return None

    @staticmethod
    def _gate_decision_from_review_contract(contract: ReviewContract) -> GateDecision:
        policy_decision = ConfidenceGatePolicy.default().evaluate(contract)
        delivery = contract.delivery_eligibility
        recommended_delivery_state = delivery.recommended_delivery_state

        if delivery.blocked_notice_required or recommended_delivery_state == "blocked_notice":
            return GateDecision(
                passed=False,
                final_decision="blocked",
                trust_score=policy_decision.trust_score,
                blocking_reasons=list(contract.blocking_reasons) or ["delivery_blocked"],
            )

        if (
            not delivery.formal_report_allowed
            and (delivery.evidence_limited_report_allowed or contract.allow_limited_delivery)
        ) or recommended_delivery_state == "evidence_limited_report":
            return GateDecision(
                passed=False,
                final_decision="evidence_limited",
                trust_score=min(policy_decision.trust_score, 60),
                blocking_reasons=list(contract.blocking_reasons),
            )

        return policy_decision

    @staticmethod
    def _review_task_outputs(result: Any) -> list[Any]:
        task_outputs = MarketReviewFlow._result_value(result, "tasks_output")
        if not isinstance(task_outputs, list):
            return []
        return task_outputs

    @classmethod
    def _review_text_from_tasks_output(cls, result: Any) -> str:
        task_outputs = cls._review_task_outputs(result)
        fallback_review = ""
        for task_output in task_outputs:
            raw_content = str(getattr(task_output, "raw", "")).strip()
            if not raw_content:
                continue
            task_name = str(getattr(task_output, "name", "")).strip()
            if task_name == "data_quality_review_task":
                return raw_content
            if not fallback_review and (
                "审查工具结构化结果摘要" in raw_content
                or "数据质量审查报告" in raw_content
                or "financial_field_completeness_tool" in raw_content
            ):
                fallback_review = raw_content
        return fallback_review

    @classmethod
    def _has_current_review_task_output(cls, result: Any) -> bool:
        return bool(cls._review_text_from_tasks_output(result))

    @staticmethod
    def _extract_ratio(text: str, pattern: str) -> float | None:
        match = re.search(pattern, text, re.IGNORECASE)
        if match is None:
            return None
        try:
            value = float(match.group(1))
        except (TypeError, ValueError):
            return None
        if "%" in match.group(0) or value > 1.0:
            return value / 100.0
        return value

    @staticmethod
    def _extract_count(text: str, label: str) -> int:
        pattern = rf"\*{{0,2}}(\d+)\*{{0,2}}\s+{re.escape(label)}"
        match = re.search(pattern, text, re.IGNORECASE)
        if match is None:
            return 0
        return int(match.group(1))

    @staticmethod
    def _review_lines(review_text: str) -> list[str]:
        return [line.strip() for line in review_text.splitlines() if line.strip()]

    @staticmethod
    def _contains_negated_blocked_marker(review_text: str) -> bool:
        normalized_text = re.sub(r"\s+", "", review_text).lower()
        negated_patterns = (
            r"非阻断级",
            r"已解除阻断",
            r"解除阻断",
            r"无需阻断",
            r"记录即可",
        )
        return any(
            re.search(pattern, normalized_text, re.IGNORECASE) for pattern in negated_patterns
        )

    @classmethod
    def _explicit_blocked_decision_from_review_text(cls, review_text: str) -> GateDecision | None:
        if not review_text.strip():
            return None
        blocked_patterns = (
            r"gate风险.{0,24}block(?:ed)?",
            r"gate阻断风险",
            r"block(?:ed)?.{0,24}gate",
            r"gate风险.{0,24}阻断",
            r"gate\s*risk.{0,24}(?:阻断|block(?:ed)?)",
            r"审查结论[：:]\s*阻断",
        )
        for line in cls._review_lines(review_text):
            if cls._contains_negated_blocked_marker(line):
                continue
            normalized_line = re.sub(r"\s+", "", line).lower()
            if any(re.search(pattern, normalized_line, re.IGNORECASE) for pattern in blocked_patterns):
                return GateDecision(
                    passed=False,
                    final_decision="blocked",
                    trust_score=0,
                    blocking_reasons=["analysis_review_marked_blocked"],
                )
        return None

    @classmethod
    def _blocked_decision_from_review_text(cls, review_text: str) -> GateDecision | None:
        if not review_text.strip():
            return None
        if cls._contains_blocked_review_marker(review_text):
            return GateDecision(
                passed=False,
                final_decision="blocked",
                trust_score=0,
                blocking_reasons=["analysis_review_marked_blocked"],
            )
        return None

    @classmethod
    def _summary_from_review_text(cls, review_text: str) -> ReviewToolSummary | None:
        if not review_text.strip():
            return None
        evidence_coverage_ratio = cls._extract_ratio(
            review_text,
            r"evidence_coverage_tool.*?覆盖率\s*\*{0,2}(\d+(?:\.\d+)?)%?\*{0,2}",
        )
        financial_coverage_score = cls._extract_ratio(
            review_text,
            r"financial_field_completeness_tool.*?(?:整体|总)覆盖率\s*\*{0,2}(\d+(?:\.\d+)?)%?\*{0,2}",
        )
        if evidence_coverage_ratio is None or financial_coverage_score is None:
            return None
        gate_financial_coverage_score = cls._extract_ratio(
            review_text,
            r"financial_field_completeness_tool.*?Gate\s*覆盖率\s*\*{0,2}(\d+(?:\.\d+)?)%?\*{0,2}",
        )
        critical_conflict_count = cls._extract_count(review_text, "critical conflicts")
        violation_count = cls._extract_count(review_text, "violations")
        unsupported_claim_count = 0
        if not re.search(r"\b无\s+unsupported claims\b", review_text, re.IGNORECASE):
            unsupported_claim_count = cls._extract_count(review_text, "unsupported claims")
        return ReviewToolSummary(
            evidence_coverage_ratio=evidence_coverage_ratio,
            financial_coverage_score=financial_coverage_score,
            gate_financial_coverage_score=gate_financial_coverage_score,
            critical_conflict_count=critical_conflict_count,
            market_policy_violations=(
                ["review_markdown_violation"] if violation_count > 0 else []
            ),
            unsupported_critical_claims=(
                ["review_markdown_unsupported_claim"] if unsupported_claim_count > 0 else []
            ),
            blocking_reasons=[],
        )

    def _summary_from_latest_metrics(self) -> ReviewToolSummary | None:
        metrics = self._latest_metrics_payload()
        if not metrics:
            return None
        financial_fields = metrics.get("financial_fields", {})
        if not isinstance(financial_fields, dict) or not financial_fields:
            return None
        extracted_fields = {
            name: field.get("normalized_value")
            for name, field in financial_fields.items()
            if isinstance(field, dict) and field.get("extracted")
        }
        gate_required_fields = FORMAL_GATE_REQUIRED_FIELDS
        gate_missing_fields = [
            field_name for field_name in gate_required_fields if field_name not in extracted_fields
        ]
        gate_coverage = (
            (len(gate_required_fields) - len(gate_missing_fields)) / len(gate_required_fields)
            if gate_required_fields
            else 1.0
        )
        return ReviewToolSummary(
            evidence_coverage_ratio=0.0,
            financial_coverage_score=float(metrics.get("financial_fields_success_rate", 0.0) or 0.0),
            gate_financial_coverage_score=gate_coverage,
            critical_conflict_count=0,
            market_policy_violations=[],
            unsupported_critical_claims=[],
            blocking_reasons=[],
        )

    @classmethod
    def _contains_blocked_review_marker(cls, review_text: str) -> bool:
        blocked_patterns = (
            r"必须修复\s*[（(]\s*(阻断级|must\s*fix|blockers?|gate\s*阻断风险|p0)",
            r"阻断级",
            r"gate\s*风险.*?(阻断|block(?:ed)?)",
            r"gate\s*risk.*?(阻断|block(?:ed)?)",
            r"⚠️?\s*\*{0,2}\s*阻断",
            r"\bp0-\d+\b",
            r"###\s*.*\bb-\d+\b",
        )
        for line in cls._review_lines(review_text):
            if cls._contains_negated_blocked_marker(line):
                continue
            if any(re.search(pattern, line, re.IGNORECASE) is not None for pattern in blocked_patterns):
                return True
        return False

    def _default_analysis_gate(self, analysis_result: Any) -> GateDecision:
        review_text = self._review_text_from_tasks_output(analysis_result)
        materialized_review_text = self._materialized_analysis_review_content()
        summary = self._structured_review_summary(analysis_result, "analysis_review_summary")
        if summary is not None:
            return ConfidenceGatePolicy.default().evaluate(summary)
        for candidate_review_text in (review_text, materialized_review_text):
            contract = self._machine_readable_review_contract_from_text(candidate_review_text)
            if contract is not None:
                return self._gate_decision_from_review_contract(contract)
        summary = self._summary_from_latest_metrics()
        if summary is not None:
            return ConfidenceGatePolicy.default().evaluate(summary)
        for candidate_review_text in (review_text, materialized_review_text):
            summary = self._summary_from_review_text(candidate_review_text)
            if summary is not None:
                return ConfidenceGatePolicy.default().evaluate(summary)
            blocked_decision = self._explicit_blocked_decision_from_review_text(candidate_review_text)
            if blocked_decision is not None:
                return blocked_decision
            blocked_decision = self._blocked_decision_from_review_text(candidate_review_text)
            if blocked_decision is not None:
                return blocked_decision
        return GateDecision(
            passed=False,
            final_decision="rerun",
            trust_score=0,
            blocking_reasons=["analysis_review_summary_missing"],
        )

    @staticmethod
    def _blocked_report_markers() -> tuple[str, ...]:
        return (
            "阻断——不得用于正式投资建议",
            "投资备忘录（已阻断）",
            "Analysis Gate 未通过",
            "分析状态：阻断（Blocked）",
            "本报告不构成正式投资建议",
        )

    @staticmethod
    def _evidence_limited_report_markers() -> tuple[str, ...]:
        return (
            "受限版备忘录",
            "Analysis Gate 未完全通过",
        )

    def _expected_report_mode(self) -> str:
        analysis_gate = self.state.analysis_gate_decision or self.state.gate_decision
        if analysis_gate is None:
            return "formal_report"
        final_decision = self._coerce_terminal_gate(analysis_gate).final_decision
        if final_decision == "evidence_limited":
            return "evidence_limited_report"
        return "formal_report"

    def _infer_report_mode(self, report_text: str) -> str:
        if any(marker in report_text for marker in self._blocked_report_markers()):
            return "blocked_notice"
        if any(marker in report_text for marker in self._evidence_limited_report_markers()):
            return "evidence_limited_report"
        return "formal_report"

    def _default_report_gate(self, report: Any) -> GateDecision:
        report_text = str(report).strip()
        if not report_text:
            report_text = self._materialized_report_content()
        report_mode = self._infer_report_mode(report_text)
        if report_mode == "blocked_notice":
            return GateDecision(
                passed=False,
                final_decision="blocked",
                trust_score=0,
                blocking_reasons=["report_marked_blocked"],
            )
        expected_mode = self._expected_report_mode()
        if expected_mode == "evidence_limited_report" and report_mode != "evidence_limited_report":
            return GateDecision(
                passed=False,
                final_decision="blocked",
                trust_score=0,
                blocking_reasons=["report_mode_mismatch"],
            )
        if report_mode == "evidence_limited_report":
            return GateDecision(
                passed=False,
                final_decision="evidence_limited",
                trust_score=60,
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
    def _is_evidence_limited_rerun(gate: GateDecision) -> bool:
        if gate.final_decision != "rerun":
            return False
        reasons = [reason for reason in gate.blocking_reasons if reason]
        return bool(reasons) and set(reasons) == {"gate_financial_coverage_incomplete"}

    @staticmethod
    def _logic_review_supports_evidence_limited_delivery(review_text: str) -> bool:
        if not review_text.strip():
            return False
        negative_patterns = (
            r"不得交付",
            r"禁止交付",
            r"不应交付",
            r"不可交付",
            r"必须阻断",
            r"阻断级",
            r"\bblockers?\b",
            r"gate\s*block",
            r"gate\s*阻断",
            r"审查结论[：:]\s*阻断",
        )
        if any(re.search(pattern, review_text, re.IGNORECASE) for pattern in negative_patterns):
            return False
        positive_patterns = (
            r"有条件通过",
            r"达到[“\"]?受限版[”\"]?的预期标准",
            r"不损坏报告的核心逻辑和整体可靠性",
        )
        return (
            "受限版" in review_text
            and any(re.search(pattern, review_text, re.IGNORECASE) for pattern in positive_patterns)
        )

    def _should_fallback_to_evidence_limited(self, gate: GateDecision) -> bool:
        if self._is_evidence_limited_rerun(gate):
            return True
        if gate.final_decision != "rerun":
            return False
        reasons = [reason for reason in gate.blocking_reasons if reason]
        if not reasons or set(reasons) != {"analysis_review_summary_missing"}:
            return False
        return self._logic_review_supports_evidence_limited_delivery(
            self._materialized_logic_review_content()
        )

    def _coerce_terminal_gate(self, gate: GateDecision) -> GateDecision:
        if gate.final_decision != "rerun":
            return gate
        if self._should_fallback_to_evidence_limited(gate):
            return GateDecision(
                passed=False,
                final_decision="evidence_limited",
                trust_score=gate.trust_score,
                blocking_reasons=list(gate.blocking_reasons),
            )
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
            "run_id": self.state.request_id,
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
        if self._should_fallback_to_evidence_limited(gate):
            self._record_stage("analysis_evidence_limited")
            self.state.analysis_gate_decision = self._coerce_terminal_gate(gate)
            self.state.gate_decision = self.state.analysis_gate_decision
            return "analysis_evidence_limited"
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

    @listen("analysis_evidence_limited")
    def write_evidence_limited_report(self) -> Any:
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

    @listen(write_evidence_limited_report)
    def review_evidence_limited_report(self, report: Any) -> GateDecision:
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

    @listen(review_evidence_limited_report)
    def finalize_evidence_limited_delivery(self, report_gate: GateDecision) -> Any:
        self.state.report_gate_decision = report_gate
        if report_gate.final_decision == "blocked":
            return self._finalize(report_gate)
        locked_gate = (
            self.state.analysis_gate_decision
            or self.state.gate_decision
            or GateDecision(
                passed=False,
                final_decision="evidence_limited",
                trust_score=report_gate.trust_score,
                blocking_reasons=["gate_financial_coverage_incomplete"],
            )
        )
        locked_gate = self._coerce_terminal_gate(locked_gate)
        if report_gate.final_decision == "evidence_limited":
            merged_reasons = list(dict.fromkeys([
                *locked_gate.blocking_reasons,
                *report_gate.blocking_reasons,
            ]))
            locked_gate = GateDecision(
                passed=False,
                final_decision="evidence_limited",
                trust_score=min(locked_gate.trust_score, report_gate.trust_score),
                blocking_reasons=merged_reasons,
            )
        return self._finalize(locked_gate)
