from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import json
import re

from pydantic import Field

from crewai.flow.flow import Flow, listen, router, start

from multi_agent.core.confidence_gate import ConfidenceGatePolicy
from multi_agent.core.formal_gate import FORMAL_GATE_REQUIRED_FIELDS
from multi_agent.core.evidence import ResearchEvidenceBundle
from multi_agent.core.market import MarketValidationResult
from multi_agent.core.delivery import DeliveryValidator
from multi_agent.core.report_document import (
    REQUIRED_SECTION_KEYS, SECTION_HEADINGS, ReportDocument, ReportGenerationContext, SourceReference,
)
from multi_agent.core.review_contracts import (
    FinalDecisionRecord,
    GateDecision,
    RepairAction,
    ReviewContract,
    ReviewToolSummary,
    review_contract_from_payload,
    review_contract_from_text,
)
from multi_agent.core.state import ResearchRunState
from multi_agent.crew import MultiAgent
from multi_agent.evaluation import current_evaluation

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
    MAX_REPAIR_ROUNDS = 3
    ANALYSIS_RERUN_MODEL_OVERRIDES = {
        "event_guidance_analyst": "deep",
        "fundamental_analyst": "deep",
        "quant_valuation_analyst": "deep",
    }
    _ANALYSIS_CONTRACT_FILE = "08_data_quality_review.json"
    _REPORT_CONTRACT_FILE = "09_logic_compliance_review.json"
    _EVIDENCE_PRODUCER_TARGETS = frozenset({
        "market_validation_analyst",
        "event_guidance_analyst",
        "fundamental_analyst",
        "quant_valuation_analyst",
    })
    _ANALYSIS_REPAIR_TARGETS = _EVIDENCE_PRODUCER_TARGETS | {"data_quality_reviewer"}
    _REPORT_REPAIR_TARGETS = frozenset({"report_writing_analyst", "logic_compliance_reviewer"})

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
        self._crew_factory = crew_factory
        self._analysis_executor = analysis_executor or self._execute_existing_crew
        self._analysis_gate = analysis_gate or self._default_analysis_gate
        self._report_writer = report_writer or self._load_materialized_report
        self._report_reviewer = report_reviewer or self._default_report_gate
        self._report_writer_injected = report_writer is not None
        self._report_reviewer_injected = report_reviewer is not None
        self._typed_report_execution: Any = None
        self._active_rerun_targets: list[str] = []
        if initial_state is not None:
            snapshot = initial_state.model_copy(deep=True)
            for key in type(snapshot).model_fields:
                setattr(self.state, key, getattr(snapshot, key))
        if self._typed_run():
            for key in (self.ANALYSIS_RERUN_KEY, "report_writing_analyst"):
                remaining = self.state.rerun_budget.get(key, self.MAX_REPAIR_ROUNDS)
                self.state.rerun_budget[key] = min(max(remaining, 0), self.MAX_REPAIR_ROUNDS)

    def _record_stage(self, stage_name: str) -> None:
        self.state.current_stage = stage_name
        self.state.stage_history.append(stage_name)

    def _execute_existing_crew(self, inputs: dict[str, Any]) -> Any:
        crew_factory = self._crew_factory or MultiAgent()
        configure = getattr(crew_factory, "configure_run", None)
        if callable(configure):
            configure(
                model_tier_overrides=inputs.get("model_tier_overrides", {}),
                rerun_targets=inputs.get("rerun_targets", []),
            )
        targets = inputs.get("rerun_targets", [])
        review_crew = getattr(crew_factory, "analysis_review_crew", None)
        targeted_crew = getattr(crew_factory, "targeted_analysis_crew", None)
        analysis_crew = getattr(crew_factory, "analysis_crew", None)
        if not callable(review_crew):
            if targets and callable(targeted_crew):
                return targeted_crew(targets).kickoff(inputs=inputs)
            if callable(analysis_crew):
                return analysis_crew().kickoff(inputs=inputs)
            return crew_factory.crew().kickoff(inputs=inputs)

        producer_targets = [
            target for target in targets if target in self._EVIDENCE_PRODUCER_TARGETS
        ]
        if producer_targets:
            if not callable(targeted_crew):
                raise ValueError("targeted analysis crew is unavailable")
            producer_result = targeted_crew(producer_targets).kickoff(inputs=inputs)
            self._ingest_typed_analysis_result(producer_result)
        elif not targets:
            if not callable(analysis_crew):
                raise ValueError("analysis crew is unavailable")
            producer_result = analysis_crew().kickoff(inputs=inputs)
            self._ingest_typed_analysis_result(producer_result)

        self._evidence_bundle_for_gate()
        reviewer_inputs = dict(inputs)
        reviewer_inputs["review_evidence_context_json"] = (
            self._review_evidence_context_json()
        )
        return review_crew().kickoff(inputs=reviewer_inputs)

    def _typed_run(self) -> bool:
        return self.state.execution_mode == "new"

    def _artifact_json_path(self, filename: str) -> Path | None:
        artifacts_dir = self.state.artifacts_dir.strip()
        return Path(artifacts_dir) / filename if artifacts_dir else None

    def _persist_json_artifact(self, filename: str, payload: dict[str, object]) -> None:
        path = self._artifact_json_path(filename)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _clear_new_run_artifacts(self) -> None:
        """Initial typed runs begin without evidence or analysis-review state."""
        self.state.evidence_bundle = None
        self.state.analysis_review_contract = None
        for filename in ("10_research_evidence.json", self._ANALYSIS_CONTRACT_FILE):
            path = self._artifact_json_path(filename)
            if path is not None and path.exists():
                path.unlink()

    def _clear_typed_rerun_artifacts(self, targets: list[str]) -> None:
        """Clear only artifacts owned by the scheduled typed repair targets."""
        if self._EVIDENCE_PRODUCER_TARGETS.intersection(targets):
            self._clear_new_run_artifacts()
            return
        self.state.analysis_review_contract = None
        review_path = self._artifact_json_path(self._ANALYSIS_CONTRACT_FILE)
        if review_path is not None and review_path.exists():
            review_path.unlink()

    def _prepare_tavily_evidence_attempt(self, targets: list[str]) -> None:
        evaluation = current_evaluation()
        if evaluation is None or not self._EVIDENCE_PRODUCER_TARGETS.intersection(targets):
            return
        if "event_guidance_analyst" in targets:
            evaluation.begin_tavily_evidence_attempt()
            return
        evaluation.preserve_tavily_evidence_snapshot()

    @classmethod
    def _structured_result_payload(cls, result: Any) -> dict[str, object]:
        if isinstance(result, dict):
            return result
        for attribute in ("json_dict", "pydantic"):
            value = getattr(result, attribute, None)
            if isinstance(value, dict):
                return value
            model_dump = getattr(value, "model_dump", None)
            if callable(model_dump):
                dumped = model_dump(mode="json")
                if isinstance(dumped, dict):
                    return dumped
        model_dump = getattr(result, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump(mode="json")
            if isinstance(dumped, dict):
                return dumped
        return {}

    @staticmethod
    def _strict_contract_payload(value: Any) -> dict[str, object] | None:
        if isinstance(value, ReviewContract):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    def _load_strict_contract(
        self, *, value: Any, filename: str, expected_stage: str
    ) -> tuple[ReviewContract | None, str | None]:
        payload = self._strict_contract_payload(value)
        if payload is None:
            path = self._artifact_json_path(filename)
            if path is not None and path.exists():
                try:
                    candidate = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return None, "invalid"
                payload = candidate if isinstance(candidate, dict) else None
        if payload is None:
            return None, "missing"
        try:
            contract = review_contract_from_payload(payload)  # type: ignore[arg-type]
        except Exception:
            return None, "invalid"
        if contract.stage != expected_stage:
            return None, "invalid"
        self._persist_json_artifact(filename, contract.model_dump(mode="json"))
        return contract, None

    def _ingest_typed_analysis_result(self, result: Any) -> None:
        payload = self._structured_result_payload(result)
        raw_bundle = payload.get("evidence_bundle")
        if raw_bundle is not None:
            try:
                bundle = ResearchEvidenceBundle.model_validate(raw_bundle)
            except Exception:
                return
            self.state.evidence_bundle = bundle
            self._persist_json_artifact("10_research_evidence.json", bundle.model_dump(mode="json"))
        elif self._typed_run():
            # A real targeted Crew writes the canonical artifact; reload that attempt's snapshot.
            self._evidence_bundle_for_gate()
        raw_contract = payload.get("analysis_review_contract")
        contract, error = self._load_strict_contract(
            value=raw_contract,
            filename=self._ANALYSIS_CONTRACT_FILE,
            expected_stage="analysis_review",
        )
        if contract is not None:
            self.state.analysis_review_contract = contract
        elif error == "invalid":
            # Keep no prior contract: a malformed fresh output must fail closed.
            self.state.analysis_review_contract = None

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

    @classmethod
    def _task_output_raw(
        cls, result: Any, task_name: str, *, fallback_index: int | None = None
    ) -> str:
        task_outputs = cls._review_task_outputs(result)
        for task_output in task_outputs:
            if str(getattr(task_output, "name", "")).strip() == task_name:
                return str(getattr(task_output, "raw", "")).strip()
        if fallback_index is not None and task_outputs:
            try:
                return str(getattr(task_outputs[fallback_index], "raw", "")).strip()
            except IndexError:
                return ""
        return ""

    def _execute_existing_report_writer(self, writer_input: dict[str, Any]) -> Any:
        crew_factory = self._crew_factory or MultiAgent()
        configure = getattr(crew_factory, "configure_run", None)
        if callable(configure):
            configure(model_tier_overrides=dict(self.state.model_tier_overrides))
        report_crew = getattr(crew_factory, "report_writer_crew", None)
        if not callable(report_crew):
            report_crew = getattr(crew_factory, "report_crew", None)
        if not callable(report_crew):
            raise ValueError("writer_payload_invalid")
        try:
            result = report_crew().kickoff(inputs=writer_input)
        except Exception as error:
            raise ValueError("writer_payload_invalid") from error
        self._typed_report_execution = result
        raw = self._task_output_raw(result, "investment_report_task", fallback_index=0)
        return raw

    def _execute_existing_report_review(self, report: ReportDocument) -> str:
        crew_factory = self._crew_factory or MultiAgent()
        configure = getattr(crew_factory, "configure_run", None)
        if callable(configure):
            configure(model_tier_overrides=dict(self.state.model_tier_overrides))
        review_crew = getattr(crew_factory, "report_review_crew", None)
        if not callable(review_crew):
            return self._task_output_raw(
                self._typed_report_execution,
                "logic_compliance_review_task",
                fallback_index=-1,
            )
        context = self.state.report_context
        result = review_crew().kickoff(inputs={
            "company_name": self.state.company_name,
            "company_ticker": self.state.input_ticker,
            "run_id": self.state.request_id,
            "REPORT_CONTEXT_JSON": context.model_dump_json() if context is not None else "{}",
            "REPORT_DOCUMENT_JSON": report.model_dump_json(),
        })
        self._typed_report_execution = result
        return self._task_output_raw(
            result,
            "logic_compliance_review_task",
            fallback_index=0,
        )

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
    def _machine_readable_review_contract_from_text(
        cls, review_text: str
    ) -> tuple[ReviewContract | None, str | None]:
        if not review_text.strip():
            return None, None
        try:
            return review_contract_from_text(review_text), None
        except Exception:
            return None, "invalid_review_contract"

    def _evidence_bundle_for_gate(self) -> ResearchEvidenceBundle | None:
        if self.state.evidence_bundle is not None and not self._typed_run():
            return self.state.evidence_bundle
        artifacts_dir = self.state.artifacts_dir.strip()
        if not artifacts_dir:
            return self.state.evidence_bundle
        evidence_path = Path(artifacts_dir) / "10_research_evidence.json"
        if not evidence_path.exists():
            if self._typed_run():
                return None
            return self.state.evidence_bundle
        try:
            bundle = ResearchEvidenceBundle.model_validate_json(
                evidence_path.read_text(encoding="utf-8")
            )
        except Exception:
            return None
        self.state.evidence_bundle = bundle
        return bundle

    @staticmethod
    def _invalid_review_contract_decision() -> GateDecision:
        action = RepairAction(
            target="data_quality_reviewer",
            code="invalid_review_contract",
            instruction="按严格 ReviewContract 契约重新生成审查 JSON。",
        )
        return GateDecision(
            passed=False,
            final_decision="blocked",
            trust_score=0,
            blocking_reasons=["invalid_review_contract"],
            repair_actions=[action],
        )

    @staticmethod
    def _missing_review_contract_decision() -> GateDecision:
        action = RepairAction(
            target="data_quality_reviewer",
            code="review_contract_missing",
            instruction="生成严格 ReviewContract JSON 后重新审查。",
        )
        return GateDecision(
            passed=False,
            final_decision="rerun",
            trust_score=0,
            blocking_reasons=["review_contract_missing"],
            repair_actions=[action],
        )

    @staticmethod
    def _typed_missing_review_contract_decision() -> GateDecision:
        action = RepairAction(
            target="data_quality_reviewer",
            code="analysis_review_contract_missing",
            instruction="生成严格 08_data_quality_review.json 后重新审查。",
        )
        return GateDecision(
            passed=False,
            final_decision="rerun",
            trust_score=0,
            blocking_reasons=["analysis_review_contract_missing"],
            repair_actions=[action],
        )

    def _has_evidence_context(self) -> bool:
        if self.state.evidence_bundle is not None:
            return True
        artifacts_dir = self.state.artifacts_dir.strip()
        return bool(artifacts_dir and (Path(artifacts_dir) / "10_research_evidence.json").exists())

    def _gate_decision_from_review_contract(self, contract: ReviewContract) -> GateDecision:
        bundle = self._evidence_bundle_for_gate()
        if bundle is None:
            return GateDecision(
                passed=False,
                final_decision="rerun",
                trust_score=0,
                blocking_reasons=["research_evidence_bundle_missing"],
                repair_actions=[
                    RepairAction(
                        target="fundamental_analyst",
                        code="research_evidence_bundle_missing",
                        instruction="生成并持久化 10_research_evidence.json 后重新审查。",
                    )
                ],
            )
        return ConfidenceGatePolicy.default().evaluate(bundle, contract)

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
        if self._typed_run():
            bundle = self._evidence_bundle_for_gate()
            raw_contract = (
                self._structured_result_payload(analysis_result).get("analysis_review_contract")
            )
            contract = self.state.analysis_review_contract
            if contract is None:
                contract, error = self._load_strict_contract(
                    value=raw_contract,
                    filename=self._ANALYSIS_CONTRACT_FILE,
                    expected_stage="analysis_review",
                )
                if contract is None and error == "missing":
                    # Crew returns the strict JSON block as task text; persist it as the
                    # canonical JSON sibling before the typed gate evaluates it.
                    review_text = self._review_text_from_tasks_output(analysis_result)
                    contract, error = self._machine_readable_review_contract_from_text(review_text)
                    if contract is not None:
                        if contract.stage != "analysis_review":
                            contract, error = None, "invalid"
                        else:
                            self._persist_json_artifact(
                                self._ANALYSIS_CONTRACT_FILE,
                                contract.model_dump(mode="json"),
                            )
                if error == "invalid":
                    return self._invalid_review_contract_decision()
                if contract is None:
                    return self._typed_missing_review_contract_decision()
                self.state.analysis_review_contract = contract
            if bundle is None:
                return GateDecision(
                    passed=False,
                    final_decision="blocked",
                    trust_score=0,
                    blocking_reasons=["evidence_bundle_missing"],
                )
            return self._gate_decision_from_review_contract(contract)
        review_text = self._review_text_from_tasks_output(analysis_result)
        materialized_review_text = self._materialized_analysis_review_content()
        for candidate_review_text in (review_text, materialized_review_text):
            contract, contract_error = self._machine_readable_review_contract_from_text(
                candidate_review_text
            )
            if contract_error is not None:
                return self._invalid_review_contract_decision()
            if contract is not None:
                return self._gate_decision_from_review_contract(contract)
        if self._has_evidence_context():
            return self._missing_review_contract_decision()
        summary = self._structured_review_summary(analysis_result, "analysis_review_summary")
        if summary is not None:
            return ConfidenceGatePolicy.default().evaluate_legacy(summary)
        summary = self._summary_from_latest_metrics()
        if summary is not None:
            return ConfidenceGatePolicy.default().evaluate_legacy(summary)
        for candidate_review_text in (review_text, materialized_review_text):
            summary = self._summary_from_review_text(candidate_review_text)
            if summary is not None:
                return ConfidenceGatePolicy.default().evaluate_legacy(summary)
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

    def _report_mode_for_gate(self, gate: GateDecision) -> str:
        return {
            "passed": "formal_report",
            "evidence_limited": "evidence_limited_report",
            "blocked": "blocked_notice",
        }.get(gate.final_decision, "blocked_notice")

    def _build_report_context(self, revision_instructions: tuple[str, ...] = ()) -> ReportGenerationContext:
        gate = self.state.analysis_gate_decision or self.state.gate_decision
        bundle = self._evidence_bundle_for_gate()
        contract = self.state.analysis_review_contract
        if gate is None or bundle is None or contract is None:
            raise ValueError("typed report context requires analysis gate, bundle, and contract")
        allowed_claim_ids = sorted(
            {
                *(f"claim:{fact.field_name}" for fact in bundle.formal_facts()),
                *(f"event:{event.event_id}" for event in bundle.events),
            }
        )
        sources = [
            SourceReference(
                source_id=f"claim:{fact.field_name}",
                title=f"{bundle.company_name} {fact.form or 'filing'}",
                url=fact.source_url or "",
                source_tag=fact.source_tag,
                field_name=fact.field_name,
            )
            for fact in bundle.formal_facts()
            if fact.source_url
        ]
        sources.extend(
            SourceReference(source_id=f"market:{index}", title="市场快照", url=item.source_url,
                            source_tag=item.source_tag, field_name="stock_price")
            for index, item in enumerate(bundle.market_snapshots)
        )
        sources.extend(
            SourceReference(source_id=f"event:{item.event_id}", title=item.title, url=item.source_url,
                            source_tag=item.source_type, field_name=None)
            for item in bundle.events if item.independently_confirmed
        )
        return ReportGenerationContext(
            company_name=self.state.company_name,
            ticker=self.state.input_ticker,
            report_mode=self._report_mode_for_gate(gate),  # type: ignore[arg-type]
            evidence_bundle=bundle,
            analysis_review_contract=contract,
            allowed_claim_ids=allowed_claim_ids,
            canonical_sources_json=tuple(
                source.model_dump_json() for source in sorted(sources, key=lambda item: item.source_id)
            ),
            revision_instructions=revision_instructions,
        )

    def _typed_writer_document(self, revision_instructions: tuple[str, ...] = ()) -> ReportDocument:
        context = self._build_report_context(revision_instructions)
        self.state.report_context = context
        writer_input = {
            "company_name": self.state.company_name,
            "company_ticker": self.state.input_ticker,
            "run_id": self.state.request_id,
            "REPORT_CONTEXT_JSON": context.model_dump_json(),
        }
        raw_payload = (
            self._report_writer(writer_input)
            if self._report_writer_injected
            else self._execute_existing_report_writer(writer_input)
        )
        payload = self._strict_contract_payload(raw_payload)
        if payload is None:
            raise ValueError("writer_payload_invalid")
        gate = self.state.analysis_gate_decision or self.state.gate_decision
        if gate is None:
            raise ValueError("writer_payload_invalid")
        try:
            return ReportDocument.from_writer_payload(
                context=context,
                writer_payload=payload,
                trust_score=gate.trust_score,
            )
        except Exception as error:
            raise ValueError("writer_payload_invalid") from error

    def _analysis_report_revision_instructions(self) -> tuple[str, ...]:
        return tuple(
            action.instruction
            for action in self.state.last_repair_actions
            if action.target == "report_writing_analyst" and action.instruction
        )

    def _typed_report_gate(self, report: Any) -> GateDecision:
        if not isinstance(report, ReportDocument):
            return GateDecision(
                passed=False,
                final_decision="blocked",
                trust_score=0,
                blocking_reasons=["writer_payload_invalid"],
            )
        raw_contract = (
            self._report_reviewer(report)
            if self._report_reviewer_injected
            else self._execute_existing_report_review(report)
        )
        if isinstance(raw_contract, str):
            contract_from_text, text_error = self._machine_readable_review_contract_from_text(
                raw_contract
            )
            if contract_from_text is not None:
                raw_contract = contract_from_text
            elif text_error is not None:
                raw_contract = {"_invalid": True}
        contract, error = self._load_strict_contract(
            value=raw_contract,
            filename=self._REPORT_CONTRACT_FILE,
            expected_stage="report_review",
        )
        if error == "missing":
            return GateDecision(
                passed=False,
                final_decision="blocked",
                trust_score=0,
                blocking_reasons=["report_review_contract_missing"],
            )
        if error == "invalid" or contract is None:
            return GateDecision(
                passed=False,
                final_decision="blocked",
                trust_score=0,
                blocking_reasons=["report_review_contract_invalid"],
            )
        self.state.report_review_contract = contract
        if (
            contract.decision.gate_outcome == "block"
            or contract.delivery_eligibility.blocked_notice_required
            or contract.blocking_reasons
        ):
            self.state.report_document = None
            return GateDecision(
                passed=False,
                final_decision="blocked",
                trust_score=0,
                blocking_reasons=["logic_reviewer_requested_block", *contract.blocking_reasons],
            )
        if (
            contract.decision.gate_outcome == "rerun"
            or contract.rerun_reasons
            or contract.repair_actions
        ):
            return GateDecision(
                passed=False,
                final_decision="rerun",
                trust_score=(self.state.analysis_gate_decision or self.state.gate_decision).trust_score,  # type: ignore[union-attr]
                blocking_reasons=["logic_reviewer_requested_rerun", *contract.rerun_reasons],
                repair_actions=contract.repair_actions or [
                    RepairAction(
                        target="report_writing_analyst",
                        code="logic_reviewer_requested_rerun",
                        instruction="根据严格逻辑审查契约修订 writer JSON。",
                    )
                ],
            )
        analysis_gate = self.state.analysis_gate_decision or self.state.gate_decision
        if analysis_gate is None:
            return GateDecision(
                passed=False,
                final_decision="blocked",
                trust_score=0,
                blocking_reasons=["analysis_gate_missing"],
            )
        expected_mode = self._report_mode_for_gate(analysis_gate)
        if report.report_mode != expected_mode:
            return GateDecision(
                passed=False,
                final_decision="blocked",
                trust_score=0,
                blocking_reasons=["report_mode_mismatch"],
            )
        return GateDecision(
            passed=analysis_gate.final_decision == "passed",
            final_decision=analysis_gate.final_decision,
            trust_score=analysis_gate.trust_score,
            blocking_reasons=list(analysis_gate.blocking_reasons),
        )

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
            "rerun_targets": list(self._active_rerun_targets),
            "review_evidence_context_json": self._review_evidence_context_json(),
        }

    def _review_evidence_context_json(self) -> str:
        """Provide reviewer-only reruns with compact, canonical evidence."""
        bundle = self.state.evidence_bundle
        if bundle is None:
            return "{}"
        actionable_gaps: list[dict[str, object]] = []
        seen_gaps: set[tuple[object, ...]] = set()
        for gap in bundle.gaps:
            key = (gap.code, gap.target, tuple(gap.fields), tuple(gap.sources))
            if key in seen_gaps:
                continue
            seen_gaps.add(key)
            actionable_gaps.append(gap.model_dump(mode="json"))
        analysis_artifacts = self._materialized_analysis_artifacts()
        valid_evidence_refs = sorted({
            *(f"claim:{fact.field_name}" for fact in bundle.financial_facts),
            *(fact.source_url for fact in bundle.financial_facts if fact.source_url),
            *(f"market:{index}" for index, _ in enumerate(bundle.market_snapshots)),
            *(snapshot.source_url for snapshot in bundle.market_snapshots),
            *(f"event:{event.event_id}" for event in bundle.events),
            *(event.source_url for event in bundle.events),
            *bundle.raw_artifact_refs,
            *analysis_artifacts,
            "10_research_evidence.json",
        })
        payload = {
            "company_name": bundle.company_name,
            "ticker": bundle.ticker,
            "market_label": bundle.market_label,
            "financial_facts": [
                fact.model_dump(mode="json") for fact in bundle.financial_facts
            ],
            "market_snapshots": [
                snapshot.model_dump(mode="json") for snapshot in bundle.market_snapshots
            ],
            "events": [event.model_dump(mode="json") for event in bundle.events],
            "tool_health": [item.model_dump(mode="json") for item in bundle.tool_health],
            "actionable_gaps": actionable_gaps,
            "artifact_refs": list(bundle.raw_artifact_refs),
            "analysis_artifacts": analysis_artifacts,
            "valid_evidence_refs": valid_evidence_refs,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _materialized_analysis_artifacts(self) -> dict[str, str]:
        artifacts_dir = self.state.artifacts_dir.strip()
        if not artifacts_dir:
            return {}
        root = Path(artifacts_dir)
        names = (
            "00_market_validation.md",
            "01_market_intelligence.md",
            "02_filing_review.md",
            "03_financial_analysis.md",
        )
        return {
            name: (root / name).read_text(encoding="utf-8")
            for name in names
            if (root / name).exists()
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
        if gate.repair_actions:
            self.state.last_repair_actions = list(gate.repair_actions)
        return gate

    @staticmethod
    def _model_tier_for_target(target: str) -> str:
        if target in {"data_quality_reviewer", "logic_compliance_reviewer"}:
            return "review"
        if target == "market_validation_analyst":
            return "fast"
        return "deep"

    def _typed_rerun_targets_for_gate(
        self, gate: GateDecision
    ) -> tuple[list[str], list[str], list[str]]:
        targets = sorted({action.target for action in gate.repair_actions})
        exhausted: list[str] = []
        runnable: list[str] = []
        invalid: list[str] = []
        for target in targets:
            if target == "report_writing_analyst":
                continue
            if target not in self._ANALYSIS_REPAIR_TARGETS:
                prefix = (
                    "invalid_repair_phase"
                    if target in self._REPORT_REPAIR_TARGETS
                    else "unsupported_repair_target"
                )
                invalid.append(f"{prefix}:{target}")
                continue
            if self.state.rerun_budget.get(self.ANALYSIS_RERUN_KEY, 0) > 0:
                runnable.append(target)
            else:
                exhausted.append(target)
        return runnable, exhausted, invalid

    def _gate_after_analysis_repair_exhaustion(self, gate: GateDecision) -> GateDecision:
        reasons = list(dict.fromkeys([
            *gate.blocking_reasons,
            f"rerun_budget_exhausted:{self.ANALYSIS_RERUN_KEY}",
        ]))
        contract = self.state.analysis_review_contract
        if (
            gate.final_decision == "rerun"
            and contract is not None
            and contract.delivery_eligibility.evidence_limited_report_allowed
        ):
            return GateDecision(
                passed=False,
                final_decision="evidence_limited",
                trust_score=gate.trust_score,
                blocking_reasons=reasons,
                repair_actions=list(gate.repair_actions),
            )
        return GateDecision(
            passed=False,
            final_decision="blocked",
            trust_score=gate.trust_score,
            blocking_reasons=reasons,
            repair_actions=list(gate.repair_actions),
        )

    def _route_after_analysis_gate(self, gate: GateDecision) -> str:
        if gate.final_decision == "passed":
            self._record_stage("analysis_passed")
            return "analysis_passed"
        if self._typed_run() and gate.final_decision == "evidence_limited":
            self._record_stage("analysis_evidence_limited")
            return "analysis_evidence_limited"
        if (
            self._typed_run()
            and gate.final_decision in {"rerun", "blocked"}
            and gate.repair_actions
        ):
            targets, exhausted, invalid = self._typed_rerun_targets_for_gate(gate)
            if invalid:
                self.state.analysis_gate_decision = GateDecision(
                    passed=False,
                    final_decision="blocked",
                    trust_score=gate.trust_score,
                    blocking_reasons=[*gate.blocking_reasons, *invalid],
                    repair_actions=list(gate.repair_actions),
                )
                self.state.gate_decision = self.state.analysis_gate_decision
                self._record_stage("analysis_blocked")
                return "analysis_blocked"
            report_repairs = [
                action
                for action in gate.repair_actions
                if action.target == "report_writing_analyst"
            ]
            if report_repairs and not targets and not exhausted and not gate.blocking_reasons:
                contract = self.state.analysis_review_contract
                outcome = (
                    "passed"
                    if contract is not None
                    and contract.delivery_eligibility.formal_report_allowed
                    else "evidence_limited"
                )
                self.state.analysis_gate_decision = GateDecision(
                    passed=outcome == "passed",
                    final_decision=outcome,
                    trust_score=gate.trust_score,
                    repair_actions=list(gate.repair_actions),
                )
                self.state.gate_decision = self.state.analysis_gate_decision
                route = "analysis_passed" if outcome == "passed" else "analysis_evidence_limited"
                self._record_stage(route)
                return route
            if targets and not exhausted:
                self._active_rerun_targets = targets
                self._record_stage("analysis_needs_rerun")
                return "analysis_needs_rerun"
            self.state.analysis_gate_decision = self._gate_after_analysis_repair_exhaustion(gate)
            self.state.gate_decision = self.state.analysis_gate_decision
            if self.state.analysis_gate_decision.final_decision == "evidence_limited":
                self._record_stage("analysis_evidence_limited")
                return "analysis_evidence_limited"
            contract = self.state.analysis_review_contract
            if contract is not None and contract.delivery_eligibility.blocked_notice_required:
                self._record_stage("analysis_blocked_write")
                return "analysis_blocked_write"
            self._record_stage("analysis_blocked")
            return "analysis_blocked"
        if self._typed_run() and gate.final_decision == "blocked" and self.state.analysis_review_contract is not None:
            self._record_stage("analysis_blocked_write")
            return "analysis_blocked_write"
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
        result = {
            "status": gate.final_decision,
            "trust_score": gate.trust_score,
            "blocking_reasons": list(gate.blocking_reasons),
            "analysis_result": self.state.analysis_result,
            "report_result": self.state.report_result,
        }
        if self._typed_run():
            result.update(
                {
                    "final_decision_record": (
                        self.state.final_decision_record.model_dump(mode="json")
                        if self.state.final_decision_record is not None
                        else None
                    ),
                    "report_document": (
                        self.state.report_document.model_dump(mode="json")
                        if self.state.report_document is not None
                        else None
                    ),
                    "analysis_review_contract": (
                        self.state.analysis_review_contract.model_dump(mode="json")
                        if self.state.analysis_review_contract is not None
                        else None
                    ),
                    "report_review_contract": (
                        self.state.report_review_contract.model_dump(mode="json")
                        if self.state.report_review_contract is not None
                        else None
                    ),
                    "repair_actions": [
                        action.model_dump(mode="json") for action in self.state.last_repair_actions
                    ],
                }
            )
        return result

    def _blocked_document(self, reasons: list[str], trust_score: int) -> ReportDocument:
        reason = "；".join(reasons) or "控制面未满足交付条件"
        return ReportDocument.model_validate({
            "company_name": self.state.company_name,
            "ticker": self.state.input_ticker,
            "report_mode": "blocked_notice",
            "title": f"{self.state.company_name} 投资研究阻断通知",
            "stance": "blocked",
            "executive_summary": f"报告已阻断：{reason}。",
            "catalysts": ["阻断状态下不提供催化剂判断。"],
            "risks": [reason],
            "sections": {
                key: {"key": key, "heading": SECTION_HEADINGS[key], "content": f"报告已阻断：{reason}。", "claim_ids": []}
                for key in REQUIRED_SECTION_KEYS
            },
            "claims": [], "sources": [], "trust_score": trust_score, "allowed_claim_ids": [],
        })

    def _finalize(self, gate: GateDecision) -> dict[str, Any]:
        gate = self._coerce_terminal_gate(gate)
        if self._typed_run() and gate.final_decision == "blocked" and self.state.report_document is None:
            self.state.report_document = self._blocked_document(list(gate.blocking_reasons), gate.trust_score)
        if self._typed_run():
            decision = FinalDecisionRecord(
                company_name=self.state.company_name,
                company_ticker=self.state.input_ticker,
                final_decision=gate.final_decision,
                final_delivery_state=self._report_mode_for_gate(gate),  # type: ignore[arg-type]
                trust_score=gate.trust_score,
                blocking_reasons=list(gate.blocking_reasons),
                analysis_review_contract=self.state.analysis_review_contract,
                report_review_contract=self.state.report_review_contract,
            )
            validation = (
                DeliveryValidator().validate(decision=decision, document=self.state.report_document)
                if self.state.report_document is not None
                else None
            )
            if validation is not None and not validation.valid:
                gate = GateDecision(
                    passed=False,
                    final_decision="blocked",
                    trust_score=0,
                    blocking_reasons=["delivery_validation_failed", *validation.errors],
                )
                decision = FinalDecisionRecord(
                    company_name=self.state.company_name,
                    company_ticker=self.state.input_ticker,
                    final_decision="blocked",
                    final_delivery_state="blocked_notice",
                    trust_score=0,
                    blocking_reasons=list(gate.blocking_reasons),
                    analysis_review_contract=self.state.analysis_review_contract,
                    report_review_contract=self.state.report_review_contract,
                )
                blocked_document = self._blocked_document(
                    list(gate.blocking_reasons), gate.trust_score
                )
                self.state.report_document = blocked_document
                self.state.report_result = blocked_document
            self.state.final_decision_record = decision
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
        if self._typed_run():
            self._clear_new_run_artifacts()
        result = self._analysis_executor(inputs)
        self.state.analysis_result = result
        self._ingest_typed_analysis_result(result)
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
        if self._typed_run():
            targets = list(self._active_rerun_targets)
            remaining_budget = self.state.rerun_budget.get(self.ANALYSIS_RERUN_KEY, 0)
            self.state.rerun_budget[self.ANALYSIS_RERUN_KEY] = max(remaining_budget - 1, 0)
            self.state.model_tier_overrides = {
                target: self._model_tier_for_target(target)  # type: ignore[dict-item]
                for target in targets
            }
            self._prepare_tavily_evidence_attempt(targets)
            self._clear_typed_rerun_artifacts(targets)
            result = self._analysis_executor(self._analysis_inputs())
            self.state.analysis_result = result
            self._ingest_typed_analysis_result(result)
            return result
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
        if self._typed_run():
            try:
                report = self._typed_writer_document(self._analysis_report_revision_instructions())
                self.state.report_document = report
            except ValueError as error:
                report = {"typed_error": str(error)}
            self.state.report_result = report
            return report
        report = self._report_writer(self.state.analysis_result)
        self.state.report_result = report
        return report

    @listen("analysis_evidence_limited")
    def write_evidence_limited_report(self) -> Any:
        self._record_stage("write_report")
        if self._typed_run():
            try:
                report = self._typed_writer_document(self._analysis_report_revision_instructions())
                self.state.report_document = report
            except ValueError as error:
                report = {"typed_error": str(error)}
            self.state.report_result = report
            return report
        report = self._report_writer(self.state.analysis_result)
        self.state.report_result = report
        return report

    @listen("analysis_blocked_write")
    def write_blocked_report(self) -> Any:
        self._record_stage("write_report")
        try:
            report = self._typed_writer_document(self._analysis_report_revision_instructions())
            self.state.report_document = report
        except ValueError as error:
            report = {"typed_error": str(error)}
        self.state.report_result = report
        return report

    @listen(write_report)
    def review_report(self, report: Any) -> GateDecision:
        self._record_stage("review_report")
        gate = self._typed_report_gate(report) if self._typed_run() else self._report_reviewer(report)
        self._ensure_gate_consistency(gate, "review_report")
        self.state.report_gate_decision = gate
        self.state.gate_decision = gate
        return gate

    @listen(write_evidence_limited_report)
    def review_evidence_limited_report(self, report: Any) -> GateDecision:
        self._record_stage("review_report")
        gate = self._typed_report_gate(report) if self._typed_run() else self._report_reviewer(report)
        self._ensure_gate_consistency(gate, "review_report")
        self.state.report_gate_decision = gate
        self.state.gate_decision = gate
        return gate

    @listen(write_blocked_report)
    def review_blocked_report(self, report: Any) -> GateDecision:
        self._record_stage("review_report")
        gate = self._typed_report_gate(report)
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
        if self._typed_run() and report_gate.final_decision == "rerun":
            while report_gate.final_decision == "rerun":
                remaining = self.state.rerun_budget.get("report_writing_analyst", 0)
                if remaining <= 0:
                    report_gate = GateDecision(
                        passed=False,
                        final_decision="blocked",
                        trust_score=report_gate.trust_score,
                        blocking_reasons=[
                            *report_gate.blocking_reasons,
                            "rerun_budget_exhausted:report_writing_analyst",
                        ],
                    )
                    break
                self.state.rerun_budget["report_writing_analyst"] = remaining - 1
                self.state.model_tier_overrides = {"report_writing_analyst": "deep"}
                feedback = tuple([
                    *self.state.report_review_contract.rerun_reasons,
                    *(action.instruction for action in self.state.report_review_contract.repair_actions),
                ]) if self.state.report_review_contract is not None else tuple(report_gate.blocking_reasons)
                try:
                    report = self._typed_writer_document(feedback)
                    self.state.report_document = report
                except ValueError:
                    report_gate = GateDecision(
                        passed=False,
                        final_decision="blocked",
                        trust_score=0,
                        blocking_reasons=["writer_payload_invalid"],
                    )
                    break
                self.state.report_result = report
                report_gate = self._typed_report_gate(report)
                self.state.report_gate_decision = report_gate
        self.state.report_gate_decision = report_gate
        return self._finalize(report_gate)

    @listen(review_evidence_limited_report)
    def finalize_evidence_limited_delivery(self, report_gate: GateDecision) -> Any:
        if self._typed_run():
            return self.finalize_delivery(report_gate)
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

    @listen(review_blocked_report)
    def finalize_blocked_report_delivery(self, report_gate: GateDecision) -> Any:
        return self.finalize_delivery(report_gate)
