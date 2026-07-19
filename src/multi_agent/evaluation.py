from __future__ import annotations

import json
import os
import re
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from multi_agent.recommendation import calculate_trust_score


_URL_PATTERN = re.compile(r"https?://[^\s)>\"']+")
_CURRENT_EVALUATION: ContextVar["WorkflowEvaluation | None"] = ContextVar(
    "current_workflow_evaluation",
    default=None,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round_metric(value: float) -> float:
    return round(value, 3)


def activate_evaluation(evaluation: "WorkflowEvaluation") -> Token:
    return _CURRENT_EVALUATION.set(evaluation)


def clear_evaluation(token: Token) -> None:
    _CURRENT_EVALUATION.reset(token)


def current_evaluation() -> "WorkflowEvaluation | None":
    return _CURRENT_EVALUATION.get()


def record_api_call(service_name: str, success: bool, status_code: int | None = None) -> None:
    evaluation = current_evaluation()
    if evaluation is not None:
        evaluation.record_api_call(
            service_name=service_name,
            success=success,
            status_code=status_code,
        )


def record_financial_fields(fields: dict[str, dict[str, Any]]) -> None:
    evaluation = current_evaluation()
    if evaluation is not None:
        evaluation.record_financial_fields(fields)


def record_tavily_payload(payload: dict[str, Any]) -> None:
    evaluation = current_evaluation()
    if evaluation is not None:
        evaluation.record_tavily_payload(payload)


def recorded_tavily_payloads() -> list[dict[str, Any]]:
    evaluation = current_evaluation()
    return evaluation.tavily_payloads() if evaluation is not None else []


def record_research_evidence(payload: dict[str, Any]) -> None:
    evaluation = current_evaluation()
    if evaluation is not None:
        evaluation.record_research_evidence(payload)


def record_task_completion_callback(task_output: Any) -> None:
    evaluation = current_evaluation()
    if evaluation is None:
        return

    task_name = str(getattr(task_output, "name", "")).strip()
    if task_name:
        evaluation.record_task_completion(task_name)


@dataclass
class WorkflowEvaluation:
    artifacts_dir: Path
    final_report_path: Path
    expected_task_outputs: dict[str, Path]
    company_name: str
    company_ticker: str
    time_source: Callable[[], float] = time.perf_counter
    started_at_iso: str = field(default_factory=_utc_now_iso)
    _started_at_seconds: float | None = field(default=None, init=False)
    _task_completion_seconds: dict[str, float] = field(default_factory=dict, init=False)
    _task_order: list[str] = field(default_factory=list, init=False)
    _api_calls: list[dict[str, Any]] = field(default_factory=list, init=False)
    _financial_fields: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _tavily_payloads: list[dict[str, Any]] = field(default_factory=list, init=False)

    def start(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._started_at_seconds = self.time_source()
        self._write_json(self.artifacts_dir / "latest_run_metrics.json", self._build_running_metrics())

    def set_company(self, company_name: str, company_ticker: str) -> None:
        self.company_name = company_name
        self.company_ticker = company_ticker

    def record_task_completion(self, task_name: str, completed_at: float | None = None) -> None:
        if self._started_at_seconds is None:
            self.start()
        if task_name not in self._task_completion_seconds:
            self._task_order.append(task_name)
        self._task_completion_seconds[task_name] = (
            completed_at if completed_at is not None else self.time_source()
        )

    def record_api_call(self, service_name: str, success: bool, status_code: int | None = None) -> None:
        self._api_calls.append(
            {
                "service_name": service_name,
                "success": success,
                "status_code": status_code,
            }
        )

    def record_financial_fields(self, fields: dict[str, dict[str, Any]]) -> None:
        self._financial_fields = json.loads(json.dumps(fields))

    def record_tavily_payload(self, payload: dict[str, Any]) -> None:
        self._tavily_payloads.append(json.loads(json.dumps(payload)))

    def tavily_payloads(self) -> list[dict[str, Any]]:
        return json.loads(json.dumps(self._tavily_payloads))

    def record_research_evidence(self, payload: dict[str, Any]) -> None:
        self._atomic_write_json(self.artifacts_dir / "10_research_evidence.json", payload)

    def finalize(self, success: bool, error_message: str | None = None) -> dict[str, Any]:
        if self._started_at_seconds is None:
            self.start()

        finished_at_seconds = self.time_source()
        latest_metrics = self._build_latest_metrics(
            success=success,
            error_message=error_message,
            finished_at_seconds=finished_at_seconds,
        )
        self._write_json(self.artifacts_dir / "latest_run_metrics.json", latest_metrics)
        summary = self._update_summary(latest_metrics)
        self._write_json(self.artifacts_dir / "evaluation_summary.json", summary)
        return latest_metrics

    def _build_running_metrics(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at_iso,
            "finished_at": None,
            "status": "running",
            "success": False,
            "error_message": None,
            "company_name": self.company_name,
            "company_ticker": self.company_ticker,
            "total_runtime_seconds": 0.0,
            "task_durations_seconds": {},
            "api_calls": {
                "total": 0,
                "failures": 0,
                "failure_rate": 0.0,
                "per_service": {},
            },
            "report_generated": False,
            "report_complete": False,
            "citation_count": 0,
            "intermediate_artifacts_complete": False,
            "artifacts": self._build_artifact_inventory(),
            "financial_fields": {},
            "financial_fields_success_rate": 0.0,
            "trust_score": {
                "score": 0.0,
                "level": "pending",
                "summary": "运行中，待生成可信度评分。",
                "breakdown": {},
            },
        }

    def _build_latest_metrics(
        self,
        *,
        success: bool,
        error_message: str | None,
        finished_at_seconds: float,
    ) -> dict[str, Any]:
        started_at_seconds = (
            self._started_at_seconds if self._started_at_seconds is not None else finished_at_seconds
        )
        task_durations = self._build_task_durations(started_at_seconds)
        api_summary = self._build_api_summary()
        artifact_inventory = self._build_artifact_inventory()
        report_generated, report_complete, citation_count = self._inspect_report()
        financial_fields_success_rate = self._calculate_financial_fields_success_rate()
        metrics = {
            "started_at": self.started_at_iso,
            "finished_at": _utc_now_iso(),
            "status": "completed" if success else "failed",
            "success": success,
            "error_message": error_message,
            "company_name": self.company_name,
            "company_ticker": self.company_ticker,
            "total_runtime_seconds": _round_metric(finished_at_seconds - started_at_seconds),
            "task_durations_seconds": task_durations,
            "api_calls": api_summary,
            "report_generated": report_generated,
            "report_complete": report_complete,
            "citation_count": citation_count,
            "intermediate_artifacts_complete": all(
                item["exists"] and item["size_bytes"] > 0 for item in artifact_inventory.values()
            ),
            "artifacts": artifact_inventory,
            "financial_fields": self._financial_fields,
            "financial_fields_success_rate": financial_fields_success_rate,
        }
        metrics["trust_score"] = calculate_trust_score(metrics)

        return metrics

    def _build_task_durations(self, started_at_seconds: float) -> dict[str, float]:
        durations: dict[str, float] = {}
        previous_mark = started_at_seconds
        for task_name in self._task_order:
            completed_at = self._task_completion_seconds.get(task_name)
            if completed_at is None:
                continue
            durations[task_name] = _round_metric(completed_at - previous_mark)
            previous_mark = completed_at
        return durations

    def _build_api_summary(self) -> dict[str, Any]:
        total = len(self._api_calls)
        failures = sum(1 for item in self._api_calls if not item["success"])
        per_service: dict[str, dict[str, Any]] = {}
        for call in self._api_calls:
            service_bucket = per_service.setdefault(
                call["service_name"],
                {"total": 0, "failures": 0, "status_codes": []},
            )
            service_bucket["total"] += 1
            if not call["success"]:
                service_bucket["failures"] += 1
            if call["status_code"] is not None:
                service_bucket["status_codes"].append(call["status_code"])

        for service_bucket in per_service.values():
            service_bucket["failure_rate"] = _round_metric(
                service_bucket["failures"] / service_bucket["total"]
            ) if service_bucket["total"] else 0.0

        return {
            "total": total,
            "failures": failures,
            "failure_rate": _round_metric(failures / total) if total else 0.0,
            "per_service": per_service,
        }

    def _build_artifact_inventory(self) -> dict[str, dict[str, Any]]:
        inventory: dict[str, dict[str, Any]] = {}
        for task_name, output_path in self.expected_task_outputs.items():
            resolved_path = output_path if output_path.is_absolute() else output_path.resolve()
            inventory[task_name] = {
                "path": str(resolved_path),
                "exists": resolved_path.exists(),
                "size_bytes": resolved_path.stat().st_size if resolved_path.exists() else 0,
            }
        return inventory

    def _inspect_report(self) -> tuple[bool, bool, int]:
        report_path = self.final_report_path
        if not report_path.is_absolute():
            report_path = report_path.resolve()
        if not report_path.exists():
            return False, False, 0

        content = report_path.read_text(encoding="utf-8").strip()
        citation_count = len(_URL_PATTERN.findall(content))
        return True, bool(content), citation_count

    def _calculate_financial_fields_success_rate(self) -> float:
        if not self._financial_fields:
            return 0.0
        extracted_count = sum(
            1 for field in self._financial_fields.values() if field.get("extracted", False)
        )
        return _round_metric(extracted_count / len(self._financial_fields))

    def _update_summary(self, latest_metrics: dict[str, Any]) -> dict[str, Any]:
        summary_path = self.artifacts_dir / "evaluation_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = {
                "total_runs": 0,
                "successful_runs": 0,
                "api_calls": {
                    "total": 0,
                    "failures": 0,
                    "failure_rate": 0.0,
                },
            }

        summary["total_runs"] += 1
        summary["successful_runs"] += 1 if latest_metrics["success"] else 0
        summary["success_rate"] = _round_metric(
            summary["successful_runs"] / summary["total_runs"]
        )

        summary["api_calls"]["total"] += latest_metrics["api_calls"]["total"]
        summary["api_calls"]["failures"] += latest_metrics["api_calls"]["failures"]
        total_api_calls = summary["api_calls"]["total"]
        summary["api_calls"]["failure_rate"] = _round_metric(
            summary["api_calls"]["failures"] / total_api_calls
        ) if total_api_calls else 0.0

        summary["latest_run_file"] = str((self.artifacts_dir / "latest_run_metrics.json").resolve())
        summary["last_updated_at"] = _utc_now_iso()
        return summary

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
