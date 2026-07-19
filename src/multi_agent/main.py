#!/usr/bin/env python
import argparse
import contextlib
import json
import os
import re
import signal
import sys
import warnings

from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from multi_agent.evaluation import WorkflowEvaluation, activate_evaluation, clear_evaluation
from multi_agent.core.report_document import (
    ReportDocument,
    ReportGenerationContext,
    render_markdown,
    render_recommendation,
    render_structured_report,
)
from multi_agent.recommendation import build_structured_recommendation, build_structured_report
from multi_agent.resolver import CompanyResolver
from multi_agent import runtime
from multi_agent.settings import InvestmentResearchSettings
from multi_agent.tools.official_sec import FatalAPIError
from multi_agent.tools.market_validation import MarketValidationService
from multi_agent.watchlist import WatchlistStore

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
_PLACEHOLDER_MARKER = "<!-- PLACEHOLDER -->"


@dataclass(frozen=True)
class RunOutputPaths:
    company_dir: Path
    run_dir: Path
    market_validation_path: Path
    market_intelligence_path: Path
    filing_review_path: Path
    financial_analysis_path: Path
    final_report_path: Path
    structured_recommendation_path: Path
    structured_report_path: Path
    runtime_log_path: Path
    data_quality_review_path: Path
    logic_compliance_review_path: Path
    final_decision_path: Path
    latest_metrics_path: Path
    evaluation_summary_path: Path
    readme_path: Path
    evidence_bundle_path: Path
    report_document_path: Path

def _crew():
    return runtime.build_crew()


def _flow(inputs: dict[str, str]):
    return runtime.build_flow(inputs)


def _use_flow_execution() -> bool:
    return runtime.use_flow_execution()


def _kickoff_workflow(inputs: dict[str, str]):
    if _use_flow_execution():
        return _flow(inputs).kickoff()
    return _crew().kickoff(inputs=inputs)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_output_path(path_value: str) -> Path:
    candidate_path = Path(path_value)
    if candidate_path.is_absolute():
        return candidate_path
    return _project_root() / candidate_path


def _now_for_output_paths() -> datetime:
    return datetime.now()


def _run_id_from_time(run_time: datetime) -> str:
    return run_time.strftime("%Y%m%d_%H%M%S")


def _run_time_from_run_id(run_id: str) -> datetime:
    return datetime.strptime(run_id, "%Y%m%d_%H%M%S")


def _slugify_for_path(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return normalized or fallback


def _build_run_output_paths(
    *,
    base_artifacts_dir: Path,
    company_name: str,
    company_ticker: str,
    run_time: datetime,
) -> RunOutputPaths:
    company_slug = _slugify_for_path(company_name, fallback="company")
    ticker_slug = _slugify_for_path(company_ticker, fallback="no_ticker")
    timestamp = run_time.strftime("%Y%m%d_%H%M%S")
    company_dir = base_artifacts_dir / f"{company_slug}__{ticker_slug}"
    run_dir = company_dir / timestamp
    return RunOutputPaths(
        company_dir=company_dir,
        run_dir=run_dir,
        market_validation_path=run_dir / "00_market_validation.md",
        market_intelligence_path=run_dir / "01_market_intelligence.md",
        filing_review_path=run_dir / "02_filing_review.md",
        financial_analysis_path=run_dir / "03_financial_analysis.md",
        final_report_path=run_dir / "04_investment_report.md",
        structured_recommendation_path=run_dir / "06_structured_recommendation.json",
        structured_report_path=run_dir / "07_structured_report.json",
        runtime_log_path=run_dir / "05_runtime.txt",
        data_quality_review_path=run_dir / "08_data_quality_review.md",
        logic_compliance_review_path=run_dir / "09_logic_compliance_review.md",
        final_decision_path=run_dir / "final_decision.json",
        latest_metrics_path=run_dir / "latest_run_metrics.json",
        evaluation_summary_path=run_dir / "evaluation_summary.json",
        readme_path=run_dir / "README.md",
        evidence_bundle_path=run_dir / "10_research_evidence.json",
        report_document_path=run_dir / "11_report_document.json",
    )


def _write_run_readme(
    output_paths: RunOutputPaths,
    *,
    company_name: str,
    company_ticker: str,
) -> None:
    company_ticker_display = company_ticker or "未解析到 ticker"
    readme_content = "\n".join(
        [
            "# 本次运行输出说明",
            "",
            f"- 公司名称：`{company_name}`",
            f"- 股票代码：`{company_ticker_display}`",
            f"- 输出目录：`{output_paths.run_dir}`",
            "",
            "## 文件清单",
            "",
            "- `00_market_validation.md`：市场验证结果",
            "- `01_market_intelligence.md`：市场情报简报",
            "- `02_filing_review.md`：监管文件复核",
            "- `03_financial_analysis.md`：财务分析结果",
            "- `04_investment_report.md`：最终投资备忘录",
            "- `05_runtime.txt`：运行日志",
            "- `06_structured_recommendation.json`：结构化投资建议与可信度评分",
            "- `07_structured_report.json`：结构化完整报告快照",
            "- `08_data_quality_review.md`：数据质量审查结果",
            "- `09_logic_compliance_review.md`：逻辑与合规审查结果",
            "- `10_research_evidence.json`：可审计的规范化研究证据包",
            "- `11_report_document.json`：报告交付的规范化单一真值",
            "- `final_decision.json`：最终状态单一真值源",
            "- `latest_run_metrics.json`：单次运行评估指标",
            "- `evaluation_summary.json`：当前目录下的评估汇总",
        ]
    )
    output_paths.readme_path.write_text(readme_content + "\n", encoding="utf-8")


def _write_markdown_file(path: Path, title: str, body: str, *, placeholder: bool) -> None:
    prefix = f"{_PLACEHOLDER_MARKER}\n\n" if placeholder else ""
    content = f"{prefix}# {title}\n\n{body.strip()}\n"
    path.write_text(content, encoding="utf-8")


def _write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _standard_markdown_outputs(output_paths: RunOutputPaths) -> tuple[tuple[Path, str], ...]:
    return (
        (output_paths.market_validation_path, "市场验证结果"),
        (output_paths.market_intelligence_path, "市场情报简报"),
        (output_paths.filing_review_path, "监管文件复核"),
        (output_paths.financial_analysis_path, "财务分析结果"),
        (output_paths.final_report_path, "投资备忘录"),
        (output_paths.data_quality_review_path, "数据质量审查结果"),
        (output_paths.logic_compliance_review_path, "逻辑与合规审查结果"),
    )


def _initialize_standard_output_files(
    output_paths: RunOutputPaths,
    *,
    company_name: str,
    company_ticker: str,
) -> None:
    company_ticker_display = company_ticker or "未解析到 ticker"
    waiting_message = f"公司：{company_name}（{company_ticker_display}）\n\n状态：运行中，结果待生成。"
    for path, title in _standard_markdown_outputs(output_paths):
        _write_markdown_file(path, title, waiting_message, placeholder=True)
    _write_json_file(
        output_paths.structured_recommendation_path,
        {
            "status": "running",
            "company_name": company_name,
            "company_ticker": company_ticker,
            "message": "运行中，结构化投资建议待生成。",
        },
    )
    _write_json_file(
        output_paths.structured_report_path,
        {
            "status": "running",
            "company_name": company_name,
            "company_ticker": company_ticker,
            "message": "运行中，结构化完整报告待生成。",
        },
    )


def _task_output_path_map(output_paths: RunOutputPaths) -> dict[str, Path]:
    return {
        "market_validation_task": output_paths.market_validation_path,
        "market_intelligence_task": output_paths.market_intelligence_path,
        "filing_review_task": output_paths.filing_review_path,
        "financial_analysis_task": output_paths.financial_analysis_path,
        "investment_report_task": output_paths.final_report_path,
        "data_quality_review_task": output_paths.data_quality_review_path,
        "logic_compliance_review_task": output_paths.logic_compliance_review_path,
    }


def _extract_task_raw_outputs(result: object) -> dict[str, str]:
    task_outputs = getattr(result, "tasks_output", None)
    extracted: dict[str, str] = {}
    if not task_outputs:
        return extracted
    for task_output in task_outputs:
        task_name = str(getattr(task_output, "name", "")).strip()
        raw_content = str(getattr(task_output, "raw", "")).strip()
        if task_name and raw_content:
            extracted[task_name] = raw_content
    return extracted


def _final_report_content_from_result(result: object) -> str:
    report_result = _workflow_result_value(result, "report_result", "")
    if isinstance(report_result, str):
        return report_result.strip()
    if report_result:
        return str(report_result).strip()
    return ""


def _is_placeholder_file(path: Path) -> bool:
    if not path.exists():
        return True
    content = path.read_text(encoding="utf-8")
    return _PLACEHOLDER_MARKER in content


def _materialize_standard_outputs(output_paths: RunOutputPaths, result: object) -> None:
    task_outputs = _extract_task_raw_outputs(result)
    for task_name, path in _task_output_path_map(output_paths).items():
        if path.exists() and not _is_placeholder_file(path):
            continue
        content = task_outputs.get(task_name, "").strip()
        if not content and task_name == "investment_report_task":
            content = str(getattr(result, "raw", "")).strip()
        if not content and task_name == "investment_report_task":
            content = _final_report_content_from_result(result)
        if content:
            path.write_text(content + "\n", encoding="utf-8")


def _workflow_result_value(result: object, key: str, default: object = "") -> object:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _final_status_from_result(result: object) -> str:
    status = str(
        _workflow_result_value(
            result,
            "status",
            _workflow_result_value(result, "final_decision", "passed"),
        )
    ).strip().lower()
    if status in {"passed", "blocked", "evidence_limited"}:
        return status
    raise ValueError(f"未知终态：{status}")


def _blocking_reasons_from_result(result: object) -> list[str]:
    blocking_reasons = _workflow_result_value(result, "blocking_reasons", [])
    if not isinstance(blocking_reasons, list):
        return []
    return [str(item).strip() for item in blocking_reasons if str(item).strip()]


def _trust_score_from_result(result: object) -> int | None:
    trust_score = _workflow_result_value(result, "trust_score", None)
    if isinstance(trust_score, dict):
        trust_score = trust_score.get("score")
    if trust_score in (None, ""):
        return None
    try:
        return int(trust_score)
    except (TypeError, ValueError):
        return None


def _describe_trust_score(score: int) -> tuple[str, str]:
    if score >= 80:
        return "high", "证据较充分，可作为高优先级研究输入。"
    if score >= 60:
        return "medium", "证据基本够用，但仍建议人工复核关键结论。"
    return "low", "证据不足，当前结果更适合作为线索而非结论。"


def _final_delivery_state_from_status(final_status: str) -> str:
    if final_status == "passed":
        return "formal_report"
    if final_status == "evidence_limited":
        return "evidence_limited_report"
    return "blocked_notice"


def _build_final_decision_record(
    *,
    company_name: str,
    company_ticker: str,
    result: object,
    final_status: str,
) -> dict[str, object]:
    return {
        "company_name": company_name,
        "company_ticker": company_ticker,
        "final_decision": final_status,
        "final_delivery_state": _final_delivery_state_from_status(final_status),
        "trust_score": _trust_score_from_result(result),
        "blocking_reasons": _blocking_reasons_from_result(result),
    }


def _write_final_decision(
    output_paths: RunOutputPaths,
    *,
    company_name: str,
    company_ticker: str,
    result: object,
    final_status: str,
) -> dict[str, object]:
    final_decision = _build_final_decision_record(
        company_name=company_name,
        company_ticker=company_ticker,
        result=result,
        final_status=final_status,
    )
    _write_json_file(output_paths.final_decision_path, final_decision)
    return final_decision


def _apply_final_decision_projection(
    recommendation: dict[str, object],
    structured_report: dict[str, object],
    *,
    final_decision: dict[str, object],
) -> None:
    final_status = str(final_decision.get("final_decision", "passed")).strip()
    final_delivery_state = str(final_decision.get("final_delivery_state", "")).strip()
    blocking_reasons = [
        str(item).strip()
        for item in final_decision.get("blocking_reasons", [])
        if str(item).strip()
    ]
    recommendation["status"] = final_status
    recommendation["final_delivery_state"] = final_delivery_state
    structured_report["status"] = final_status
    structured_report["final_decision"] = final_status
    structured_report["final_delivery_state"] = final_delivery_state
    if final_status == "blocked":
        recommendation["stance"] = "blocked"
        recommendation["stance_label"] = "阻断"
        structured_report["stance"] = "blocked"
        structured_report["stance_label"] = "阻断"
    elif final_status == "evidence_limited":
        recommendation["stance"] = "watch"
        recommendation["stance_label"] = "证据受限"
        structured_report["stance"] = "watch"
        structured_report["stance_label"] = "证据受限"
    if blocking_reasons:
        recommendation["blocking_reasons"] = blocking_reasons
        structured_report["blocking_reasons"] = blocking_reasons


def _write_blocked_report(output_paths: RunOutputPaths, result: object) -> None:
    body_lines = [
        "状态：硬门控未通过，正式投资备忘录未放行。",
        "",
    ]
    trust_score = _trust_score_from_result(result)
    if trust_score is not None:
        body_lines.append(f"trust_score：{trust_score}")
        body_lines.append("")
    body_lines.append("阻断原因：")
    blocking_reasons = _blocking_reasons_from_result(result)
    if blocking_reasons:
        body_lines.extend(f"- {item}" for item in blocking_reasons)
    else:
        body_lines.append("- 未提供阻断原因。")
    _write_markdown_file(
        output_paths.final_report_path,
        "投资备忘录（已阻断）",
        "\n".join(body_lines),
        placeholder=False,
    )


def _write_evidence_limited_report(output_paths: RunOutputPaths, result: object) -> None:
    body_lines = [
        "状态：已生成证据受限版备忘录，正式投资备忘录暂未放行。",
        "",
    ]
    trust_score = _trust_score_from_result(result)
    if trust_score is not None:
        body_lines.append(f"trust_score：{trust_score}")
        body_lines.append("")
    blocking_reasons = _blocking_reasons_from_result(result)
    body_lines.append("限制说明：")
    if blocking_reasons:
        body_lines.extend(f"- {item}" for item in blocking_reasons)
    else:
        body_lines.append("- 关键 formal report 字段未完全闭合，已按 evidence-limited 交付。")
    _write_markdown_file(
        output_paths.final_report_path,
        "受限版投资备忘录",
        "\n".join(body_lines),
        placeholder=False,
    )


def _write_failure_outputs(output_paths: RunOutputPaths, *, error_message: str) -> None:
    failure_message = f"状态：运行失败。\n\n原因：{error_message}"
    for path, title in _standard_markdown_outputs(output_paths):
        _write_markdown_file(path, title, failure_message, placeholder=False)


def _write_failure_recommendation_output(
    output_paths: RunOutputPaths,
    *,
    company_name: str,
    company_ticker: str,
    error_message: str,
) -> None:
    _write_json_file(
        output_paths.structured_recommendation_path,
        {
            "status": "failed",
            "company_name": company_name,
            "company_ticker": company_ticker,
            "message": error_message,
        },
    )
    _write_json_file(
        output_paths.structured_report_path,
        {
            "status": "failed",
            "company_name": company_name,
            "company_ticker": company_ticker,
            "message": error_message,
        },
    )


def _validate_successful_outputs(output_paths: RunOutputPaths, *, final_status: str) -> None:
    if final_status not in {"passed", "blocked", "evidence_limited"}:
        raise RuntimeError(f"不支持的工作流结束状态：{final_status}")
    for path, _ in _standard_markdown_outputs(output_paths):
        if not path.exists() or not path.read_text(encoding="utf-8").strip() or _is_placeholder_file(path):
            raise RuntimeError(f"运行结束但未生成规范输出文件：{path.name}")


@contextlib.contextmanager
def _temporary_env(overrides: dict[str, str]):
    original_values = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, original in original_values.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


@contextlib.contextmanager
def _graceful_termination_signals():
    previous_handlers: dict[int, object] = {}

    def _raise_keyboard_interrupt(_signum, _frame):
        raise KeyboardInterrupt()

    try:
        for signum in (signal.SIGTERM,):
            previous_handlers[signum] = signal.signal(signum, _raise_keyboard_interrupt)
        yield
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)


def _create_evaluation(
    output_paths: RunOutputPaths,
    *,
    company_name: str,
    company_ticker: str,
) -> WorkflowEvaluation:
    return WorkflowEvaluation(
        artifacts_dir=output_paths.run_dir,
        final_report_path=output_paths.final_report_path,
        expected_task_outputs={
            "market_validation_task": output_paths.market_validation_path,
            "market_intelligence_task": output_paths.market_intelligence_path,
            "filing_review_task": output_paths.filing_review_path,
            "financial_analysis_task": output_paths.financial_analysis_path,
            "data_quality_review_task": output_paths.data_quality_review_path,
            "logic_compliance_review_task": output_paths.logic_compliance_review_path,
        },
        company_name=company_name,
        company_ticker=company_ticker,
    )


def _resolve_watchlist_path(settings: InvestmentResearchSettings) -> Path:
    return _resolve_output_path(settings.watchlist_path)


def _write_structured_outputs(
    output_paths: RunOutputPaths,
    *,
    final_decision: dict[str, object],
    latest_metrics: dict[str, object],
    company_name: str,
    company_ticker: str,
    watchlist_path: Path,
    save_to_watchlist: bool,
) -> dict[str, object]:
    del latest_metrics, company_name, company_ticker
    document = _load_report_document(output_paths.report_document_path)
    expected_mode = str(final_decision.get("final_delivery_state", "")).strip()
    if document.report_mode != expected_mode:
        raise ValueError(
            "report document mode does not match final delivery state: "
            f"{document.report_mode} != {expected_mode}"
        )
    recommendation = render_recommendation(document)
    structured_report = render_structured_report(document)
    _apply_final_decision_projection(
        recommendation,
        structured_report,
        final_decision=final_decision,
    )
    _write_json_file(output_paths.structured_recommendation_path, recommendation)
    _write_json_file(output_paths.structured_report_path, structured_report)
    if save_to_watchlist:
        WatchlistStore(watchlist_path).upsert(recommendation)
    return recommendation


def _load_report_document(path: Path) -> ReportDocument:
    if not path.exists():
        raise ValueError(f"new run is missing canonical report document: {path.name}")
    try:
        return ReportDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"invalid canonical report document: {path.name}") from error


def _materialize_new_run_report_document(
    output_paths: RunOutputPaths,
    *,
    result: object,
    final_status: str,
) -> ReportDocument:
    """Build every new report from locked context and untrusted writer JSON only."""
    raw_context = _workflow_result_value(result, "report_context", None)
    raw_payload = _workflow_result_value(result, "report_writer_payload", None)
    if raw_context is None or not isinstance(raw_payload, dict):
        raise ValueError(
            "new run requires report_context and report_writer_payload; "
            "direct report_document input is not accepted"
        )
    context = ReportGenerationContext.model_validate(raw_context)
    expected_mode = _final_delivery_state_from_status(final_status)
    if context.report_mode != expected_mode:
        raise ValueError(
            "report context mode does not match final delivery state: "
            f"{context.report_mode} != {expected_mode}"
        )
    trust_score = _trust_score_from_result(result)
    if trust_score is None:
        raise ValueError("new run requires an integer trust_score for its report document")
    document = ReportDocument.from_writer_payload(
        context=context,
        writer_payload=raw_payload,
        trust_score=trust_score,
    )
    _write_json_file(output_paths.report_document_path, document.model_dump(mode="json"))
    output_paths.final_report_path.write_text(render_markdown(document), encoding="utf-8")
    return document


def _finalize_successful_result(output_paths: RunOutputPaths, result: object) -> str:
    final_status = _final_status_from_result(result)
    _materialize_standard_outputs(output_paths, result)
    _validate_successful_outputs(output_paths, final_status=final_status)
    return final_status


def _annotate_latest_metrics(
    output_paths: RunOutputPaths,
    latest_metrics: dict[str, object],
    *,
    result: object,
    final_decision: dict[str, object],
) -> dict[str, object]:
    annotated_metrics = dict(latest_metrics)
    final_status = str(final_decision.get("final_decision", "passed")).strip()
    annotated_metrics["final_status"] = final_status
    annotated_metrics["final_delivery_state"] = final_decision.get("final_delivery_state", "")
    annotated_metrics["final_decision"] = final_decision

    blocking_reasons = _blocking_reasons_from_result(result)
    if blocking_reasons:
        annotated_metrics["blocking_reasons"] = blocking_reasons

    trust_score = _trust_score_from_result(result)
    if trust_score is not None:
        trust_level, trust_summary = _describe_trust_score(trust_score)
        existing_trust_score = annotated_metrics.get("trust_score")
        breakdown = {}
        if isinstance(existing_trust_score, dict):
            breakdown = dict(existing_trust_score.get("breakdown", {}))
        annotated_metrics["trust_score"] = {
            "score": trust_score,
            "level": trust_level,
            "summary": trust_summary,
            "breakdown": breakdown,
        }

    _write_json_file(output_paths.latest_metrics_path, annotated_metrics)
    return annotated_metrics


def _print_watchlist(path: Path) -> None:
    items = WatchlistStore(path).list_items()
    if not items:
        print("当前 watchlist 为空。")
        return
    print(f"当前 watchlist：{path}")
    for index, item in enumerate(items, start=1):
        print(
            f"{index}. {item.get('company_name', '')} "
            f"({item.get('company_ticker', '')}) | {item.get('stance_label', '观察')} | "
            f"trust_score={item.get('trust_score', 0)}"
        )


def _load_json_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_existing_run_dirs(base_artifacts_dir: Path) -> list[Path]:
    run_dirs: list[Path] = []
    if not base_artifacts_dir.exists():
        return run_dirs
    for company_dir in sorted(base_artifacts_dir.iterdir()):
        if not company_dir.is_dir():
            continue
        for run_dir in sorted(company_dir.iterdir()):
            if run_dir.is_dir():
                run_dirs.append(run_dir)
    return run_dirs


def _rebuild_watchlist_from_artifacts(
    *,
    base_artifacts_dir: Path,
    watchlist_path: Path,
) -> int:
    watchlist_path.parent.mkdir(parents=True, exist_ok=True)
    watchlist_path.write_text(json.dumps({"items": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    store = WatchlistStore(watchlist_path)
    rebuilt_count = 0

    for run_dir in _iter_existing_run_dirs(base_artifacts_dir):
        report_path = run_dir / "04_investment_report.md"
        if not report_path.exists() or not report_path.read_text(encoding="utf-8").strip():
            continue

        latest_metrics = _load_json_file(run_dir / "latest_run_metrics.json")
        existing_recommendation = _load_json_file(run_dir / "06_structured_recommendation.json")
        company_name = str(
            latest_metrics.get("company_name")
            or existing_recommendation.get("company_name")
            or ""
        ).strip()
        company_ticker = str(
            latest_metrics.get("company_ticker")
            or existing_recommendation.get("company_ticker")
            or ""
        ).strip()
        if not company_name or not company_ticker:
            continue

        report_document_path = run_dir / "11_report_document.json"
        if report_document_path.exists():
            document = _load_report_document(report_document_path)
            recommendation = render_recommendation(document)
            structured_report = render_structured_report(document)
        else:
            # Historical runs predate ReportDocument and are rebuilt from Markdown only here.
            recommendation = build_structured_recommendation(
                company_name=company_name,
                company_ticker=company_ticker,
                report_path=report_path,
                metrics=latest_metrics,
            )
            structured_report = build_structured_report(
                company_name=company_name,
                company_ticker=company_ticker,
                report_path=report_path,
                metrics=latest_metrics,
            )
        final_decision = _load_json_file(run_dir / "final_decision.json")
        if final_decision:
            _apply_final_decision_projection(
                recommendation,
                structured_report,
                final_decision=final_decision,
            )
        _write_json_file(run_dir / "06_structured_recommendation.json", recommendation)
        _write_json_file(run_dir / "07_structured_report.json", structured_report)
        store.upsert(recommendation)
        rebuilt_count += 1

    return rebuilt_count


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行自动化投研 CrewAI 工作流。")
    parser.add_argument(
        "--company-name",
        default=os.getenv("DEFAULT_COMPANY_NAME", "Apple Inc."),
        help="需要调研的公司名称。",
    )
    parser.add_argument(
        "--company-ticker",
        default="",
        help="可选，上市公司的股票代码；如果不填，系统会自动尝试解析。",
    )
    parser.add_argument(
        "--save-to-watchlist",
        action="store_true",
        help="运行成功后，将结构化投资建议保存到 watchlist。",
    )
    parser.add_argument(
        "--watchlist-list",
        action="store_true",
        help="只查看当前 watchlist，不执行新的投研任务。",
    )
    parser.add_argument(
        "--watchlist-rebuild",
        action="store_true",
        help="基于已有 artifacts 重新生成结构化建议和 watchlist，不执行新的投研任务。",
    )
    return parser


def _workflow_inputs(company_name: str, company_ticker: str) -> dict[str, str]:
    # 统一在入口处准备工作流输入，方便 CLI、测试和触发器复用同一套参数。
    settings = InvestmentResearchSettings.from_env()
    resolved_company = CompanyResolver(settings=settings).resolve(company_name, company_ticker)
    market_validation = MarketValidationService(settings).validate(
        company_name=resolved_company.normalized_name,
        ticker=resolved_company.ticker,
        exchange=resolved_company.exchange,
    )
    artifacts_dir = _project_root() / settings.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    local_pdf_path = settings.local_filing_pdf_path.strip()
    resolved_local_pdf_path = Path(local_pdf_path).expanduser() if local_pdf_path else None
    if resolved_local_pdf_path and not resolved_local_pdf_path.is_absolute():
        resolved_local_pdf_path = _project_root() / resolved_local_pdf_path
    local_pdf_available = "yes" if resolved_local_pdf_path and resolved_local_pdf_path.exists() else "no"
    run_id = _run_id_from_time(_now_for_output_paths())

    return {
        "company_name": resolved_company.normalized_name,
        "company_ticker": resolved_company.ticker,
        "run_id": run_id,
        "company_market_label": market_validation.market_label,
        "market_resolution_status": market_validation.resolution_status,
        "current_year": str(datetime.now().year),
        "artifacts_dir": settings.artifacts_dir,
        "final_report_path": settings.final_report_path,
        "local_filing_pdf_path": str(resolved_local_pdf_path) if resolved_local_pdf_path else "未提供本地 PDF 文件",
        "local_filing_pdf_available": local_pdf_available,
    }


def _ensure_run_id(inputs: dict[str, str]) -> dict[str, str]:
    normalized_inputs = dict(inputs)
    run_id = str(normalized_inputs.get("run_id", "")).strip()
    if not run_id:
        normalized_inputs["run_id"] = _run_id_from_time(_now_for_output_paths())
    return normalized_inputs


def _raise_user_facing_runtime_error(error: Exception) -> None:
    """把内部异常转换成对终端用户更友好的中文退出信息。"""
    raise SystemExit(f"程序已终止：{error}")


def run():
    """运行自动化投研主流程。"""
    args = _build_parser().parse_args()
    evaluation = None
    token = None

    try:
        settings = InvestmentResearchSettings.from_env()
        if getattr(args, "watchlist_list", False):
            _print_watchlist(_resolve_watchlist_path(settings))
            return
        if getattr(args, "watchlist_rebuild", False):
            rebuilt_count = _rebuild_watchlist_from_artifacts(
                base_artifacts_dir=_resolve_output_path(settings.artifacts_dir),
                watchlist_path=_resolve_watchlist_path(settings),
            )
            print(f"watchlist 重建完成，共处理 {rebuilt_count} 个运行目录。")
            return
        inputs = _ensure_run_id(_workflow_inputs(args.company_name, args.company_ticker))
        output_paths = _build_run_output_paths(
            base_artifacts_dir=_resolve_output_path(settings.artifacts_dir),
            company_name=inputs["company_name"],
            company_ticker=inputs["company_ticker"],
            run_time=_run_time_from_run_id(inputs["run_id"]),
        )
        output_paths.run_dir.mkdir(parents=True, exist_ok=True)
        _write_run_readme(
            output_paths,
            company_name=inputs["company_name"],
            company_ticker=inputs["company_ticker"],
        )
        _initialize_standard_output_files(
            output_paths,
            company_name=inputs["company_name"],
            company_ticker=inputs["company_ticker"],
        )
        evaluation = _create_evaluation(
            output_paths,
            company_name=inputs["company_name"],
            company_ticker=inputs["company_ticker"],
        )
        evaluation.start()
        token = activate_evaluation(evaluation)
        inputs["artifacts_dir"] = str(output_paths.run_dir)
        inputs["final_report_path"] = str(output_paths.final_report_path)
        print(f"本次输出目录：{output_paths.run_dir}")
        print(f"最终报告路径：{output_paths.final_report_path}")
        with _temporary_env(
            {
                "ARTIFACTS_DIR": str(output_paths.run_dir),
                "FINAL_REPORT_PATH": str(output_paths.final_report_path),
                "COMPANY_MARKET_LABEL": inputs.get("company_market_label", ""),
                "RUN_ID": inputs.get("run_id", ""),
            }
        ):
            with _graceful_termination_signals():
                result = _kickoff_workflow(inputs)
        final_status = _final_status_from_result(result)
        _materialize_new_run_report_document(
            output_paths, result=result, final_status=final_status
        )
        final_status = _finalize_successful_result(output_paths, result)
        final_decision = _write_final_decision(
            output_paths,
            company_name=inputs["company_name"],
            company_ticker=inputs["company_ticker"],
            result=result,
            final_status=final_status,
        )
        latest_metrics = evaluation.finalize(success=True)
        latest_metrics = _annotate_latest_metrics(
            output_paths,
            latest_metrics,
            result=result,
            final_decision=final_decision,
        )
        _write_structured_outputs(
            output_paths,
            final_decision=final_decision,
            latest_metrics=latest_metrics,
            company_name=inputs["company_name"],
            company_ticker=inputs["company_ticker"],
            watchlist_path=_resolve_watchlist_path(settings),
            save_to_watchlist=getattr(args, "save_to_watchlist", False),
        )
        print(f"运行完成，文件已写入：{output_paths.run_dir}")
    except (FatalAPIError, ValueError) as error:
        if evaluation is not None:
            _write_failure_outputs(output_paths, error_message=f"程序已终止：{error}")
            _write_failure_recommendation_output(
                output_paths,
                company_name=inputs["company_name"],
                company_ticker=inputs["company_ticker"],
                error_message=f"程序已终止：{error}",
            )
        if evaluation is not None:
            evaluation.finalize(success=False, error_message=str(error))
        _raise_user_facing_runtime_error(error)
    except KeyboardInterrupt:
        if evaluation is not None:
            _write_failure_outputs(output_paths, error_message="运行被中断。")
            _write_failure_recommendation_output(
                output_paths,
                company_name=inputs["company_name"],
                company_ticker=inputs["company_ticker"],
                error_message="运行被中断。",
            )
        if evaluation is not None:
            evaluation.finalize(success=False, error_message="运行被中断。")
        _raise_user_facing_runtime_error(RuntimeError("运行被中断。"))
    except Exception as error:
        if evaluation is not None:
            _write_failure_outputs(
                output_paths,
                error_message=f"运行投研工作流时发生未预期错误：{error}",
            )
            _write_failure_recommendation_output(
                output_paths,
                company_name=inputs["company_name"],
                company_ticker=inputs["company_ticker"],
                error_message=f"运行投研工作流时发生未预期错误：{error}",
            )
        if evaluation is not None:
            evaluation.finalize(success=False, error_message=str(error))
        _raise_user_facing_runtime_error(
            RuntimeError(f"运行投研工作流时发生未预期错误：{error}")
        )
    finally:
        if token is not None:
            clear_evaluation(token)


def train():
    """训练当前 Crew 配置。"""
    args = _build_parser().parse_args(sys.argv[3:])
    try:
        inputs = _workflow_inputs(args.company_name, args.company_ticker)
        _crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
    except (FatalAPIError, ValueError) as error:
        _raise_user_facing_runtime_error(error)
    except Exception as error:
        _raise_user_facing_runtime_error(RuntimeError(f"训练过程中发生错误：{error}"))

def replay():
    """从指定任务回放 Crew 执行。"""
    try:
        _crew().replay(task_id=sys.argv[1])
    except (FatalAPIError, ValueError) as error:
        _raise_user_facing_runtime_error(error)
    except Exception as error:
        _raise_user_facing_runtime_error(RuntimeError(f"回放执行时发生错误：{error}"))

def test():
    """测试当前 Crew 执行效果。"""
    args = _build_parser().parse_args(sys.argv[3:])
    try:
        inputs = _workflow_inputs(args.company_name, args.company_ticker)
        _crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)
    except (FatalAPIError, ValueError) as error:
        _raise_user_facing_runtime_error(error)
    except Exception as error:
        _raise_user_facing_runtime_error(RuntimeError(f"测试执行时发生错误：{error}"))

def run_trigger_payload(trigger_payload: dict[str, object]):
    """使用结构化触发器参数运行工作流。"""
    evaluation = None
    token = None
    try:
        settings = InvestmentResearchSettings.from_env()
        workflow_inputs = _ensure_run_id(
            _workflow_inputs(
                trigger_payload.get("company_name", settings.company_name),
                trigger_payload.get("company_ticker", settings.company_ticker),
            )
        )
        output_paths = _build_run_output_paths(
            base_artifacts_dir=_resolve_output_path(settings.artifacts_dir),
            company_name=workflow_inputs["company_name"],
            company_ticker=workflow_inputs["company_ticker"],
            run_time=_run_time_from_run_id(workflow_inputs["run_id"]),
        )
        output_paths.run_dir.mkdir(parents=True, exist_ok=True)
        _write_run_readme(
            output_paths,
            company_name=workflow_inputs["company_name"],
            company_ticker=workflow_inputs["company_ticker"],
        )
        _initialize_standard_output_files(
            output_paths,
            company_name=workflow_inputs["company_name"],
            company_ticker=workflow_inputs["company_ticker"],
        )
        evaluation = _create_evaluation(
            output_paths,
            company_name=workflow_inputs["company_name"],
            company_ticker=workflow_inputs["company_ticker"],
        )
        evaluation.start()
        token = activate_evaluation(evaluation)
        workflow_inputs["artifacts_dir"] = str(output_paths.run_dir)
        workflow_inputs["final_report_path"] = str(output_paths.final_report_path)
        print(f"本次输出目录：{output_paths.run_dir}")
        print(f"最终报告路径：{output_paths.final_report_path}")
        inputs = {
            "crewai_trigger_payload": trigger_payload,
            "company_name": workflow_inputs["company_name"],
            "company_ticker": workflow_inputs["company_ticker"],
            "run_id": workflow_inputs["run_id"],
            "company_market_label": workflow_inputs.get("company_market_label", ""),
            "market_resolution_status": workflow_inputs.get("market_resolution_status", ""),
            "current_year": str(datetime.now().year),
            "artifacts_dir": workflow_inputs["artifacts_dir"],
            "final_report_path": workflow_inputs["final_report_path"],
            "local_filing_pdf_path": workflow_inputs["local_filing_pdf_path"],
            "local_filing_pdf_available": workflow_inputs["local_filing_pdf_available"],
        }
        with _temporary_env(
            {
                "ARTIFACTS_DIR": str(output_paths.run_dir),
                "FINAL_REPORT_PATH": str(output_paths.final_report_path),
                "COMPANY_MARKET_LABEL": workflow_inputs.get("company_market_label", ""),
                "RUN_ID": workflow_inputs.get("run_id", ""),
            }
        ):
            with _graceful_termination_signals():
                result = _kickoff_workflow(inputs)
        final_status = _final_status_from_result(result)
        _materialize_new_run_report_document(
            output_paths, result=result, final_status=final_status
        )
        final_status = _finalize_successful_result(output_paths, result)
        final_decision = _write_final_decision(
            output_paths,
            company_name=workflow_inputs["company_name"],
            company_ticker=workflow_inputs["company_ticker"],
            result=result,
            final_status=final_status,
        )
        latest_metrics = evaluation.finalize(success=True)
        latest_metrics = _annotate_latest_metrics(
            output_paths,
            latest_metrics,
            result=result,
            final_decision=final_decision,
        )
        _write_structured_outputs(
            output_paths,
            final_decision=final_decision,
            latest_metrics=latest_metrics,
            company_name=workflow_inputs["company_name"],
            company_ticker=workflow_inputs["company_ticker"],
            watchlist_path=_resolve_watchlist_path(settings),
            save_to_watchlist=bool(trigger_payload.get("save_to_watchlist", False)),
        )
        print(f"运行完成，文件已写入：{output_paths.run_dir}")
        return result
    except (FatalAPIError, ValueError) as error:
        if evaluation is not None:
            _write_failure_outputs(output_paths, error_message=f"程序已终止：{error}")
            _write_failure_recommendation_output(
                output_paths,
                company_name=workflow_inputs["company_name"],
                company_ticker=workflow_inputs["company_ticker"],
                error_message=f"程序已终止：{error}",
            )
        if evaluation is not None:
            evaluation.finalize(success=False, error_message=str(error))
        _raise_user_facing_runtime_error(error)
    except KeyboardInterrupt:
        if evaluation is not None:
            _write_failure_outputs(output_paths, error_message="运行被中断。")
            _write_failure_recommendation_output(
                output_paths,
                company_name=workflow_inputs["company_name"],
                company_ticker=workflow_inputs["company_ticker"],
                error_message="运行被中断。",
            )
        if evaluation is not None:
            evaluation.finalize(success=False, error_message="运行被中断。")
        _raise_user_facing_runtime_error(RuntimeError("运行被中断。"))
    except Exception as error:
        if evaluation is not None:
            _write_failure_outputs(
                output_paths,
                error_message=f"触发器运行时发生未预期错误：{error}",
            )
            _write_failure_recommendation_output(
                output_paths,
                company_name=workflow_inputs["company_name"],
                company_ticker=workflow_inputs["company_ticker"],
                error_message=f"触发器运行时发生未预期错误：{error}",
            )
        if evaluation is not None:
            evaluation.finalize(success=False, error_message=str(error))
        _raise_user_facing_runtime_error(
            RuntimeError(f"触发器运行时发生未预期错误：{error}")
        )
    finally:
        if token is not None:
            clear_evaluation(token)


def run_with_trigger():
    """使用外部触发器参数运行 Crew。"""
    if len(sys.argv) < 2:
        _raise_user_facing_runtime_error(RuntimeError("未提供触发器 JSON 参数。"))

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        _raise_user_facing_runtime_error(RuntimeError("触发器参数不是合法 JSON。"))

    return run_trigger_payload(trigger_payload)


if __name__ == "__main__":
    run()
