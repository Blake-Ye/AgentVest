#!/usr/bin/env python
import argparse
import contextlib
import json
import os
import re
import sys
import warnings

from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from multi_agent.evaluation import WorkflowEvaluation, activate_evaluation, clear_evaluation
from multi_agent.recommendation import build_structured_recommendation, build_structured_report
from multi_agent.resolver import CompanyResolver
from multi_agent.settings import InvestmentResearchSettings
from multi_agent.tools.investment_tools import FatalAPIError
from multi_agent.watchlist import WatchlistStore

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
_PLACEHOLDER_MARKER = "<!-- PLACEHOLDER -->"


@dataclass(frozen=True)
class RunOutputPaths:
    company_dir: Path
    run_dir: Path
    market_intelligence_path: Path
    filing_review_path: Path
    financial_analysis_path: Path
    final_report_path: Path
    structured_recommendation_path: Path
    structured_report_path: Path
    runtime_log_path: Path
    latest_metrics_path: Path
    evaluation_summary_path: Path
    readme_path: Path


def _prepare_runtime_env() -> None:
    """将 CrewAI 的运行时数据固定到项目目录，避免污染系统环境。"""
    project_root = Path(__file__).resolve().parents[2]
    local_home = project_root / ".crewai_home"
    (local_home / "Library" / "Application Support").mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(local_home)
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")


def _crew():
    _prepare_runtime_env()
    from multi_agent.crew import MultiAgent

    return MultiAgent().crew()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_output_path(path_value: str) -> Path:
    candidate_path = Path(path_value)
    if candidate_path.is_absolute():
        return candidate_path
    return _project_root() / candidate_path


def _now_for_output_paths() -> datetime:
    return datetime.now()


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
        market_intelligence_path=run_dir / "01_market_intelligence.md",
        filing_review_path=run_dir / "02_filing_review.md",
        financial_analysis_path=run_dir / "03_financial_analysis.md",
        final_report_path=run_dir / "04_investment_report.md",
        structured_recommendation_path=run_dir / "06_structured_recommendation.json",
        structured_report_path=run_dir / "07_structured_report.json",
        runtime_log_path=run_dir / "05_runtime.txt",
        latest_metrics_path=run_dir / "latest_run_metrics.json",
        evaluation_summary_path=run_dir / "evaluation_summary.json",
        readme_path=run_dir / "README.md",
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
            "- `01_market_intelligence.md`：市场情报简报",
            "- `02_filing_review.md`：监管文件复核",
            "- `03_financial_analysis.md`：财务分析结果",
            "- `04_investment_report.md`：最终投资备忘录",
            "- `05_runtime.txt`：运行日志",
            "- `06_structured_recommendation.json`：结构化投资建议与可信度评分",
            "- `07_structured_report.json`：结构化完整报告快照",
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


def _initialize_standard_output_files(
    output_paths: RunOutputPaths,
    *,
    company_name: str,
    company_ticker: str,
) -> None:
    company_ticker_display = company_ticker or "未解析到 ticker"
    waiting_message = f"公司：{company_name}（{company_ticker_display}）\n\n状态：运行中，结果待生成。"
    _write_markdown_file(
        output_paths.market_intelligence_path,
        "市场情报简报",
        waiting_message,
        placeholder=True,
    )
    _write_markdown_file(
        output_paths.filing_review_path,
        "监管文件复核",
        waiting_message,
        placeholder=True,
    )
    _write_markdown_file(
        output_paths.financial_analysis_path,
        "财务分析结果",
        waiting_message,
        placeholder=True,
    )
    _write_markdown_file(
        output_paths.final_report_path,
        "投资备忘录",
        waiting_message,
        placeholder=True,
    )
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
        "market_intelligence_task": output_paths.market_intelligence_path,
        "filing_review_task": output_paths.filing_review_path,
        "financial_analysis_task": output_paths.financial_analysis_path,
        "investment_report_task": output_paths.final_report_path,
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
        if content:
            path.write_text(content + "\n", encoding="utf-8")


def _write_failure_outputs(output_paths: RunOutputPaths, *, error_message: str) -> None:
    failure_message = f"状态：运行失败。\n\n原因：{error_message}"
    for path, title in (
        (output_paths.market_intelligence_path, "市场情报简报"),
        (output_paths.filing_review_path, "监管文件复核"),
        (output_paths.financial_analysis_path, "财务分析结果"),
        (output_paths.final_report_path, "投资备忘录"),
    ):
        if _is_placeholder_file(path):
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


def _validate_successful_outputs(output_paths: RunOutputPaths) -> None:
    for path in (
        output_paths.market_intelligence_path,
        output_paths.filing_review_path,
        output_paths.financial_analysis_path,
        output_paths.final_report_path,
    ):
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
            "market_intelligence_task": output_paths.market_intelligence_path,
            "filing_review_task": output_paths.filing_review_path,
            "financial_analysis_task": output_paths.financial_analysis_path,
        },
        company_name=company_name,
        company_ticker=company_ticker,
    )


def _resolve_watchlist_path(settings: InvestmentResearchSettings) -> Path:
    return _resolve_output_path(settings.watchlist_path)


def _write_structured_outputs(
    output_paths: RunOutputPaths,
    *,
    latest_metrics: dict[str, object],
    company_name: str,
    company_ticker: str,
    watchlist_path: Path,
    save_to_watchlist: bool,
) -> dict[str, object]:
    recommendation = build_structured_recommendation(
        company_name=company_name,
        company_ticker=company_ticker,
        report_path=output_paths.final_report_path,
        metrics=latest_metrics,
    )
    structured_report = build_structured_report(
        company_name=company_name,
        company_ticker=company_ticker,
        report_path=output_paths.final_report_path,
        metrics=latest_metrics,
    )
    _write_json_file(output_paths.structured_recommendation_path, recommendation)
    _write_json_file(output_paths.structured_report_path, structured_report)
    if save_to_watchlist:
        WatchlistStore(watchlist_path).upsert(recommendation)
    return recommendation


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
    artifacts_dir = _project_root() / settings.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    local_pdf_path = settings.local_filing_pdf_path.strip()
    resolved_local_pdf_path = Path(local_pdf_path).expanduser() if local_pdf_path else None
    if resolved_local_pdf_path and not resolved_local_pdf_path.is_absolute():
        resolved_local_pdf_path = _project_root() / resolved_local_pdf_path
    local_pdf_available = "yes" if resolved_local_pdf_path and resolved_local_pdf_path.exists() else "no"

    return {
        "company_name": resolved_company.normalized_name,
        "company_ticker": resolved_company.ticker,
        "current_year": str(datetime.now().year),
        "artifacts_dir": settings.artifacts_dir,
        "final_report_path": settings.final_report_path,
        "local_filing_pdf_path": str(resolved_local_pdf_path) if resolved_local_pdf_path else "未提供本地 PDF 文件",
        "local_filing_pdf_available": local_pdf_available,
    }


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
        inputs = _workflow_inputs(args.company_name, args.company_ticker)
        output_paths = _build_run_output_paths(
            base_artifacts_dir=_resolve_output_path(settings.artifacts_dir),
            company_name=inputs["company_name"],
            company_ticker=inputs["company_ticker"],
            run_time=_now_for_output_paths(),
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
            }
        ):
            result = _crew().kickoff(inputs=inputs)
        _materialize_standard_outputs(output_paths, result)
        _validate_successful_outputs(output_paths)
        latest_metrics = evaluation.finalize(success=True)
        _write_structured_outputs(
            output_paths,
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

def run_with_trigger():
    """使用外部触发器参数运行 Crew。"""
    import json

    evaluation = None
    token = None
    if len(sys.argv) < 2:
        _raise_user_facing_runtime_error(RuntimeError("未提供触发器 JSON 参数。"))

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        _raise_user_facing_runtime_error(RuntimeError("触发器参数不是合法 JSON。"))

    try:
        settings = InvestmentResearchSettings.from_env()
        workflow_inputs = _workflow_inputs(
            trigger_payload.get("company_name", settings.company_name),
            trigger_payload.get("company_ticker", settings.company_ticker),
        )
        output_paths = _build_run_output_paths(
            base_artifacts_dir=_resolve_output_path(settings.artifacts_dir),
            company_name=workflow_inputs["company_name"],
            company_ticker=workflow_inputs["company_ticker"],
            run_time=_now_for_output_paths(),
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
            }
        ):
            result = _crew().kickoff(inputs=inputs)
        _materialize_standard_outputs(output_paths, result)
        _validate_successful_outputs(output_paths)
        latest_metrics = evaluation.finalize(success=True)
        _write_structured_outputs(
            output_paths,
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


if __name__ == "__main__":
    run()
