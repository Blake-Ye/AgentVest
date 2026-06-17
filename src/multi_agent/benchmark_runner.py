from __future__ import annotations

import json
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from multi_agent.benchmark_dataset import BenchmarkSample, load_benchmark_dataset
from multi_agent.benchmark_factual import FactualChecker
from multi_agent.benchmark_judge import LLMJudgeClient
from multi_agent.benchmark_metrics import aggregate_benchmark_summary
from multi_agent.settings import BenchmarkSettings

# 这一层是 benchmark 编排器：
# 对每个样本先跑主工作流拿报告，再跑 factual check，再可选跑 LLM judge，最后统一写 JSON。


@dataclass(frozen=True)
class GeneratedArtifacts:
    """记录单个样本跑完主工作流后产出的关键文件路径。"""

    run_dir: Path
    final_report_path: Path
    structured_report_path: Path
    structured_recommendation_path: Path
    latest_metrics_path: Path


def _build_sample_metrics(
    *,
    factual_result: dict[str, Any],
    judge_result: dict[str, Any] | None,
) -> dict[str, float | None]:
    """把 factual 和 judge 结果折叠成统一 metrics 结构。"""

    if judge_result is None:
        overall_quality_score = float(factual_result["factual_accuracy"])
        return {
            "factual_accuracy": float(factual_result["factual_accuracy"]),
            "risk_recall": None,
            "catalyst_recall": None,
            "hallucination_score": None,
            "overall_quality_score": round(overall_quality_score, 3),
        }

    return {
        "factual_accuracy": float(factual_result["factual_accuracy"]),
        "risk_recall": float(judge_result["risk_recall"]),
        "catalyst_recall": float(judge_result["catalyst_recall"]),
        "hallucination_score": float(judge_result["hallucination_score"]),
        "overall_quality_score": float(judge_result["overall_quality_score"]),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _default_report_generator(sample: BenchmarkSample, artifacts_root: Path) -> GeneratedArtifacts:
    """复用现有主工作流生成 benchmark 需要的真实报告与结构化产物。"""

    from multi_agent import main as workflow_main
    from multi_agent.evaluation import activate_evaluation, clear_evaluation
    from multi_agent.settings import InvestmentResearchSettings

    settings = InvestmentResearchSettings.from_env()
    output_paths = workflow_main._build_run_output_paths(
        base_artifacts_dir=artifacts_root,
        company_name=sample.company_name,
        company_ticker=sample.ticker,
        run_time=workflow_main._now_for_output_paths(),
    )
    output_paths.run_dir.mkdir(parents=True, exist_ok=True)
    workflow_main._write_run_readme(
        output_paths,
        company_name=sample.company_name,
        company_ticker=sample.ticker,
    )
    workflow_main._initialize_standard_output_files(
        output_paths,
        company_name=sample.company_name,
        company_ticker=sample.ticker,
    )
    evaluation = workflow_main._create_evaluation(
        output_paths,
        company_name=sample.company_name,
        company_ticker=sample.ticker,
    )
    evaluation.start()
    token = activate_evaluation(evaluation)

    try:
        # benchmark 不单独造报告，而是直接复用生产主流程，避免评测环境和真实运行脱节。
        inputs = workflow_main._workflow_inputs(sample.company_name, sample.ticker)
        inputs["artifacts_dir"] = str(output_paths.run_dir)
        inputs["final_report_path"] = str(output_paths.final_report_path)
        with workflow_main._temporary_env(
            {
                "ARTIFACTS_DIR": str(output_paths.run_dir),
                "FINAL_REPORT_PATH": str(output_paths.final_report_path),
            }
        ):
            result = workflow_main._crew().kickoff(inputs=inputs)
        workflow_main._materialize_standard_outputs(output_paths, result)
        workflow_main._validate_successful_outputs(output_paths)
        latest_metrics = evaluation.finalize(success=True)
        workflow_main._write_structured_outputs(
            output_paths,
            latest_metrics=latest_metrics,
            company_name=sample.company_name,
            company_ticker=sample.ticker,
            watchlist_path=workflow_main._resolve_watchlist_path(settings),
            save_to_watchlist=False,
        )
    except Exception as error:
        workflow_main._write_failure_outputs(output_paths, error_message=str(error))
        workflow_main._write_failure_recommendation_output(
            output_paths,
            company_name=sample.company_name,
            company_ticker=sample.ticker,
            error_message=str(error),
        )
        evaluation.finalize(success=False, error_message=str(error))
        raise
    finally:
        clear_evaluation(token)

    return GeneratedArtifacts(
        run_dir=output_paths.run_dir,
        final_report_path=output_paths.final_report_path,
        structured_report_path=output_paths.structured_report_path,
        structured_recommendation_path=output_paths.structured_recommendation_path,
        latest_metrics_path=output_paths.latest_metrics_path,
    )


class BenchmarkRunner:
    """执行整套 benchmark，并把单样本明细与总体 summary 一起落盘。"""

    def __init__(
        self,
        *,
        factual_checker: FactualChecker | None = None,
        judge_client: LLMJudgeClient | None = None,
        report_generator: Callable[[BenchmarkSample, Path], GeneratedArtifacts] | None = None,
        inter_sample_delay_seconds: float = 0.0,
    ) -> None:
        self.factual_checker = factual_checker or FactualChecker()
        self.judge_client = judge_client
        self.report_generator = report_generator or _default_report_generator
        self.inter_sample_delay_seconds = max(inter_sample_delay_seconds, 0.0)

    def run(self, *, samples: list[BenchmarkSample], output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        details_dir = output_dir / "details"
        workflow_artifacts_dir = output_dir / "artifacts"
        sample_results: list[dict[str, Any]] = []

        for sample_index, sample in enumerate(samples):
            detail_path = details_dir / f"{sample.sample_id}.json"
            try:
                generated = self.report_generator(sample, workflow_artifacts_dir)
                report_text = generated.final_report_path.read_text(encoding="utf-8")
                factual_result = self.factual_checker.evaluate(sample=sample, report_text=report_text)

                llm_judge: dict[str, Any]
                judge_result: dict[str, Any] | None = None
                if self.judge_client is None:
                    llm_judge = {
                        "status": "skipped",
                        "reason": "未启用 LLM judge。",
                    }
                elif not self.judge_client.is_available:
                    llm_judge = {
                        "status": "unavailable",
                        "reason": "LLM judge 配置不可用，自动降级为 factual-only。",
                    }
                else:
                    try:
                        judge_result = self.judge_client.judge(sample=sample, report_text=report_text)
                        llm_judge = {
                            "status": "completed",
                            "result": judge_result,
                        }
                    except Exception as error:
                        # Judge 出问题时不让整条 benchmark 失败，而是优雅降级为 factual-only。
                        llm_judge = {
                            "status": "unavailable",
                            "reason": f"LLM judge 调用失败，已自动降级：{error}",
                        }

                sample_result = {
                    "sample_id": sample.sample_id,
                    "company_name": sample.company_name,
                    "ticker": sample.ticker,
                    "status": "completed",
                    "expected_facts": sample.expected_facts,
                    "expected_key_points": sample.expected_key_points,
                    "metrics": _build_sample_metrics(
                        factual_result=factual_result,
                        judge_result=judge_result,
                    ),
                    "factual_check": factual_result,
                    "llm_judge": llm_judge,
                    "generated_artifacts": {
                        "run_dir": str(generated.run_dir),
                        "final_report_path": str(generated.final_report_path),
                        "structured_report_path": str(generated.structured_report_path),
                        "structured_recommendation_path": str(generated.structured_recommendation_path),
                        "latest_metrics_path": str(generated.latest_metrics_path),
                    },
                    "workflow_metrics": _read_json(generated.latest_metrics_path),
                }
            except Exception as error:
                sample_result = {
                    "sample_id": sample.sample_id,
                    "company_name": sample.company_name,
                    "ticker": sample.ticker,
                    "status": "failed",
                    "error_message": str(error),
                    "expected_facts": sample.expected_facts,
                    "expected_key_points": sample.expected_key_points,
                    "metrics": {
                        metric_name: None
                        for metric_name in (
                            "factual_accuracy",
                            "risk_recall",
                            "catalyst_recall",
                            "hallucination_score",
                            "overall_quality_score",
                        )
                    },
                    "factual_check": None,
                    "llm_judge": {
                        "status": "skipped",
                        "reason": f"样本执行失败：{error}",
                    },
                    "generated_artifacts": {
                        "run_dir": None,
                        "final_report_path": None,
                        "structured_report_path": None,
                        "structured_recommendation_path": None,
                        "latest_metrics_path": None,
                    },
                    "workflow_metrics": {
                        "status": "failed",
                        "success": False,
                        "error_message": str(error),
                    },
                }

            _write_json(detail_path, sample_result)
            sample_results.append(sample_result)
            if self.inter_sample_delay_seconds > 0 and sample_index < len(samples) - 1:
                time.sleep(self.inter_sample_delay_seconds)

        summary = aggregate_benchmark_summary(sample_results)
        summary["detail_files"] = [
            str((details_dir / f"{sample_result['sample_id']}.json").resolve())
            for sample_result in sample_results
        ]
        summary["output_dir"] = str(output_dir.resolve())

        summary_path = output_dir / "benchmark_summary.json"
        combined_results_path = output_dir / "benchmark_results.json"
        _write_json(summary_path, summary)
        _write_json(
            combined_results_path,
            {
                "summary": summary,
                "samples": sample_results,
            },
        )
        return {
            "summary_path": str(summary_path.resolve()),
            "results_path": str(combined_results_path.resolve()),
            "summary": summary,
            "samples": sample_results,
        }


def run_benchmark(
    *,
    dataset_path: Path,
    output_dir: Path,
    judge_model: str | None,
    factual_only: bool,
) -> dict[str, Any]:
    """CLI 和外部调用统一入口，负责装配 settings、dataset、judge 和 runner。"""

    benchmark_settings = BenchmarkSettings.from_env()
    samples = load_benchmark_dataset(dataset_path)
    judge_client = None
    if not factual_only:
        judge_client = LLMJudgeClient(
            model=judge_model or benchmark_settings.judge_model,
            api_key=benchmark_settings.judge_api_key,
            base_url=benchmark_settings.judge_base_url,
            timeout_seconds=benchmark_settings.llm_judge_timeout_seconds,
        )

    runner = BenchmarkRunner(
        judge_client=judge_client,
        inter_sample_delay_seconds=benchmark_settings.inter_sample_delay_seconds,
    )
    result = runner.run(samples=samples, output_dir=output_dir)
    result["config"] = {
        "dataset_path": str(dataset_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "judge_model": judge_model or benchmark_settings.judge_model,
        "factual_only": factual_only,
    }
    return result
