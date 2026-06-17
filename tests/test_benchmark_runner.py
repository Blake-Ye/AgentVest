import json
from pathlib import Path
from unittest.mock import patch

from multi_agent.benchmark_dataset import BenchmarkSample
from multi_agent.benchmark_runner import BenchmarkRunner, aggregate_benchmark_summary


def test_aggregate_benchmark_summary_averages_available_metrics() -> None:
    summary = aggregate_benchmark_summary(
        [
            {
                "company_name": "Apple Inc.",
                "ticker": "AAPL",
                "status": "completed",
                "llm_judge": {"status": "completed"},
                "metrics": {
                    "factual_accuracy": 1.0,
                    "risk_recall": 0.8,
                    "catalyst_recall": 0.6,
                    "hallucination_score": 0.9,
                    "overall_quality_score": 0.85,
                },
            },
            {
                "company_name": "Alibaba Group Holding Ltd",
                "ticker": "BABA",
                "status": "completed",
                "llm_judge": {"status": "skipped"},
                "metrics": {
                    "factual_accuracy": 0.5,
                    "risk_recall": None,
                    "catalyst_recall": None,
                    "hallucination_score": None,
                    "overall_quality_score": 0.5,
                },
            },
        ]
    )

    assert summary["sample_count"] == 2
    assert summary["completed_count"] == 2
    assert summary["completion_rate"] == 1.0
    assert summary["judge_completed_count"] == 1
    assert summary["judge_execution_rate"] == 0.5
    assert summary["metrics"]["factual_accuracy"]["mean"] == 0.75
    assert summary["metrics"]["factual_accuracy"]["stddev"] == 0.25
    assert summary["metrics"]["factual_accuracy"]["min"] == 0.5
    assert summary["metrics"]["factual_accuracy"]["max"] == 1.0
    assert summary["metrics"]["risk_recall"]["mean"] == 0.8
    assert summary["metrics"]["risk_recall"]["count"] == 1
    assert summary["metrics"]["risk_recall"]["stddev"] == 0.0
    assert summary["metrics"]["overall_quality_score"]["mean"] == 0.675


def test_benchmark_runner_keeps_failed_sample_result_shape_consistent(
    tmp_path: Path,
) -> None:
    sample = BenchmarkSample(
        company_name="Apple Inc.",
        ticker="AAPL",
        expected_facts={"form_type": "10-K"},
        expected_key_points={"risks": [], "catalysts": [], "thesis": []},
    )

    def _failing_report_generator(sample: BenchmarkSample, artifacts_root: Path):
        raise RuntimeError("SEC Mapping API 返回 429")

    runner = BenchmarkRunner(report_generator=_failing_report_generator)

    result = runner.run(samples=[sample], output_dir=tmp_path / "benchmark")

    failed_sample = result["samples"][0]
    assert failed_sample["status"] == "failed"
    assert failed_sample["error_message"] == "SEC Mapping API 返回 429"
    assert failed_sample["factual_check"] is None
    assert failed_sample["llm_judge"] == {
        "status": "skipped",
        "reason": "样本执行失败：SEC Mapping API 返回 429",
    }
    assert failed_sample["generated_artifacts"] == {
        "run_dir": None,
        "final_report_path": None,
        "structured_report_path": None,
        "structured_recommendation_path": None,
        "latest_metrics_path": None,
    }
    assert failed_sample["workflow_metrics"] == {
        "status": "failed",
        "success": False,
        "error_message": "SEC Mapping API 返回 429",
    }

    detail_path = tmp_path / "benchmark" / "details" / f"{sample.sample_id}.json"
    persisted = json.loads(detail_path.read_text(encoding="utf-8"))
    assert persisted["generated_artifacts"] == failed_sample["generated_artifacts"]
    assert persisted["workflow_metrics"] == failed_sample["workflow_metrics"]


def test_benchmark_runner_applies_minimal_throttle_between_samples(
    tmp_path: Path,
) -> None:
    samples = [
        BenchmarkSample(
            company_name="Apple Inc.",
            ticker="AAPL",
            expected_facts={"form_type": "10-K"},
            expected_key_points={"risks": [], "catalysts": [], "thesis": []},
        ),
        BenchmarkSample(
            company_name="Alibaba Group Holding Ltd",
            ticker="BABA",
            expected_facts={"form_type": "20-F"},
            expected_key_points={"risks": [], "catalysts": [], "thesis": []},
        ),
    ]

    def _report_generator(sample: BenchmarkSample, artifacts_root: Path):
        run_dir = artifacts_root / sample.sample_id
        run_dir.mkdir(parents=True, exist_ok=True)
        final_report_path = run_dir / "04_investment_report.md"
        final_report_path.write_text("# report", encoding="utf-8")
        latest_metrics_path = run_dir / "latest_run_metrics.json"
        latest_metrics_path.write_text(
            json.dumps({"status": "completed", "success": True}, ensure_ascii=False),
            encoding="utf-8",
        )
        structured_report_path = run_dir / "07_structured_report.json"
        structured_report_path.write_text("{}", encoding="utf-8")
        structured_recommendation_path = run_dir / "06_structured_recommendation.json"
        structured_recommendation_path.write_text("{}", encoding="utf-8")
        return type("Generated", (), {
            "run_dir": run_dir,
            "final_report_path": final_report_path,
            "structured_report_path": structured_report_path,
            "structured_recommendation_path": structured_recommendation_path,
            "latest_metrics_path": latest_metrics_path,
        })()

    class StubFactualChecker:
        def evaluate(self, *, sample: BenchmarkSample, report_text: str):
            return {"factual_accuracy": 1.0}

    runner = BenchmarkRunner(
        factual_checker=StubFactualChecker(),
        report_generator=_report_generator,
        inter_sample_delay_seconds=0.25,
    )
    sleep_calls: list[float] = []

    with patch("time.sleep", side_effect=lambda seconds: sleep_calls.append(seconds)):
        result = runner.run(samples=samples, output_dir=tmp_path / "benchmark")

    assert [item["status"] for item in result["samples"]] == ["completed", "completed"]
    assert sleep_calls == [0.25]
