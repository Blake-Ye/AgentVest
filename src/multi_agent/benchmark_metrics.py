from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any

# 这一层只做聚合，不关心单条样本如何生成。
# 它把 sample-level metrics 汇总成 dashboard / API 适合消费的 summary。

_METRIC_NAMES = (
    "factual_accuracy",
    "risk_recall",
    "catalyst_recall",
    "hallucination_score",
    "overall_quality_score",
)


def round_metric(value: float) -> float:
    return round(value, 3)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round_metric(sum(values) / len(values))


def stddev(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return round_metric(statistics.pstdev(values))


def value_range(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None}
    return {
        "min": round_metric(min(values)),
        "max": round_metric(max(values)),
    }


def aggregate_benchmark_summary(sample_results: list[dict[str, Any]]) -> dict[str, Any]:
    """按指标计算均值与样本数，并统计运行完成/失败/降级情况。"""

    metrics_summary: dict[str, dict[str, Any]] = {}
    for metric_name in _METRIC_NAMES:
        values = [
            float(result["metrics"][metric_name])
            for result in sample_results
            if result.get("status") == "completed" and result.get("metrics", {}).get(metric_name) is not None
        ]
        metric_range = value_range(values)
        metrics_summary[metric_name] = {
            "mean": mean(values),
            "count": len(values),
            "stddev": stddev(values),
            "min": metric_range["min"],
            "max": metric_range["max"],
        }

    completed_count = sum(1 for result in sample_results if result.get("status") == "completed")
    failed_count = sum(1 for result in sample_results if result.get("status") == "failed")
    sample_count = len(sample_results)
    judge_completed_count = sum(
        1
        for result in sample_results
        if result.get("llm_judge", {}).get("status") == "completed"
    )
    judge_skipped_count = sum(
        1
        for result in sample_results
        if result.get("llm_judge", {}).get("status") in {"skipped", "unavailable"}
    )
    judge_execution_rate = round_metric(judge_completed_count / completed_count) if completed_count else 0.0
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": sample_count,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "completion_rate": round_metric(completed_count / sample_count) if sample_count else 0.0,
        "judge_completed_count": judge_completed_count,
        "judge_skipped_count": judge_skipped_count,
        "judge_execution_rate": judge_execution_rate,
        "metrics": metrics_summary,
    }
