from __future__ import annotations

from pathlib import Path
import re


def _tasks_yaml_text() -> str:
    tasks_path = Path(__file__).resolve().parents[1] / "src" / "multi_agent" / "config" / "tasks.yaml"
    return tasks_path.read_text(encoding="utf-8")


def test_task_prompts_require_machine_readable_json_contract() -> None:
    content = _tasks_yaml_text()

    assert "PART A: MACHINE_READABLE_JSON" in content
    assert "PART B: HUMAN_READABLE_MARKDOWN" in content
    assert "AgentVest Review Contract" in content


def test_investment_report_prompt_restricts_writing_to_report_mode() -> None:
    content = _tasks_yaml_text()

    assert "REPORT_MODE" in content
    assert "formal_report" in content
    assert "evidence_limited_report" in content
    assert "blocked_notice" in content
    assert "claim-to-source binding" in content
    assert "REPORT_CONTEXT_JSON" in content
    assert "10_research_evidence.json" in content
    assert "08_data_quality_review.json" in content
    assert "09_logic_compliance_review.json" in content
    assert "executive_summary、business_overview、recent_events" in content
    assert "不得输出 final_decision" in content


def test_financial_analysis_prompt_forbids_unsupported_external_benchmarks() -> None:
    content = re.sub(r"\s+", "", _tasks_yaml_text())

    assert "不得引入未在工具返回中出现的历史年份财务数字" in content
    assert "不得引用“行业典型区间”" in content
    assert "不得把假设性DCF、可比公司倍数、历史均值回归区间写成正式结论" in content
