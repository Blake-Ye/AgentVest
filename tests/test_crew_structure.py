import os
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from crewai.memory.storage import kickoff_task_outputs_storage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent.crew import MultiAgent
from multi_agent.tools.review_tools import EvidenceCoverageTool
from multi_agent.tools.tavily_search import TavilySearchInput


@pytest.fixture
def writable_crewai_storage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    storage_dir = tmp_path / ".crewai_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(storage_dir))
    monkeypatch.setattr(kickoff_task_outputs_storage, "db_storage_path", lambda: storage_dir)
    return storage_dir


def test_multi_agent_crew_has_three_specialists(
    monkeypatch: pytest.MonkeyPatch, writable_crewai_storage: Path
) -> None:
    monkeypatch.setenv("FAST_MODEL", "qwen-plus")
    monkeypatch.setenv("DEEP_MODEL", "qwen-plus")
    monkeypatch.setenv("REVIEW_MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")

    crew = MultiAgent().crew()
    agent_roles = [agent.role for agent in crew.agents]
    task_descriptions = [task.description for task in crew.tasks]

    assert len(crew.agents) == 7
    assert any("市场验证分析师" in role for role in agent_roles)
    assert any("基本面分析师" in role for role in agent_roles)
    assert any("估值分析师" in role for role in agent_roles)
    assert any("事件与指引分析师" in role for role in agent_roles)
    assert any("数据质量审查员" in role for role in agent_roles)
    assert any("报告撰写分析师" in role for role in agent_roles)
    assert any("逻辑与合规审查员" in role for role in agent_roles)
    assert len(crew.tasks) == 7
    assert any("市场验证" in description for description in task_descriptions)
    assert any("投资备忘录" in description for description in task_descriptions)
    assert all(any("\u4e00" <= char <= "\u9fff" for char in description) for description in task_descriptions)

    for agent in crew.agents:
        assert "Read Local Artifact" not in [tool.name for tool in agent.tools]
        assert "Write Local Artifact" not in [tool.name for tool in agent.tools]

    event_agent = next(agent for agent in crew.agents if "事件与指引分析师" in agent.role)
    tavily_tool = next(tool for tool in event_agent.tools if tool.name == "Tavily Search Intelligence")
    assert "SEC Filing Content" in [tool.name for tool in event_agent.tools]
    assert tavily_tool.args_schema is TavilySearchInput
    assert {"query", "topic", "market_label", "company_name"}.issubset(
        tavily_tool.args_schema.model_fields.keys()
    )
    assert Path(os.environ["CREWAI_STORAGE_DIR"]) == writable_crewai_storage

    market_agent = next(agent for agent in crew.agents if "市场验证分析师" in agent.role)
    market_tool_names = [tool.name for tool in market_agent.tools]
    assert "Market Validation" in market_tool_names

    fundamental_agent = next(agent for agent in crew.agents if "基本面分析师" in agent.role)
    fundamental_tool_names = [tool.name for tool in fundamental_agent.tools]
    assert "SEC Filing Search" in fundamental_tool_names
    assert "SEC Filing Content" in fundamental_tool_names
    assert "SEC Company Facts" in fundamental_tool_names

    quant_agent = next(agent for agent in crew.agents if "估值分析师" in agent.role)
    quant_tool_names = [tool.name for tool in quant_agent.tools]
    assert "Financial Metrics Calculator" in quant_tool_names

    review_agent = next(agent for agent in crew.agents if "数据质量审查员" in agent.role)
    assert review_agent.llm.model == "qwen-plus"

    financial_tool_names = [tool.name for tool in fundamental_agent.tools]
    assert "Extract PDF Text" not in financial_tool_names


def test_financial_agent_only_exposes_pdf_tool_when_local_file_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, writable_crewai_storage: Path
) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setenv("FAST_MODEL", "qwen-plus")
    monkeypatch.setenv("DEEP_MODEL", "qwen-plus")
    monkeypatch.setenv("REVIEW_MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")
    monkeypatch.setenv("LOCAL_FILING_PDF_PATH", str(pdf_path))

    crew = MultiAgent().crew()
    fundamental_agent = next(agent for agent in crew.agents if "基本面分析师" in agent.role)
    financial_tool_names = [tool.name for tool in fundamental_agent.tools]

    assert "Extract PDF Text" in financial_tool_names
    assert "Read Local Artifact" not in financial_tool_names
    assert "Write Local Artifact" not in financial_tool_names
    assert Path(os.environ["CREWAI_STORAGE_DIR"]) == writable_crewai_storage


def test_crew_sets_writable_default_crewai_storage_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, writable_crewai_storage: Path
) -> None:
    monkeypatch.setenv("FAST_MODEL", "qwen-plus")
    monkeypatch.setenv("DEEP_MODEL", "qwen-plus")
    monkeypatch.setenv("REVIEW_MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    MultiAgent().crew()

    storage_dir = Path(os.environ["CREWAI_STORAGE_DIR"])
    assert storage_dir.exists()
    assert storage_dir.is_dir()
    assert storage_dir == writable_crewai_storage


def test_task_output_paths_are_relative_for_crewai_when_under_project_root(
    monkeypatch: pytest.MonkeyPatch, writable_crewai_storage: Path
) -> None:
    monkeypatch.setenv("FAST_MODEL", "qwen-plus")
    monkeypatch.setenv("DEEP_MODEL", "qwen-plus")
    monkeypatch.setenv("REVIEW_MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")
    project_root = Path(__file__).resolve().parents[1]
    run_dir = project_root / "var" / "runs" / "test_company__tst" / "20260622_999999"
    monkeypatch.setenv("ARTIFACTS_DIR", str(run_dir))
    monkeypatch.setenv("FINAL_REPORT_PATH", str(run_dir / "04_investment_report.md"))

    workflow = MultiAgent()

    assert workflow._task_output_file(  # type: ignore[attr-defined]
        run_dir / "00_market_validation.md"
    ) == "var/runs/test_company__tst/20260622_999999/00_market_validation.md"
    assert workflow._task_output_file(  # type: ignore[attr-defined]
        run_dir / "04_investment_report.md"
    ) == "var/runs/test_company__tst/20260622_999999/04_investment_report.md"


def test_evidence_coverage_tool_flags_unsupported_claims() -> None:
    tool = EvidenceCoverageTool()

    result = tool._run(
        claims=[
            {"claim": "营收增速改善", "evidence_refs": ["news-1"]},
            {"claim": "利润率显著扩张", "evidence_refs": []},
        ],
        valid_evidence_refs=["news-1"],
    )

    assert result["evidence_coverage_ratio"] == 0.5
    assert result["unsupported_claims"] == ["利润率显著扩张"]


def test_evidence_coverage_tool_rejects_unknown_references() -> None:
    result = EvidenceCoverageTool()._run(
        claims=[{"claim": "无来源结论", "evidence_refs": ["invented-ref"]}],
        valid_evidence_refs=["claim:revenue"],
    )

    assert result["evidence_coverage_ratio"] == 0.0
    assert result["unsupported_claims"] == ["无来源结论"]


def test_evidence_coverage_tool_schema_requires_explicit_claim_fields() -> None:
    tool = EvidenceCoverageTool()

    with pytest.raises(ValidationError):
        tool.args_schema.model_validate(
            {
                "claims": [
                    {"evidence_refs": ["news-1"]},
                ]
            }
        )


def test_data_quality_reviewer_exposes_expected_review_tools(
    monkeypatch: pytest.MonkeyPatch, writable_crewai_storage: Path
) -> None:
    monkeypatch.setenv("FAST_MODEL", "qwen-plus")
    monkeypatch.setenv("DEEP_MODEL", "qwen-plus")
    monkeypatch.setenv("REVIEW_MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")

    crew = MultiAgent().crew()
    review_agent = next(agent for agent in crew.agents if "数据质量审查员" in agent.role)
    review_tool_names = [tool.name for tool in review_agent.tools]

    assert set(review_tool_names) == {
        "evidence_coverage_tool",
        "cross_source_consistency_tool",
        "market_tool_policy_audit_tool",
        "financial_field_completeness_tool",
    }


def test_reviewer_prompts_reference_gate_and_tool_outputs() -> None:
    agents_yaml = Path("src/multi_agent/config/agents.yaml").read_text(encoding="utf-8")
    tasks_yaml = Path("src/multi_agent/config/tasks.yaml").read_text(encoding="utf-8")

    assert "不要主观决定是否放行" in agents_yaml
    assert "放行由 gate policy 决定" in agents_yaml
    assert "必须引用 reviewer tools 的结构化结果" in tasks_yaml
    assert "仅当 analysis gate 已通过时" in tasks_yaml
    assert "若 gate 未通过，不得生成正式投资建议" in tasks_yaml
    assert "必须输出阻断说明而非正式报告" in tasks_yaml


def test_writer_prompt_requires_revision_instructions_to_be_applied() -> None:
    tasks_yaml = Path("src/multi_agent/config/tasks.yaml").read_text(encoding="utf-8")

    writer_prompt = yaml.safe_load(tasks_yaml)["investment_report_task"]["description"]
    assert "{REPORT_CONTEXT_JSON}" in writer_prompt
    assert "REPORT_CONTEXT_JSON.revision_instructions" in writer_prompt
    assert "每一条 repair action" in writer_prompt
    assert "每一条 rerun reason" in writer_prompt
    assert "不得忽略 revision_instructions" in writer_prompt
    assert "实质性修订" in writer_prompt


def test_flow_crews_preserve_seven_agent_topology_and_allow_targeted_override(
    monkeypatch: pytest.MonkeyPatch, writable_crewai_storage: Path
) -> None:
    monkeypatch.setenv("FAST_MODEL", "fast-model")
    monkeypatch.setenv("DEEP_MODEL", "deep-model")
    monkeypatch.setenv("REVIEW_MODEL", "review-model")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")

    workflow = MultiAgent()
    workflow.configure_run(
        model_tier_overrides={"quant_valuation_analyst": "deep"},
        rerun_targets=["quant_valuation_analyst"],
    )

    assert len(workflow.analysis_crew().tasks) == 4
    assert len(workflow.analysis_review_crew().tasks) == 1
    assert len(workflow.report_writer_crew().tasks) == 1
    assert len(workflow.report_review_crew().tasks) == 1
    assert len(workflow.report_crew().tasks) == 2
    assert len(workflow.crew().tasks) == 7
    assert len(workflow.targeted_analysis_crew(["quant_valuation_analyst"]).tasks) == 1
    assert workflow.quant_valuation_analyst().llm.model == "deep-model"


def test_targeted_financial_repair_preserves_current_task_context_chain(
    monkeypatch: pytest.MonkeyPatch, writable_crewai_storage: Path
) -> None:
    monkeypatch.setenv("FAST_MODEL", "fast-model")
    monkeypatch.setenv("DEEP_MODEL", "deep-model")
    monkeypatch.setenv("REVIEW_MODEL", "review-model")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")

    targeted_tasks = MultiAgent().targeted_analysis_crew(["quant_valuation_analyst"]).tasks
    tasks_by_output = {Path(task.output_file).name: task for task in targeted_tasks}
    financial = tasks_by_output["03_financial_analysis.md"]
    assert financial.context == []


def test_analysis_reviewer_runs_in_a_separate_context_free_crew(
    monkeypatch: pytest.MonkeyPatch, writable_crewai_storage: Path
) -> None:
    monkeypatch.setenv("FAST_MODEL", "fast-model")
    monkeypatch.setenv("DEEP_MODEL", "deep-model")
    monkeypatch.setenv("REVIEW_MODEL", "review-model")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")

    workflow = MultiAgent()
    producer_tasks = workflow.targeted_analysis_crew(["quant_valuation_analyst"]).tasks
    reviewer_tasks = workflow.analysis_review_crew().tasks

    assert [Path(task.output_file).name for task in producer_tasks] == [
        "03_financial_analysis.md"
    ]
    assert [Path(task.output_file).name for task in reviewer_tasks] == [
        "08_data_quality_review.md"
    ]
    assert reviewer_tasks[0].context == []


def test_targeted_reviewer_has_stable_task_name_for_metrics(
    monkeypatch: pytest.MonkeyPatch, writable_crewai_storage: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")

    task = MultiAgent().targeted_analysis_crew(["data_quality_reviewer"]).tasks[0]
    report_tasks = MultiAgent().report_crew().tasks

    assert task.name == "data_quality_review_task"
    assert [item.name for item in report_tasks] == [
        "investment_report_task",
        "logic_compliance_review_task",
    ]


@pytest.mark.parametrize(
    ("target", "expected_outputs"),
    (
        ("market_validation_analyst", ("00_market_validation.md",)),
        ("event_guidance_analyst", ("01_market_intelligence.md",)),
        ("fundamental_analyst", ("02_filing_review.md", "03_financial_analysis.md")),
        ("quant_valuation_analyst", ("03_financial_analysis.md",)),
    ),
)
def test_targeted_evidence_repairs_run_only_required_producers(
    target: str,
    expected_outputs: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
    writable_crewai_storage: Path,
) -> None:
    monkeypatch.setenv("FAST_MODEL", "fast-model")
    monkeypatch.setenv("DEEP_MODEL", "deep-model")
    monkeypatch.setenv("REVIEW_MODEL", "review-model")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")

    tasks = MultiAgent().targeted_analysis_crew([target]).tasks

    assert tuple(Path(task.output_file).name for task in tasks) == expected_outputs
    if target in {"fundamental_analyst", "quant_valuation_analyst"}:
        quant_task = next(task for task in tasks if Path(task.output_file).name == "03_financial_analysis.md")
        assert "Financial Metrics Calculator" in [tool.name for tool in quant_task.agent.tools]
