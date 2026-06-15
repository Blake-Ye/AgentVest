import pytest

from multi_agent.crew import MultiAgent


def test_multi_agent_crew_has_three_specialists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("SEC_API_KEY", "sec-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")

    crew = MultiAgent().crew()
    agent_roles = [agent.role for agent in crew.agents]
    task_descriptions = [task.description for task in crew.tasks]

    assert len(crew.agents) == 3
    assert any("信息搜集分析师" in role for role in agent_roles)
    assert any("财报分析师" in role for role in agent_roles)
    assert any("报告撰写分析师" in role for role in agent_roles)
    assert len(crew.tasks) == 4
    assert any("投资备忘录" in description for description in task_descriptions)
    assert all(any("\u4e00" <= char <= "\u9fff" for char in description) for description in task_descriptions)

    for agent in crew.agents:
        assert "Read Local Artifact" not in [tool.name for tool in agent.tools]
        assert "Write Local Artifact" not in [tool.name for tool in agent.tools]

    financial_agent = next(agent for agent in crew.agents if "财报分析师" in agent.role)
    financial_tool_names = [tool.name for tool in financial_agent.tools]
    assert "Extract PDF Text" not in financial_tool_names


def test_financial_agent_only_exposes_pdf_tool_when_local_file_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setenv("MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("SEC_API_KEY", "sec-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")
    monkeypatch.setenv("LOCAL_FILING_PDF_PATH", str(pdf_path))

    crew = MultiAgent().crew()
    financial_agent = next(agent for agent in crew.agents if "财报分析师" in agent.role)
    financial_tool_names = [tool.name for tool in financial_agent.tools]

    assert "Extract PDF Text" in financial_tool_names
    assert "Read Local Artifact" not in financial_tool_names
    assert "Write Local Artifact" not in financial_tool_names
