import json
import subprocess
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import pytest

from multi_agent.settings import InvestmentResearchSettings
from multi_agent.main import (
    _build_run_output_paths,
    _prepare_workflow_context,
    _raise_user_facing_runtime_error,
    _workflow_inputs,
)
from multi_agent.resolver import CompanyResolution
from multi_agent.tools.investment_tools import FatalAPIError


def test_module_entrypoint_exposes_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "multi_agent.main", "--help"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src"},
    )

    assert completed.returncode == 0
    assert "company-name" in completed.stdout


def test_raise_user_facing_runtime_error_exits_with_chinese_message() -> None:
    with pytest.raises(SystemExit) as exc_info:
        _raise_user_facing_runtime_error(FatalAPIError("外部服务返回 403，程序已终止。"))

    assert "程序已终止" in str(exc_info.value)


def test_workflow_inputs_auto_resolve_company_name_when_ticker_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubResolver:
        def __init__(self, settings):
            self.settings = settings

        def resolve(self, company_name: str, ticker: str = "") -> CompanyResolution:
            assert company_name == "阿里"
            assert ticker == ""
            return CompanyResolution(
                user_input=company_name,
                normalized_name="Alibaba Group Holding Ltd",
                ticker="BABA",
                entity_type="public_company",
                parent_company="Alibaba Group Holding Ltd",
                exchange="NYSE",
                confidence=0.98,
            )

    monkeypatch.setenv("MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")
    monkeypatch.setattr("multi_agent.main.CompanyResolver", StubResolver)

    inputs = _workflow_inputs("阿里", "")

    assert inputs["company_name"] == "Alibaba Group Holding Ltd"
    assert inputs["company_ticker"] == "BABA"


def test_prepare_workflow_context_tracks_alias_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubResolver:
        def __init__(self, settings):
            self.settings = settings

        def resolve(self, company_name: str, ticker: str = "") -> CompanyResolution:
            assert company_name == "苹果"
            assert ticker == ""
            return CompanyResolution(
                user_input=company_name,
                normalized_name="Apple Inc.",
                ticker="AAPL",
                entity_type="public_company",
                parent_company="Apple Inc.",
                exchange="NASDAQ",
                confidence=0.99,
                resolution_source="alias",
                resolution_steps=(
                    "检查内置别名表",
                    "命中“苹果” -> Apple Inc. / AAPL",
                    "无需触发小模型兜底",
                ),
            )

    monkeypatch.setenv("MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")
    monkeypatch.setattr("multi_agent.main.CompanyResolver", StubResolver)

    context = _prepare_workflow_context("苹果", "")

    assert context.workflow_inputs["company_name"] == "Apple Inc."
    assert context.workflow_inputs["company_ticker"] == "AAPL"
    assert context.resolution.resolution_source == "alias"
    assert context.resolution.resolution_steps[0] == "检查内置别名表"


def test_run_writes_evaluation_artifacts_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from multi_agent import main

    settings = InvestmentResearchSettings(
        model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        search_provider="auto",
        serper_api_key="serper-key",
        serpapi_api_key="",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
        final_report_path="report.md",
        watchlist_path="artifacts/watchlist.json",
    )

    class StubParser:
        def parse_args(self):
            return Namespace(company_name="Apple Inc.", company_ticker="AAPL")

    class StubCrew:
        def kickoff(self, inputs):
            artifacts_dir = Path(inputs["artifacts_dir"])
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "01_market_intelligence.md",
                "02_filing_review.md",
                "03_financial_analysis.md",
            ):
                (artifacts_dir / name).write_text("# artifact\n", encoding="utf-8")
            Path(inputs["final_report_path"]).write_text(
                "参考 https://example.com/report",
                encoding="utf-8",
            )
            return "ok"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(main, "_build_parser", lambda: StubParser())
    monkeypatch.setattr(main, "_crew", lambda: StubCrew())
    monkeypatch.setattr(main, "_workflow_inputs", lambda *_: {"company_name": "Apple Inc.", "company_ticker": "AAPL"})
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(main, "_now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))

    main.run()

    run_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
    latest_metrics = (run_dir / "latest_run_metrics.json").read_text(encoding="utf-8")
    assert "Apple Inc." in latest_metrics
    assert '"success": true' in latest_metrics
    assert (run_dir / "04_investment_report.md").exists()
    assert (run_dir / "evaluation_summary.json").exists()
    readme_content = (run_dir / "README.md").read_text(encoding="utf-8")
    assert "Apple Inc." in readme_content
    assert "04_investment_report.md" in readme_content


def test_run_writes_structured_recommendation_and_watchlist_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from multi_agent import main

    settings = InvestmentResearchSettings(
        model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        search_provider="auto",
        serper_api_key="serper-key",
        serpapi_api_key="",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
        final_report_path="report.md",
        watchlist_path="artifacts/watchlist.json",
    )

    class StubParser:
        def parse_args(self):
            return Namespace(
                company_name="Alibaba Group Holding Ltd",
                company_ticker="BABA",
                save_to_watchlist=True,
                watchlist_list=False,
            )

    class StubCrew:
        def kickoff(self, inputs):
            artifacts_dir = Path(inputs["artifacts_dir"])
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "01_market_intelligence.md",
                "02_filing_review.md",
                "03_financial_analysis.md",
            ):
                (artifacts_dir / name).write_text("# artifact\n", encoding="utf-8")
            Path(inputs["final_report_path"]).write_text(
                "\n".join(
                    [
                        "# 投资备忘录",
                        "",
                        "## 执行摘要",
                        "",
                        "阿里云利润率改善，现金流保持稳健。",
                        "",
                        "## 催化剂",
                        "",
                        "- 云业务恢复",
                        "- 回购计划",
                        "",
                        "## 风险",
                        "",
                        "- 宏观需求疲弱",
                        "",
                        "## 投资建议",
                        "",
                        "建议增持。",
                        "",
                        "参考 https://example.com/report",
                    ]
                ),
                encoding="utf-8",
            )
            return "ok"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(main, "_build_parser", lambda: StubParser())
    monkeypatch.setattr(main, "_crew", lambda: StubCrew())
    monkeypatch.setattr(
        main,
        "build_structured_recommendation",
        lambda **_: {
            "generated_at": "2026-06-17T00:00:00+00:00",
            "company_name": "Alibaba Group Holding Ltd",
            "company_ticker": "BABA",
            "stance": "buy",
            "stance_label": "增持",
            "trust_score": 87.0,
            "trust_level": "high",
            "trust_summary": "证据较充分，可作为高优先级研究输入。",
            "summary": "阿里云利润率改善，现金流保持稳健。",
            "catalysts": ["云业务恢复", "回购计划"],
            "risks": ["宏观需求疲弱"],
            "next_actions": ["继续跟踪"],
            "source_report_path": str(
                tmp_path / "artifacts" / "alibaba_group_holding_ltd__baba" / "20260615_103045" / "04_investment_report.md"
            ),
        },
    )
    monkeypatch.setattr(
        main,
        "build_structured_report",
        lambda **_: {
            "generated_at": "2026-06-17T00:00:00+00:00",
            "company_name": "Alibaba Group Holding Ltd",
            "company_ticker": "BABA",
            "summary": "阿里云利润率改善，现金流保持稳健。",
            "stance": "buy",
            "stance_label": "增持",
            "trust_score": 87.0,
            "trust_level": "high",
            "trust_summary": "证据较充分，可作为高优先级研究输入。",
            "catalysts": ["云业务恢复", "回购计划"],
            "risks": ["宏观需求疲弱"],
            "next_actions": ["继续跟踪"],
            "sections": {"investment_recommendation": "建议增持。"},
            "citation_urls": ["https://example.com/report"],
            "validation": {"has_summary": True},
            "source_report_path": str(
                tmp_path / "artifacts" / "alibaba_group_holding_ltd__baba" / "20260615_103045" / "04_investment_report.md"
            ),
        },
    )
    monkeypatch.setattr(
        main,
        "_workflow_inputs",
        lambda *_: {"company_name": "Alibaba Group Holding Ltd", "company_ticker": "BABA"},
    )
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(main, "_now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))

    main.run()

    run_dir = tmp_path / "artifacts" / "alibaba_group_holding_ltd__baba" / "20260615_103045"
    recommendation_path = run_dir / "06_structured_recommendation.json"
    structured_report_path = run_dir / "07_structured_report.json"
    assert recommendation_path.exists()
    assert structured_report_path.exists()
    recommendation_content = recommendation_path.read_text(encoding="utf-8")
    structured_report_content = structured_report_path.read_text(encoding="utf-8")
    assert '"stance": "buy"' in recommendation_content
    assert '"trust_score"' in recommendation_content
    assert '"sections"' in structured_report_content
    assert '"validation"' in structured_report_content
    latest_metrics = (run_dir / "latest_run_metrics.json").read_text(encoding="utf-8")
    assert '"trust_score"' in latest_metrics
    watchlist_content = (tmp_path / "artifacts" / "watchlist.json").read_text(encoding="utf-8")
    assert "BABA" in watchlist_content
    assert "增持" in watchlist_content


def test_run_rebuilds_watchlist_from_existing_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from multi_agent import main

    settings = InvestmentResearchSettings(
        model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        search_provider="auto",
        serper_api_key="serper-key",
        serpapi_api_key="",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
        final_report_path="report.md",
        watchlist_path="artifacts/watchlist.json",
    )

    class StubParser:
        def parse_args(self):
            return Namespace(
                company_name="",
                company_ticker="",
                save_to_watchlist=False,
                watchlist_list=False,
                watchlist_rebuild=True,
            )

    run_dir = tmp_path / "artifacts" / "tencent_holdings_limited__0700_hk" / "20260615_202646"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "04_investment_report.md").write_text(
        "\n".join(
            [
                "# 投资备忘录",
                "",
                "## ✅ 执行摘要",
                "",
                "腾讯进入 AI 商业化与视频号电商兑现并行阶段。",
                "",
                "## ⚠️ 风险：具象化、有依据、可跟踪",
                "",
                "| 风险类别 | 具体风险 | 依据与现状 |",
                "|----------|-----------|------------|",
                "| 监管风险 | 海外监管收紧 | 审批节奏存在不确定性 |",
                "",
                "## 🚀 催化剂：2026年内可验证、有明确时间表",
                "",
                "| 催化剂 | 触发条件 | 验证方式 |",
                "|---------|------------|------------|",
                "| TokenHub放量 | 企业客户续约率提升 | 中报披露 |",
                "",
                "## 💡 投资建议与置信度",
                "",
                "维持增持，继续观察执行兑现情况。",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "latest_run_metrics.json").write_text(
        json.dumps(
            {
                "company_name": "Tencent Holdings Limited",
                "company_ticker": "0700.HK",
                "trust_score": {
                    "score": 78.0,
                    "level": "medium",
                    "summary": "证据基本够用，但仍建议人工复核关键结论。",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "watchlist.json").write_text(
        json.dumps({"items": []}, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(main, "_build_parser", lambda: StubParser())
    monkeypatch.setattr(
        main,
        "build_structured_recommendation",
        lambda **_: {
            "generated_at": "2026-06-17T00:00:00+00:00",
            "company_name": "Tencent Holdings Limited",
            "company_ticker": "0700.HK",
            "stance": "buy",
            "stance_label": "增持",
            "trust_score": 78.0,
            "trust_level": "medium",
            "trust_summary": "证据基本够用，但仍建议人工复核关键结论。",
            "summary": "腾讯进入 AI 商业化与视频号电商兑现并行阶段。",
            "catalysts": ["TokenHub放量"],
            "risks": ["海外监管收紧"],
            "next_actions": ["继续跟踪"],
            "source_report_path": str(run_dir / "04_investment_report.md"),
        },
    )
    monkeypatch.setattr(
        main,
        "build_structured_report",
        lambda **_: {
            "generated_at": "2026-06-17T00:00:00+00:00",
            "company_name": "Tencent Holdings Limited",
            "company_ticker": "0700.HK",
            "summary": "腾讯进入 AI 商业化与视频号电商兑现并行阶段。",
            "stance": "buy",
            "stance_label": "增持",
            "trust_score": 78.0,
            "trust_level": "medium",
            "trust_summary": "证据基本够用，但仍建议人工复核关键结论。",
            "catalysts": ["TokenHub放量"],
            "risks": ["海外监管收紧"],
            "next_actions": ["继续跟踪"],
            "sections": {"investment_recommendation": "维持增持，继续观察执行兑现情况。"},
            "citation_urls": [],
            "validation": {"has_summary": True},
            "source_report_path": str(run_dir / "04_investment_report.md"),
        },
    )
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))

    main.run()

    recommendation = json.loads((run_dir / "06_structured_recommendation.json").read_text(encoding="utf-8"))
    structured_report = json.loads((run_dir / "07_structured_report.json").read_text(encoding="utf-8"))
    assert recommendation["company_ticker"] == "0700.HK"
    assert recommendation["summary"] == "腾讯进入 AI 商业化与视频号电商兑现并行阶段。"
    assert recommendation["catalysts"] == ["TokenHub放量"]
    assert recommendation["risks"] == ["海外监管收紧"]
    assert structured_report["sections"]["investment_recommendation"].startswith("维持增持")
    assert structured_report["validation"]["has_summary"] is True

    watchlist = json.loads((tmp_path / "artifacts" / "watchlist.json").read_text(encoding="utf-8"))
    assert len(watchlist["items"]) == 1
    assert watchlist["items"][0]["company_ticker"] == "0700.HK"
    assert watchlist["items"][0]["catalysts"] == ["TokenHub放量"]


def test_run_backfills_standard_output_files_when_crew_does_not_write_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from multi_agent import main

    settings = InvestmentResearchSettings(
        model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        search_provider="auto",
        serper_api_key="serper-key",
        serpapi_api_key="",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
        final_report_path="report.md",
    )

    class StubParser:
        def parse_args(self):
            return Namespace(company_name="Apple Inc.", company_ticker="AAPL")

    class StubTaskOutput:
        def __init__(self, name: str, raw: str) -> None:
            self.name = name
            self.raw = raw

    class StubCrewResult:
        raw = "# 最终报告\n\n参考 https://example.com/report"
        tasks_output = [
            StubTaskOutput("market_intelligence_task", "# 市场情报\n\n参考 https://example.com/market"),
            StubTaskOutput("filing_review_task", "# Filing 复核\n\n参考 https://example.com/filing"),
            StubTaskOutput("financial_analysis_task", "# 财务分析\n\n参考 https://example.com/financial"),
            StubTaskOutput("investment_report_task", "# 投资备忘录\n\n参考 https://example.com/report"),
        ]

    class StubCrew:
        def kickoff(self, inputs):
            return StubCrewResult()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(main, "_build_parser", lambda: StubParser())
    monkeypatch.setattr(main, "_crew", lambda: StubCrew())
    monkeypatch.setattr(main, "_workflow_inputs", lambda *_: {"company_name": "Apple Inc.", "company_ticker": "AAPL"})
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(main, "_now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))

    main.run()

    run_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
    assert "市场情报" in (run_dir / "01_market_intelligence.md").read_text(encoding="utf-8")
    assert "Filing 复核" in (run_dir / "02_filing_review.md").read_text(encoding="utf-8")
    assert "财务分析" in (run_dir / "03_financial_analysis.md").read_text(encoding="utf-8")
    assert "投资备忘录" in (run_dir / "04_investment_report.md").read_text(encoding="utf-8")
    latest_metrics = (run_dir / "latest_run_metrics.json").read_text(encoding="utf-8")
    assert '"report_generated": true' in latest_metrics
    assert '"report_complete": true' in latest_metrics


def test_run_writes_failed_evaluation_artifacts_on_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from multi_agent import main

    settings = InvestmentResearchSettings(
        model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        search_provider="auto",
        serper_api_key="serper-key",
        serpapi_api_key="",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
        final_report_path="report.md",
    )

    class StubParser:
        def parse_args(self):
            return Namespace(company_name="Apple Inc.", company_ticker="AAPL")

    class FailingCrew:
        def kickoff(self, inputs):
            raise FatalAPIError("外部服务返回 403，程序已终止。")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(main, "_build_parser", lambda: StubParser())
    monkeypatch.setattr(main, "_crew", lambda: FailingCrew())
    monkeypatch.setattr(main, "_workflow_inputs", lambda *_: {"company_name": "Apple Inc.", "company_ticker": "AAPL"})
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(main, "_now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))

    with pytest.raises(SystemExit):
        main.run()

    run_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
    latest_metrics = (run_dir / "latest_run_metrics.json").read_text(encoding="utf-8")
    assert '"success": false' in latest_metrics
    assert "403" in latest_metrics
    assert (run_dir / "README.md").exists()
    failure_report = (run_dir / "04_investment_report.md").read_text(encoding="utf-8")
    assert "程序已终止" in failure_report
    assert "403" in failure_report


def test_run_writes_failed_evaluation_artifacts_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from multi_agent import main

    settings = InvestmentResearchSettings(
        model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        search_provider="auto",
        serper_api_key="serper-key",
        serpapi_api_key="",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
        final_report_path="report.md",
    )

    class StubParser:
        def parse_args(self):
            return Namespace(company_name="Apple Inc.", company_ticker="AAPL")

    class InterruptedCrew:
        def kickoff(self, inputs):
            raise KeyboardInterrupt()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(main, "_build_parser", lambda: StubParser())
    monkeypatch.setattr(main, "_crew", lambda: InterruptedCrew())
    monkeypatch.setattr(main, "_workflow_inputs", lambda *_: {"company_name": "Apple Inc.", "company_ticker": "AAPL"})
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(main, "_now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))

    with pytest.raises(SystemExit):
        main.run()

    run_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
    latest_metrics = (run_dir / "latest_run_metrics.json").read_text(encoding="utf-8")
    assert '"status": "failed"' in latest_metrics
    assert "运行被中断" in latest_metrics
    assert (run_dir / "README.md").exists()
    failure_report = (run_dir / "04_investment_report.md").read_text(encoding="utf-8")
    assert "运行被中断" in failure_report


def test_run_with_trigger_reuses_execution_pipeline_and_passes_trigger_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from multi_agent import main

    settings = InvestmentResearchSettings(
        model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        search_provider="auto",
        serper_api_key="serper-key",
        serpapi_api_key="",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
        final_report_path="report.md",
    )
    captured_inputs: dict[str, object] = {}

    class StubCrew:
        def kickoff(self, inputs):
            captured_inputs.update(inputs)
            artifacts_dir = Path(inputs["artifacts_dir"])
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "01_market_intelligence.md",
                "02_filing_review.md",
                "03_financial_analysis.md",
            ):
                (artifacts_dir / name).write_text("# artifact\n", encoding="utf-8")
            Path(inputs["final_report_path"]).write_text(
                "# 投资备忘录\n\n参考 https://example.com/report",
                encoding="utf-8",
            )
            return {"status": "ok"}

    trigger_payload = {
        "company_name": "Apple Inc.",
        "company_ticker": "AAPL",
        "save_to_watchlist": True,
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(main, "_crew", lambda: StubCrew())
    monkeypatch.setattr(
        main,
        "_workflow_inputs",
        lambda *_: {
            "company_name": "Apple Inc.",
            "company_ticker": "AAPL",
            "local_filing_pdf_path": "未提供本地 PDF 文件",
            "local_filing_pdf_available": "no",
        },
    )
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(main, "_now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))
    monkeypatch.setattr(sys, "argv", ["run_with_trigger", json.dumps(trigger_payload)])

    result = main.run_with_trigger()

    assert result == {"status": "ok"}
    assert captured_inputs["crewai_trigger_payload"] == trigger_payload
    assert captured_inputs["company_name"] == "Apple Inc."
    assert captured_inputs["company_ticker"] == "AAPL"
    assert captured_inputs["local_filing_pdf_available"] == "no"


def test_build_run_output_paths_groups_all_outputs_in_company_folder(tmp_path: Path) -> None:
    output_paths = _build_run_output_paths(
        base_artifacts_dir=tmp_path / "artifacts",
        company_name="Shenzhen Inovance Technology Co., Ltd.",
        company_ticker="300124.SZ",
        run_time=datetime(2026, 6, 15, 9, 8, 7),
    )

    assert output_paths.run_dir == (
        tmp_path / "artifacts" / "shenzhen_inovance_technology_co_ltd__300124_sz" / "20260615_090807"
    )
    assert output_paths.company_dir == (
        tmp_path / "artifacts" / "shenzhen_inovance_technology_co_ltd__300124_sz"
    )
    assert output_paths.market_intelligence_path.name == "01_market_intelligence.md"
    assert output_paths.filing_review_path.name == "02_filing_review.md"
    assert output_paths.financial_analysis_path.name == "03_financial_analysis.md"
    assert output_paths.final_report_path.name == "04_investment_report.md"
    assert output_paths.runtime_log_path.name == "05_runtime.txt"
    assert output_paths.latest_metrics_path.name == "latest_run_metrics.json"
    assert output_paths.readme_path.name == "README.md"
