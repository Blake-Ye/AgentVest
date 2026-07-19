import json
import signal
import subprocess
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent.core.market import MarketValidationResult, build_tool_policy
from multi_agent.core.report_document import ReportDocument
from multi_agent.settings import InvestmentResearchSettings
from multi_agent.main import (
    _build_run_output_paths,
    _final_status_from_result,
    _raise_user_facing_runtime_error,
    _workflow_inputs,
)
from multi_agent.resolver import CompanyResolution
from multi_agent.tools.investment_tools import FatalAPIError


def _typed_workflow_result(
    *,
    status: str = "passed",
    company_name: str = "Apple Inc.",
    company_ticker: str = "AAPL",
    trust_score: int = 91,
    **extra: object,
) -> dict[str, object]:
    mode, stance, summary, conclusion = {
        "passed": ("formal_report", "hold", "Formal evidence supports monitoring.", "Maintain hold."),
        "evidence_limited": (
            "evidence_limited_report",
            "watch",
            "证据受限，继续观察。",
            "证据受限，待补证后复核。",
        ),
        "blocked": ("blocked_notice", "blocked", "报告已阻断，不能形成投资结论。", "报告已阻断，不提供可执行投资建议。"),
    }[status]
    claim_ids = ["revenue"] if status == "passed" else []
    document = ReportDocument.model_validate(
        {
            "company_name": company_name,
            "ticker": company_ticker,
            "report_mode": mode,
            "title": f"{company_name} Investment Research",
            "stance": stance,
            "executive_summary": summary,
            "catalysts": ["Services growth"],
            "risks": ["Demand volatility"],
            "sections": {
                key: {
                    "key": key,
                    "heading": key,
                    "content": conclusion if key == "investment_conclusion" else f"{key} content",
                    "claim_ids": claim_ids if key in {"executive_summary", "financial_analysis", "investment_conclusion"} else [],
                }
                for key in (
                    "executive_summary", "business_overview", "recent_events", "financial_analysis",
                    "key_risks", "investment_conclusion", "source_index",
                )
            },
            "claims": [
                {"claim_id": "revenue", "text": "Revenue is supported by the filing.", "critical": True, "source_ids": ["sec-10k"]}
            ] if status == "passed" else [],
            "sources": [
                {"source_id": "sec-10k", "title": "Apple 10-K", "url": "https://www.sec.gov/Archives/example", "source_tag": "sec_filing"}
            ] if status == "passed" else [],
            "trust_score": trust_score,
            "allowed_claim_ids": ["revenue"] if status == "passed" else [],
        }
    )
    return {
        "status": status,
        "trust_score": trust_score,
        "report_document": document.model_dump(mode="json"),
        "final_decision_record": {
            "final_decision": status,
            "final_delivery_state": mode,
            "trust_score": trust_score,
        },
        **extra,
    }


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

    class StubMarketValidationService:
        def __init__(self, settings):
            self.settings = settings

        def validate(self, *, company_name: str, ticker: str = "", exchange: str = "") -> MarketValidationResult:
            assert company_name == "Alibaba Group Holding Ltd"
            assert ticker == "BABA"
            assert exchange == "NYSE"
            return MarketValidationResult(
                market_label="US",
                confidence=0.99,
                resolution_status="confirmed",
                evidence=["exchange=NYSE"],
                requires_human_confirmation=False,
                tool_policy=build_tool_policy("US"),
            )

    monkeypatch.setenv("MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")
    monkeypatch.setenv("SEC_API_KEY", "sec-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")
    monkeypatch.setattr("multi_agent.main.CompanyResolver", StubResolver)
    monkeypatch.setattr("multi_agent.main.MarketValidationService", StubMarketValidationService)

    inputs = _workflow_inputs("阿里", "")

    assert inputs["company_name"] == "Alibaba Group Holding Ltd"
    assert inputs["company_ticker"] == "BABA"
    assert inputs["company_market_label"] == "US"
    assert inputs["market_resolution_status"] == "confirmed"


def test_workflow_inputs_includes_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubResolver:
        def __init__(self, settings):
            self.settings = settings

        def resolve(self, company_name: str, ticker: str = "") -> CompanyResolution:
            return CompanyResolution(
                user_input=company_name,
                normalized_name="Apple Inc.",
                ticker="AAPL",
                entity_type="public_company",
                parent_company="Apple Inc.",
                exchange="NASDAQ",
                confidence=0.99,
            )

    class StubMarketValidationService:
        def __init__(self, settings):
            self.settings = settings

        def validate(self, *, company_name: str, ticker: str = "", exchange: str = "") -> MarketValidationResult:
            assert company_name == "Apple Inc."
            assert ticker == "AAPL"
            assert exchange == "NASDAQ"
            return MarketValidationResult(
                market_label="US",
                confidence=0.99,
                resolution_status="confirmed",
                evidence=["exchange=NASDAQ"],
                requires_human_confirmation=False,
                tool_policy=build_tool_policy("US"),
            )

    monkeypatch.setenv("MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")
    monkeypatch.setenv("SEC_API_KEY", "sec-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")
    monkeypatch.setattr("multi_agent.main.CompanyResolver", StubResolver)
    monkeypatch.setattr("multi_agent.main.MarketValidationService", StubMarketValidationService)
    monkeypatch.setattr("multi_agent.main._now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))

    inputs = _workflow_inputs("Apple Inc.", "AAPL")

    assert inputs["run_id"] == "20260615_103045"


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
        sec_api_key="sec-key",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
        final_report_path="report.md",
    )

    class StubParser:
        def parse_args(self):
            return Namespace(company_name="Apple Inc.", company_ticker="AAPL")

    class StubCrew:
        def kickoff(self, inputs):
            artifacts_dir = Path(inputs["artifacts_dir"])
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "00_market_validation.md",
                "01_market_intelligence.md",
                "02_filing_review.md",
                "03_financial_analysis.md",
                "08_data_quality_review.md",
                "09_logic_compliance_review.md",
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
    monkeypatch.setattr(
        main,
        "_kickoff_workflow",
        lambda inputs: (StubCrew().kickoff(inputs), _typed_workflow_result())[1],
    )
    monkeypatch.setattr(main, "_workflow_inputs", lambda *_: {"company_name": "Apple Inc.", "company_ticker": "AAPL"})
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(main, "_now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))

    main.run()

    run_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
    latest_metrics = (run_dir / "latest_run_metrics.json").read_text(encoding="utf-8")
    assert "Apple Inc." in latest_metrics
    assert '"success": true' in latest_metrics
    assert (run_dir / "00_market_validation.md").exists()
    assert (run_dir / "04_investment_report.md").exists()
    assert (run_dir / "08_data_quality_review.md").exists()
    assert (run_dir / "09_logic_compliance_review.md").exists()
    assert (run_dir / "evaluation_summary.json").exists()
    readme_content = (run_dir / "README.md").read_text(encoding="utf-8")
    assert "Apple Inc." in readme_content
    assert "00_market_validation.md" in readme_content
    assert "04_investment_report.md" in readme_content
    assert "09_logic_compliance_review.md" in readme_content


def test_run_uses_flow_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from multi_agent import main

    settings = InvestmentResearchSettings(
        model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        search_provider="tavily",
        tavily_api_key="tvly-key",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
        final_report_path="report.md",
    )

    class StubParser:
        def parse_args(self):
            return Namespace(company_name="Apple Inc.", company_ticker="AAPL")

    class StubFlow:
        def kickoff(self):
            artifacts_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "00_market_validation.md",
                "01_market_intelligence.md",
                "02_filing_review.md",
                "03_financial_analysis.md",
                "08_data_quality_review.md",
                "09_logic_compliance_review.md",
            ):
                (artifacts_dir / name).write_text("# artifact\n", encoding="utf-8")
            (artifacts_dir / "04_investment_report.md").write_text(
                "参考 https://example.com/report",
                encoding="utf-8",
            )
            return "ok"

    class StubFlowFactory:
        def __init__(self, initial_state):
            self.initial_state = initial_state

        def kickoff(self):
            assert self.initial_state.company_name == "Apple Inc."
            assert self.initial_state.company_ticker == "AAPL"
            return StubFlow().kickoff()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_FLOW_EXECUTION", "1")
    monkeypatch.setattr(main, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(main, "_build_parser", lambda: StubParser())
    monkeypatch.setattr(
        main,
        "_workflow_inputs",
        lambda *_: {
            "company_name": "Apple Inc.",
            "company_ticker": "AAPL",
            "company_market_label": "US",
            "market_resolution_status": "confirmed",
        },
    )
    monkeypatch.setattr(
        main,
        "_kickoff_workflow",
        lambda _inputs: (StubFlow().kickoff(), _typed_workflow_result())[1],
    )
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(main, "_now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))

    main.run()

    run_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
    assert (run_dir / "00_market_validation.md").exists()
    assert (run_dir / "04_investment_report.md").exists()
    assert (run_dir / "09_logic_compliance_review.md").exists()


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
        sec_api_key="sec-key",
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
                "00_market_validation.md",
                "01_market_intelligence.md",
                "02_filing_review.md",
                "03_financial_analysis.md",
                "08_data_quality_review.md",
                "09_logic_compliance_review.md",
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
    monkeypatch.setattr(
        main,
        "_kickoff_workflow",
        lambda inputs: (
            StubCrew().kickoff(inputs),
            _typed_workflow_result(
                company_name="Alibaba Group Holding Ltd", company_ticker="BABA"
            ),
        )[1],
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
    assert '"stance": "hold"' in recommendation_content
    assert '"trust_score"' in recommendation_content
    assert '"sections"' in structured_report_content
    assert '"final_delivery_state": "formal_report"' in structured_report_content
    latest_metrics = (run_dir / "latest_run_metrics.json").read_text(encoding="utf-8")
    assert '"trust_score"' in latest_metrics
    watchlist_content = (tmp_path / "artifacts" / "watchlist.json").read_text(encoding="utf-8")
    assert "BABA" in watchlist_content
    assert "持有" in watchlist_content


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
        sec_api_key="sec-key",
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


def test_run_rebuilds_watchlist_using_final_decision_projection_when_available(
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
        sec_api_key="sec-key",
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

    run_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_202646"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "04_investment_report.md").write_text(
        "\n".join(
            [
                "# 投资备忘录",
                "",
                "## 执行摘要",
                "",
                "苹果现金流稳健，估值看起来具备上行空间。",
                "",
                "## 催化剂",
                "",
                "- 服务业务扩张",
                "",
                "## 风险",
                "",
                "- 宏观需求承压",
                "",
                "## 投资建议",
                "",
                "建议增持。",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "latest_run_metrics.json").write_text(
        json.dumps(
            {
                "company_name": "Apple Inc.",
                "company_ticker": "AAPL",
                "trust_score": {"score": 74, "level": "medium", "summary": "evidence limited"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "final_decision.json").write_text(
        json.dumps(
            {
                "company_name": "Apple Inc.",
                "company_ticker": "AAPL",
                "final_decision": "evidence_limited",
                "final_delivery_state": "evidence_limited_report",
                "trust_score": 74,
                "blocking_reasons": ["gate_financial_coverage_incomplete"],
            },
            ensure_ascii=False,
            indent=2,
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
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))

    main.run()

    recommendation = json.loads((run_dir / "06_structured_recommendation.json").read_text(encoding="utf-8"))
    structured_report = json.loads((run_dir / "07_structured_report.json").read_text(encoding="utf-8"))
    watchlist = json.loads((tmp_path / "artifacts" / "watchlist.json").read_text(encoding="utf-8"))

    assert recommendation["status"] == "evidence_limited"
    assert recommendation["stance"] == "watch"
    assert recommendation["stance_label"] == "证据受限"
    assert structured_report["status"] == "evidence_limited"
    assert structured_report["final_decision"] == "evidence_limited"
    assert watchlist["items"][0]["status"] == "evidence_limited"


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
        sec_api_key="sec-key",
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
            StubTaskOutput("market_validation_task", "# 市场验证\n\nUS"),
            StubTaskOutput("market_intelligence_task", "# 市场情报\n\n参考 https://example.com/market"),
            StubTaskOutput("filing_review_task", "# Filing 复核\n\n参考 https://example.com/filing"),
            StubTaskOutput("financial_analysis_task", "# 财务分析\n\n参考 https://example.com/financial"),
            StubTaskOutput("investment_report_task", "# 投资备忘录\n\n参考 https://example.com/report"),
            StubTaskOutput("data_quality_review_task", "# 数据质量审查\n\n无阻塞问题"),
            StubTaskOutput("logic_compliance_review_task", "# 逻辑与合规审查\n\n无阻塞问题"),
        ]

    class StubCrew:
        def kickoff(self, inputs):
            return StubCrewResult()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(main, "_build_parser", lambda: StubParser())
    def _stub_kickoff_workflow(inputs):
        legacy = StubCrew().kickoff(inputs)
        return _typed_workflow_result(
            tasks_output=legacy.tasks_output,
            raw=legacy.raw,
        )

    monkeypatch.setattr(main, "_kickoff_workflow", _stub_kickoff_workflow)
    monkeypatch.setattr(main, "_workflow_inputs", lambda *_: {"company_name": "Apple Inc.", "company_ticker": "AAPL"})
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(main, "_now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))

    main.run()

    run_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
    assert "市场验证" in (run_dir / "00_market_validation.md").read_text(encoding="utf-8")
    assert "市场情报" in (run_dir / "01_market_intelligence.md").read_text(encoding="utf-8")
    assert "Filing 复核" in (run_dir / "02_filing_review.md").read_text(encoding="utf-8")
    assert "财务分析" in (run_dir / "03_financial_analysis.md").read_text(encoding="utf-8")
    assert "Investment Research" in (run_dir / "04_investment_report.md").read_text(encoding="utf-8")
    assert "数据质量审查" in (run_dir / "08_data_quality_review.md").read_text(encoding="utf-8")
    assert "逻辑与合规审查" in (run_dir / "09_logic_compliance_review.md").read_text(encoding="utf-8")
    latest_metrics = (run_dir / "latest_run_metrics.json").read_text(encoding="utf-8")
    assert '"report_generated": true' in latest_metrics
    assert '"report_complete": true' in latest_metrics


def test_run_writes_blocked_outputs_when_gate_fails(
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
        sec_api_key="sec-key",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
        final_report_path="report.md",
    )

    class StubParser:
        def parse_args(self):
            return Namespace(company_name="Apple Inc.", company_ticker="AAPL")

    blocked_result = _typed_workflow_result(status="blocked", trust_score=68)

    def _stub_kickoff_workflow(inputs):
        artifacts_dir = Path(inputs["artifacts_dir"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        for name, content in (
            ("00_market_validation.md", "# 市场验证结果\n\n市场已确认。"),
            ("01_market_intelligence.md", "# 市场情报简报\n\n已完成情报汇总。"),
            ("02_filing_review.md", "# 监管文件复核\n\n已完成文件复核。"),
            ("03_financial_analysis.md", "# 财务分析结果\n\n已完成财务分析。"),
            ("08_data_quality_review.md", "# 数据质量审查结果\n\n存在阻断问题。"),
            ("09_logic_compliance_review.md", "# 逻辑与合规审查结果\n\n存在阻断问题。"),
        ):
            (artifacts_dir / name).write_text(content, encoding="utf-8")
        return blocked_result

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(main, "_build_parser", lambda: StubParser())
    monkeypatch.setattr(main, "_workflow_inputs", lambda *_: {"company_name": "Apple Inc.", "company_ticker": "AAPL"})
    monkeypatch.setattr(main, "_kickoff_workflow", _stub_kickoff_workflow)
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(main, "_now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))

    main.run()

    run_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
    report_content = (run_dir / "04_investment_report.md").read_text(encoding="utf-8")
    recommendation = json.loads((run_dir / "06_structured_recommendation.json").read_text(encoding="utf-8"))
    structured_report = json.loads((run_dir / "07_structured_report.json").read_text(encoding="utf-8"))
    latest_metrics = (run_dir / "latest_run_metrics.json").read_text(encoding="utf-8")

    assert "阻断" in report_content
    assert "报告已阻断" in report_content
    assert recommendation["status"] == "blocked"
    assert recommendation["stance"] == "blocked"
    assert recommendation["stance_label"] == "阻断"
    assert structured_report["status"] == "blocked"
    assert structured_report["final_decision"] == "blocked"
    assert '"status": "completed"' in latest_metrics


def test_run_overwrites_existing_formal_report_when_gate_fails(
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
        sec_api_key="sec-key",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
        final_report_path="report.md",
    )

    class StubParser:
        def parse_args(self):
            return Namespace(company_name="Apple Inc.", company_ticker="AAPL")

    blocked_result = _typed_workflow_result(status="blocked", trust_score=52)

    def _stub_kickoff_workflow(inputs):
        artifacts_dir = Path(inputs["artifacts_dir"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        for name, content in (
            ("00_market_validation.md", "# 市场验证结果\n\n市场已确认。"),
            ("01_market_intelligence.md", "# 市场情报简报\n\n已完成情报汇总。"),
            ("02_filing_review.md", "# 监管文件复核\n\n已完成文件复核。"),
            ("03_financial_analysis.md", "# 财务分析结果\n\n已完成财务分析。"),
            ("08_data_quality_review.md", "# 数据质量审查结果\n\n存在阻断问题。"),
            ("09_logic_compliance_review.md", "# 逻辑与合规审查结果\n\n存在阻断问题。"),
        ):
            (artifacts_dir / name).write_text(content, encoding="utf-8")
        Path(inputs["final_report_path"]).write_text(
            "# 投资备忘录\n\n这是运行中先写出的正式报告，不应在 blocked 时保留。\n",
            encoding="utf-8",
        )
        return blocked_result

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(main, "_build_parser", lambda: StubParser())
    monkeypatch.setattr(main, "_workflow_inputs", lambda *_: {"company_name": "Apple Inc.", "company_ticker": "AAPL"})
    monkeypatch.setattr(main, "_kickoff_workflow", _stub_kickoff_workflow)
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(main, "_now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))

    main.run()

    run_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
    report_content = (run_dir / "04_investment_report.md").read_text(encoding="utf-8")

    assert "已阻断" in report_content
    assert "报告已阻断" in report_content
    assert "不应在 blocked 时保留" not in report_content


def test_run_allows_formal_report_after_rerun_passes(
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
        sec_api_key="sec-key",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
        final_report_path="report.md",
    )

    class StubParser:
        def parse_args(self):
            return Namespace(company_name="Apple Inc.", company_ticker="AAPL")

    final_result = _typed_workflow_result(status="passed", trust_score=88)

    def _stub_kickoff_workflow(inputs):
        artifacts_dir = Path(inputs["artifacts_dir"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        for name, content in (
            ("00_market_validation.md", "# 市场验证结果\n\n市场已确认。"),
            ("01_market_intelligence.md", "# 市场情报简报\n\n已完成情报汇总。"),
            ("02_filing_review.md", "# 监管文件复核\n\n已完成文件复核。"),
            ("03_financial_analysis.md", "# 财务分析结果\n\n已完成财务分析。"),
            ("08_data_quality_review.md", "# 数据质量审查结果\n\n已完成修复复核。"),
            ("09_logic_compliance_review.md", "# 逻辑与合规审查结果\n\n允许正式交付。"),
        ):
            (artifacts_dir / name).write_text(content, encoding="utf-8")
        return final_result

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(main, "_build_parser", lambda: StubParser())
    monkeypatch.setattr(main, "_workflow_inputs", lambda *_: {"company_name": "Apple Inc.", "company_ticker": "AAPL"})
    monkeypatch.setattr(main, "_kickoff_workflow", _stub_kickoff_workflow)
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(main, "_now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))

    main.run()

    run_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
    report_content = (run_dir / "04_investment_report.md").read_text(encoding="utf-8")
    recommendation = json.loads((run_dir / "06_structured_recommendation.json").read_text(encoding="utf-8"))
    structured_report = json.loads((run_dir / "07_structured_report.json").read_text(encoding="utf-8"))
    latest_metrics = json.loads((run_dir / "latest_run_metrics.json").read_text(encoding="utf-8"))

    assert "<!-- PLACEHOLDER -->" not in report_content
    assert "Maintain hold" in report_content
    assert recommendation["status"] == "passed"
    assert recommendation["stance"] == "hold"
    assert recommendation["stance_label"] == "持有"
    assert structured_report["status"] == "passed"
    assert structured_report["final_decision"] == "passed"
    assert structured_report["sections"]["investment_conclusion"].startswith("Maintain hold.")
    assert latest_metrics["final_status"] == "passed"
    assert latest_metrics["trust_score"]["score"] == 88


def test_run_preserves_evidence_limited_report_body_when_workflow_already_generated_one(
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
        sec_api_key="sec-key",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
        final_report_path="report.md",
    )

    class StubParser:
        def parse_args(self):
            return Namespace(company_name="Apple Inc.", company_ticker="AAPL")

    detailed_report = "\n".join(
        [
            "# 受限版投资备忘录",
            "",
            "**状态**：⚠️ 受限版备忘录 — Analysis Gate 未完全通过",
            "",
            "## 执行摘要",
            "",
            "这是一份保留正文的 evidence-limited 报告。",
        ]
    )
    final_result = _typed_workflow_result(status="evidence_limited", trust_score=74)

    def _stub_kickoff_workflow(inputs):
        artifacts_dir = Path(inputs["artifacts_dir"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        for name, content in (
            ("00_market_validation.md", "# 市场验证结果\n\n市场已确认。"),
            ("01_market_intelligence.md", "# 市场情报简报\n\n已完成情报汇总。"),
            ("02_filing_review.md", "# 监管文件复核\n\n已完成文件复核。"),
            ("03_financial_analysis.md", "# 财务分析结果\n\n已完成财务分析。"),
            ("08_data_quality_review.md", "# 数据质量审查结果\n\nformal 证据未闭合。"),
            ("09_logic_compliance_review.md", "# 逻辑与合规审查结果\n\n只允许受限交付。"),
        ):
            (artifacts_dir / name).write_text(content, encoding="utf-8")
        return final_result

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(main, "_build_parser", lambda: StubParser())
    monkeypatch.setattr(main, "_workflow_inputs", lambda *_: {"company_name": "Apple Inc.", "company_ticker": "AAPL"})
    monkeypatch.setattr(main, "_kickoff_workflow", _stub_kickoff_workflow)
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(main, "_now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))

    main.run()

    run_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
    report_content = (run_dir / "04_investment_report.md").read_text(encoding="utf-8")

    assert "证据受限，待补证后复核。" in report_content
    assert "这是一份保留正文的 evidence-limited 报告。" not in report_content


def test_run_writes_final_decision_and_projects_outputs_from_it(
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
        sec_api_key="sec-key",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
        final_report_path="report.md",
    )

    class StubParser:
        def parse_args(self):
            return Namespace(company_name="Apple Inc.", company_ticker="AAPL")

    final_result = _typed_workflow_result(status="evidence_limited", trust_score=74)

    def _stub_kickoff_workflow(inputs):
        artifacts_dir = Path(inputs["artifacts_dir"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        for name, content in (
            ("00_market_validation.md", "# 市场验证结果\n\n市场已确认。"),
            ("01_market_intelligence.md", "# 市场情报简报\n\n已完成情报汇总。"),
            ("02_filing_review.md", "# 监管文件复核\n\n已完成文件复核。"),
            ("03_financial_analysis.md", "# 财务分析结果\n\n已完成财务分析。"),
            ("08_data_quality_review.md", "# 数据质量审查结果\n\nformal 证据未闭合。"),
            ("09_logic_compliance_review.md", "# 逻辑与合规审查结果\n\n只允许受限交付。"),
        ):
            (artifacts_dir / name).write_text(content, encoding="utf-8")
        return final_result

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(main, "_build_parser", lambda: StubParser())
    monkeypatch.setattr(main, "_workflow_inputs", lambda *_: {"company_name": "Apple Inc.", "company_ticker": "AAPL"})
    monkeypatch.setattr(main, "_kickoff_workflow", _stub_kickoff_workflow)
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(main, "_now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))

    main.run()

    run_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
    final_decision = json.loads((run_dir / "final_decision.json").read_text(encoding="utf-8"))
    recommendation = json.loads((run_dir / "06_structured_recommendation.json").read_text(encoding="utf-8"))
    structured_report = json.loads((run_dir / "07_structured_report.json").read_text(encoding="utf-8"))
    latest_metrics = json.loads((run_dir / "latest_run_metrics.json").read_text(encoding="utf-8"))

    assert final_decision["final_decision"] == "evidence_limited"
    assert final_decision["final_delivery_state"] == "evidence_limited_report"
    assert recommendation["status"] == final_decision["final_decision"]
    assert structured_report["status"] == final_decision["final_decision"]
    assert latest_metrics["final_status"] == final_decision["final_decision"]


def test_final_status_from_result_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="未知"):
        _final_status_from_result({"status": "pending_review"})


def test_final_status_from_result_accepts_evidence_limited() -> None:
    assert _final_status_from_result({"status": "evidence_limited"}) == "evidence_limited"


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
        sec_api_key="sec-key",
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
    monkeypatch.setattr(main, "_kickoff_workflow", lambda inputs: FailingCrew().kickoff(inputs))
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
    assert "程序已终止" in (run_dir / "00_market_validation.md").read_text(encoding="utf-8")
    failure_report = (run_dir / "04_investment_report.md").read_text(encoding="utf-8")
    assert "程序已终止" in failure_report
    assert "403" in failure_report
    assert "程序已终止" in (run_dir / "09_logic_compliance_review.md").read_text(encoding="utf-8")


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
        sec_api_key="sec-key",
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
    monkeypatch.setattr(main, "_kickoff_workflow", lambda inputs: InterruptedCrew().kickoff(inputs))
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
    assert "运行被中断" in (run_dir / "00_market_validation.md").read_text(encoding="utf-8")
    failure_report = (run_dir / "04_investment_report.md").read_text(encoding="utf-8")
    assert "运行被中断" in failure_report
    assert "运行被中断" in (run_dir / "09_logic_compliance_review.md").read_text(encoding="utf-8")


def test_run_treats_sigterm_like_keyboard_interrupt(
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
        sec_api_key="sec-key",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
        final_report_path="report.md",
    )

    class StubParser:
        def parse_args(self):
            return Namespace(company_name="Apple Inc.", company_ticker="AAPL")

    registered_handlers: dict[signal.Signals, object] = {}

    def _stub_signal(sig: signal.Signals, handler: object) -> object:
        previous = registered_handlers.get(sig, signal.SIG_DFL)
        registered_handlers[sig] = handler
        return previous

    def _stub_kickoff_workflow(_inputs):
        sigterm_handler = registered_handlers[signal.SIGTERM]
        assert callable(sigterm_handler)
        sigterm_handler(signal.SIGTERM, None)
        raise AssertionError("SIGTERM handler should interrupt execution before continuing")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(main, "_build_parser", lambda: StubParser())
    monkeypatch.setattr(main, "_workflow_inputs", lambda *_: {"company_name": "Apple Inc.", "company_ticker": "AAPL"})
    monkeypatch.setattr(main, "_kickoff_workflow", _stub_kickoff_workflow)
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(main, "_now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))
    monkeypatch.setattr(main.signal, "signal", _stub_signal)

    with pytest.raises(SystemExit):
        main.run()

    run_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
    latest_metrics = (run_dir / "latest_run_metrics.json").read_text(encoding="utf-8")
    assert '"status": "failed"' in latest_metrics
    assert "运行被中断" in latest_metrics
    assert "运行被中断" in (run_dir / "04_investment_report.md").read_text(encoding="utf-8")


def test_run_overwrites_existing_formal_markdown_outputs_on_unexpected_error(
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
        sec_api_key="sec-key",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
        final_report_path="report.md",
    )

    class StubParser:
        def parse_args(self):
            return Namespace(company_name="Apple Inc.", company_ticker="AAPL")

    class PartiallyFailingCrew:
        def kickoff(self, inputs):
            artifacts_dir = Path(inputs["artifacts_dir"])
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            for name, content in (
                ("00_market_validation.md", "# 市场验证结果\n\n这是失败前写出的正式内容。"),
                ("01_market_intelligence.md", "# 市场情报简报\n\n这是失败前写出的正式内容。"),
                ("02_filing_review.md", "# 监管文件复核\n\n这是失败前写出的正式内容。"),
                ("03_financial_analysis.md", "# 财务分析结果\n\n这是失败前写出的正式内容。"),
                ("08_data_quality_review.md", "# 数据质量审查结果\n\n这是失败前写出的正式内容。"),
                ("09_logic_compliance_review.md", "# 逻辑与合规审查结果\n\n这是失败前写出的正式内容。"),
            ):
                (artifacts_dir / name).write_text(content, encoding="utf-8")
            Path(inputs["final_report_path"]).write_text(
                "# 投资备忘录\n\n这是失败前写出的正式报告，不应在异常后保留。\n",
                encoding="utf-8",
            )
            raise RuntimeError("模拟未预期错误")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(main, "_build_parser", lambda: StubParser())
    monkeypatch.setattr(
        main, "_kickoff_workflow", lambda inputs: PartiallyFailingCrew().kickoff(inputs)
    )
    monkeypatch.setattr(main, "_workflow_inputs", lambda *_: {"company_name": "Apple Inc.", "company_ticker": "AAPL"})
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(main, "_now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))

    with pytest.raises(SystemExit):
        main.run()

    run_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
    for name in (
        "00_market_validation.md",
        "01_market_intelligence.md",
        "02_filing_review.md",
        "03_financial_analysis.md",
        "04_investment_report.md",
        "08_data_quality_review.md",
        "09_logic_compliance_review.md",
    ):
        content = (run_dir / name).read_text(encoding="utf-8")
        assert "运行投研工作流时发生未预期错误：模拟未预期错误" in content
        assert "这是失败前写出的正式内容" not in content
        assert "这是失败前写出的正式报告，不应在异常后保留" not in content


def test_run_with_trigger_uses_flow_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from multi_agent import main

    settings = InvestmentResearchSettings(
        model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        search_provider="tavily",
        tavily_api_key="tvly-key",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
        final_report_path="report.md",
    )

    class StubFlow:
        def kickoff(self):
            artifacts_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "00_market_validation.md",
                "01_market_intelligence.md",
                "02_filing_review.md",
                "03_financial_analysis.md",
                "08_data_quality_review.md",
                "09_logic_compliance_review.md",
            ):
                (artifacts_dir / name).write_text("# artifact\n", encoding="utf-8")
            (artifacts_dir / "04_investment_report.md").write_text(
                "参考 https://example.com/report",
                encoding="utf-8",
            )
            return "ok"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_FLOW_EXECUTION", "1")
    monkeypatch.setattr(main, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(
        main,
        "_workflow_inputs",
        lambda *_: {
            "company_name": "Apple Inc.",
            "company_ticker": "AAPL",
            "company_market_label": "US",
            "market_resolution_status": "confirmed",
            "local_filing_pdf_path": "",
            "local_filing_pdf_available": "no",
        },
    )
    monkeypatch.setattr(
        main,
        "_kickoff_workflow",
        lambda _inputs: (StubFlow().kickoff(), _typed_workflow_result())[1],
    )
    monkeypatch.setattr(main, "_now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multi_agent.main",
            json.dumps({"company_name": "Apple Inc.", "company_ticker": "AAPL"}),
        ],
    )

    result = main.run_with_trigger()

    assert result["status"] == "passed"
    run_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
    assert (run_dir / "09_logic_compliance_review.md").exists()


def test_run_trigger_payload_treats_sigterm_like_keyboard_interrupt(
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
        sec_api_key="sec-key",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
        final_report_path="report.md",
    )

    registered_handlers: dict[signal.Signals, object] = {}

    def _stub_signal(sig: signal.Signals, handler: object) -> object:
        previous = registered_handlers.get(sig, signal.SIG_DFL)
        registered_handlers[sig] = handler
        return previous

    def _stub_kickoff_workflow(_inputs):
        sigterm_handler = registered_handlers[signal.SIGTERM]
        assert callable(sigterm_handler)
        sigterm_handler(signal.SIGTERM, None)
        raise AssertionError("SIGTERM handler should interrupt trigger execution before continuing")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(main.InvestmentResearchSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(
        main,
        "_workflow_inputs",
        lambda *_: {
            "company_name": "Apple Inc.",
            "company_ticker": "AAPL",
            "company_market_label": "US",
            "market_resolution_status": "confirmed",
            "local_filing_pdf_path": "",
            "local_filing_pdf_available": "no",
        },
    )
    monkeypatch.setattr(main, "_kickoff_workflow", _stub_kickoff_workflow)
    monkeypatch.setattr(main, "_now_for_output_paths", lambda: datetime(2026, 6, 15, 10, 30, 45))
    monkeypatch.setattr(main.signal, "signal", _stub_signal)

    with pytest.raises(SystemExit):
        main.run_trigger_payload({"company_name": "Apple Inc.", "company_ticker": "AAPL"})

    run_dir = tmp_path / "artifacts" / "apple_inc__aapl" / "20260615_103045"
    latest_metrics = (run_dir / "latest_run_metrics.json").read_text(encoding="utf-8")
    assert '"status": "failed"' in latest_metrics
    assert "运行被中断" in latest_metrics
    assert "运行被中断" in (run_dir / "04_investment_report.md").read_text(encoding="utf-8")


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
    assert output_paths.market_validation_path.name == "00_market_validation.md"
    assert output_paths.market_intelligence_path.name == "01_market_intelligence.md"
    assert output_paths.filing_review_path.name == "02_filing_review.md"
    assert output_paths.financial_analysis_path.name == "03_financial_analysis.md"
    assert output_paths.final_report_path.name == "04_investment_report.md"
    assert output_paths.runtime_log_path.name == "05_runtime.txt"
    assert output_paths.data_quality_review_path.name == "08_data_quality_review.md"
    assert output_paths.logic_compliance_review_path.name == "09_logic_compliance_review.md"
    assert output_paths.latest_metrics_path.name == "latest_run_metrics.json"
    assert output_paths.readme_path.name == "README.md"
