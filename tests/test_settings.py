import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_agent.core.artifact_paths import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_FINAL_REPORT_PATH,
    DEFAULT_LATEST_DIR,
    DEFAULT_RUNS_DIR,
    DEFAULT_WATCHLIST_PATH,
    RunArtifactPaths,
    build_run_artifact_paths,
)
from multi_agent.settings import InvestmentResearchSettings


def _set_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAST_MODEL", "qwen-v4-flash")
    monkeypatch.setenv("DEEP_MODEL", "qwen-v4-pro")
    monkeypatch.setenv("REVIEW_MODEL", "qwen-v4-pro")
    monkeypatch.delenv("COMPANY_RESOLVER_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")


def _disable_dotenv_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("multi_agent.settings._bootstrap_env", lambda: None)


def test_settings_load_values_from_dotenv_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=file-llm-key",
                "OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                "FAST_MODEL=file-fast-model",
                "DEEP_MODEL=file-deep-model",
                "REVIEW_MODEL=file-review-model",
                "TAVILY_API_KEY=file-tavily-key",
                "SEC_API_EMAIL=analyst@example.com",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("FAST_MODEL", raising=False)
    monkeypatch.delenv("DEEP_MODEL", raising=False)
    monkeypatch.delenv("REVIEW_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SEC_API_EMAIL", raising=False)

    settings = InvestmentResearchSettings.from_env()

    assert settings.fast_model == "file-fast-model"
    assert settings.deep_model == "file-deep-model"
    assert settings.review_model == "file-review-model"
    assert settings.openai_api_key == "file-llm-key"
    assert settings.tavily_api_key == "file-tavily-key"
    assert settings.sec_api_email == "analyst@example.com"


def test_settings_load_values_from_dotenv_with_export_and_quotes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                'export FAST_MODEL="quoted-fast-model"',
                "export DEEP_MODEL='quoted-deep-model'",
                "REVIEW_MODEL=quoted-review-model",
                'OPENAI_API_KEY="quoted-llm-key"',
                "OPENAI_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'",
                'export TAVILY_API_KEY="quoted-tavily-key"',
                "SEC_API_EMAIL='analyst@example.com'",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("FAST_MODEL", raising=False)
    monkeypatch.delenv("DEEP_MODEL", raising=False)
    monkeypatch.delenv("REVIEW_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SEC_API_EMAIL", raising=False)

    settings = InvestmentResearchSettings.from_env()

    assert settings.fast_model == "quoted-fast-model"
    assert settings.deep_model == "quoted-deep-model"
    assert settings.review_model == "quoted-review-model"
    assert settings.openai_api_key == "quoted-llm-key"
    assert settings.openai_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert settings.tavily_api_key == "quoted-tavily-key"
    assert settings.sec_api_email == "analyst@example.com"


def test_settings_load_model_tiers_and_tavily_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_dotenv_bootstrap(monkeypatch)
    _set_required_env(monkeypatch)

    settings = InvestmentResearchSettings.from_env()

    assert settings.fast_model == "qwen-v4-flash"
    assert settings.deep_model == "qwen-v4-pro"
    assert settings.review_model == "qwen-v4-pro"
    assert settings.tavily_api_key == "tvly-key"
    assert settings.sec_api_email == "analyst@example.com"
    assert settings.company_resolver_model == "qwen-v4-flash"


def test_settings_company_resolver_model_can_override_fast_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_dotenv_bootstrap(monkeypatch)
    _set_required_env(monkeypatch)
    monkeypatch.setenv("COMPANY_RESOLVER_MODEL", "resolver-model")

    settings = InvestmentResearchSettings.from_env()

    assert settings.company_resolver_model == "resolver-model"


def test_settings_default_company_ticker_can_be_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    _disable_dotenv_bootstrap(monkeypatch)
    _set_required_env(monkeypatch)
    monkeypatch.setenv("DEFAULT_COMPANY_TICKER", "")

    settings = InvestmentResearchSettings.from_env()

    assert settings.company_ticker == ""


def test_settings_loads_runtime_company_market_label_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_dotenv_bootstrap(monkeypatch)
    _set_required_env(monkeypatch)
    monkeypatch.setenv("COMPANY_MARKET_LABEL", "hk")

    settings = InvestmentResearchSettings.from_env()

    assert settings.company_market_label == "HK"


def test_settings_use_default_artifact_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_dotenv_bootstrap(monkeypatch)
    _set_required_env(monkeypatch)
    monkeypatch.delenv("ARTIFACTS_DIR", raising=False)
    monkeypatch.delenv("ARTIFACT_ROOT", raising=False)
    monkeypatch.delenv("FINAL_REPORT_PATH", raising=False)
    monkeypatch.delenv("RUNS_DIR", raising=False)
    monkeypatch.delenv("LATEST_DIR", raising=False)
    monkeypatch.delenv("WATCHLIST_PATH", raising=False)

    settings = InvestmentResearchSettings.from_env()

    assert settings.artifact_root == DEFAULT_ARTIFACT_ROOT
    assert settings.runs_dir == DEFAULT_RUNS_DIR
    assert settings.latest_dir == DEFAULT_LATEST_DIR
    assert settings.watchlist_path == DEFAULT_WATCHLIST_PATH
    assert settings.final_report_path == DEFAULT_FINAL_REPORT_PATH


def test_settings_explicit_artifact_paths_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_dotenv_bootstrap(monkeypatch)
    _set_required_env(monkeypatch)
    monkeypatch.setenv("ARTIFACT_ROOT", "var/artifacts")
    monkeypatch.setenv("RUNS_DIR", "var/artifacts/runs")
    monkeypatch.setenv("LATEST_DIR", "var/artifacts/latest")
    monkeypatch.setenv("WATCHLIST_PATH", "var/artifacts/watchlist.json")

    settings = InvestmentResearchSettings.from_env()

    assert settings.artifact_root == "var/artifacts"
    assert settings.runs_dir == "var/artifacts/runs"
    assert settings.latest_dir == "var/artifacts/latest"
    assert settings.watchlist_path == "var/artifacts/watchlist.json"


def test_settings_accepts_legacy_artifacts_and_final_report_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_dotenv_bootstrap(monkeypatch)
    _set_required_env(monkeypatch)
    monkeypatch.delenv("ARTIFACT_ROOT", raising=False)
    monkeypatch.delenv("RUNS_DIR", raising=False)
    monkeypatch.delenv("LATEST_DIR", raising=False)
    monkeypatch.delenv("WATCHLIST_PATH", raising=False)
    monkeypatch.setenv("ARTIFACTS_DIR", "legacy-output")
    monkeypatch.setenv("FINAL_REPORT_PATH", "legacy-output/final.md")

    settings = InvestmentResearchSettings.from_env()

    assert settings.runs_dir == "legacy-output"
    assert settings.artifacts_dir == "legacy-output"
    assert settings.final_report_path == "legacy-output/final.md"


def test_settings_derives_final_report_path_from_legacy_artifacts_dir_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_dotenv_bootstrap(monkeypatch)
    _set_required_env(monkeypatch)
    monkeypatch.delenv("ARTIFACT_ROOT", raising=False)
    monkeypatch.delenv("RUNS_DIR", raising=False)
    monkeypatch.delenv("LATEST_DIR", raising=False)
    monkeypatch.delenv("WATCHLIST_PATH", raising=False)
    monkeypatch.delenv("FINAL_REPORT_PATH", raising=False)
    monkeypatch.setenv("ARTIFACTS_DIR", "legacy-output")

    settings = InvestmentResearchSettings.from_env()

    assert settings.runs_dir == "legacy-output"
    assert settings.final_report_path == "legacy-output/04_investment_report.md"


def test_settings_accepts_legacy_constructor_parameters() -> None:
    settings = InvestmentResearchSettings(
        model="qwen-plus",
        company_resolver_model="qwen-plus",
        openai_api_key="llm-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        search_provider="auto",
        serper_api_key="legacy-serper-key",
        serpapi_api_key="legacy-serpapi-key",
        sec_api_key="legacy-sec-key",
        sec_api_email="analyst@example.com",
        artifacts_dir="artifacts",
    )

    assert settings.fast_model == "qwen-plus"
    assert settings.deep_model == "qwen-plus"
    assert settings.review_model == "qwen-plus"
    assert settings.model == "qwen-plus"
    assert settings.search_provider == "auto"
    assert settings.serper_api_key == "legacy-serper-key"
    assert settings.serpapi_api_key == "legacy-serpapi-key"
    assert settings.sec_api_key == "legacy-sec-key"
    assert settings.tavily_api_key == ""
    assert settings.artifacts_dir == "artifacts"
    assert settings.runs_dir == "artifacts"


def test_settings_from_env_accepts_model_with_tavily_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_dotenv_bootstrap(monkeypatch)
    monkeypatch.delenv("FAST_MODEL", raising=False)
    monkeypatch.delenv("DEEP_MODEL", raising=False)
    monkeypatch.delenv("REVIEW_MODEL", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")
    monkeypatch.setenv("MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("SEC_API_KEY", "legacy-sec-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")

    settings = InvestmentResearchSettings.from_env()

    assert settings.fast_model == "qwen-plus"
    assert settings.deep_model == "qwen-plus"
    assert settings.review_model == "qwen-plus"
    assert settings.tavily_api_key == "tvly-key"
    assert settings.serper_api_key == ""
    assert settings.sec_api_key == "legacy-sec-key"


def test_settings_from_env_rejects_legacy_search_env_without_tavily(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_dotenv_bootstrap(monkeypatch)
    monkeypatch.delenv("FAST_MODEL", raising=False)
    monkeypatch.delenv("DEEP_MODEL", raising=False)
    monkeypatch.delenv("REVIEW_MODEL", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("SERPER_API_KEY", "legacy-serper-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")

    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        InvestmentResearchSettings.from_env()


def test_build_run_artifact_paths_returns_expected_files(tmp_path: Path) -> None:
    paths = build_run_artifact_paths(tmp_path / "run-001")

    assert paths == RunArtifactPaths(
        company_dir=tmp_path,
        run_dir=tmp_path / "run-001",
        market_intelligence_path=tmp_path / "run-001" / "01_market_intelligence.md",
        filing_review_path=tmp_path / "run-001" / "02_filing_review.md",
        financial_analysis_path=tmp_path / "run-001" / "03_financial_analysis.md",
        final_report_path=tmp_path / "run-001" / "04_investment_report.md",
        runtime_log_path=tmp_path / "run-001" / "05_runtime.txt",
        structured_recommendation_path=tmp_path / "run-001" / "06_structured_recommendation.json",
        structured_report_path=tmp_path / "run-001" / "07_structured_report.json",
        latest_metrics_path=tmp_path / "run-001" / "latest_run_metrics.json",
        evaluation_summary_path=tmp_path / "run-001" / "evaluation_summary.json",
        readme_path=tmp_path / "run-001" / "README.md",
        evidence_bundle_path=tmp_path / "run-001" / "10_research_evidence.json",
        report_document_path=tmp_path / "run-001" / "11_report_document.json",
        final_decision_path=tmp_path / "run-001" / "final_decision.json",
    )
    assert paths.market_validation_json == paths.market_intelligence_path
    assert paths.final_report_md == paths.final_report_path
    assert paths.final_report_json == paths.structured_report_path
