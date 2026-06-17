import pytest

from multi_agent.settings import InvestmentResearchSettings


def test_settings_load_values_from_dotenv_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=file-llm-key",
                "OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                "SERPER_API_KEY=file-serper-key",
                "SEC_API_EMAIL=analyst@example.com",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.delenv("SEC_API_EMAIL", raising=False)

    settings = InvestmentResearchSettings.from_env()

    assert settings.openai_api_key == "file-llm-key"
    assert settings.serper_api_key == "file-serper-key"
    assert settings.sec_api_email == "analyst@example.com"


def test_settings_require_external_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")

    settings = InvestmentResearchSettings.from_env()

    assert settings.model == "qwen-plus"
    assert settings.serper_api_key == "serper-key"
    assert settings.sec_api_email == "analyst@example.com"
    assert settings.max_search_results == 5


def test_settings_accept_serpapi_key_without_serper_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.setenv("SERPAPI_API_KEY", "serpapi-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")

    settings = InvestmentResearchSettings.from_env()

    assert settings.serpapi_api_key == "serpapi-key"
    assert settings.search_provider == "auto"


def test_settings_require_sec_api_email_without_extra_sec_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")

    settings = InvestmentResearchSettings.from_env()

    assert settings.sec_api_email == "analyst@example.com"


def test_settings_default_company_ticker_can_be_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "MODEL=qwen-plus",
                "OPENAI_API_KEY=llm-key",
                "OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                "SERPER_API_KEY=serper-key",
                "SEC_API_EMAIL=analyst@example.com",
                "ARTIFACTS_DIR=outputs/artifacts",
                "FINAL_REPORT_PATH=outputs/report.md",
                "WATCHLIST_PATH=outputs/watchlist/watchlist.json",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")
    monkeypatch.setenv("DEFAULT_COMPANY_TICKER", "")

    settings = InvestmentResearchSettings.from_env()

    assert settings.company_ticker == ""


def test_settings_use_outputs_directory_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=llm-key",
                "OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                "SERPER_API_KEY=serper-key",
                "SEC_API_EMAIL=analyst@example.com",
                "ARTIFACTS_DIR=outputs/artifacts",
                "FINAL_REPORT_PATH=outputs/report.md",
                "WATCHLIST_PATH=outputs/watchlist/watchlist.json",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")
    monkeypatch.delenv("ARTIFACTS_DIR", raising=False)
    monkeypatch.delenv("FINAL_REPORT_PATH", raising=False)
    monkeypatch.delenv("WATCHLIST_PATH", raising=False)

    settings = InvestmentResearchSettings.from_env()

    assert settings.artifacts_dir == "outputs/artifacts"
    assert settings.final_report_path == "outputs/report.md"
    assert settings.watchlist_path == "outputs/watchlist/watchlist.json"
