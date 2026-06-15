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
                "SEC_API_KEY=file-sec-key",
                "SEC_API_EMAIL=analyst@example.com",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.delenv("SEC_API_KEY", raising=False)
    monkeypatch.delenv("SEC_API_EMAIL", raising=False)

    settings = InvestmentResearchSettings.from_env()

    assert settings.openai_api_key == "file-llm-key"
    assert settings.serper_api_key == "file-serper-key"
    assert settings.sec_api_key == "file-sec-key"
    assert settings.sec_api_email == "analyst@example.com"


def test_settings_require_external_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("SEC_API_KEY", "sec-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")

    settings = InvestmentResearchSettings.from_env()

    assert settings.model == "qwen-plus"
    assert settings.serper_api_key == "serper-key"
    assert settings.sec_api_key == "sec-key"
    assert settings.sec_api_email == "analyst@example.com"
    assert settings.max_search_results == 5


def test_settings_accept_serpapi_key_without_serper_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.setenv("SERPAPI_API_KEY", "serpapi-key")
    monkeypatch.setenv("SEC_API_KEY", "sec-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")

    settings = InvestmentResearchSettings.from_env()

    assert settings.serpapi_api_key == "serpapi-key"
    assert settings.search_provider == "auto"


def test_settings_default_company_ticker_can_be_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL", "qwen-plus")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("SEC_API_KEY", "sec-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")
    monkeypatch.setenv("DEFAULT_COMPANY_TICKER", "")

    settings = InvestmentResearchSettings.from_env()

    assert settings.company_ticker == ""
