from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv_file(dotenv_path: Path) -> None:
    # 只在环境变量不存在时注入，避免覆盖用户已经显式导出的值。
    if not dotenv_path.exists():
        return

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _bootstrap_env() -> None:
    # 优先读取当前工作目录，便于测试或多环境运行；再回退到项目根目录。
    current_dir_env = Path.cwd() / ".env"
    project_root_env = Path(__file__).resolve().parents[2] / ".env"
    _load_dotenv_file(current_dir_env)
    if current_dir_env != project_root_env:
        _load_dotenv_file(project_root_env)


def _read_env(name: str, *, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if value is None:
        if required:
            raise ValueError(f"Missing required environment variable: {name}")
        return ""

    value = value.strip()
    if required and not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class InvestmentResearchSettings:
    """从环境变量读取的项目配置。"""

    model: str
    company_resolver_model: str
    openai_api_key: str
    openai_base_url: str
    search_provider: str
    serper_api_key: str
    serpapi_api_key: str
    sec_api_email: str
    max_search_results: int = 5
    http_timeout_seconds: int = 20
    max_http_retries: int = 3
    artifacts_dir: str = "outputs/artifacts"
    final_report_path: str = "outputs/report.md"
    watchlist_path: str = "outputs/watchlist/watchlist.json"
    company_name: str = "Apple Inc."
    company_ticker: str = ""
    local_filing_pdf_path: str = ""

    @classmethod
    def from_env(cls) -> "InvestmentResearchSettings":
        _bootstrap_env()
        required_names = [
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "SEC_API_EMAIL",
        ]
        missing_names = [name for name in required_names if not os.getenv(name, "").strip()]
        if not (os.getenv("SERPER_API_KEY", "").strip() or os.getenv("SERPAPI_API_KEY", "").strip()):
            missing_names.append("SERPER_API_KEY or SERPAPI_API_KEY")
        if missing_names:
            raise ValueError(
                "Missing required environment variables: "
                + ", ".join(missing_names)
                + ". Please fill them in your .env file."
            )

        return cls(
            model=_read_env("MODEL", default="qwen-plus"),
            company_resolver_model=_read_env("COMPANY_RESOLVER_MODEL", default=_read_env("MODEL", default="qwen-plus")),
            openai_api_key=_read_env("OPENAI_API_KEY", required=True),
            openai_base_url=_read_env("OPENAI_BASE_URL", required=True),
            search_provider=_read_env("SEARCH_PROVIDER", default="auto").lower(),
            serper_api_key=_read_env("SERPER_API_KEY", default=""),
            serpapi_api_key=_read_env("SERPAPI_API_KEY", default=""),
            sec_api_email=_read_env("SEC_API_EMAIL", required=True),
            max_search_results=int(_read_env("MAX_SEARCH_RESULTS", default="5")),
            http_timeout_seconds=int(_read_env("HTTP_TIMEOUT_SECONDS", default="20")),
            max_http_retries=int(_read_env("MAX_HTTP_RETRIES", default="3")),
            artifacts_dir=_read_env("ARTIFACTS_DIR", default="outputs/artifacts"),
            final_report_path=_read_env("FINAL_REPORT_PATH", default="outputs/report.md"),
            watchlist_path=_read_env(
                "WATCHLIST_PATH",
                default="outputs/watchlist/watchlist.json",
            ),
            company_name=_read_env("DEFAULT_COMPANY_NAME", default="Apple Inc."),
            company_ticker=_read_env("DEFAULT_COMPANY_TICKER", default=""),
            local_filing_pdf_path=_read_env("LOCAL_FILING_PDF_PATH", default=""),
        )


@dataclass(frozen=True)
class BenchmarkSettings:
    """Benchmark 相关配置，默认复用现有 OpenAI 兼容接口配置。"""

    judge_model: str
    judge_api_key: str
    judge_base_url: str
    output_dir: str = "outputs/benchmark"
    llm_judge_timeout_seconds: int = 60
    inter_sample_delay_seconds: float = 1.0

    @classmethod
    def from_env(cls) -> "BenchmarkSettings":
        _bootstrap_env()
        return cls(
            judge_model=_read_env(
                "BENCHMARK_JUDGE_MODEL",
                default=_read_env("MODEL", default="qwen-plus"),
            ),
            judge_api_key=_read_env(
                "BENCHMARK_JUDGE_API_KEY",
                default=_read_env("OPENAI_API_KEY", default=""),
            ),
            judge_base_url=_read_env(
                "BENCHMARK_JUDGE_BASE_URL",
                default=_read_env("OPENAI_BASE_URL", default=""),
            ),
            output_dir=_read_env("BENCHMARK_OUTPUT_DIR", default="outputs/benchmark"),
            llm_judge_timeout_seconds=int(
                _read_env("BENCHMARK_JUDGE_TIMEOUT_SECONDS", default="60")
            ),
            inter_sample_delay_seconds=float(
                _read_env("BENCHMARK_INTER_SAMPLE_DELAY_SECONDS", default="1.0")
            ),
        )
