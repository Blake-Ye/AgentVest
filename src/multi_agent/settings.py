from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from multi_agent.core.artifact_paths import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_FINAL_REPORT_PATH,
    DEFAULT_LATEST_DIR,
    DEFAULT_RUNS_DIR,
    DEFAULT_WATCHLIST_PATH,
    build_run_artifact_paths,
    build_default_latest_dir,
    build_default_runs_dir,
    build_default_watchlist_path,
)


def _parse_dotenv_assignment(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None

    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()

    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]

    return key, value


def _load_dotenv_file(dotenv_path: Path) -> None:
    # 只在环境变量不存在时注入，避免覆盖用户已经显式导出的值。
    if not dotenv_path.exists():
        return

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        assignment = _parse_dotenv_assignment(line)
        if assignment is None:
            continue

        key, value = assignment
        os.environ.setdefault(key, value)


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


def _read_env_any(names: tuple[str, ...], *, default: str | None = None, required: bool = False) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()

    if required:
        raise ValueError(f"Missing required environment variable(s): {', '.join(names)}")
    return (default or "").strip()


@dataclass(frozen=True, init=False)
class InvestmentResearchSettings:
    """从环境变量读取的项目配置。"""

    fast_model: str
    deep_model: str
    review_model: str
    company_resolver_model: str
    openai_api_key: str
    openai_base_url: str
    tavily_api_key: str
    sec_api_email: str
    sec_api_key: str = ""
    serper_api_key: str = ""
    serpapi_api_key: str = ""
    _search_provider: str = "tavily"
    max_search_results: int = 5
    http_timeout_seconds: int = 20
    max_http_retries: int = 3
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    runs_dir: str = DEFAULT_RUNS_DIR
    latest_dir: str = DEFAULT_LATEST_DIR
    final_report_path: str = DEFAULT_FINAL_REPORT_PATH
    watchlist_path: str = DEFAULT_WATCHLIST_PATH
    company_name: str = "Apple Inc."
    company_ticker: str = ""
    company_market_label: str = ""
    local_filing_pdf_path: str = ""

    def __init__(
        self,
        *,
        fast_model: str | None = None,
        deep_model: str | None = None,
        review_model: str | None = None,
        company_resolver_model: str | None = None,
        openai_api_key: str,
        openai_base_url: str,
        tavily_api_key: str | None = None,
        sec_api_email: str,
        sec_api_key: str = "",
        serper_api_key: str = "",
        serpapi_api_key: str = "",
        search_provider: str | None = None,
        max_search_results: int = 5,
        http_timeout_seconds: int = 20,
        max_http_retries: int = 3,
        artifact_root: str = DEFAULT_ARTIFACT_ROOT,
        runs_dir: str | None = None,
        latest_dir: str | None = None,
        final_report_path: str | Path | None = None,
        watchlist_path: str | None = None,
        company_name: str = "Apple Inc.",
        company_ticker: str = "",
        company_market_label: str = "",
        local_filing_pdf_path: str = "",
        model: str | None = None,
        artifacts_dir: str | None = None,
    ) -> None:
        resolved_fast_model = (fast_model or model or "").strip()
        resolved_deep_model = (deep_model or model or "").strip()
        resolved_review_model = (review_model or model or "").strip()
        missing_names = [
            name
            for name, value in (
                ("fast_model/model", resolved_fast_model),
                ("deep_model/model", resolved_deep_model),
                ("review_model/model", resolved_review_model),
                ("openai_api_key", openai_api_key.strip()),
                ("openai_base_url", openai_base_url.strip()),
                ("sec_api_email", sec_api_email.strip()),
            )
            if not value
        ]
        if missing_names:
            raise TypeError(f"Missing required settings values: {', '.join(missing_names)}")

        resolved_artifact_root = str(artifact_root)
        resolved_runs_dir = str(runs_dir or artifacts_dir or build_default_runs_dir(resolved_artifact_root))
        resolved_latest_dir = str(latest_dir or build_default_latest_dir(resolved_artifact_root))
        resolved_watchlist_path = str(watchlist_path or build_default_watchlist_path(resolved_artifact_root))
        resolved_final_report_path = str(
            final_report_path
            if final_report_path is not None
            else build_run_artifact_paths(Path(resolved_runs_dir)).final_report_path
        )
        resolved_tavily_api_key = (tavily_api_key or "").strip()

        object.__setattr__(self, "fast_model", resolved_fast_model)
        object.__setattr__(self, "deep_model", resolved_deep_model)
        object.__setattr__(self, "review_model", resolved_review_model)
        object.__setattr__(
            self,
            "company_resolver_model",
            (company_resolver_model or resolved_fast_model).strip(),
        )
        object.__setattr__(self, "openai_api_key", openai_api_key.strip())
        object.__setattr__(self, "openai_base_url", openai_base_url.strip())
        object.__setattr__(self, "tavily_api_key", resolved_tavily_api_key)
        object.__setattr__(self, "sec_api_email", sec_api_email.strip())
        object.__setattr__(self, "sec_api_key", sec_api_key.strip())
        object.__setattr__(self, "serper_api_key", serper_api_key.strip())
        object.__setattr__(self, "serpapi_api_key", serpapi_api_key.strip())
        object.__setattr__(self, "_search_provider", (search_provider or "tavily").strip() or "tavily")
        object.__setattr__(self, "max_search_results", int(max_search_results))
        object.__setattr__(self, "http_timeout_seconds", int(http_timeout_seconds))
        object.__setattr__(self, "max_http_retries", int(max_http_retries))
        object.__setattr__(self, "artifact_root", resolved_artifact_root)
        object.__setattr__(self, "runs_dir", resolved_runs_dir)
        object.__setattr__(self, "latest_dir", resolved_latest_dir)
        object.__setattr__(self, "final_report_path", resolved_final_report_path)
        object.__setattr__(self, "watchlist_path", resolved_watchlist_path)
        object.__setattr__(self, "company_name", company_name)
        object.__setattr__(self, "company_ticker", company_ticker)
        object.__setattr__(self, "company_market_label", company_market_label.strip().upper())
        object.__setattr__(self, "local_filing_pdf_path", local_filing_pdf_path)

    @classmethod
    def from_env(cls) -> "InvestmentResearchSettings":
        _bootstrap_env()
        required_groups = [
            ("FAST_MODEL", ("FAST_MODEL", "MODEL")),
            ("DEEP_MODEL", ("DEEP_MODEL", "MODEL")),
            ("REVIEW_MODEL", ("REVIEW_MODEL", "MODEL")),
            ("OPENAI_API_KEY", ("OPENAI_API_KEY",)),
            ("OPENAI_BASE_URL", ("OPENAI_BASE_URL",)),
            ("TAVILY_API_KEY", ("TAVILY_API_KEY",)),
            ("SEC_API_EMAIL", ("SEC_API_EMAIL",)),
        ]
        missing_names = [
            public_name
            for public_name, candidates in required_groups
            if not any(os.getenv(candidate, "").strip() for candidate in candidates)
        ]
        if missing_names:
            raise ValueError(
                "Missing required environment variables: "
                + ", ".join(missing_names)
                + ". Please fill them in your .env file."
            )

        artifact_root = _read_env("ARTIFACT_ROOT", default=DEFAULT_ARTIFACT_ROOT)
        legacy_artifacts_dir = _read_env("ARTIFACTS_DIR", default="")
        runs_dir = _read_env("RUNS_DIR", default=legacy_artifacts_dir or build_default_runs_dir(artifact_root))
        final_report_path = _read_env(
            "FINAL_REPORT_PATH",
            default=str(build_run_artifact_paths(Path(runs_dir)).final_report_path),
        )
        return cls(
            fast_model=_read_env_any(("FAST_MODEL", "MODEL"), required=True),
            deep_model=_read_env_any(("DEEP_MODEL", "MODEL"), required=True),
            review_model=_read_env_any(("REVIEW_MODEL", "MODEL"), required=True),
            company_resolver_model=_read_env(
                "COMPANY_RESOLVER_MODEL",
                default=_read_env_any(("FAST_MODEL", "MODEL"), required=True),
            ),
            openai_api_key=_read_env("OPENAI_API_KEY", required=True),
            openai_base_url=_read_env("OPENAI_BASE_URL", required=True),
            tavily_api_key=_read_env("TAVILY_API_KEY", required=True),
            sec_api_email=_read_env("SEC_API_EMAIL", required=True),
            sec_api_key=_read_env("SEC_API_KEY", default=""),
            serper_api_key=_read_env("SERPER_API_KEY", default=""),
            serpapi_api_key=_read_env("SERPAPI_API_KEY", default=""),
            search_provider=_read_env(
                "SEARCH_PROVIDER",
                default="tavily",
            ),
            max_search_results=int(_read_env("MAX_SEARCH_RESULTS", default="5")),
            http_timeout_seconds=int(_read_env("HTTP_TIMEOUT_SECONDS", default="20")),
            max_http_retries=int(_read_env("MAX_HTTP_RETRIES", default="3")),
            artifact_root=artifact_root,
            runs_dir=runs_dir,
            latest_dir=_read_env("LATEST_DIR", default=build_default_latest_dir(artifact_root)),
            final_report_path=final_report_path,
            watchlist_path=_read_env("WATCHLIST_PATH", default=build_default_watchlist_path(artifact_root)),
            company_name=_read_env("DEFAULT_COMPANY_NAME", default="Apple Inc."),
            company_ticker=_read_env("DEFAULT_COMPANY_TICKER", default=""),
            company_market_label=_read_env("COMPANY_MARKET_LABEL", default=""),
            local_filing_pdf_path=_read_env("LOCAL_FILING_PDF_PATH", default=""),
        )

    @property
    def model(self) -> str:
        return self.deep_model

    @property
    def search_provider(self) -> str:
        return self._search_provider

    @property
    def artifacts_dir(self) -> str:
        return self.runs_dir
