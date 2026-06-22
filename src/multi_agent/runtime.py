from __future__ import annotations

import os
from pathlib import Path


def prepare_runtime_env() -> None:
    """将 CrewAI 的运行时数据固定到项目目录，避免污染系统环境。"""
    project_root = Path(__file__).resolve().parents[2]
    local_home = project_root / ".crewai_home"
    (local_home / "Library" / "Application Support").mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(local_home)
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")


def build_crew():
    prepare_runtime_env()
    from multi_agent.crew import MultiAgent

    return MultiAgent().crew()


def build_flow(inputs: dict[str, str]):
    prepare_runtime_env()
    from multi_agent.core.market import MarketValidationResult, build_tool_policy
    from multi_agent.flows.market_review_flow import MarketReviewFlow, MarketReviewFlowState

    market_label = inputs.get("company_market_label", "").strip().upper() or "UNRESOLVED"
    resolution_status = inputs.get("market_resolution_status", "").strip().lower() or "unresolved"
    confidence = 0.95 if resolution_status == "confirmed" else 0.6 if resolution_status == "tentative" else 0.2
    market_validation = MarketValidationResult(
        market_label=market_label,  # type: ignore[arg-type]
        confidence=confidence,
        resolution_status=resolution_status,  # type: ignore[arg-type]
        evidence=[f"runtime_input.market_label={market_label}"],
        requires_human_confirmation=market_label == "UNRESOLVED" or resolution_status != "confirmed",
        tool_policy=build_tool_policy(market_label),  # type: ignore[arg-type]
    )

    return MarketReviewFlow(
        initial_state=MarketReviewFlowState(
            company_name=inputs.get("company_name", ""),
            input_ticker=inputs.get("company_ticker", ""),
            current_year=inputs.get("current_year", ""),
            artifacts_dir=inputs.get("artifacts_dir", ""),
            final_report_path=inputs.get("final_report_path", ""),
            local_filing_pdf_path=inputs.get("local_filing_pdf_path", "未提供本地 PDF 文件"),
            local_filing_pdf_available=inputs.get("local_filing_pdf_available", "no"),
            market_validation=market_validation,
        )
    )


def use_flow_execution() -> bool:
    return os.getenv("USE_FLOW_EXECUTION", "").strip().lower() in {"1", "true", "yes", "on"}


def kickoff_workflow(inputs: dict[str, str]):
    if use_flow_execution():
        return build_flow(inputs).kickoff()
    return build_crew().kickoff(inputs=inputs)
