from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from multi_agent.api_models import ArtifactInfo, CreateJobRequest, JobAcceptedResponse
from multi_agent.job_store import InMemoryJobStore, SQLiteJobStore
from multi_agent.main import WorkflowExecutionRequest, _execute_workflow, _workflow_inputs
from multi_agent.settings import InvestmentResearchSettings

JobRunner = Callable[[str, CreateJobRequest, InMemoryJobStore], None]
STANDARD_ARTIFACTS = {
    "01_market_intelligence.md",
    "02_filing_review.md",
    "03_financial_analysis.md",
    "04_investment_report.md",
    "06_structured_recommendation.json",
    "07_structured_report.json",
    "latest_run_metrics.json",
    "evaluation_summary.json",
}


def default_data_dir() -> Path:
    return Path(os.getenv("MULTI_AGENT_DATA_DIR", "outputs/api"))


def _resolve_latest_run_dir(settings: InvestmentResearchSettings, company_name: str, company_ticker: str) -> Path | None:
    company_slug = company_name.lower().replace(" ", "_")
    ticker_slug = company_ticker.lower().replace(".", "_")
    company_dir = Path(settings.artifacts_dir) / f"{company_slug}__{ticker_slug}"
    if not company_dir.exists():
        return None
    run_dirs = sorted([path for path in company_dir.iterdir() if path.is_dir()])
    if not run_dirs:
        return None
    return run_dirs[-1]


def run_research_job(job_id: str, request: CreateJobRequest, job_store: InMemoryJobStore) -> None:
    job_store.update_job(job_id, status="running")
    try:
        settings = InvestmentResearchSettings.from_env()
        workflow_inputs = _workflow_inputs(request.company_name, request.company_ticker)
        execution_request = WorkflowExecutionRequest(
            workflow_inputs=workflow_inputs,
            save_to_watchlist=request.save_to_watchlist,
        )
        _execute_workflow(
            settings=settings,
            request=execution_request,
            unexpected_error_prefix="API 任务运行时发生未预期错误",
        )
        latest_run_dir = _resolve_latest_run_dir(
            settings,
            workflow_inputs["company_name"],
            workflow_inputs["company_ticker"],
        )
        report_path = latest_run_dir / "04_investment_report.md" if latest_run_dir else None
        job_store.update_job(
            job_id,
            status="completed",
            run_dir=str(latest_run_dir) if latest_run_dir else None,
            report_path=str(report_path) if report_path and report_path.exists() else None,
        )
    except BaseException as exc:  # pragma: no cover - background thread safety
        job_store.update_job(job_id, status="failed", error_message=str(exc))


def create_app(
    *,
    job_store: InMemoryJobStore | SQLiteJobStore | None = None,
    job_runner: JobRunner | None = None,
) -> FastAPI:
    app = FastAPI(title="Multi-Agent Investment Research")
    store = job_store or SQLiteJobStore(default_data_dir() / "jobs.db")
    executor = ThreadPoolExecutor(max_workers=2)
    runner = job_runner or run_research_job
    dashboard_path = Path(__file__).with_name("dashboard.html")

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(dashboard_path.read_text(encoding="utf-8"))

    @app.post("/api/jobs", response_model=JobAcceptedResponse, status_code=202)
    def submit_job(request: CreateJobRequest) -> JobAcceptedResponse:
        job = store.create_job(request)
        executor.submit(runner, job.job_id, request, store)
        return JobAcceptedResponse(job_id=job.job_id, status=job.status)

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, object]]:
        return [job.model_dump() for job in store.list_jobs()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, object]:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.model_dump()

    @app.get("/api/jobs/{job_id}/artifacts")
    def list_artifacts(job_id: str) -> list[dict[str, object]]:
        job = store.get_job(job_id)
        if job is None or not job.run_dir:
            raise HTTPException(status_code=404, detail="job artifacts not found")
        run_dir = Path(job.run_dir)
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail="job artifacts not found")
        artifacts = [
            ArtifactInfo(name=path.name, size_bytes=path.stat().st_size).model_dump()
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name in STANDARD_ARTIFACTS
        ]
        return artifacts

    @app.get("/api/jobs/{job_id}/artifacts/{artifact_name}")
    def download_artifact(job_id: str, artifact_name: str) -> FileResponse:
        if artifact_name not in STANDARD_ARTIFACTS:
            raise HTTPException(status_code=404, detail="artifact not found")
        job = store.get_job(job_id)
        if job is None or not job.run_dir:
            raise HTTPException(status_code=404, detail="job artifacts not found")
        artifact_path = Path(job.run_dir) / artifact_name
        if not artifact_path.exists() or not artifact_path.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(artifact_path)

    return app


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(create_app(), host="0.0.0.0", port=port)
