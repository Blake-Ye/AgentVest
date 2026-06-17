from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest

import multi_agent.api as api_module
from multi_agent.api import create_app
from multi_agent.api_models import CreateJobRequest
from multi_agent.job_store import InMemoryJobStore, SQLiteJobStore


def test_submit_job_returns_job_id() -> None:
    job_store = InMemoryJobStore()

    def stub_runner(job_id: str, request: CreateJobRequest, store: InMemoryJobStore) -> None:
        store.update_job(job_id, status="completed")

    client = TestClient(create_app(job_store=job_store, job_runner=stub_runner))

    response = client.post(
        "/api/jobs",
        json={"company_name": "Apple Inc."},
    )

    assert response.status_code == 202
    payload = response.json()
    assert "job_id" in payload
    assert payload["status"] in {"queued", "running"}


def test_dashboard_page_is_served() -> None:
    client = TestClient(create_app(job_runner=lambda *_: None))

    response = client.get("/")

    assert response.status_code == 200
    assert "Investment Research Dashboard" in response.text
    assert "开始研究" in response.text
    assert "展开技术细节" in response.text
    assert "自动校正结果" in response.text


def test_job_detail_exposes_resolution_metadata() -> None:
    job_store = InMemoryJobStore()
    job = job_store.create_job(CreateJobRequest(company_name="苹果"))
    job_store.update_job(
        job.job_id,
        status="running",
        resolved_company_name="Apple Inc.",
        resolved_company_ticker="AAPL",
        resolution_source="alias",
        resolution_confidence=0.99,
        resolution_entity_type="public_company",
        resolution_steps=[
            "检查内置别名表",
            "命中“苹果” -> Apple Inc. / AAPL",
            "无需触发小模型兜底",
        ],
    )

    client = TestClient(create_app(job_store=job_store, job_runner=lambda *_: None))

    response = client.get(f"/api/jobs/{job.job_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["company_name"] == "苹果"
    assert payload["resolved_company_name"] == "Apple Inc."
    assert payload["resolved_company_ticker"] == "AAPL"
    assert payload["resolution_source"] == "alias"
    assert payload["resolution_confidence"] == 0.99
    assert payload["resolution_steps"][0] == "检查内置别名表"


def test_job_timeline_endpoint_returns_stage_events() -> None:
    job_store = InMemoryJobStore()
    job = job_store.create_job(CreateJobRequest(company_name="谷歌"))
    job_store.update_job(
        job.job_id,
        timeline_events=[
            {
                "stage_key": "resolution",
                "stage_label": "输入校正",
                "status": "completed",
                "summary": "已识别为 Alphabet Inc. / GOOGL",
                "details": ["别名命中：谷歌"],
                "agent_name": "CompanyResolver",
                "tool_name": "alias-map",
                "timestamp": "2026-06-17T00:00:00+00:00",
            }
        ],
    )

    client = TestClient(create_app(job_store=job_store, job_runner=lambda *_: None))

    response = client.get(f"/api/jobs/{job.job_id}/timeline")

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["stage_key"] == "resolution"
    assert payload[0]["status"] == "completed"
    assert payload[0]["details"] == ["别名命中：谷歌"]


def test_sqlite_job_store_persists_jobs(tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.db"
    created = SQLiteJobStore(db_path).create_job(
        CreateJobRequest(company_name="Apple Inc.", company_ticker="AAPL")
    )

    reloaded = SQLiteJobStore(db_path).get_job(created.job_id)

    assert reloaded is not None
    assert reloaded.company_name == "Apple Inc."


def test_artifact_list_endpoint_returns_standard_files(tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.db"
    store = SQLiteJobStore(db_path)
    job = store.create_job(CreateJobRequest(company_name="Apple Inc.", company_ticker="AAPL"))
    run_dir = tmp_path / "outputs" / "apple_inc__aapl" / "20260617_120000"
    run_dir.mkdir(parents=True)
    (run_dir / "04_investment_report.md").write_text("# 投资备忘录\n", encoding="utf-8")
    (run_dir / "notes.txt").write_text("debug", encoding="utf-8")
    store.update_job(job.job_id, status="completed", run_dir=str(run_dir))

    client = TestClient(create_app(job_store=store, job_runner=lambda *_: None))

    response = client.get(f"/api/jobs/{job.job_id}/artifacts")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["name"] == "04_investment_report.md"


def test_artifact_download_endpoint_returns_file_content(tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.db"
    store = SQLiteJobStore(db_path)
    job = store.create_job(CreateJobRequest(company_name="Apple Inc.", company_ticker="AAPL"))
    run_dir = tmp_path / "outputs" / "apple_inc__aapl" / "20260617_120000"
    run_dir.mkdir(parents=True)
    artifact_path = run_dir / "04_investment_report.md"
    artifact_path.write_text("# 投资备忘录\n\n这是测试报告。", encoding="utf-8")
    store.update_job(job.job_id, status="completed", run_dir=str(run_dir))

    client = TestClient(create_app(job_store=store, job_runner=lambda *_: None))

    response = client.get(f"/api/jobs/{job.job_id}/artifacts/04_investment_report.md")

    assert response.status_code == 200
    assert "投资备忘录" in response.text


def test_create_app_uses_configurable_job_db_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MULTI_AGENT_DATA_DIR", str(tmp_path / "data"))

    app = create_app()

    assert (tmp_path / "data" / "jobs.db").exists()
    assert app.title == "Multi-Agent Investment Research"


def test_api_main_reads_port_env(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_uvicorn = MagicMock()
    monkeypatch.setenv("PORT", "18000")
    monkeypatch.setattr(api_module, "uvicorn", mock_uvicorn, raising=False)

    api_module.main()

    mock_uvicorn.run.assert_called_once()
    _, kwargs = mock_uvicorn.run.call_args
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 18000


def test_render_blueprint_declares_web_service_runtime_contract() -> None:
    render_yaml = Path(__file__).resolve().parents[1] / "render.yaml"

    assert render_yaml.exists()
    content = render_yaml.read_text(encoding="utf-8")
    assert "type: web" in content
    assert "runtime: python" in content
    assert "buildCommand: pip install ." in content
    assert "python -m uvicorn multi_agent.api:create_app --factory" in content
    assert "key: MULTI_AGENT_DATA_DIR" in content
