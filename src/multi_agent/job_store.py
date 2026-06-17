from __future__ import annotations

import json
import sqlite3
from threading import Lock
from pathlib import Path

from multi_agent.api_models import CreateJobRequest, JobTimelineEvent, ResearchJob


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, ResearchJob] = {}
        self._lock = Lock()

    def create_job(self, request: CreateJobRequest) -> ResearchJob:
        job = ResearchJob(
            company_name=request.company_name,
            company_ticker=request.company_ticker,
            save_to_watchlist=request.save_to_watchlist,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> ResearchJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[ResearchJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def update_job(self, job_id: str, **updates: object) -> ResearchJob:
        with self._lock:
            current = self._jobs[job_id]
            updated = ResearchJob.model_validate({**current.touch().model_dump(), **updates})
            self._jobs[job_id] = updated
            return updated


class SQLiteJobStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    company_ticker TEXT,
                    save_to_watchlist INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error_message TEXT,
                    run_dir TEXT,
                    report_path TEXT,
                    resolved_company_name TEXT,
                    resolved_company_ticker TEXT,
                    resolution_source TEXT,
                    resolution_confidence REAL,
                    resolution_entity_type TEXT,
                    resolution_steps TEXT NOT NULL DEFAULT '[]',
                    timeline_events TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            existing_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
            }
            required_columns = {
                "resolved_company_name": "TEXT",
                "resolved_company_ticker": "TEXT",
                "resolution_source": "TEXT",
                "resolution_confidence": "REAL",
                "resolution_entity_type": "TEXT",
                "resolution_steps": "TEXT NOT NULL DEFAULT '[]'",
                "timeline_events": "TEXT NOT NULL DEFAULT '[]'",
            }
            for column_name, column_type in required_columns.items():
                if column_name not in existing_columns:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {column_name} {column_type}")

    @staticmethod
    def _from_row(row: sqlite3.Row | None) -> ResearchJob | None:
        if row is None:
            return None
        return ResearchJob(
            job_id=row["job_id"],
            company_name=row["company_name"],
            company_ticker=row["company_ticker"],
            save_to_watchlist=bool(row["save_to_watchlist"]),
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            error_message=row["error_message"],
            run_dir=row["run_dir"],
            report_path=row["report_path"],
            resolved_company_name=row["resolved_company_name"],
            resolved_company_ticker=row["resolved_company_ticker"],
            resolution_source=row["resolution_source"],
            resolution_confidence=row["resolution_confidence"],
            resolution_entity_type=row["resolution_entity_type"],
            resolution_steps=json.loads(row["resolution_steps"] or "[]"),
            timeline_events=[
                JobTimelineEvent.model_validate(event)
                for event in json.loads(row["timeline_events"] or "[]")
            ],
        )

    def create_job(self, request: CreateJobRequest) -> ResearchJob:
        job = ResearchJob(
            company_name=request.company_name,
            company_ticker=request.company_ticker,
            save_to_watchlist=request.save_to_watchlist,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, company_name, company_ticker, save_to_watchlist, status,
                    created_at, updated_at, error_message, run_dir, report_path,
                    resolved_company_name, resolved_company_ticker, resolution_source,
                    resolution_confidence, resolution_entity_type, resolution_steps, timeline_events
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.company_name,
                    job.company_ticker,
                    int(job.save_to_watchlist),
                    job.status,
                    job.created_at,
                    job.updated_at,
                    job.error_message,
                    job.run_dir,
                    job.report_path,
                    job.resolved_company_name,
                    job.resolved_company_ticker,
                    job.resolution_source,
                    job.resolution_confidence,
                    job.resolution_entity_type,
                    json.dumps(job.resolution_steps, ensure_ascii=False),
                    json.dumps([event.model_dump() for event in job.timeline_events], ensure_ascii=False),
                ),
            )
        return job

    def get_job(self, job_id: str) -> ResearchJob | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._from_row(row)

    def list_jobs(self) -> list[ResearchJob]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [job for row in rows if (job := self._from_row(row)) is not None]

    def update_job(self, job_id: str, **updates: object) -> ResearchJob:
        current = self.get_job(job_id)
        if current is None:
            raise KeyError(job_id)
        updated = ResearchJob.model_validate({**current.touch().model_dump(), **updates})
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET company_name = ?, company_ticker = ?, save_to_watchlist = ?, status = ?,
                    created_at = ?, updated_at = ?, error_message = ?, run_dir = ?, report_path = ?,
                    resolved_company_name = ?, resolved_company_ticker = ?, resolution_source = ?,
                    resolution_confidence = ?, resolution_entity_type = ?, resolution_steps = ?,
                    timeline_events = ?
                WHERE job_id = ?
                """,
                (
                    updated.company_name,
                    updated.company_ticker,
                    int(updated.save_to_watchlist),
                    updated.status,
                    updated.created_at,
                    updated.updated_at,
                    updated.error_message,
                    updated.run_dir,
                    updated.report_path,
                    updated.resolved_company_name,
                    updated.resolved_company_ticker,
                    updated.resolution_source,
                    updated.resolution_confidence,
                    updated.resolution_entity_type,
                    json.dumps(updated.resolution_steps, ensure_ascii=False),
                    json.dumps(
                        [event.model_dump() for event in updated.timeline_events],
                        ensure_ascii=False,
                    ),
                    updated.job_id,
                ),
            )
        return updated
