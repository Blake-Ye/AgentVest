from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


JobStatus = Literal["queued", "running", "completed", "failed"]


class CreateJobRequest(BaseModel):
    company_name: str
    company_ticker: str | None = None
    save_to_watchlist: bool = False


class JobAcceptedResponse(BaseModel):
    job_id: str
    status: JobStatus


class ArtifactInfo(BaseModel):
    name: str
    size_bytes: int


class ResearchJob(BaseModel):
    job_id: str = Field(default_factory=lambda: uuid4().hex)
    company_name: str
    company_ticker: str | None = None
    save_to_watchlist: bool = False
    status: JobStatus = "queued"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error_message: str | None = None
    run_dir: str | None = None
    report_path: str | None = None

    def touch(self, *, status: JobStatus | None = None) -> "ResearchJob":
        updates = {"updated_at": datetime.now(timezone.utc).isoformat()}
        if status is not None:
            updates["status"] = status
        return self.model_copy(update=updates)
