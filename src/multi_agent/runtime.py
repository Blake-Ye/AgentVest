from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CrewAIRuntimePaths:
    runtime_root: Path
    app_support_dir: Path
    storage_dir: Path


def prepare_runtime_env(*, base_dir: Path | None = None) -> CrewAIRuntimePaths:
    """将 CrewAI 的运行时写盘位置固定到项目目录，且不覆盖进程级 HOME。"""
    runtime_base_dir = base_dir or Path.cwd()
    runtime_root = runtime_base_dir / ".crewai_home"
    app_support_dir = runtime_root / "Library" / "Application Support"
    storage_dir = runtime_root / "storage"
    app_support_dir.mkdir(parents=True, exist_ok=True)
    storage_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    return CrewAIRuntimePaths(
        runtime_root=runtime_root,
        app_support_dir=app_support_dir,
        storage_dir=storage_dir,
    )
