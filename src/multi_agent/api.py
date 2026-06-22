from __future__ import annotations

from typing import Any

from multi_agent.main import run_trigger_payload


def execute_trigger_payload(trigger_payload: dict[str, Any]) -> Any:
    """供外部 API 或脚本复用的最小入口。"""
    return run_trigger_payload(trigger_payload)
