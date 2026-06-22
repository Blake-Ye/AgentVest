from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ReviewToolSummary(BaseModel):
    evidence_coverage_ratio: float
    financial_coverage_score: float
    critical_conflict_count: int = 0
    market_policy_violations: list[str] = Field(default_factory=list)
    unsupported_critical_claims: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)


GateControlDecision = Literal["passed", "rerun", "blocked"]


class GateDecision(BaseModel):
    passed: bool
    final_decision: GateControlDecision
    trust_score: int
    blocking_reasons: list[str] = Field(default_factory=list)
