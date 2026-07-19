from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, Field

from multi_agent.core.evidence import ResearchEvidenceBundle
from multi_agent.core.market import MarketValidationResult
from multi_agent.core.review_contracts import GateDecision

ModelTier: TypeAlias = Literal["fast", "deep", "review"]
FinalDecision: TypeAlias = Literal["passed", "blocked", "evidence_limited"]
RoutedAgentName: TypeAlias = Literal[
    "market_validation_analyst",
    "event_guidance_analyst",
    "fundamental_analyst",
    "quant_valuation_analyst",
    "report_writing_analyst",
    "data_quality_reviewer",
    "logic_compliance_reviewer",
]


class EvidenceItem(BaseModel):
    source_id: str
    source_type: str
    title: str
    url: str
    summary: str


class ResearchRunState(BaseModel):
    request_id: str
    company_name: str
    input_ticker: str = ""
    input_exchange: str = ""
    market_validation: MarketValidationResult | None = None
    evidence_ledger: list[EvidenceItem] = Field(default_factory=list)
    analysis_outputs: dict[str, dict[str, object]] = Field(default_factory=dict)
    review_tool_outputs: dict[str, dict[str, object]] = Field(default_factory=dict)
    review_findings: dict[str, dict[str, object]] = Field(default_factory=dict)
    rerun_budget: dict[str, int] = Field(default_factory=dict)
    model_tier_overrides: dict[RoutedAgentName, ModelTier] = Field(default_factory=dict)
    gate_decision: GateDecision | None = None
    final_decision: FinalDecision | None = None
    evidence_bundle: ResearchEvidenceBundle | None = None
