from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


MarketLabel = Literal["US", "EU", "HK", "UNRESOLVED"]
ResolutionStatus = Literal["confirmed", "tentative", "unresolved"]


@dataclass(frozen=True)
class ToolPolicy:
    sec_allowed: bool
    tavily_allowed: bool
    transcripts_allowed: bool
    valuation_allowed: bool
    ownership_allowed: bool
    blocked_conclusions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MarketValidationResult:
    market_label: MarketLabel
    confidence: float
    resolution_status: ResolutionStatus
    evidence: list[str]
    requires_human_confirmation: bool
    tool_policy: ToolPolicy

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["tool_policy"] = self.tool_policy.as_dict()
        return payload


def build_tool_policy(market_label: MarketLabel) -> ToolPolicy:
    if market_label == "US":
        return ToolPolicy(
            sec_allowed=True,
            tavily_allowed=True,
            transcripts_allowed=True,
            valuation_allowed=True,
            ownership_allowed=True,
            blocked_conclusions=[],
        )
    if market_label in {"EU", "HK"}:
        return ToolPolicy(
            sec_allowed=False,
            tavily_allowed=True,
            transcripts_allowed=False,
            valuation_allowed=True,
            ownership_allowed=False,
            blocked_conclusions=["SEC based fundamental conclusions"],
        )
    return ToolPolicy(
        sec_allowed=False,
        tavily_allowed=True,
        transcripts_allowed=False,
        valuation_allowed=False,
        ownership_allowed=False,
        blocked_conclusions=[
            "investment recommendation",
            "valuation conclusion",
        ],
    )
