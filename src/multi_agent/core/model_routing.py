from __future__ import annotations

from typing import cast

from multi_agent.core.state import ModelTier, RoutedAgentName
from multi_agent.settings import InvestmentResearchSettings

MODEL_TIERS: tuple[ModelTier, ...] = ("fast", "deep", "review")
DEFAULT_AGENT_TIERS: dict[RoutedAgentName, ModelTier] = {
    "market_validation_analyst": "fast",
    "event_guidance_analyst": "fast",
    "fundamental_analyst": "deep",
    "quant_valuation_analyst": "deep",
    "report_writing_analyst": "deep",
    "data_quality_reviewer": "review",
    "logic_compliance_reviewer": "review",
}


class ModelRouter:
    def __init__(self, settings: InvestmentResearchSettings) -> None:
        self.settings = settings

    def _validate_tier(self, tier: str, *, error_prefix: str = "Unknown model tier") -> ModelTier:
        if tier not in MODEL_TIERS:
            raise ValueError(f"{error_prefix}: {tier}")
        return cast(ModelTier, tier)

    def _validate_agent_name(
        self,
        agent_name: str,
        *,
        error_prefix: str = "Unknown agent for model routing",
    ) -> RoutedAgentName:
        if agent_name not in DEFAULT_AGENT_TIERS:
            raise ValueError(f"{error_prefix}: {agent_name}")
        return cast(RoutedAgentName, agent_name)

    def _validate_overrides(
        self,
        overrides: dict[str, str] | None,
    ) -> dict[RoutedAgentName, ModelTier]:
        validated: dict[RoutedAgentName, ModelTier] = {}
        for agent_name, tier in (overrides or {}).items():
            validated_agent = self._validate_agent_name(
                agent_name,
                error_prefix="Unknown agent in model tier overrides",
            )
            validated[validated_agent] = self._validate_tier(
                tier,
                error_prefix=f"Unknown model tier override for agent {validated_agent}",
            )
        return validated

    def for_tier(self, tier: str) -> str:
        validated_tier = self._validate_tier(tier)
        mapping = {
            "fast": self.settings.fast_model,
            "deep": self.settings.deep_model,
            "review": self.settings.review_model,
        }
        return mapping[validated_tier]

    def default_tier_for_agent(self, agent_name: str) -> ModelTier:
        validated_agent = self._validate_agent_name(agent_name)
        return DEFAULT_AGENT_TIERS[validated_agent]

    def for_agent(self, agent_name: str, overrides: dict[str, str] | None = None) -> str:
        validated_agent = self._validate_agent_name(agent_name)
        validated_overrides = self._validate_overrides(overrides)
        tier = validated_overrides.get(validated_agent) or DEFAULT_AGENT_TIERS[validated_agent]
        return self.for_tier(tier)
