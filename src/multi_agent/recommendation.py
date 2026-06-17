from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from crewai import Agent, LLM

from multi_agent.settings import InvestmentResearchSettings

_URL_PATTERN = re.compile(r"https?://[^\s)>\"']+")


def _round_score(value: float) -> float:
    return round(value, 1)


def calculate_trust_score(metrics: dict[str, Any]) -> dict[str, Any]:
    citation_count = min(int(metrics.get("citation_count", 0) or 0), 4)
    financial_rate = float(metrics.get("financial_fields_success_rate", 0.0) or 0.0)
    failure_rate = float(metrics.get("api_calls", {}).get("failure_rate", 0.0) or 0.0)

    breakdown = {
        "report_completeness": 30.0 if metrics.get("report_complete") else 0.0,
        "citations": _round_score(citation_count / 4 * 20) if citation_count else 0.0,
        "artifact_completeness": 20.0 if metrics.get("intermediate_artifacts_complete") else 0.0,
        "financial_coverage": _round_score(financial_rate * 20),
        "api_reliability": 10.0 if failure_rate <= 0.1 else _round_score(max(0.0, (1 - failure_rate) * 10)),
    }
    score = _round_score(sum(breakdown.values()))
    if score >= 80:
        level = "high"
        summary = "证据较充分，可作为高优先级研究输入。"
    elif score >= 60:
        level = "medium"
        summary = "证据基本够用，但仍建议人工复核关键结论。"
    else:
        level = "low"
        summary = "证据不足，当前结果更适合作为线索而非结论。"

    return {
        "score": score,
        "level": level,
        "summary": summary,
        "breakdown": breakdown,
    }


class GeneratedStructuredContent(BaseModel):
    summary: str = Field(default="")
    stance: str = Field(default="watch")
    stance_label: str = Field(default="观察")
    catalysts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    sections: dict[str, str] = Field(
        default_factory=lambda: {
            "business_overview": "",
            "recent_updates": "",
            "financial_analysis": "",
            "investment_recommendation": "",
        }
    )
    citation_urls: list[str] = Field(default_factory=list)


class StructuredRecommendation(BaseModel):
    generated_at: str
    company_name: str
    company_ticker: str
    stance: str
    stance_label: str
    trust_score: float
    trust_level: str
    trust_summary: str
    summary: str
    catalysts: list[str]
    risks: list[str]
    next_actions: list[str]
    source_report_path: str


class StructuredReport(BaseModel):
    generated_at: str
    company_name: str
    company_ticker: str
    summary: str
    stance: str
    stance_label: str
    trust_score: float
    trust_level: str
    trust_summary: str
    catalysts: list[str]
    risks: list[str]
    next_actions: list[str]
    sections: dict[str, str]
    citation_urls: list[str]
    validation: dict[str, Any]
    source_report_path: str


class StructuredOutputs(BaseModel):
    recommendation: StructuredRecommendation
    report: StructuredReport


def _normalize_stance(stance: str) -> tuple[str, str]:
    normalized = stance.strip().lower()
    if normalized in {"buy", "增持", "买入"}:
        return "buy", "增持"
    if normalized in {"hold", "中性", "持有"}:
        return "hold", "中性"
    if normalized in {"sell", "减持", "卖出"}:
        return "sell", "减持"
    return "watch", "观察"


def _default_next_actions() -> list[str]:
    return [
        "复核最新一季财报和关键经营指标。",
        "跟踪重大催化剂是否兑现。",
        "将核心风险加入后续监控列表。",
    ]


def _build_generation_prompt(
    *,
    company_name: str,
    company_ticker: str,
    report_content: str,
    metrics: dict[str, Any],
    task_outputs: dict[str, str] | None,
) -> str:
    return "\n\n".join(
        [
            "请基于以下投研工作流结果，生成严格的结构化 JSON 内容。",
            "要求：",
            "1. stance 只能是 buy/hold/sell/watch 之一。",
            "2. stance_label 必须与 stance 对应，分别为 增持/中性/减持/观察。",
            "3. catalysts、risks、next_actions 使用简洁中文数组。",
            "4. sections 必须包含 business_overview、recent_updates、financial_analysis、investment_recommendation 四个键。",
            "5. citation_urls 只保留真实 URL。",
            "6. 不要输出 markdown，不要解释，只返回符合 schema 的结构化内容。",
            f"company_name: {company_name}",
            f"company_ticker: {company_ticker}",
            f"metrics: {json.dumps(metrics, ensure_ascii=False, indent=2)}",
            f"task_outputs: {json.dumps(task_outputs or {}, ensure_ascii=False, indent=2)}",
            f"report_content:\n{report_content}",
        ]
    )


def _generate_structured_content_with_agent(
    *,
    company_name: str,
    company_ticker: str,
    report_content: str,
    metrics: dict[str, Any],
    task_outputs: dict[str, str] | None,
) -> GeneratedStructuredContent:
    settings = InvestmentResearchSettings.from_env()
    llm = LLM(
        model=settings.model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0,
    )
    generator = Agent(
        role="投研结构化输出分析师",
        goal="把投研工作流结果转换成稳定的结构化 JSON 输出",
        backstory="擅长把长文本研究结论压缩为可程序消费的结构化结果。",
        llm=llm,
        verbose=False,
    )
    result = generator.kickoff(
        _build_generation_prompt(
            company_name=company_name,
            company_ticker=company_ticker,
            report_content=report_content,
            metrics=metrics,
            task_outputs=task_outputs,
        ),
        response_format=GeneratedStructuredContent,
    )
    if result.pydantic is None:
        raise RuntimeError("结构化输出生成失败：未返回 Pydantic 结果。")
    return result.pydantic


def _fallback_structured_content(report_content: str) -> GeneratedStructuredContent:
    citation_urls = _URL_PATTERN.findall(report_content)
    return GeneratedStructuredContent(
        summary="",
        stance="watch",
        stance_label="观察",
        catalysts=[],
        risks=[],
        next_actions=_default_next_actions(),
        sections={
            "business_overview": "",
            "recent_updates": "",
            "financial_analysis": "",
            "investment_recommendation": report_content.strip(),
        },
        citation_urls=citation_urls,
    )


def _generate_structured_outputs(
    *,
    company_name: str,
    company_ticker: str,
    report_path: Path,
    metrics: dict[str, Any],
    task_outputs: dict[str, str] | None = None,
) -> StructuredOutputs:
    report_content = report_path.read_text(encoding="utf-8")
    trust_score = metrics.get("trust_score") or calculate_trust_score(metrics)
    try:
        generated = _generate_structured_content_with_agent(
            company_name=company_name,
            company_ticker=company_ticker,
            report_content=report_content,
            metrics=metrics,
            task_outputs=task_outputs,
        )
    except Exception:
        generated = _fallback_structured_content(report_content)

    stance, stance_label = _normalize_stance(generated.stance or generated.stance_label)
    next_actions = generated.next_actions or _default_next_actions()
    sections = {
        "business_overview": generated.sections.get("business_overview", ""),
        "recent_updates": generated.sections.get("recent_updates", ""),
        "financial_analysis": generated.sections.get("financial_analysis", ""),
        "investment_recommendation": generated.sections.get("investment_recommendation", ""),
    }
    recommendation = StructuredRecommendation(
        generated_at=datetime.now(timezone.utc).isoformat(),
        company_name=company_name,
        company_ticker=company_ticker,
        stance=stance,
        stance_label=stance_label,
        trust_score=trust_score["score"],
        trust_level=trust_score["level"],
        trust_summary=trust_score["summary"],
        summary=generated.summary,
        catalysts=generated.catalysts,
        risks=generated.risks,
        next_actions=next_actions,
        source_report_path=str(report_path.resolve()),
    )
    report = StructuredReport(
        generated_at=recommendation.generated_at,
        company_name=company_name,
        company_ticker=company_ticker,
        summary=generated.summary,
        stance=stance,
        stance_label=stance_label,
        trust_score=trust_score["score"],
        trust_level=trust_score["level"],
        trust_summary=trust_score["summary"],
        catalysts=generated.catalysts,
        risks=generated.risks,
        next_actions=next_actions,
        sections=sections,
        citation_urls=generated.citation_urls,
        validation={
            "has_summary": bool(generated.summary),
            "has_catalysts": bool(generated.catalysts),
            "has_risks": bool(generated.risks),
            "has_investment_recommendation": bool(sections["investment_recommendation"]),
            "citation_count": len(generated.citation_urls),
        },
        source_report_path=str(report_path.resolve()),
    )
    return StructuredOutputs(recommendation=recommendation, report=report)


def build_structured_report(
    *,
    company_name: str,
    company_ticker: str,
    report_path: Path,
    metrics: dict[str, Any],
    task_outputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    structured_outputs = _generate_structured_outputs(
        company_name=company_name,
        company_ticker=company_ticker,
        report_path=report_path,
        metrics=metrics,
        task_outputs=task_outputs,
    )
    return structured_outputs.report.model_dump()


def build_structured_recommendation(
    *,
    company_name: str,
    company_ticker: str,
    report_path: Path,
    metrics: dict[str, Any],
    task_outputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    structured_outputs = _generate_structured_outputs(
        company_name=company_name,
        company_ticker=company_ticker,
        report_path=report_path,
        metrics=metrics,
        task_outputs=task_outputs,
    )
    return structured_outputs.recommendation.model_dump()
