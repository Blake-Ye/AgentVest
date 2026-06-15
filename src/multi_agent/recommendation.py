from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _normalize_heading_text(value: str) -> str:
    text = re.sub(r"^#+\s*", "", value.strip())
    text = re.sub(r"[*_`>#-]+", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff：:]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "", text)
    return text.lower()


def _extract_section_lines(report_content: str, section_name: str) -> list[str]:
    lines = report_content.splitlines()
    target_heading = _normalize_heading_text(section_name)
    collected: list[str] = []
    capture = False

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("##"):
            current_heading = _normalize_heading_text(stripped_line)
            if capture:
                break
            if current_heading.startswith(target_heading):
                capture = True
            continue
        if capture and stripped_line:
            collected.append(stripped_line)

    return collected


def _extract_section_summary(report_content: str, section_name: str) -> str:
    for line in _extract_section_lines(report_content, section_name):
        if not line.startswith("-") and not line.startswith("*"):
            return line
    return ""


def _extract_section_text(report_content: str, section_name: str) -> str:
    return "\n".join(_extract_section_lines(report_content, section_name)).strip()


def _extract_section_bullets(report_content: str, section_name: str) -> list[str]:
    bullets: list[str] = []
    section_lines = _extract_section_lines(report_content, section_name)
    for line in section_lines:
        if line.startswith("-") or line.startswith("*"):
            bullets.append(line[1:].strip())
    if bullets:
        return bullets

    return _extract_table_items(section_lines, section_name=section_name)


def _split_markdown_row(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def _is_markdown_separator(line: str) -> bool:
    return bool(re.fullmatch(r"[\|\-\:\s]+", line.strip()))


def _extract_table_items(section_lines: list[str], *, section_name: str) -> list[str]:
    table_lines = [line for line in section_lines if line.startswith("|")]
    if len(table_lines) < 2:
        return []

    headers = _split_markdown_row(table_lines[0])
    target_headers = {
        "催化剂": ("催化剂",),
        "风险": ("具体风险", "风险"),
    }.get(section_name, (section_name,))
    header_index = 0
    best_score = -1
    for index, header in enumerate(headers):
        normalized_header = _normalize_heading_text(header)
        for candidate in target_headers:
            normalized_candidate = _normalize_heading_text(candidate)
            if normalized_header == normalized_candidate:
                score = 3
            elif normalized_header.startswith(normalized_candidate):
                score = 2
            elif normalized_candidate in normalized_header:
                score = 1
            else:
                score = 0
            if score > best_score:
                best_score = score
                header_index = index

    extracted: list[str] = []
    for line in table_lines[1:]:
        if _is_markdown_separator(line):
            continue
        columns = _split_markdown_row(line)
        if header_index < len(columns) and columns[header_index]:
            extracted.append(columns[header_index])
    return extracted


def _infer_stance(report_content: str) -> tuple[str, str]:
    section_text = "\n".join(_extract_section_lines(report_content, "投资建议")) or report_content
    sell_keywords = ("建议卖出", "卖出", "减持", "underperform", "negative")
    buy_keywords = ("建议买入", "建议增持", "买入", "增持", "outperform", "positive")
    hold_keywords = ("建议持有", "持有", "中性", "观望", "hold", "neutral")
    if any(keyword in section_text for keyword in sell_keywords):
        return "sell", "减持"
    if any(keyword in section_text for keyword in buy_keywords):
        return "buy", "增持"
    if any(keyword in section_text for keyword in hold_keywords):
        return "hold", "中性"
    return "watch", "观察"


def build_structured_report(
    *,
    company_name: str,
    company_ticker: str,
    report_path: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    report_content = report_path.read_text(encoding="utf-8")
    recommendation = build_structured_recommendation(
        company_name=company_name,
        company_ticker=company_ticker,
        report_path=report_path,
        metrics=metrics,
    )
    sections = {
        "business_overview": _extract_section_text(report_content, "业务概览"),
        "recent_updates": _extract_section_text(report_content, "近期动态"),
        "financial_analysis": _extract_section_text(report_content, "财务分析"),
        "investment_recommendation": _extract_section_text(report_content, "投资建议"),
    }
    citation_urls = _URL_PATTERN.findall(report_content)

    return {
        "generated_at": recommendation["generated_at"],
        "company_name": company_name,
        "company_ticker": company_ticker,
        "summary": recommendation["summary"],
        "stance": recommendation["stance"],
        "stance_label": recommendation["stance_label"],
        "trust_score": recommendation["trust_score"],
        "trust_level": recommendation["trust_level"],
        "trust_summary": recommendation["trust_summary"],
        "catalysts": recommendation["catalysts"],
        "risks": recommendation["risks"],
        "next_actions": recommendation["next_actions"],
        "sections": sections,
        "citation_urls": citation_urls,
        "validation": {
            "has_summary": bool(recommendation["summary"]),
            "has_catalysts": bool(recommendation["catalysts"]),
            "has_risks": bool(recommendation["risks"]),
            "has_investment_recommendation": bool(sections["investment_recommendation"]),
            "citation_count": len(citation_urls),
        },
        "source_report_path": recommendation["source_report_path"],
    }


def build_structured_recommendation(
    *,
    company_name: str,
    company_ticker: str,
    report_path: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    report_content = report_path.read_text(encoding="utf-8")
    trust_score = metrics.get("trust_score") or calculate_trust_score(metrics)
    stance, stance_label = _infer_stance(report_content)
    summary = _extract_section_summary(report_content, "执行摘要")
    catalysts = _extract_section_bullets(report_content, "催化剂")
    risks = _extract_section_bullets(report_content, "风险")

    next_actions = [
        "复核最新一季财报和关键经营指标。",
        "跟踪重大催化剂是否兑现。",
        "将核心风险加入后续监控列表。",
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "company_name": company_name,
        "company_ticker": company_ticker,
        "stance": stance,
        "stance_label": stance_label,
        "trust_score": trust_score["score"],
        "trust_level": trust_score["level"],
        "trust_summary": trust_score["summary"],
        "summary": summary,
        "catalysts": catalysts,
        "risks": risks,
        "next_actions": next_actions,
        "source_report_path": str(report_path.resolve()),
    }
