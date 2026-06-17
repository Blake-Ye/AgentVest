from __future__ import annotations

import re

from typing import Any

from multi_agent.benchmark_dataset import BenchmarkSample

# 这一层实现规则型 benchmark：
# 先用别名定位报告里的候选行，再抽取数字/文本，与 expected_facts 做程序化比对。

_ROUND_DIGITS = 3
_FORM_TYPE_PATTERN = re.compile(r"\b(10-K|10-Q|20-F|6-K|8-K|S-1)\b", flags=re.IGNORECASE)
_NUMBER_PATTERN = re.compile(
    r"(?P<number>-?\d[\d,]*(?:\.\d+)?)\s*(?P<unit>trillion|billion|million|thousand|bn|mn|mm|m|b|k)?",
    flags=re.IGNORECASE,
)

_DEFAULT_ALIASES = {
    "revenue": ["revenue", "revenues", "营收", "收入"],
    "net_income": ["net income", "净利润", "归母净利润"],
    "form_type": ["form type", "filing type", "表单类型"],
    "filing_year": ["filing year", "fiscal year", "报告年份", "财年"],
}

_UNIT_MULTIPLIERS = {
    "trillion": 1_000_000_000_000,
    "billion": 1_000_000_000,
    "million": 1_000_000,
    "thousand": 1_000,
    "bn": 1_000_000_000,
    "b": 1_000_000_000,
    "mn": 1_000_000,
    "mm": 1_000_000,
    "m": 1_000_000,
    "k": 1_000,
}


def _round_metric(value: float) -> float:
    return round(value, _ROUND_DIGITS)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_text(text: str) -> str:
    return _normalize_whitespace(text).casefold()


def _default_aliases(field_name: str) -> list[str]:
    return _DEFAULT_ALIASES.get(field_name, [field_name.replace("_", " ")])


def _coerce_fact_definition(field_name: str, raw_definition: Any) -> dict[str, Any]:
    """兼容简写和完整写法，把 fact 定义统一成带 type/tolerance/aliases 的结构。"""

    if isinstance(raw_definition, dict):
        expected_value = raw_definition.get("value")
        if expected_value is None:
            raise ValueError(f"expected_facts.{field_name} 缺少 value。")
        fact_type = str(raw_definition.get("type", "")).strip() or (
            "number" if isinstance(expected_value, (int, float)) else "text"
        )
        tolerance = float(raw_definition.get("tolerance", 0.05))
        aliases = raw_definition.get("aliases") or _default_aliases(field_name)
        return {
            "type": fact_type,
            "expected_value": expected_value,
            "tolerance": tolerance,
            "aliases": [str(item) for item in aliases],
        }
    return {
        "type": "number" if isinstance(raw_definition, (int, float)) else "text",
        "expected_value": raw_definition,
        "tolerance": 0.05,
        "aliases": _default_aliases(field_name),
    }


def _lines_with_aliases(report_text: str, aliases: list[str]) -> list[str]:
    lowered_aliases = [alias.casefold() for alias in aliases]
    return [
        line.strip()
        for line in report_text.splitlines()
        if line.strip() and any(alias in line.casefold() for alias in lowered_aliases)
    ]


def _parse_numeric_token(token_match: re.Match[str]) -> int:
    number = float(token_match.group("number").replace(",", ""))
    unit = (token_match.group("unit") or "").lower()
    multiplier = _UNIT_MULTIPLIERS.get(unit, 1)
    return int(round(number * multiplier))


def _extract_numeric_value(report_text: str, aliases: list[str]) -> int | None:
    """找到第一条候选行后立即抽取数字，保持实现简单且可预测。"""

    for line in _lines_with_aliases(report_text, aliases):
        for match in _NUMBER_PATTERN.finditer(line):
            return _parse_numeric_token(match)
    return None


def _extract_form_type(report_text: str) -> str | None:
    match = _FORM_TYPE_PATTERN.search(report_text)
    if not match:
        return None
    return match.group(1).upper()


def _extract_year(report_text: str, aliases: list[str]) -> int | None:
    for line in _lines_with_aliases(report_text, aliases):
        year_match = re.search(r"\b(20\d{2})\b", line)
        if year_match:
            return int(year_match.group(1))
    return None


def _extract_text_value(report_text: str, field_name: str, aliases: list[str]) -> str | int | None:
    """文本字段走轻量规则提取；form_type 和 filing_year 单独走专用 parser。"""

    if field_name == "form_type":
        return _extract_form_type(report_text)
    if field_name == "filing_year":
        return _extract_year(report_text, aliases)

    lines = _lines_with_aliases(report_text, aliases)
    if not lines:
        return None
    line = lines[0]
    for alias in aliases:
        pattern = re.compile(re.escape(alias), flags=re.IGNORECASE)
        line = pattern.sub("", line)
    cleaned = line.strip(" :：-")
    return _normalize_whitespace(cleaned) or None


def _compare_number(expected_value: float, actual_value: int | None, tolerance: float) -> tuple[bool, str]:
    if actual_value is None:
        return False, "报告中未提取到对应数值。"
    baseline = abs(float(expected_value))
    if baseline == 0:
        return actual_value == 0, "期望值为 0，按精确匹配处理。"
    relative_error = abs(actual_value - float(expected_value)) / baseline
    if relative_error <= tolerance:
        return True, f"相对误差 {relative_error:.3f}，在容差 {tolerance:.3f} 内。"
    return False, f"相对误差 {relative_error:.3f}，超过容差 {tolerance:.3f}。"


class FactualChecker:
    """对结构化事实做程序化校验，输出字段级明细和总体 factual_accuracy。"""

    def evaluate(self, *, sample: BenchmarkSample, report_text: str) -> dict[str, Any]:
        # 每个 fact 都会返回 matched / expected / actual / reason，便于后续排查错因。
        checks: dict[str, dict[str, Any]] = {}
        matched_count = 0

        for field_name, raw_definition in sample.expected_facts.items():
            fact_definition = _coerce_fact_definition(field_name, raw_definition)
            aliases = fact_definition["aliases"]
            expected_value = fact_definition["expected_value"]

            if fact_definition["type"] == "number":
                actual_value = _extract_numeric_value(report_text, aliases)
                matched, reason = _compare_number(
                    expected_value=float(expected_value),
                    actual_value=actual_value,
                    tolerance=float(fact_definition["tolerance"]),
                )
            else:
                actual_value = _extract_text_value(report_text, field_name, aliases)
                matched = actual_value is not None and _normalize_text(str(actual_value)) == _normalize_text(
                    str(expected_value)
                )
                reason = "文本完全匹配。" if matched else "文本不匹配或报告中不存在。"

            matched_count += 1 if matched else 0
            checks[field_name] = {
                "matched": matched,
                "expected_value": expected_value,
                "actual_value": actual_value,
                "aliases": aliases,
                "reason": reason,
            }

        fields_checked = len(checks)
        factual_accuracy = _round_metric(matched_count / fields_checked) if fields_checked else 0.0
        return {
            "factual_accuracy": factual_accuracy,
            "fields_checked": fields_checked,
            "fields_matched": matched_count,
            "checks": checks,
        }
