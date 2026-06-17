from __future__ import annotations

import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 这一层只负责把 JSONL golden dataset 规范化成统一样本对象。
# 它不做评分，作用是确保后续 factual check / judge 拿到稳定输入。


@dataclass(frozen=True)
class BenchmarkSample:
    """单条 benchmark 样本。

    `expected_facts` 给规则型校验使用，`expected_key_points` 给 LLM judge 使用。
    """

    company_name: str
    ticker: str
    expected_facts: dict[str, Any]
    expected_key_points: dict[str, list[str]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_id(self) -> str:
        company_slug = "".join(
            character.lower() if character.isalnum() else "_"
            for character in self.company_name.strip()
        ).strip("_")
        ticker_slug = "".join(
            character.lower() if character.isalnum() else "_"
            for character in self.ticker.strip()
        ).strip("_")
        return f"{company_slug or 'company'}__{ticker_slug or 'ticker'}"


def _normalize_key_points(raw_value: Any) -> dict[str, list[str]]:
    """把 risks / catalysts / thesis 统一整理成字符串数组。"""

    if raw_value is None:
        raw_value = {}
    if not isinstance(raw_value, dict):
        raise ValueError("expected_key_points 必须是对象。")

    normalized: dict[str, list[str]] = {}
    for key in ("risks", "catalysts", "thesis"):
        value = raw_value.get(key, [])
        if value is None:
            normalized[key] = []
            continue
        if isinstance(value, str):
            normalized[key] = [value.strip()] if value.strip() else []
            continue
        if not isinstance(value, list):
            raise ValueError(f"expected_key_points.{key} 必须是字符串或字符串数组。")
        normalized[key] = [str(item).strip() for item in value if str(item).strip()]
    return normalized


def _normalize_facts(raw_value: Any) -> dict[str, Any]:
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise ValueError("expected_facts 必须是对象。")
    return raw_value


def _sample_from_record(record: dict[str, Any]) -> BenchmarkSample:
    """把一行 JSONL 记录转成 BenchmarkSample，并把其余字段收进 metadata。"""

    company_name = str(record.get("company_name", "")).strip()
    ticker = str(record.get("ticker", "")).strip()
    if not company_name:
        raise ValueError("每条 benchmark 样本都必须包含 company_name。")
    if not ticker:
        raise ValueError("每条 benchmark 样本都必须包含 ticker。")

    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"company_name", "ticker", "expected_facts", "expected_key_points"}
    }
    return BenchmarkSample(
        company_name=company_name,
        ticker=ticker,
        expected_facts=_normalize_facts(record.get("expected_facts")),
        expected_key_points=_normalize_key_points(record.get("expected_key_points")),
        metadata=metadata,
    )


def load_benchmark_dataset(dataset_path: Path) -> list[BenchmarkSample]:
    """按行读取 JSONL 数据集，遇到坏行时带行号报错，方便修数据。"""

    if not dataset_path.exists():
        raise FileNotFoundError(f"Benchmark dataset 不存在：{dataset_path}")

    samples: list[BenchmarkSample] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(f"JSONL 第 {line_number} 行不是合法 JSON：{error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"JSONL 第 {line_number} 行必须是对象。")
        try:
            samples.append(_sample_from_record(record))
        except ValueError as error:
            raise ValueError(f"JSONL 第 {line_number} 行格式错误：{error}") from error

    if not samples:
        raise ValueError("Benchmark dataset 为空，至少需要一条样本。")
    return samples
