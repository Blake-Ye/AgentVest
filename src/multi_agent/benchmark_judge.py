from __future__ import annotations

import json
import re

from typing import Any

import requests

from multi_agent.benchmark_dataset import BenchmarkSample

# 这一层实现 LLM-as-a-Judge：
# 用固定 JSON schema 约束 Judge 输出，避免自由文本结果难以被程序消费。

_JSON_BLOCK_PATTERN = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", flags=re.IGNORECASE)


def build_judge_messages(*, sample: BenchmarkSample, report_text: str) -> list[dict[str, str]]:
    """构造 Judge prompt，把 expected_key_points 和完整报告一起交给评审模型。"""

    expected_key_points = json.dumps(sample.expected_key_points, ensure_ascii=False, indent=2)
    return [
        {
            "role": "system",
            "content": (
                "你是投研 benchmark 的严格评审员。"
                "请只输出 JSON，不要输出解释性前缀。"
                "所有分数都使用 0 到 1 之间的小数，1 表示最好。"
                "hallucination_score 越高表示幻觉风险越低。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"公司：{sample.company_name} ({sample.ticker})\n"
                "请根据 expected_key_points 评估报告内容质量。\n\n"
                f"expected_key_points:\n{expected_key_points}\n\n"
                f"report_text:\n{report_text}\n\n"
                "返回以下 JSON 结构：\n"
                "{\n"
                '  "risk_recall": 0.0,\n'
                '  "catalyst_recall": 0.0,\n'
                '  "summary_quality_score": 0.0,\n'
                '  "hallucination_score": 0.0,\n'
                '  "overall_quality_score": 0.0,\n'
                '  "matched_risks": [],\n'
                '  "missed_risks": [],\n'
                '  "matched_catalysts": [],\n'
                '  "missed_catalysts": [],\n'
                '  "judge_summary": ""\n'
                "}\n"
            ),
        },
    ]


def parse_judge_response(raw_response: str) -> dict[str, Any]:
    """兼容纯 JSON 和 ```json fenced block`` 两种返回格式。"""

    stripped = raw_response.strip()
    fenced_match = _JSON_BLOCK_PATTERN.search(stripped)
    json_payload = fenced_match.group(1) if fenced_match else stripped
    if not json_payload.startswith("{"):
        start = json_payload.find("{")
        end = json_payload.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("LLM judge 响应中未找到 JSON 对象。")
        json_payload = json_payload[start : end + 1]

    parsed = json.loads(json_payload)
    return {
        "risk_recall": float(parsed["risk_recall"]),
        "catalyst_recall": float(parsed["catalyst_recall"]),
        "summary_quality_score": float(parsed["summary_quality_score"]),
        "hallucination_score": float(parsed["hallucination_score"]),
        "overall_quality_score": float(parsed["overall_quality_score"]),
        "matched_risks": list(parsed.get("matched_risks", [])),
        "missed_risks": list(parsed.get("missed_risks", [])),
        "matched_catalysts": list(parsed.get("matched_catalysts", [])),
        "missed_catalysts": list(parsed.get("missed_catalysts", [])),
        "judge_summary": str(parsed.get("judge_summary", "")).strip(),
    }


class LLMJudgeClient:
    """最小 OpenAI-compatible Judge 客户端，只负责发送请求并解析结构化结果。"""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout_seconds: int = 60,
    ) -> None:
        self.model = model.strip()
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @property
    def is_available(self) -> bool:
        return bool(self.model and self.api_key and self.base_url)

    def judge(self, *, sample: BenchmarkSample, report_text: str) -> dict[str, Any]:
        # 这里故意把 temperature 固定为 0，尽量减少同一报告重复评测时的波动。
        if not self.is_available:
            raise RuntimeError("LLM judge 未配置可用的 model/api_key/base_url。")

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0,
                "messages": build_judge_messages(sample=sample, report_text=report_text),
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        content = str(payload["choices"][0]["message"]["content"])
        return parse_judge_response(content)
