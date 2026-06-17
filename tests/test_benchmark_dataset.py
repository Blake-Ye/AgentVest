import json
from pathlib import Path

from multi_agent.benchmark_dataset import load_benchmark_dataset


def test_load_benchmark_dataset_normalizes_jsonl_records(tmp_path: Path) -> None:
    dataset_path = tmp_path / "benchmark.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "company_name": "Apple Inc.",
                        "ticker": "AAPL",
                        "expected_facts": {
                            "revenue": {"value": 391_000_000_000, "tolerance": 0.02},
                            "net_income": {"value": 93_700_000_000, "tolerance": 0.02},
                            "form_type": "10-K",
                            "filing_year": 2024,
                        },
                        "expected_key_points": {
                            "risks": ["iPhone 增速放缓"],
                            "catalysts": ["服务业务扩张"],
                            "thesis": ["自由现金流稳健"],
                        },
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "company_name": "Alibaba Group Holding Ltd",
                        "ticker": "BABA",
                        "expected_facts": {"form_type": "20-F"},
                        "expected_key_points": {"risks": ["竞争加剧"]},
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )

    samples = load_benchmark_dataset(dataset_path)

    assert len(samples) == 2
    assert samples[0].company_name == "Apple Inc."
    assert samples[0].expected_facts["revenue"]["value"] == 391_000_000_000
    assert samples[0].expected_key_points["catalysts"] == ["服务业务扩张"]
    assert samples[1].ticker == "BABA"
    assert samples[1].expected_key_points["catalysts"] == []
    assert samples[1].expected_key_points["thesis"] == []
