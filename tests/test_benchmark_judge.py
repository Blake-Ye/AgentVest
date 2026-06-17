from multi_agent.benchmark_judge import parse_judge_response


def test_parse_judge_response_extracts_json_from_markdown_block() -> None:
    raw_response = """
    下面是评分结果：

    ```json
    {
      "risk_recall": 0.8,
      "catalyst_recall": 0.6,
      "summary_quality_score": 0.9,
      "hallucination_score": 0.7,
      "overall_quality_score": 0.75,
      "matched_risks": ["竞争加剧"],
      "missed_risks": [],
      "matched_catalysts": ["回购"],
      "missed_catalysts": ["云业务改善"],
      "judge_summary": "风险覆盖较完整。"
    }
    ```
    """

    parsed = parse_judge_response(raw_response)

    assert parsed["risk_recall"] == 0.8
    assert parsed["catalyst_recall"] == 0.6
    assert parsed["hallucination_score"] == 0.7
    assert parsed["matched_catalysts"] == ["回购"]
