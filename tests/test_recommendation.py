import json
from pathlib import Path

import pytest

from multi_agent.recommendation import (
    StructuredOutputs,
    StructuredRecommendation,
    StructuredReport,
    build_structured_report,
    build_structured_recommendation,
    calculate_trust_score,
)


def test_calculate_trust_score_returns_weighted_breakdown() -> None:
    metrics = {
        "report_complete": True,
        "citation_count": 3,
        "intermediate_artifacts_complete": True,
        "financial_fields_success_rate": 0.6,
        "api_calls": {"failure_rate": 0.1},
    }

    trust_score = calculate_trust_score(metrics)

    assert trust_score["score"] == 87.0
    assert trust_score["level"] == "high"
    assert trust_score["breakdown"] == {
        "report_completeness": 30.0,
        "citations": 15.0,
        "artifact_completeness": 20.0,
        "financial_coverage": 12.0,
        "api_reliability": 10.0,
    }


def test_build_structured_recommendation_uses_json_first_generator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "04_investment_report.md"
    report_path.write_text("# 投资备忘录", encoding="utf-8")
    metrics = {
        "trust_score": {
            "score": 87.0,
            "level": "high",
            "summary": "证据较充分，可作为高优先级研究输入。",
        }
    }

    monkeypatch.setattr(
        "multi_agent.recommendation._generate_structured_outputs",
        lambda **_: StructuredOutputs(
            recommendation=StructuredRecommendation(
                generated_at="2026-06-17T00:00:00+00:00",
                company_name="Alibaba Group Holding Ltd",
                company_ticker="BABA",
                stance="buy",
                stance_label="增持",
                trust_score=87.0,
                trust_level="high",
                trust_summary="证据较充分，可作为高优先级研究输入。",
                summary="公司基本面稳健，短期受云业务恢复和回购计划支撑。",
                catalysts=["云业务利润率改善", "新一轮回购计划"],
                risks=["宏观需求复苏低于预期", "海外监管不确定性"],
                next_actions=["继续跟踪"],
                source_report_path=str(report_path.resolve()),
            ),
            report=StructuredReport(
                generated_at="2026-06-17T00:00:00+00:00",
                company_name="Alibaba Group Holding Ltd",
                company_ticker="BABA",
                summary="公司基本面稳健，短期受云业务恢复和回购计划支撑。",
                stance="buy",
                stance_label="增持",
                trust_score=87.0,
                trust_level="high",
                trust_summary="证据较充分，可作为高优先级研究输入。",
                catalysts=["云业务利润率改善", "新一轮回购计划"],
                risks=["宏观需求复苏低于预期", "海外监管不确定性"],
                next_actions=["继续跟踪"],
                sections={},
                citation_urls=[],
                validation={
                    "has_summary": True,
                    "has_catalysts": True,
                    "has_risks": True,
                    "has_investment_recommendation": True,
                    "citation_count": 0,
                },
                source_report_path=str(report_path.resolve()),
            ),
        ),
    )

    recommendation = build_structured_recommendation(
        company_name="Alibaba Group Holding Ltd",
        company_ticker="BABA",
        report_path=report_path,
        metrics=metrics,
    )

    assert recommendation["company_name"] == "Alibaba Group Holding Ltd"
    assert recommendation["company_ticker"] == "BABA"
    assert recommendation["stance"] == "buy"
    assert recommendation["stance_label"] == "增持"
    assert recommendation["trust_score"] == 87.0
    assert recommendation["summary"] == "公司基本面稳健，短期受云业务恢复和回购计划支撑。"
    assert recommendation["catalysts"] == ["云业务利润率改善", "新一轮回购计划"]
    assert recommendation["risks"] == ["宏观需求复苏低于预期", "海外监管不确定性"]
    assert recommendation["source_report_path"] == str(report_path.resolve())

    serialized = json.dumps(recommendation, ensure_ascii=False)
    assert "证据较充分" in serialized


def test_build_structured_recommendation_handles_decorated_headings_and_tables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    report_path = tmp_path / "04_investment_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# 投资备忘录",
                "",
                "## ✅ 执行摘要  ",
                "",
                "公司进入盈利能力改善与新业务兑现并行阶段。",
                "",
                "## ⚠️ 风险：具象化、有依据、可跟踪",
                "",
                "| 风险类别 | 具体风险 | 依据与现状 |",
                "|----------|-----------|------------|",
                "| 监管风险 | 海外监管收紧 | 审批节奏存在不确定性 |",
                "| 竞争风险 | 云厂商价格战 | 短期利润率承压 |",
                "",
                "## 🚀 催化剂：2026年内可验证、有明确时间表",
                "",
                "| 催化剂 | 触发条件 | 验证方式 |",
                "|---------|------------|------------|",
                "| 新产品放量 | 企业客户续约率提升 | 中报披露 |",
                "| 回购加速 | 董事会扩大授权 | 公告验证 |",
                "",
                "## 💡 投资建议与置信度",
                "",
                "维持增持，继续观察执行兑现情况。",
            ]
        ),
        encoding="utf-8",
    )
    metrics = {
        "trust_score": {
            "score": 72.0,
            "level": "medium",
            "summary": "证据基本够用，但仍建议人工复核关键结论。",
        }
    }

    monkeypatch.setattr(
        "multi_agent.recommendation._generate_structured_outputs",
        lambda **_: StructuredOutputs(
            recommendation=StructuredRecommendation(
                generated_at="2026-06-17T00:00:00+00:00",
                company_name="Tencent Holdings Limited",
                company_ticker="0700.HK",
                stance="buy",
                stance_label="增持",
                trust_score=72.0,
                trust_level="medium",
                trust_summary="证据基本够用，但仍建议人工复核关键结论。",
                summary="公司进入盈利能力改善与新业务兑现并行阶段。",
                catalysts=["新产品放量", "回购加速"],
                risks=["海外监管收紧", "云厂商价格战"],
                next_actions=["继续观察"],
                source_report_path=str(report_path.resolve()),
            ),
            report=StructuredReport(
                generated_at="2026-06-17T00:00:00+00:00",
                company_name="Tencent Holdings Limited",
                company_ticker="0700.HK",
                summary="公司进入盈利能力改善与新业务兑现并行阶段。",
                stance="buy",
                stance_label="增持",
                trust_score=72.0,
                trust_level="medium",
                trust_summary="证据基本够用，但仍建议人工复核关键结论。",
                catalysts=["新产品放量", "回购加速"],
                risks=["海外监管收紧", "云厂商价格战"],
                next_actions=["继续观察"],
                sections={},
                citation_urls=[],
                validation={},
                source_report_path=str(report_path.resolve()),
            ),
        ),
    )

    recommendation = build_structured_recommendation(
        company_name="Tencent Holdings Limited",
        company_ticker="0700.HK",
        report_path=report_path,
        metrics=metrics,
    )

    assert recommendation["summary"] == "公司进入盈利能力改善与新业务兑现并行阶段。"
    assert recommendation["catalysts"] == ["新产品放量", "回购加速"]
    assert recommendation["risks"] == ["海外监管收紧", "云厂商价格战"]
    assert recommendation["stance"] == "buy"


def test_build_structured_report_exports_sections_and_citations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "04_investment_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# 投资备忘录",
                "",
                "## ✅ 执行摘要",
                "",
                "公司进入 AI 商业化兑现阶段。",
                "",
                "## 🧩 业务概览：双引擎战略落地验证",
                "",
                "核心业务结构持续优化，广告与云业务占比提升。",
                "",
                "## 📈 近期动态：AI商业化进入可验证阶段",
                "",
                "- 发布新的企业智能体产品",
                "- 海外拓展加速",
                "",
                "## 📊 财务分析：高质量现金流兑现期",
                "",
                "自由现金流维持高位，利润率稳定改善。",
                "",
                "## ⚠️ 风险：具象化、有依据、可跟踪",
                "",
                "| 风险类别 | 具体风险 | 依据与现状 |",
                "|----------|-----------|------------|",
                "| 监管风险 | 海外监管收紧 | 审批节奏存在不确定性 |",
                "",
                "## 🚀 催化剂：2026年内可验证、有明确时间表",
                "",
                "| 催化剂 | 触发条件 | 验证方式 |",
                "|---------|------------|------------|",
                "| TokenHub放量 | 企业客户续约率提升 | 中报披露 |",
                "",
                "## 💡 投资建议与置信度",
                "",
                "建议增持，继续观察执行兑现。",
                "",
                "参考 https://example.com/report",
                "参考 https://example.com/source2",
            ]
        ),
        encoding="utf-8",
    )
    metrics = {
        "trust_score": {
            "score": 78.0,
            "level": "medium",
            "summary": "证据基本够用，但仍建议人工复核关键结论。",
        }
    }

    monkeypatch.setattr(
        "multi_agent.recommendation._generate_structured_outputs",
        lambda **_: StructuredOutputs(
            recommendation=StructuredRecommendation(
                generated_at="2026-06-17T00:00:00+00:00",
                company_name="Tencent Holdings Limited",
                company_ticker="0700.HK",
                stance="buy",
                stance_label="增持",
                trust_score=78.0,
                trust_level="medium",
                trust_summary="证据基本够用，但仍建议人工复核关键结论。",
                summary="公司进入 AI 商业化兑现阶段。",
                catalysts=["TokenHub放量"],
                risks=["海外监管收紧"],
                next_actions=["继续观察"],
                source_report_path=str(report_path.resolve()),
            ),
            report=StructuredReport(
                generated_at="2026-06-17T00:00:00+00:00",
                company_name="Tencent Holdings Limited",
                company_ticker="0700.HK",
                summary="公司进入 AI 商业化兑现阶段。",
                stance="buy",
                stance_label="增持",
                trust_score=78.0,
                trust_level="medium",
                trust_summary="证据基本够用，但仍建议人工复核关键结论。",
                catalysts=["TokenHub放量"],
                risks=["海外监管收紧"],
                next_actions=["继续观察"],
                sections={
                    "business_overview": "核心业务结构持续优化，广告与云业务占比提升。",
                    "recent_updates": "- 发布新的企业智能体产品\n- 海外拓展加速",
                    "financial_analysis": "自由现金流维持高位，利润率稳定改善。",
                    "investment_recommendation": "建议增持，继续观察执行兑现。",
                },
                citation_urls=[
                    "https://example.com/report",
                    "https://example.com/source2",
                ],
                validation={
                    "has_summary": True,
                    "has_catalysts": True,
                    "has_risks": True,
                    "has_investment_recommendation": True,
                    "citation_count": 2,
                },
                source_report_path=str(report_path.resolve()),
            ),
        ),
    )

    structured_report = build_structured_report(
        company_name="Tencent Holdings Limited",
        company_ticker="0700.HK",
        report_path=report_path,
        metrics=metrics,
    )

    assert structured_report["company_ticker"] == "0700.HK"
    assert structured_report["summary"] == "公司进入 AI 商业化兑现阶段。"
    assert structured_report["sections"]["business_overview"].startswith("核心业务结构持续优化")
    assert structured_report["sections"]["recent_updates"].startswith("- 发布新的企业智能体产品")
    assert structured_report["sections"]["financial_analysis"].startswith("自由现金流维持高位")
    assert structured_report["sections"]["investment_recommendation"].startswith("建议增持")
    assert structured_report["catalysts"] == ["TokenHub放量"]
    assert structured_report["risks"] == ["海外监管收紧"]
    assert structured_report["citation_urls"] == [
        "https://example.com/report",
        "https://example.com/source2",
    ]
    assert structured_report["validation"] == {
        "has_summary": True,
        "has_catalysts": True,
        "has_risks": True,
        "has_investment_recommendation": True,
        "citation_count": 2,
    }
