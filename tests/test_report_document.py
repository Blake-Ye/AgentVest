from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from multi_agent.core.evidence import FinancialFact, ResearchEvidenceBundle
from multi_agent.core.report_document import (
    REQUIRED_SECTION_KEYS,
    ReportDocument,
    ReportGenerationContext,
    SourceReference,
    render_markdown,
    render_recommendation,
    render_structured_report,
)
from multi_agent.core.review_contracts import (
    CoverageSummary,
    DeliveryEligibility,
    FailureTaxonomy,
    ReviewContract,
)
from multi_agent.recommendation import (
    render_recommendation as exported_render_recommendation,
)


def _formal_context() -> ReportGenerationContext:
    fact = FinancialFact(
        field_name="revenue",
        value=416_161_000_000,
        unit="USD",
        period_start=date(2024, 9, 29),
        period_end=date(2025, 9, 27),
        fiscal_year=2025,
        fiscal_period="FY",
        form="10-K",
        accession="0000320193-25-000079",
        filed_at=date(2025, 10, 31),
        source_url="https://www.sec.gov/Archives/edgar/data/320193/aapl-20250927.htm",
        source_tag="sec_companyfacts",
    )
    review = ReviewContract(
        stage="report",
        delivery_eligibility=DeliveryEligibility(formal_report_allowed=True),
        failure_taxonomy=FailureTaxonomy(primary_class="none"),
        coverage_summary=CoverageSummary(
            evidence_coverage_ratio=1.0,
            financial_coverage_score=1.0,
            claim_binding_ratio=1.0,
        ),
    )
    return ReportGenerationContext(
        company_name="Apple Inc.",
        ticker="AAPL",
        report_mode="formal_report",
        evidence_bundle=ResearchEvidenceBundle(
            company_name="Apple Inc.", ticker="AAPL", financial_facts=[fact]
        ),
        analysis_review_contract=review,
        allowed_claim_ids=["apple-revenue", "apple-services"],
        canonical_sources_json=(
            SourceReference(
                source_id="sec-10k",
                title="Apple 2025 Form 10-K",
                url="https://www.sec.gov/Archives/edgar/data/320193/aapl-20250927.htm",
                source_tag="sec_filing",
                field_name="revenue",
            ).model_dump_json(),
        ),
    )


def _formal_payload() -> dict[str, object]:
    sections = {
        "executive_summary": {
            "heading": "执行摘要",
            "content": "Apple 的现金流与服务业务提供了清晰的基本面支撑。",
            "claim_ids": ["apple-revenue"],
        },
        "business_overview": {
            "heading": "公司与业务概览",
            "content": "硬件生态与服务业务共同构成收入基础。",
            "claim_ids": [],
        },
        "recent_events": {
            "heading": "近期事件与催化剂",
            "content": "服务业务扩张是未来观察催化剂。",
            "claim_ids": ["apple-services"],
        },
        "financial_analysis": {
            "heading": "财务分析与估值",
            "content": "FY2025 收入为 416.2 十亿美元，来自已审计年报。",
            "claim_ids": ["apple-revenue"],
        },
        "key_risks": {
            "heading": "关键风险",
            "content": "宏观需求和监管变化可能影响业绩兑现。",
            "claim_ids": [],
        },
        "investment_conclusion": {
            "heading": "投资结论",
            "content": "基于当前正式证据，维持买入观点。",
            "claim_ids": ["apple-revenue"],
        },
        "source_index": {
            "heading": "来源索引",
            "content": "以下来源支持本报告的关键结论。",
            "claim_ids": [],
        },
    }
    return {
        "title": "Apple Inc. 投资研究报告",
        "stance": "buy",
        "executive_summary": "Apple 的正式证据支持持续跟踪其服务业务与现金流。",
        "catalysts": ["服务业务增长", "资本回报延续"],
        "risks": ["监管压力", "终端需求波动"],
        "sections": sections,
        "claims": [
            {
                "claim_id": "apple-revenue",
                "text": "FY2025 收入为 416.2 十亿美元。",
                "critical": True,
                "source_ids": ["sec-10k"],
            },
            {
                "claim_id": "apple-services",
                "text": "服务业务是重要增长催化剂。",
                "critical": True,
                "source_ids": ["sec-10k"],
            },
        ],
        "sources": [
            {
                "source_id": "sec-10k",
                "title": "Apple 2025 Form 10-K",
                "url": "https://www.sec.gov/Archives/edgar/data/320193/aapl-20250927.htm",
                "source_tag": "sec_filing",
                "field_name": "revenue",
            }
        ],
    }


@pytest.fixture
def formal_apple_document() -> ReportDocument:
    return ReportDocument.from_writer_payload(
        context=_formal_context(), writer_payload=_formal_payload(), trust_score=91
    )


def test_all_outputs_share_the_same_canonical_document(
    formal_apple_document: ReportDocument,
) -> None:
    markdown = render_markdown(formal_apple_document)
    recommendation = render_recommendation(formal_apple_document)
    report = render_structured_report(formal_apple_document)

    assert list(formal_apple_document.sections) == list(REQUIRED_SECTION_KEYS)
    assert recommendation["summary"] == formal_apple_document.executive_summary
    assert recommendation["catalysts"] == formal_apple_document.catalysts
    assert recommendation["risks"] == formal_apple_document.risks
    assert report["sections"]["financial_analysis"] == (
        formal_apple_document.sections["financial_analysis"].content
    )
    assert "## 财务分析与估值" in markdown


def test_formal_document_rejects_unbound_or_unknown_critical_claim(
    formal_apple_document: ReportDocument,
) -> None:
    payload = formal_apple_document.model_dump()
    payload["claims"][0]["source_ids"] = []
    with pytest.raises(ValidationError, match="critical claim"):
        ReportDocument.model_validate(payload)
    payload = formal_apple_document.model_dump()
    payload["claims"][0]["claim_id"] = "unknown-claim"
    with pytest.raises(ValidationError, match="allowed_claim_ids"):
        ReportDocument.model_validate(payload)


@pytest.mark.parametrize("field", ["url", "title"])
def test_writer_cannot_forge_or_invent_canonical_sources(field: str) -> None:
    payload = _formal_payload()
    payload["sources"][0][field] = "https://evil.example/forged"  # type: ignore[index]
    with pytest.raises(ValueError, match="forged canonical source"):
        ReportDocument.from_writer_payload(
            context=_formal_context(), writer_payload=payload, trust_score=91
        )


def test_canonical_event_source_is_accepted_and_forgery_is_rejected() -> None:
    context = _formal_context().model_copy(update={
        "allowed_claim_ids": ("event:launch",),
        "canonical_sources_json": (SourceReference(
            source_id="event:launch", title="Launch event", url="https://example.com/launch",
            source_tag="news",
        ).model_dump_json(),),
    })
    payload = _formal_payload()
    payload["claims"] = [{"claim_id": "event:launch", "text": "Launch is confirmed.", "critical": True, "source_ids": ["event:launch"]}]
    for key, section in payload["sections"].items():
        section["claim_ids"] = ["event:launch"] if key != "source_index" else []
    payload["sources"] = [{"source_id": "event:launch", "title": "Launch event", "url": "https://example.com/launch", "source_tag": "news"}]
    assert ReportDocument.from_writer_payload(context=context, writer_payload=payload, trust_score=91).sources[0].source_id == "event:launch"
    payload["sources"][0]["url"] = "https://evil.example/launch"
    with pytest.raises(ValueError, match="forged canonical source"):
        ReportDocument.from_writer_payload(context=context, writer_payload=payload, trust_score=91)



    payload = _formal_payload()
    payload["sources"][0]["source_id"] = "unknown"  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown canonical source"):
        ReportDocument.from_writer_payload(
            context=_formal_context(), writer_payload=payload, trust_score=91
        )



@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("claims", [], "source-bound critical claims"),
        ("sources", [], "source-bound critical claims"),
    ],
)
def test_formal_document_requires_source_bound_critical_claims(
    formal_apple_document: ReportDocument,
    field: str,
    value: list[object],
    message: str,
) -> None:
    payload = formal_apple_document.model_dump()
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        ReportDocument.model_validate(payload)


@pytest.mark.parametrize(
    "section_key",
    ("executive_summary", "financial_analysis", "investment_conclusion"),
)
def test_formal_document_requires_critical_claims_in_each_core_section(
    formal_apple_document: ReportDocument, section_key: str
) -> None:
    payload = formal_apple_document.model_dump()
    payload["sections"][section_key]["claim_ids"] = []

    with pytest.raises(ValidationError, match=f"core section {section_key}"):
        ReportDocument.model_validate(payload)


def test_claim_source_ids_are_deduplicated_before_rendering(
    formal_apple_document: ReportDocument,
) -> None:
    payload = formal_apple_document.model_dump()
    payload["claims"][0]["source_ids"] = ["sec-10k", "sec-10k"]

    document = ReportDocument.model_validate(payload)

    assert document.claims[0].source_ids == ["sec-10k"]
    assert render_markdown(document).count("[sec-10k](") == render_markdown(
        formal_apple_document
    ).count("[sec-10k](")


def test_formal_document_rejects_unbound_noncritical_claim(
    formal_apple_document: ReportDocument,
) -> None:
    payload = formal_apple_document.model_dump()
    payload["claims"].append({
        "claim_id": "claim:unsupported",
        "text": "未经来源支持的普通结论。",
        "critical": False,
        "source_ids": [],
    })
    payload["sections"]["business_overview"]["claim_ids"].append("claim:unsupported")

    with pytest.raises(ValidationError, match="must bind at least one source"):
        ReportDocument.model_validate(payload)


@pytest.mark.parametrize(
    ("mode", "stance", "field", "value"),
    [
        ("evidence_limited_report", "watch", "executive_summary", "建议买入该股票。证据受限。"),
        ("evidence_limited_report", "watch", "investment_conclusion", "Strong buy。"),
        ("blocked_notice", "blocked", "executive_summary", "本报告已阻断，建议卖出。"),
        ("blocked_notice", "blocked", "financial_analysis", "维持增持评级。"),
    ],
)
def test_limited_or_blocked_document_rejects_actionable_recommendation_language(
    formal_apple_document: ReportDocument,
    mode: str,
    stance: str,
    field: str,
    value: str,
) -> None:
    payload = formal_apple_document.model_dump()
    payload["report_mode"] = mode
    payload["stance"] = stance
    payload["executive_summary"] = "证据受限，继续观察。" if mode == "evidence_limited_report" else "报告已阻断。"
    if field in payload["sections"]:
        payload["sections"][field]["content"] = value
    else:
        payload[field] = value

    with pytest.raises(ValidationError, match="actionable recommendation"):
        ReportDocument.model_validate(payload)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("claims",), "duplicate", "claim_id"),
        (("sources",), "duplicate", "source_id"),
        (("sections", "executive_summary"), "missing", "exactly"),
        (("sections", "unexpected"), "extra", "exactly"),
        (("sections", "financial_analysis", "content"), "empty", "non-empty"),
    ],
)
def test_document_rejects_duplicate_or_invalid_sections(
    formal_apple_document: ReportDocument,
    path: tuple[str, ...],
    value: str,
    message: str,
) -> None:
    payload = formal_apple_document.model_dump()
    if path == ("claims",):
        payload["claims"].append(dict(payload["claims"][0]))
    elif path == ("sources",):
        payload["sources"].append(dict(payload["sources"][0]))
    elif value == "missing":
        payload["sections"].pop(path[1])
    elif value == "extra":
        payload["sections"][path[1]] = {
            "key": path[1],
            "heading": "额外章节",
            "content": "不应出现。",
            "claim_ids": [],
        }
    else:
        payload["sections"][path[1]][path[2]] = ""

    with pytest.raises(ValidationError, match=message):
        ReportDocument.model_validate(payload)


@pytest.mark.parametrize(
    ("mode", "stance", "summary", "message"),
    [
        ("formal_report", "watch", "正式报告", "formal_report"),
        ("evidence_limited_report", "buy", "证据受限，不能作为正式投资建议。", "evidence_limited_report"),
        ("evidence_limited_report", "watch", "仍需继续观察。", "证据受限"),
        ("blocked_notice", "watch", "报告已阻断，不能形成投资结论。", "blocked_notice"),
        ("blocked_notice", "blocked", "无法交付。", "阻断"),
    ],
)
def test_report_mode_requires_matching_stance_and_disclosure(
    formal_apple_document: ReportDocument,
    mode: str,
    stance: str,
    summary: str,
    message: str,
) -> None:
    payload = formal_apple_document.model_dump()
    payload["report_mode"] = mode
    payload["stance"] = stance
    payload["executive_summary"] = summary
    with pytest.raises(ValidationError, match=message):
        ReportDocument.model_validate(payload)


def test_rendering_is_deterministic_and_exposes_source_anchors(
    formal_apple_document: ReportDocument,
) -> None:
    first = render_markdown(formal_apple_document)
    second = render_markdown(formal_apple_document)

    assert first == second
    positions = [first.index(f"## {formal_apple_document.sections[key].heading}") for key in REQUIRED_SECTION_KEYS]
    assert positions == sorted(positions)
    assert "[sec-10k](https://www.sec.gov/Archives/edgar/data/320193/aapl-20250927.htm)" in first
    assert "[apple-revenue]" in first


def test_new_projections_do_not_parse_decorated_markdown_headings(
    formal_apple_document: ReportDocument,
) -> None:
    document = formal_apple_document.model_copy(
        update={
            "sections": {
                **formal_apple_document.sections,
                "financial_analysis": formal_apple_document.sections[
                    "financial_analysis"
                ].model_copy(update={"heading": "*** 任意修饰标题 ***"}),
            }
        }
    )

    report = render_structured_report(document)
    recommendation = render_recommendation(document)

    assert report["sections"]["financial_analysis"].startswith("FY2025 收入")
    assert recommendation["summary"] == document.executive_summary
    assert exported_render_recommendation(document) == recommendation
