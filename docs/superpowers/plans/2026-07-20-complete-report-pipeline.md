# AgentVest Complete Report Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make AgentVest produce a complete, auditable, internally consistent formal investment report, proven by a real Apple Inc. / AAPL run.

**Architecture:** Preserve CrewAI as the agent execution engine and `MarketReviewFlow` as the control plane. Introduce a typed evidence bundle before analysis, a typed report document after the gate, and a delivery validator that renders Markdown and both JSON outputs from one object under one final-decision truth.

**Tech Stack:** Python 3.10-3.13, Pydantic v2, CrewAI 1.14.6, requests, pytest 9.

## Global Constraints

- Financial claims used for formal delivery must carry value, unit, period, filing identity, filed date, source URL, and source tag.
- `final_decision.json` is the single terminal-state truth; report mode and every projection must match it.
- New runs may not use Markdown heading inference to decide gate state or build structured output.
- Missing or ambiguous evidence must cause targeted rerun, evidence-limited delivery, or blocking; never guessed data.
- The formal gate uses exact thresholds `evidence_coverage_ratio >= 0.80`, `financial_coverage_score >= 0.80`, `claim_binding_ratio >= 0.75`, and complete formal-required fields.
- Tests must not make accidental network or LLM calls.
- No CrewAI rewrite, database, non-US source expansion, or frontend work.
- Completion requires a real AAPL run with `passed + formal_report`, seven non-empty report sections, non-empty structured summary/catalysts/risks, auditable financial periods and sources, clean reviewer contracts, and the full test suite passing.

## File Structure

- Create `src/multi_agent/core/evidence.py`: evidence models, SEC fact normalization, period compatibility, source and field diagnostics.
- Create `src/multi_agent/core/report_document.py`: report models and deterministic Markdown/recommendation/report renderers.
- Create `src/multi_agent/core/delivery.py`: final-state construction and cross-artifact delivery validation.
- Modify `src/multi_agent/core/artifact_paths.py`: add evidence, report-document, review-contract, and final-decision paths.
- Modify `src/multi_agent/core/state.py`: persist the typed evidence bundle, review contracts, report context, report document, and final decision.
- Modify `src/multi_agent/core/review_contracts.py`: strict repair actions and delivery invariants.
- Modify `src/multi_agent/core/confidence_gate.py`: evaluate evidence semantics plus the review contract.
- Modify `src/multi_agent/tools/investment_tools.py`: return provenance-complete financial facts instead of flattened values only.
- Modify `src/multi_agent/flows/market_review_flow.py`: build evidence, lock report mode, consume strict contracts, and validate delivery without prose inference.
- Modify `src/multi_agent/recommendation.py`: keep the historical Markdown parser, but add typed-document projections for new runs.
- Modify `src/multi_agent/main.py`: write all final artifacts once from the validated delivery package.
- Modify `src/multi_agent/runtime.py`: inject run paths and prevent tests from falling through to a real workflow.
- Modify `src/multi_agent/config/tasks.yaml`: require the evidence and report JSON contracts in agent outputs.
- Create `tests/fixtures/apple/companyfacts.json`, `tests/fixtures/apple/filing.html`, `tests/fixtures/apple/quote.json`, and `tests/fixtures/apple/news.json`: deterministic Apple evidence.
- Create `tests/test_evidence.py`, `tests/test_report_document.py`, `tests/test_delivery.py`, and `tests/test_apple_pipeline.py`: contract, renderer, validator, and end-to-end coverage.
- Modify `tests/test_review_gate.py`, `tests/test_flow_routing.py`, `tests/test_main_cli.py`, `tests/test_recommendation.py`, and `tests/test_runtime.py`: new control-plane behavior and zero-network test isolation.

---

### Task 1: Provenance-Complete Evidence Contract

**Files:**
- Create: `src/multi_agent/core/evidence.py`
- Modify: `src/multi_agent/core/state.py`
- Create: `tests/test_evidence.py`
- Create: `tests/fixtures/apple/companyfacts.json`

**Interfaces:**
- Consumes: raw SEC Company Facts `dict[str, object]`, normalized company identity, quote/news payloads.
- Produces: `FinancialFact`, `MarketSnapshotEvidence`, `EventEvidence`, `ToolHealthRecord`, `EvidenceGap`, `ResearchEvidenceBundle`, `EvidenceNormalizer.normalize_company_facts(company_name: str, ticker: str, payload: dict[str, object]) -> ResearchEvidenceBundle`, and `periods_are_compatible(left: FinancialFact, right: FinancialFact) -> bool`.

- [ ] **Step 1: Write failing model and normalization tests**

```python
def test_apple_revenue_fact_keeps_period_and_filing_provenance(apple_companyfacts):
    bundle = EvidenceNormalizer().normalize_company_facts(
        company_name="Apple Inc.", ticker="AAPL", payload=apple_companyfacts
    )
    revenue = bundle.require_fact("revenue")
    assert revenue.unit == "USD"
    assert revenue.period_end == date(2025, 9, 27)
    assert revenue.fiscal_year == 2025
    assert revenue.form == "10-K"
    assert revenue.accession == "0000320193-25-000079"
    assert revenue.source_url.startswith("https://www.sec.gov/Archives/")
    assert revenue.formal_eligible is True


def test_periodless_fact_is_not_formal_eligible():
    fact = FinancialFact(field_name="revenue", value=10, unit="USD", source_tag="legacy")
    assert fact.formal_eligible is False
    assert "period_end_missing" in fact.quality_flags


def test_financial_facts_from_different_fiscal_years_are_not_period_compatible():
    current_period = FinancialFact(
        field_name="revenue", value=100, unit="USD",
        period_start=date(2024, 9, 29), period_end=date(2025, 9, 27),
        fiscal_year=2025, fiscal_period="FY", form="10-K",
        accession="0000320193-25-000079", filed_at=date(2025, 10, 31),
        source_url="https://www.sec.gov/Archives/edgar/data/320193/filing.htm",
        source_tag="sec_companyfacts",
    )
    stale_period = current_period.model_copy(
        update={
            "field_name": "diluted_shares",
            "value": 15,
            "unit": "shares",
            "period_start": date(2022, 9, 25),
            "period_end": date(2023, 9, 30),
            "fiscal_year": 2023,
            "accession": "0000320193-23-000106",
            "filed_at": date(2023, 11, 3),
        }
    )
    assert periods_are_compatible(current_period, stale_period) is False
```

- [ ] **Step 2: Run tests and verify the contract does not exist yet**

Run: `PYTHONPATH=src python -m pytest tests/test_evidence.py -q`

Expected: FAIL during import because `multi_agent.core.evidence` is missing.

- [ ] **Step 3: Implement strict Pydantic models and normalizer**

```python
class FinancialFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field_name: str
    value: float
    unit: str
    period_start: date | None = None
    period_end: date | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    form: str | None = None
    accession: str | None = None
    filed_at: date | None = None
    source_url: str | None = None
    source_tag: str
    taxonomy_concept: str | None = None
    quality_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def derive_quality_flags(self) -> "FinancialFact":
        required = ("period_end", "form", "accession", "filed_at", "source_url")
        self.quality_flags = sorted(
            set(self.quality_flags)
            | {f"{name}_missing" for name in required if getattr(self, name) in (None, "")}
        )
        return self

    @property
    def formal_eligible(self) -> bool:
        return not self.quality_flags and bool(self.unit and self.source_tag)


EvidenceTarget = Literal[
    "market_validation_analyst", "event_guidance_analyst", "fundamental_analyst",
    "quant_valuation_analyst", "report_writing_analyst",
    "data_quality_reviewer", "logic_compliance_reviewer",
]


class MarketSnapshotEvidence(BaseModel):
    price: float
    currency: str
    observed_at: datetime
    source_url: str
    source_tag: str
    diluted_shares_period_end: date | None = None


class EventEvidence(BaseModel):
    event_id: str
    title: str
    occurred_at: datetime | None = None
    published_at: datetime | None = None
    source_url: str
    source_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    independently_confirmed: bool = False


class ToolHealthRecord(BaseModel):
    tool_name: str
    status: Literal["healthy", "degraded", "failed"]
    error_type: str = ""
    service_name: str = ""
    http_status: int | None = None
    message: str = ""


class EvidenceGap(BaseModel):
    code: str
    target: EvidenceTarget
    fields: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    message: str


class ResearchEvidenceBundle(BaseModel):
    company_name: str
    ticker: str
    market_label: str = "US"
    financial_facts: list[FinancialFact] = Field(default_factory=list)
    market_snapshots: list[MarketSnapshotEvidence] = Field(default_factory=list)
    events: list[EventEvidence] = Field(default_factory=list)
    tool_health: list[ToolHealthRecord] = Field(default_factory=list)
    gaps: list[EvidenceGap] = Field(default_factory=list)
    raw_artifact_refs: list[str] = Field(default_factory=list)

    def require_fact(self, field_name: str) -> FinancialFact:
        matches = [fact for fact in self.financial_facts if fact.field_name == field_name]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one fact for {field_name}, got {len(matches)}")
        return matches[0]

    def formal_facts(self) -> list[FinancialFact]:
        return [fact for fact in self.financial_facts if fact.formal_eligible]

    def tool_status(self, tool_name: str) -> str:
        matches = [item.status for item in self.tool_health if item.tool_name == tool_name]
        return matches[-1] if matches else "failed"

    def without_fact(self, field_name: str) -> "ResearchEvidenceBundle":
        return self.model_copy(
            update={
                "financial_facts": [
                    fact for fact in self.financial_facts if fact.field_name != field_name
                ]
            }
        )
```

Use SEC `end`, `start`, `fy`, `fp`, `form`, `accn`, `filed`, and `frame` directly. Construct filing URLs from CIK/accession, select the newest filed fact within the newest compatible fiscal period, and record rejected candidates as gaps instead of silently discarding them.

- [ ] **Step 4: Extend run state with typed fields**

```python
class ResearchRunState(BaseModel):
    request_id: str
    company_name: str
    input_ticker: str = ""
    input_exchange: str = ""
    market_validation: MarketValidationResult | None = None
    evidence_ledger: list[EvidenceItem] = Field(default_factory=list)
    analysis_outputs: dict[str, dict[str, object]] = Field(default_factory=dict)
    review_tool_outputs: dict[str, dict[str, object]] = Field(default_factory=dict)
    review_findings: dict[str, dict[str, object]] = Field(default_factory=dict)
    rerun_budget: dict[str, int] = Field(default_factory=dict)
    model_tier_overrides: dict[RoutedAgentName, ModelTier] = Field(default_factory=dict)
    gate_decision: GateDecision | None = None
    final_decision: FinalDecision | None = None
    evidence_bundle: ResearchEvidenceBundle | None = None
```

Import `ResearchEvidenceBundle` directly from `multi_agent.core.evidence`; no `dict[str, object]` substitute is allowed for the new field.

- [ ] **Step 5: Run evidence tests**

Run: `PYTHONPATH=src python -m pytest tests/test_evidence.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the evidence contract**

```bash
git add src/multi_agent/core/evidence.py src/multi_agent/core/state.py tests/test_evidence.py tests/fixtures/apple/companyfacts.json
git commit -m "feat: add provenance complete evidence contract"
```

### Task 2: Normalize Tool Results Into One Evidence Bundle

**Files:**
- Modify: `src/multi_agent/tools/investment_tools.py`
- Modify: `src/multi_agent/tools/official_sec.py`
- Modify: `src/multi_agent/tools/tavily_search.py`
- Modify: `src/multi_agent/core/artifact_paths.py`
- Create: `tests/fixtures/apple/filing.html`
- Create: `tests/fixtures/apple/quote.json`
- Create: `tests/fixtures/apple/news.json`
- Modify: `tests/test_tools.py`
- Modify: `tests/test_official_sec_tools.py`
- Modify: `tests/test_tavily_tool.py`

**Interfaces:**
- Consumes: `OfficialSecService`, Tavily responses, the models from Task 1.
- Produces: `build_research_evidence_bundle(company_name: str, ticker: str, company_facts: dict[str, object], filing_html: str, quote_payload: dict[str, object], tavily_payloads: list[dict[str, object]]) -> ResearchEvidenceBundle`, plus `RunArtifactPaths.evidence_bundle_path` at `10_research_evidence.json`.

- [ ] **Step 1: Write failing tests for Apple tool normalization**

```python
def test_build_bundle_contains_all_formal_gate_facts(apple_sources):
    bundle = build_research_evidence_bundle(**apple_sources)
    assert {fact.field_name for fact in bundle.formal_facts()} >= {
        "revenue", "cash_and_equivalents", "total_debt",
        "diluted_shares", "segment_revenue_services",
    }
    quote = bundle.market_snapshots[0]
    assert quote.price > 0
    assert quote.observed_at is not None
    assert quote.source_url


def test_tavily_failure_is_recorded_as_degraded_not_hidden(
    apple_companyfacts, apple_filing_html, apple_quote
):
    bundle = build_research_evidence_bundle(
        company_name="Apple Inc.",
        ticker="AAPL",
        company_facts=apple_companyfacts,
        filing_html=apple_filing_html,
        quote_payload=apple_quote,
        tavily_payloads=[{"status": "degraded", "results": []}],
    )
    assert bundle.tool_status("tavily") == "degraded"
    assert any(gap.code == "independent_event_sources_insufficient" for gap in bundle.gaps)
```

- [ ] **Step 2: Run the focused tests**

Run: `PYTHONPATH=src python -m pytest tests/test_tools.py tests/test_official_sec_tools.py tests/test_tavily_tool.py -q`

Expected: FAIL because tool outputs do not expose a bundle or complete provenance.

- [ ] **Step 3: Refactor financial extraction to retain source entries**

Make `_extract_latest_fact(company_facts: dict[str, Any], tags: list[str], units: Sequence[str] = ("USD",)) -> FinancialFieldExtraction` preserve the selected SEC row in `FinancialFieldExtraction.fact: FinancialFact | None` rather than only `normalized_value`. Keep the current `financial_fields` keys for backward compatibility, and add a canonical `evidence_bundle` key to `FinancialMetricsTool._run()`:

```python
return json.dumps(
    {
        "metrics": metrics,
        "financial_fields": legacy_metadata,
        "market_snapshot": market_snapshot,
        "segment_snapshot": segment_snapshot,
        "formal_gate_snapshot": build_formal_gate_snapshot(legacy_metadata),
        "evidence_bundle": bundle.model_dump(mode="json"),
    },
    indent=2,
    ensure_ascii=False,
)
```

Do not swallow filing or quote exceptions without recording `ToolHealthRecord` and `EvidenceGap`. Re-raise fatal SEC identity/company-facts failures; represent quote failure as degraded evidence.

- [ ] **Step 4: Persist the evidence artifact path**

```python
@dataclass(frozen=True)
class RunArtifactPaths:
    company_dir: Path
    run_dir: Path
    market_intelligence_path: Path
    filing_review_path: Path
    financial_analysis_path: Path
    final_report_path: Path
    runtime_log_path: Path
    structured_recommendation_path: Path
    structured_report_path: Path
    latest_metrics_path: Path
    evaluation_summary_path: Path
    readme_path: Path
    evidence_bundle_path: Path
    report_document_path: Path
    final_decision_path: Path

# build_run_artifact_paths
evidence_bundle_path=run_dir / "10_research_evidence.json"
report_document_path=run_dir / "11_report_document.json"
final_decision_path=run_dir / "final_decision.json"
```

- [ ] **Step 5: Run tool and fixture tests**

Run: `PYTHONPATH=src python -m pytest tests/test_tools.py tests/test_official_sec_tools.py tests/test_tavily_tool.py tests/test_investment_tools_regressions.py -q`

Expected: PASS with no network access.

- [ ] **Step 6: Commit tool normalization**

```bash
git add src/multi_agent/tools/investment_tools.py src/multi_agent/tools/official_sec.py src/multi_agent/tools/tavily_search.py src/multi_agent/core/artifact_paths.py tests/fixtures/apple tests/test_tools.py tests/test_official_sec_tools.py tests/test_tavily_tool.py
git commit -m "feat: normalize research tools into evidence bundle"
```

### Task 3: Semantic Gate and Machine-Readable Repair Actions

**Files:**
- Modify: `src/multi_agent/core/review_contracts.py`
- Modify: `src/multi_agent/core/confidence_gate.py`
- Modify: `src/multi_agent/core/formal_gate.py`
- Modify: `src/multi_agent/tools/review_tools.py`
- Modify: `tests/test_review_gate.py`
- Modify: `tests/test_gate_snapshots.py`

**Interfaces:**
- Consumes: `ResearchEvidenceBundle` and `ReviewContract`.
- Produces: `RepairAction`, `FormalDeliveryDiagnostics`, strict `ReviewContract`, `diagnose_formal_delivery(bundle: ResearchEvidenceBundle, required_fields: list[str]) -> FormalDeliveryDiagnostics`, and `ConfidenceGatePolicy.evaluate(bundle: ResearchEvidenceBundle, contract: ReviewContract) -> GateDecision`.

- [ ] **Step 1: Write failing semantic-gate tests**

```python
def test_gate_blocks_cross_period_valuation(complete_contract, bundle_with_mixed_periods):
    result = ConfidenceGatePolicy.default().evaluate(bundle_with_mixed_periods, complete_contract)
    assert result.final_decision == "blocked"
    assert "valuation_period_mismatch" in result.blocking_reasons


def test_gate_requests_targeted_rerun_for_missing_services_fact(complete_contract, apple_bundle):
    bundle = apple_bundle.without_fact("segment_revenue_services")
    result = ConfidenceGatePolicy.default().evaluate(bundle, complete_contract)
    assert result.final_decision == "rerun"
    assert result.repair_actions == [
        RepairAction(target="fundamental_analyst", code="missing_financial_fact", fields=["segment_revenue_services"])
    ]


def test_gate_does_not_pass_when_reviewer_disallows_formal(apple_bundle, limited_contract):
    result = ConfidenceGatePolicy.default().evaluate(apple_bundle, limited_contract)
    assert result.final_decision == "evidence_limited"
```

- [ ] **Step 2: Run gate tests and verify API mismatch**

Run: `PYTHONPATH=src python -m pytest tests/test_review_gate.py tests/test_gate_snapshots.py -q`

Expected: FAIL because `evaluate` currently accepts only a summary/contract and uses `0.60` thresholds.

- [ ] **Step 3: Make contracts strict and repairable**

```python
class RepairAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: EvidenceTarget
    code: str
    fields: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    instruction: str = ""


class ReviewContract(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    stage: ReviewStage = Field(validation_alias=AliasChoices("stage", "review_stage"))
    reviewer_name: str
    decision: ReviewDecision
    delivery_eligibility: DeliveryEligibility
    failure_taxonomy: FailureTaxonomy
    coverage_summary: CoverageSummary
    tool_health_summary: ToolHealthSummary
    blocking_reasons: list[str] = Field(default_factory=list)
    rerun_reasons: list[str] = Field(default_factory=list)
    allow_limited_delivery: bool = False
    review_summary: ReviewSummary
    artifact_refs: list[dict[str, str]] = Field(default_factory=list)
    findings: list[dict[str, object]] = Field(default_factory=list)
    repair_actions: list[RepairAction] = Field(default_factory=list)


class FormalDeliveryDiagnostics(BaseModel):
    blocking_reasons: list[str] = Field(default_factory=list)
    repair_actions: list[RepairAction] = Field(default_factory=list)


def diagnose_formal_delivery(
    bundle: ResearchEvidenceBundle,
    required_fields: list[str],
) -> FormalDeliveryDiagnostics:
    facts_by_name = {
        field_name: [fact for fact in bundle.financial_facts if fact.field_name == field_name]
        for field_name in required_fields
    }
    missing = [
        field_name
        for field_name in required_fields
        if not any(fact.formal_eligible for fact in facts_by_name[field_name])
    ]
    actions = []
    if missing:
        actions.append(
            RepairAction(
                target="fundamental_analyst",
                code="missing_financial_fact",
                fields=missing,
                instruction="补齐 SEC 期间、filing 标识和来源 URL 后重新审查。",
            )
        )
    blockers = [
        "source_conflict"
        for facts in facts_by_name.values()
        if len({(fact.period_end, fact.value) for fact in facts if fact.formal_eligible}) > 1
        and len({fact.period_end for fact in facts if fact.formal_eligible}) == 1
    ]
    revenue = next(iter(facts_by_name.get("revenue", [])), None)
    diluted_shares = next(iter(facts_by_name.get("diluted_shares", [])), None)
    if revenue is not None and diluted_shares is not None and not periods_are_compatible(revenue, diluted_shares):
        blockers.append("valuation_period_mismatch")
    return FormalDeliveryDiagnostics(
        blocking_reasons=sorted(set(blockers)),
        repair_actions=actions,
    )
```

Reject malformed new contracts. Move compatibility normalization into `parse_legacy_review_contract(payload: dict[str, object]) -> ReviewContract`, used only when rebuilding historical artifacts.

- [ ] **Step 4: Evaluate evidence semantics at exact thresholds**

```python
@dataclass(frozen=True)
class ConfidenceGatePolicy:
    min_evidence_coverage_ratio: float = 0.80
    min_financial_coverage_score: float = 0.80
    min_claim_binding_ratio: float = 0.75
    min_trust_score: int = 75

    def evaluate(self, bundle: ResearchEvidenceBundle, contract: ReviewContract) -> GateDecision:
        diagnostics = diagnose_formal_delivery(bundle, FORMAL_GATE_REQUIRED_FIELDS)
        blocking_reasons = list(diagnostics.blocking_reasons)
        blocking_reasons.extend(contract.coverage_summary.market_policy_violations)
        blocking_reasons.extend(contract.coverage_summary.unsupported_critical_claims)
        if contract.delivery_eligibility.blocked_notice_required:
            blocking_reasons.append("delivery_blocked")
        repair_actions = list(contract.repair_actions) + list(diagnostics.repair_actions)
        trust_score = self.calculate_trust_score(bundle, contract)
        coverage = contract.coverage_summary
        if (
            coverage.evidence_coverage_ratio < self.min_evidence_coverage_ratio
            or coverage.financial_coverage_score < self.min_financial_coverage_score
            or coverage.claim_binding_ratio < self.min_claim_binding_ratio
            or trust_score < self.min_trust_score
        ):
            repair_actions.append(
                RepairAction(
                    target="data_quality_reviewer",
                    code="coverage_below_formal_threshold",
                    instruction="重新计算结构化覆盖率并补齐 claim-source 绑定。",
                )
            )
        if blocking_reasons:
            outcome = "blocked"
        elif repair_actions:
            outcome = "rerun"
        elif not contract.delivery_eligibility.formal_report_allowed:
            outcome = (
                "evidence_limited"
                if contract.delivery_eligibility.evidence_limited_report_allowed
                else "blocked"
            )
        else:
            outcome = "passed"
        return GateDecision(
            passed=outcome == "passed",
            final_decision=outcome,
            trust_score=trust_score,
            blocking_reasons=sorted(set(blocking_reasons)),
            repair_actions=repair_actions,
        )

    def calculate_trust_score(
        self, bundle: ResearchEvidenceBundle, contract: ReviewContract
    ) -> int:
        coverage = contract.coverage_summary
        base = (
            coverage.evidence_coverage_ratio * 35
            + coverage.financial_coverage_score * 35
            + coverage.claim_binding_ratio * 30
        )
        degraded_count = sum(item.status == "degraded" for item in bundle.tool_health)
        failed_count = sum(item.status == "failed" for item in bundle.tool_health)
        return int(max(0, min(100, base - degraded_count * 5 - failed_count * 20)))
```

Add `repair_actions: list[RepairAction]` to `GateDecision`. High prose completeness must not compensate for any blocker or repair action shown above.

- [ ] **Step 5: Run gate tests**

Run: `PYTHONPATH=src python -m pytest tests/test_review_gate.py tests/test_gate_snapshots.py -q`

Expected: PASS.

- [ ] **Step 6: Commit semantic gate behavior**

```bash
git add src/multi_agent/core/review_contracts.py src/multi_agent/core/confidence_gate.py src/multi_agent/core/formal_gate.py src/multi_agent/tools/review_tools.py tests/test_review_gate.py tests/test_gate_snapshots.py
git commit -m "feat: gate formal reports on evidence semantics"
```

### Task 4: Canonical Report Document and Deterministic Renderers

**Files:**
- Create: `src/multi_agent/core/report_document.py`
- Modify: `src/multi_agent/core/state.py`
- Modify: `src/multi_agent/recommendation.py`
- Create: `tests/test_report_document.py`
- Modify: `tests/test_recommendation.py`

**Interfaces:**
- Consumes: locked `ReportGenerationContext`, structured writer payload, and final trust score.
- Produces: `ReportGenerationContext`, `ReportSection`, `ReportClaim`, `SourceReference`, `ReportDocument`, `render_markdown`, `render_recommendation`, and `render_structured_report`.

- [ ] **Step 1: Write failing renderer tests**

```python
def test_all_outputs_share_the_same_canonical_document(formal_apple_document):
    markdown = render_markdown(formal_apple_document)
    recommendation = render_recommendation(formal_apple_document)
    report = render_structured_report(formal_apple_document)
    assert list(formal_apple_document.sections) == list(REQUIRED_SECTION_KEYS)
    assert recommendation["summary"] == formal_apple_document.executive_summary
    assert recommendation["catalysts"] == formal_apple_document.catalysts
    assert recommendation["risks"] == formal_apple_document.risks
    assert report["sections"]["financial_analysis"]
    assert "## 财务分析与估值" in markdown


def test_formal_document_rejects_unbound_critical_claim():
    payload = formal_apple_document.model_dump()
    payload["claims"][0]["source_ids"] = []
    with pytest.raises(ValidationError, match="critical claim"):
        ReportDocument.model_validate(payload)
```

- [ ] **Step 2: Run renderer tests**

Run: `PYTHONPATH=src python -m pytest tests/test_report_document.py tests/test_recommendation.py -q`

Expected: FAIL because the canonical report module does not exist.

- [ ] **Step 3: Implement canonical models and validation**

```python
ReportMode = Literal["formal_report", "evidence_limited_report", "blocked_notice"]
REQUIRED_SECTION_KEYS = (
    "executive_summary", "business_overview", "recent_events",
    "financial_analysis", "key_risks", "investment_conclusion", "source_index",
)

class ReportGenerationContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    company_name: str
    ticker: str
    report_mode: ReportMode
    evidence_bundle: ResearchEvidenceBundle
    analysis_review_contract: ReviewContract
    allowed_claim_ids: list[str]


class ReportSection(BaseModel):
    key: str
    heading: str
    content: str
    claim_ids: list[str] = Field(default_factory=list)


class ReportClaim(BaseModel):
    claim_id: str
    text: str
    critical: bool = False
    source_ids: list[str] = Field(default_factory=list)


class SourceReference(BaseModel):
    source_id: str
    title: str
    url: str
    source_tag: str
    field_name: str | None = None


class ReportDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    company_name: str
    ticker: str
    report_mode: ReportMode
    title: str
    stance: Literal["buy", "hold", "sell", "watch", "blocked"]
    executive_summary: str
    catalysts: list[str]
    risks: list[str]
    sections: dict[str, ReportSection]
    claims: list[ReportClaim]
    sources: list[SourceReference]
    trust_score: int
```

In the same step, add these fields to `ResearchRunState` after importing their exact model types:

```python
analysis_review_contract: ReviewContract | None = None
report_review_contract: ReviewContract | None = None
report_context: ReportGenerationContext | None = None
report_document: ReportDocument | None = None
final_decision_record: FinalDecisionRecord | None = None
```

Validate exact seven section keys and non-empty formal content. Validate every critical claim against `allowed_claim_ids` and at least one existing source ID. For limited/blocked modes, validate stance and disclosure strength.

- [ ] **Step 4: Add deterministic projections**

`render_markdown(document: ReportDocument) -> str` emits the seven Chinese headings in fixed order and source anchors from IDs. `render_recommendation(document: ReportDocument) -> dict[str, object]` and `render_structured_report(document: ReportDocument) -> dict[str, object]` read model fields directly. Keep the existing `build_structured_recommendation(*, company_name: str, company_ticker: str, report_path: Path, metrics: dict[str, Any]) -> dict[str, Any]` and `build_structured_report(*, company_name: str, company_ticker: str, report_path: Path, metrics: dict[str, Any]) -> dict[str, Any]` under a `legacy` compatibility comment for historical rebuild only.

- [ ] **Step 5: Run renderer tests**

Run: `PYTHONPATH=src python -m pytest tests/test_report_document.py tests/test_recommendation.py -q`

Expected: PASS, including a regression proving decorated Markdown headings are irrelevant for new output.

- [ ] **Step 6: Commit canonical report rendering**

```bash
git add src/multi_agent/core/report_document.py src/multi_agent/core/state.py src/multi_agent/recommendation.py tests/test_report_document.py tests/test_recommendation.py
git commit -m "feat: render all report artifacts from one document"
```

### Task 5: Single-Truth Delivery Validation

**Files:**
- Create: `src/multi_agent/core/delivery.py`
- Create: `tests/test_delivery.py`
- Modify: `src/multi_agent/main.py`
- Modify: `tests/test_main_cli.py`

**Interfaces:**
- Consumes: `FinalDecisionRecord`, `ReportDocument`, rendered Markdown, recommendation JSON, and structured-report JSON.
- Produces: `DeliveryPackage`, `DeliveryValidationResult`, `DeliveryValidator.validate(decision: FinalDecisionRecord, document: ReportDocument) -> DeliveryValidationResult`, and `write_delivery_package(paths: RunArtifactPaths, package: DeliveryPackage) -> None`.

- [ ] **Step 1: Write failing mismatch and completeness tests**

```python
def test_validator_rejects_limited_body_with_passed_decision(formal_document):
    decision = FinalDecisionRecord(final_decision="passed", final_delivery_state="formal_report", trust_score=90)
    document = formal_document.model_copy(update={"report_mode": "evidence_limited_report"})
    result = DeliveryValidator().validate(decision=decision, document=document)
    assert result.valid is False
    assert "report_mode_mismatch" in result.errors


def test_validator_rejects_empty_structured_sections(formal_package):
    formal_package.structured_report["sections"]["financial_analysis"] = ""
    assert "section_empty:financial_analysis" in DeliveryValidator().validate_package(formal_package).errors
```

- [ ] **Step 2: Run validator tests**

Run: `PYTHONPATH=src python -m pytest tests/test_delivery.py -q`

Expected: FAIL because `multi_agent.core.delivery` is missing.

- [ ] **Step 3: Implement state mapping and validation**

```python
EXPECTED_DELIVERY_STATE = {
    "passed": "formal_report",
    "evidence_limited": "evidence_limited_report",
    "blocked": "blocked_notice",
}


class DeliveryValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)


class DeliveryPackage(BaseModel):
    decision: FinalDecisionRecord
    document: ReportDocument
    markdown: str
    recommendation: dict[str, object]
    structured_report: dict[str, object]


class DeliveryValidator:
    def validate(self, *, decision: FinalDecisionRecord, document: ReportDocument) -> DeliveryValidationResult:
        errors = []
        expected = EXPECTED_DELIVERY_STATE[decision.final_decision]
        if decision.final_delivery_state != expected or document.report_mode != expected:
            errors.append("report_mode_mismatch")
        missing_sections = [key for key in REQUIRED_SECTION_KEYS if not document.sections[key].content.strip()]
        errors.extend(f"section_empty:{key}" for key in missing_sections)
        source_ids = {source.source_id for source in document.sources}
        for claim in document.claims:
            if claim.critical and not set(claim.source_ids).intersection(source_ids):
                errors.append(f"source_unbound:{claim.claim_id}")
        allowed_stances = {
            "formal_report": {"buy", "hold", "sell"},
            "evidence_limited_report": {"watch"},
            "blocked_notice": {"blocked"},
        }
        if document.stance not in allowed_stances[expected]:
            errors.append("stance_mismatch")
        return DeliveryValidationResult(valid=not errors, errors=sorted(set(errors)))

    def validate_package(self, package: DeliveryPackage) -> DeliveryValidationResult:
        result = self.validate(decision=package.decision, document=package.document)
        errors = list(result.errors)
        for key in REQUIRED_SECTION_KEYS:
            value = str(package.structured_report["sections"].get(key, "")).strip()
            if not value:
                errors.append(f"section_empty:{key}")
        if package.recommendation.get("status") != package.decision.final_decision:
            errors.append("recommendation_decision_mismatch")
        if package.structured_report.get("final_delivery_state") != package.decision.final_delivery_state:
            errors.append("structured_report_delivery_state_mismatch")
        return DeliveryValidationResult(valid=not errors, errors=sorted(set(errors)))
```

`write_delivery_package` writes `11_report_document.json`, `04_investment_report.md`, `06_structured_recommendation.json`, `07_structured_report.json`, and `final_decision.json` from the already validated package. It must not patch stance or status after rendering.

- [ ] **Step 4: Replace CLI projection and overwrite paths**

Change `_finalize_successful_result` and `_write_structured_outputs` so new workflow results must contain serialized `final_decision_record` and `report_document`. Remove `_apply_final_decision_projection` from the new-run path. Preserve old Markdown parsing only in `--watchlist-rebuild` when `11_report_document.json` is absent.

Patch CLI tests to stub `_kickoff_workflow`, never `_crew`, when flow execution is default. Add an autouse guard that raises on `requests.Session.request` and CrewAI kickoff unless a test explicitly stubs it.

- [ ] **Step 5: Run delivery and CLI tests**

Run: `PYTHONPATH=src python -m pytest tests/test_delivery.py tests/test_main_cli.py tests/test_runtime.py -q`

Expected: PASS with no DNS, HTTP, or LLM attempt.

- [ ] **Step 6: Commit single-truth delivery**

```bash
git add src/multi_agent/core/delivery.py src/multi_agent/main.py tests/test_delivery.py tests/test_main_cli.py tests/test_runtime.py
git commit -m "feat: validate and persist one delivery truth"
```

### Task 6: Flow Integration and Targeted Reruns

**Files:**
- Modify: `src/multi_agent/flows/market_review_flow.py`
- Modify: `src/multi_agent/runtime.py`
- Modify: `src/multi_agent/crew.py`
- Modify: `src/multi_agent/config/tasks.yaml`
- Modify: `tests/test_flow_routing.py`
- Modify: `tests/test_task_prompts.py`

**Interfaces:**
- Consumes: Task 1-5 contracts, CrewAI task output, artifact paths.
- Produces: a Flow result containing `status`, `final_decision_record`, `report_document`, `analysis_review_contract`, `report_review_contract`, and `repair_actions`.

- [ ] **Step 1: Write failing Flow tests for structured routing**

```python
def test_flow_passes_only_from_bundle_and_contract(formal_bundle, formal_contract, formal_document):
    flow = build_test_flow(bundle=formal_bundle, contract=formal_contract, document=formal_document)
    result = flow.kickoff()
    assert result["status"] == "passed"
    assert result["final_decision_record"]["final_delivery_state"] == "formal_report"


def test_rerun_uses_gate_repair_target(missing_quote_bundle, rerun_contract):
    flow = build_test_flow(bundle=missing_quote_bundle, contract=rerun_contract, rerun_budget={"quote": 1})
    flow.kickoff()
    assert flow.state.last_repair_actions[0].target == "quant_valuation_analyst"
    assert flow.state.model_tier_overrides == {"quant_valuation_analyst": "deep"}


def test_flow_ignores_prose_pass_marker_without_contract():
    result = build_test_flow(review_text="审查通过", contract=None).kickoff()
    assert result["status"] == "blocked"
    assert "analysis_review_contract_missing" in result["blocking_reasons"]
```

- [ ] **Step 2: Run Flow and prompt tests**

Run: `PYTHONPATH=src python -m pytest tests/test_flow_routing.py tests/test_task_prompts.py tests/test_crew_structure.py -q`

Expected: FAIL because Flow still parses prose and reruns every analysis agent together.

- [ ] **Step 3: Replace prose gates with structured state transitions**

At analysis completion, load `10_research_evidence.json` and strict `08_data_quality_review.json` (add this JSON sibling to Crew outputs). Call `ConfidenceGatePolicy.evaluate(bundle, contract)`. Build `model_tier_overrides` only from `GateDecision.repair_actions`, decrement per-target budget, and block after exhaustion unless the contract explicitly allows limited delivery.

After gate terminalization, create immutable `ReportGenerationContext` and pass its JSON in `REPORT_CONTEXT_JSON`. Require the writer to return a JSON object matching `ReportDocument`; render the Markdown in Python. Parse `09_logic_compliance_review.json` strictly and run `DeliveryValidator` before `_finalize`.

- [ ] **Step 4: Update Crew tasks and prompts**

Add JSON outputs `08_data_quality_review.json` and `09_logic_compliance_review.json` through task callbacks or dedicated strict-output tasks without changing the seven-agent topology. Prompts must name `10_research_evidence.json`, `REPORT_CONTEXT_JSON`, exact section keys, and the prohibition on choosing final state. Remove language that makes Markdown the machine contract.

- [ ] **Step 5: Run Flow and Crew tests**

Run: `PYTHONPATH=src python -m pytest tests/test_flow_routing.py tests/test_task_prompts.py tests/test_crew_structure.py -q`

Expected: PASS and no test reads a prose marker to decide terminal state.

- [ ] **Step 6: Commit Flow integration**

```bash
git add src/multi_agent/flows/market_review_flow.py src/multi_agent/runtime.py src/multi_agent/crew.py src/multi_agent/config/tasks.yaml tests/test_flow_routing.py tests/test_task_prompts.py
git commit -m "feat: integrate evidence contracts through report delivery"
```

### Task 7: Offline Apple End-to-End Regression

**Files:**
- Create: `tests/test_apple_pipeline.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_evaluation.py`
- Modify: `src/multi_agent/evaluation.py`

**Interfaces:**
- Consumes: Apple fixture payloads and stubbed analysis/review/writer outputs.
- Produces: a complete temporary run directory and assertions over every acceptance artifact.

- [ ] **Step 1: Write the failing Apple end-to-end test**

```python
def test_apple_fixture_run_produces_complete_formal_delivery(tmp_path, apple_pipeline):
    result, run_dir = apple_pipeline.run(tmp_path)
    decision = load_json(run_dir / "final_decision.json")
    recommendation = load_json(run_dir / "06_structured_recommendation.json")
    report = load_json(run_dir / "07_structured_report.json")
    evidence = load_json(run_dir / "10_research_evidence.json")
    assert result["status"] == "passed"
    assert decision["final_delivery_state"] == "formal_report"
    assert recommendation["summary"] and recommendation["catalysts"] and recommendation["risks"]
    assert all(report["sections"].values())
    assert all(report["validation"].values())
    assert all(fact["source_url"] and fact["period_end"] for fact in evidence["financial_facts"] if fact["field_name"] in FORMAL_GATE_REQUIRED_FIELDS)
```

- [ ] **Step 2: Run the Apple integration test**

Run: `PYTHONPATH=src python -m pytest tests/test_apple_pipeline.py -q`

Expected: FAIL until the new pipeline writes all artifacts atomically.

- [ ] **Step 3: Add evaluation checks for semantic completeness**

Extend `WorkflowEvaluation` metrics with `report_sections_complete`, `structured_outputs_complete`, `formal_fact_provenance_complete`, `decision_projection_consistent`, and `delivery_validation_passed`. A run marked `success=True` must require all five when `final_decision == "passed"`.

- [ ] **Step 4: Add global network isolation for tests**

```python
@pytest.fixture(autouse=True)
def forbid_unexpected_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("unexpected network call; inject a fake service")
    monkeypatch.setattr(requests.sessions.Session, "request", fail)
```

Tests that intentionally verify request construction inject their own fake session and do not disable this guard globally.

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=src python -m pytest -q`

Expected: all tests PASS; zero network/LLM attempts and no stale `0.60` gate assertions.

- [ ] **Step 6: Commit the offline acceptance harness**

```bash
git add tests/test_apple_pipeline.py tests/conftest.py tests/test_evaluation.py src/multi_agent/evaluation.py
git commit -m "test: prove complete Apple report pipeline offline"
```

### Task 8: Real Apple Run and Acceptance Loop

**Files:**
- Modify only files directly implicated by machine-readable diagnostics from the live run.
- Verify outputs under configured `RUNS_DIR` for the new AAPL run.

**Interfaces:**
- Consumes: configured `.env`, live SEC/Tavily/quote/LLM services, and all prior tasks.
- Produces: one real AAPL run satisfying every design acceptance condition.

- [ ] **Step 1: Validate configuration without printing secrets**

Run:

```bash
PYTHONPATH=src python -c 'from multi_agent.settings import InvestmentResearchSettings as S; s=S.from_env(); print({"models": [s.fast_model,s.deep_model,s.review_model], "base_url": s.openai_base_url, "tavily": bool(s.tavily_api_key), "sec_email": bool(s.sec_api_email), "runs_dir": s.runs_dir})'
```

Expected: all three models are non-empty, Tavily and SEC flags are `True`, and a writable runs directory is shown.

- [ ] **Step 2: Run Apple through the real CLI**

Run: `PYTHONPATH=src python -m multi_agent.main --company-name "Apple Inc." --company-ticker AAPL`

Expected: exit code 0 and a newly printed run directory.

- [ ] **Step 3: Execute the acceptance checker**

Run: `PYTHONPATH=src python scripts/validate_run.py --latest-company apple_inc__aapl`

The checker must exit 0 only when:

```text
final_decision=passed
final_delivery_state=formal_report
standard_artifacts_complete=true
seven_sections_complete=true
structured_recommendation_complete=true
structured_report_complete=true
formal_fact_provenance_complete=true
review_contracts_clean=true
delivery_projection_consistent=true
```

Create `scripts/validate_run.py` in this step with a `validate_run(run_dir: Path) -> list[str]` function that returns explicit error codes and a CLI that exits 1 when the list is non-empty.

- [ ] **Step 4: Diagnose and repair any failed acceptance condition**

For each checker error, add a failing fixture or unit regression that reproduces the exact failure, run it red, make the smallest architecture-aligned correction, run it green, then repeat Steps 2-3. Examples of exact mappings:

```text
fact_period_missing -> tests/test_evidence.py + EvidenceNormalizer
analysis_review_contract_missing -> tests/test_flow_routing.py + strict review parser
report_mode_mismatch -> tests/test_delivery.py + DeliveryValidator
section_empty:* -> tests/test_report_document.py + writer JSON normalization
source_unbound:* -> tests/test_report_document.py + claim/source validation
```

- [ ] **Step 5: Run final verification after the live checker passes**

Run: `PYTHONPATH=src python -m pytest -q`

Expected: all tests PASS.

Run: `PYTHONPATH=src python scripts/validate_run.py --latest-company apple_inc__aapl`

Expected: exit code 0 with all nine acceptance lines set to the values above.

- [ ] **Step 6: Commit live-run fixes and validator**

```bash
git add scripts/validate_run.py src/multi_agent tests
git commit -m "fix: close Apple formal report acceptance gaps"
```

Do not commit `.env`, API keys, runtime logs, or generated `var/runs` artifacts unless the user explicitly requests fixture promotion.
