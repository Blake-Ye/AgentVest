# Reviewer Hard Gate With Confidence Scoring 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为当前投研工作流增加基于 reviewer 工具与结构化评分卡的硬门控，只有置信度和证据完整性达到阈值时才允许输出正式投资报告。

**架构：** 保留现有 7 个 Agent 分工，但把 reviewer 从“只写意见”升级为“工具计算结构化审查结果 + Gate 规则硬判定 + Flow 驳回回流”。正式报告不再由 `review_model` 主观批准，而是由 `EvidenceCoverageTool`、`CrossSourceConsistencyTool`、`MarketToolPolicyAuditTool`、`FinancialFieldCompletenessTool` 和 `ConfidenceGatePolicy` 共同决定是否放行。

**技术栈：** Python 3.10+、CrewAI Flow、Pydantic、pytest、现有 Tavily / Official SEC / MarketValidation 工具链

---

## 反思与优化方向

### 对 `2026-06-21-agent-recurrent-loop-plan.md` 的反思

1. 旧计划解决了“驳回后回流到谁”，但没有定义“为什么通过”。
2. 旧计划让 Reviewer 输出 `passed / rerun_targets / requires_model_escalation`，这已经比纯 markdown 审查更强，但 `passed` 仍然过度依赖 LLM 主观判断。
3. 旧计划没有处理 CLI 成功语义：如果报告被硬门控拦截，当前 `main.py` 会把“没有正式报告文件”当成运行失败，而不是“审查未通过但流程成功完成”。
4. 旧计划没有区分“分析质量门禁”和“交付门禁”：
   - 分析层应决定是否需要局部重跑
   - 交付层应决定是否允许物化正式 `04_investment_report.md`
5. 旧计划把 recurrent 和 gate 混在一起讨论，容易导致第一次落地时改动面过大、CLI 回归风险过高。

### 本次优化原则

1. **先把 gate 做成独立控制面，再让 recurrent 消费 gate 结果。**
2. **置信度来自工具计算，不来自 reviewer 自评。**
3. **未通过 gate 不算系统异常。** CLI 应输出“阻断结论”和阻断原因，而不是崩溃。
4. **正式报告与阻断报告分离语义。**
   - `通过`：输出正式投资备忘录
   - `阻断`：输出 gate 失败说明文件和结构化失败原因
5. **Flow 是主路径，Crew 保持兼容层。**

## 文件结构

**创建：**
- `src/multi_agent/core/review_contracts.py`
- `src/multi_agent/core/confidence_gate.py`
- `src/multi_agent/tools/review_tools.py`
- `tests/test_review_gate.py`

**修改：**
- `src/multi_agent/core/state.py`
- `src/multi_agent/core/model_routing.py`
- `src/multi_agent/flows/market_review_flow.py`
- `src/multi_agent/config/agents.yaml`
- `src/multi_agent/config/tasks.yaml`
- `src/multi_agent/crew.py`
- `src/multi_agent/runtime.py`
- `src/multi_agent/main.py`
- `tests/test_flow_routing.py`
- `tests/test_crew_structure.py`
- `tests/test_main_cli.py`

## 文件职责

- `src/multi_agent/core/review_contracts.py`
  - 定义 reviewer 工具输出、review 结果、gate 决策结果的 Pydantic 契约。
- `src/multi_agent/core/confidence_gate.py`
  - 实现硬门控规则，例如证据覆盖率阈值、冲突阻断、市场工具误用阻断、最终 `trust_score` 计算。
- `src/multi_agent/tools/review_tools.py`
  - 实现 reviewer 专用工具，负责把已有分析产物转成结构化评分输入。
- `src/multi_agent/core/state.py`
  - 扩展 Flow 运行态，记录分析输出、review 工具产物、gate 决策、rerun budget。
- `src/multi_agent/flows/market_review_flow.py`
  - 从“调用 Crew 壳层”升级为显式状态机：分析 -> 数据质量 gate -> 重跑或写作 -> 逻辑 gate -> 交付或阻断。
- `src/multi_agent/config/agents.yaml`
  - 重写 reviewer 与 writer 的目标，明确 reviewer 不拍板、只解释工具结果。
- `src/multi_agent/config/tasks.yaml`
  - 重写 reviewer task 的 expected output，要求输出结构化 gate 解释。
- `src/multi_agent/crew.py`
  - 保持兼容层；给 reviewer agent 挂载 review 工具；新增动态模型 tier 解析。
- `src/multi_agent/main.py`
  - 区分 `passed` 与 `blocked` 两种成功结束语义，更新文件物化和 JSON 输出逻辑。

## Gate 判定规则

默认硬门控阈值：

- `evidence_coverage_ratio >= 0.80`
- `financial_coverage_score >= 0.80`
- `critical_conflict_count == 0`
- `market_policy_violations == 0`
- `unsupported_critical_claims == 0`
- `trust_score >= 75`

其中 `trust_score` 采用 0-100 线性分数：

- `trust_score = int(max(0, min(100, evidence_coverage_ratio * 50 + financial_coverage_score * 50)))`
- `trust_score` 用于总览展示与阈值补充，不替代 evidence / financial / conflict / policy / unsupported 的独立阻断规则

默认输出级别：

- `pass`
  - 允许输出正式 `04_investment_report.md`
  - `06_structured_recommendation.json.status = "passed"`
- `blocked`
  - 不允许输出正式投资结论
  - `04_investment_report.md` 写入“阻断说明”，而不是占位符
  - `06_structured_recommendation.json.status = "blocked"`
  - `07_structured_report.json.final_decision = "blocked"`

## 任务 1：建立 reviewer 工具与 gate 契约

**文件：**
- 创建：`src/multi_agent/core/review_contracts.py`
- 创建：`src/multi_agent/core/confidence_gate.py`
- 测试：`tests/test_review_gate.py`

- [ ] **步骤 1：编写失败的 gate 契约测试**

```python
from multi_agent.core.confidence_gate import ConfidenceGatePolicy
from multi_agent.core.review_contracts import ReviewToolSummary


def test_confidence_gate_blocks_when_evidence_coverage_below_threshold() -> None:
    summary = ReviewToolSummary(
        evidence_coverage_ratio=0.6,
        financial_coverage_score=0.9,
        critical_conflict_count=0,
        market_policy_violations=[],
        unsupported_critical_claims=["growth will accelerate"],
        blocking_reasons=[],
    )

    decision = ConfidenceGatePolicy.default().evaluate(summary)

    assert decision.passed is False
    assert decision.final_decision == "blocked"
    assert "evidence_coverage_ratio<0.80" in decision.blocking_reasons
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_review_gate.py -k evidence_coverage_below_threshold -v`
预期：FAIL，报错缺少 `ConfidenceGatePolicy` 或 `ReviewToolSummary`

- [ ] **步骤 3：实现最少 reviewer 契约**

```python
from pydantic import BaseModel, Field


class ReviewToolSummary(BaseModel):
    evidence_coverage_ratio: float
    financial_coverage_score: float
    critical_conflict_count: int = 0
    market_policy_violations: list[str] = Field(default_factory=list)
    unsupported_critical_claims: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)


class GateDecision(BaseModel):
    passed: bool
    final_decision: str
    trust_score: int
    blocking_reasons: list[str] = Field(default_factory=list)
```

- [ ] **步骤 4：实现最少 gate 规则**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceGatePolicy:
    min_evidence_coverage_ratio: float = 0.80
    min_financial_coverage_score: float = 0.80
    min_trust_score: int = 75

    @classmethod
    def default(cls) -> "ConfidenceGatePolicy":
        return cls()

    def evaluate(self, summary: ReviewToolSummary) -> GateDecision:
        blocking_reasons = list(summary.blocking_reasons)
        if summary.evidence_coverage_ratio < self.min_evidence_coverage_ratio:
            blocking_reasons.append("evidence_coverage_ratio<0.80")
        if summary.financial_coverage_score < self.min_financial_coverage_score:
            blocking_reasons.append("financial_coverage_score<0.80")
        if summary.critical_conflict_count > 0:
            blocking_reasons.append("critical_conflict_count>0")
        if summary.market_policy_violations:
            blocking_reasons.append("market_policy_violations>0")
        if summary.unsupported_critical_claims:
            blocking_reasons.append("unsupported_critical_claims>0")

        trust_score = int(
            max(0, min(100, summary.evidence_coverage_ratio * 50 + summary.financial_coverage_score * 50))
        )
        if trust_score < self.min_trust_score:
            blocking_reasons.append("trust_score<75")

        return GateDecision(
            passed=not blocking_reasons,
            final_decision="passed" if not blocking_reasons else "blocked",
            trust_score=trust_score,
            blocking_reasons=blocking_reasons,
        )
```

- [ ] **步骤 5：运行测试验证通过**

运行：`python3 -m pytest tests/test_review_gate.py -v`
预期：PASS

### 任务 2：为 reviewer 增加结构化工具，而不是只读 markdown 发意见

**文件：**
- 创建：`src/multi_agent/tools/review_tools.py`
- 修改：`src/multi_agent/crew.py`
- 测试：`tests/test_crew_structure.py`

- [ ] **步骤 1：编写失败的 reviewer 工具测试**

```python
from multi_agent.tools.review_tools import EvidenceCoverageTool


def test_evidence_coverage_tool_flags_unsupported_claims() -> None:
    tool = EvidenceCoverageTool()
    result = tool._run(
        claims=[
            {"claim": "营收增速改善", "evidence_refs": ["news-1"]},
            {"claim": "利润率显著扩张", "evidence_refs": []},
        ]
    )

    assert result["evidence_coverage_ratio"] == 0.5
    assert result["unsupported_claims"] == ["利润率显著扩张"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_crew_structure.py -k evidence_coverage_tool -v`
预期：FAIL，报错缺少 `EvidenceCoverageTool`

- [ ] **步骤 3：实现最少 reviewer 工具**

```python
from crewai.tools import BaseTool


class EvidenceCoverageTool(BaseTool):
    name: str = "evidence_coverage_tool"
    description: str = "根据 claim 与 evidence_refs 计算证据覆盖率和无证据结论。"

    def _run(self, claims: list[dict[str, object]]) -> dict[str, object]:
        total = len(claims)
        unsupported = [
            str(item.get("claim", "")).strip()
            for item in claims
            if not item.get("evidence_refs")
        ]
        covered = total - len(unsupported)
        ratio = 0.0 if total == 0 else covered / total
        return {
            "evidence_coverage_ratio": ratio,
            "unsupported_claims": unsupported,
        }
```

- [ ] **步骤 4：把 reviewer 工具挂到 `data_quality_reviewer`**

```python
from multi_agent.tools.review_tools import (
    CrossSourceConsistencyTool,
    EvidenceCoverageTool,
    FinancialFieldCompletenessTool,
    MarketToolPolicyAuditTool,
)


@agent
def data_quality_reviewer(self) -> Agent:
    return Agent(
        config=self.agents_config["data_quality_reviewer"],  # type: ignore[index]
        llm=self._llm_for_tier("review"),
        tools=[
            EvidenceCoverageTool(),
            CrossSourceConsistencyTool(),
            MarketToolPolicyAuditTool(),
            FinancialFieldCompletenessTool(),
        ],
        max_retry_limit=3,
        verbose=True,
    )
```

- [ ] **步骤 5：运行测试验证通过**

运行：`python3 -m pytest tests/test_crew_structure.py -k evidence_coverage_tool -v`
预期：PASS

### 任务 3：扩展运行状态与动态模型路由，让 recurrent 真正消费 gate 结果

**文件：**
- 修改：`src/multi_agent/core/state.py`
- 修改：`src/multi_agent/core/model_routing.py`
- 测试：`tests/test_flow_routing.py`

- [ ] **步骤 1：编写失败的状态与路由测试**

```python
from multi_agent.core.state import ResearchRunState
from multi_agent.core.review_contracts import GateDecision


def test_research_run_state_tracks_gate_and_rerun_budget() -> None:
    state = ResearchRunState(
        request_id="run-001",
        company_name="Microsoft Corporation",
        rerun_budget={"event_guidance": 1},
        gate_decision=GateDecision(
            passed=False,
            final_decision="blocked",
            trust_score=68,
            blocking_reasons=["evidence_coverage_ratio<0.80"],
        ),
    )

    dumped = state.model_dump()
    assert dumped["gate_decision"]["final_decision"] == "blocked"
    assert dumped["rerun_budget"]["event_guidance"] == 1
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_flow_routing.py -k tracks_gate_and_rerun_budget -v`
预期：FAIL，报错 `gate_decision` 或 `rerun_budget` 字段不存在

- [ ] **步骤 3：扩展状态**

```python
from pydantic import BaseModel, Field


class ResearchRunState(BaseModel):
    request_id: str
    company_name: str
    input_ticker: str = ""
    input_exchange: str = ""
    market_validation: MarketValidationResult | None = None
    evidence_ledger: list[EvidenceItem] = Field(default_factory=list)
    analysis_outputs: dict[str, dict] = Field(default_factory=dict)
    review_tool_outputs: dict[str, dict] = Field(default_factory=dict)
    review_findings: dict[str, dict] = Field(default_factory=dict)
    rerun_budget: dict[str, int] = Field(default_factory=dict)
    model_tier_overrides: dict[str, str] = Field(default_factory=dict)
    gate_decision: GateDecision | None = None
    final_decision: str = ""
```

- [ ] **步骤 4：扩展动态路由**

```python
class ModelRouter:
    def default_tier_for_agent(self, agent_name: str) -> str:
        defaults = {
            "market_validation_analyst": "fast",
            "event_guidance_analyst": "fast",
            "fundamental_analyst": "deep",
            "quant_valuation_analyst": "deep",
            "report_writing_analyst": "deep",
            "data_quality_reviewer": "review",
            "logic_compliance_reviewer": "review",
        }
        return defaults[agent_name]

    def for_agent(self, agent_name: str, overrides: dict[str, str] | None = None) -> str:
        overrides = overrides or {}
        tier = overrides.get(agent_name, self.default_tier_for_agent(agent_name))
        return self.for_tier(tier)
```

- [ ] **步骤 5：运行测试验证通过**

运行：`python3 -m pytest tests/test_flow_routing.py -k "tracks_gate_and_rerun_budget or model_router" -v`
预期：PASS

### 任务 4：把 Flow 从“调用 Crew”升级为“分析 gate + 写作 gate”的状态机

**文件：**
- 修改：`src/multi_agent/flows/market_review_flow.py`
- 测试：`tests/test_flow_routing.py`

- [ ] **步骤 1：编写失败的 gate 流程测试**

```python
def test_flow_blocks_report_when_gate_decision_fails() -> None:
    flow = MarketReviewFlow(...)
    result = flow.kickoff()

    assert flow.state.final_decision == "blocked"
    assert result["status"] == "blocked"
    assert "evidence_coverage_ratio<0.80" in result["blocking_reasons"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_flow_routing.py -k blocks_report_when_gate_decision_fails -v`
预期：FAIL，当前 Flow 只会 `execute_existing_crew`

- [ ] **步骤 3：拆成显式阶段**

```python
class MarketReviewFlow(Flow[ResearchRunState]):
    @start()
    def validate_market(self):
        ...

    @listen(validate_market)
    def run_analysis(self):
        ...

    @listen(run_analysis)
    def review_analysis(self):
        ...

    @listen(review_analysis)
    def apply_analysis_gate(self, review_result):
        ...
```

- [ ] **步骤 4：增加分析闭环与模型升级**

```python
    @listen(apply_analysis_gate)
    def rerun_analysis_if_needed(self, gate):
        if gate.passed:
            return "analysis_passed"
        if self.state.rerun_budget.get("analysis", 0) >= 1:
            self.state.final_decision = "blocked"
            return {"status": "blocked", "blocking_reasons": gate.blocking_reasons}
        self.state.rerun_budget["analysis"] = self.state.rerun_budget.get("analysis", 0) + 1
        self.state.model_tier_overrides["event_guidance_analyst"] = "deep"
        return self.run_analysis()
```

- [ ] **步骤 5：增加写作闭环与最终 gate**

```python
    @listen("analysis_passed")
    def write_report(self):
        ...

    @listen(write_report)
    def review_report(self, report):
        ...

    @listen(review_report)
    def finalize_delivery(self, report_gate):
        self.state.gate_decision = report_gate
        self.state.final_decision = report_gate.final_decision
        return {
            "status": report_gate.final_decision,
            "trust_score": report_gate.trust_score,
            "blocking_reasons": report_gate.blocking_reasons,
        }
```

- [ ] **步骤 6：运行测试验证通过**

运行：`python3 -m pytest tests/test_flow_routing.py -v`
预期：PASS

### 任务 5：重写 reviewer / writer 提示词，使其服务 gate 而不是替代 gate

**文件：**
- 修改：`src/multi_agent/config/agents.yaml`
- 修改：`src/multi_agent/config/tasks.yaml`
- 测试：`tests/test_crew_structure.py`

- [ ] **步骤 1：编写失败的 prompt 契约测试**

```python
def test_reviewer_prompts_reference_gate_and_tool_outputs() -> None:
    agents_yaml = Path("src/multi_agent/config/agents.yaml").read_text(encoding="utf-8")
    tasks_yaml = Path("src/multi_agent/config/tasks.yaml").read_text(encoding="utf-8")

    assert "不要主观决定是否放行" in agents_yaml
    assert "必须引用 reviewer tools 的结构化结果" in tasks_yaml
    assert "若 gate 未通过，不得生成正式投资建议" in tasks_yaml
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_crew_structure.py -k gate_and_tool_outputs -v`
预期：FAIL，当前 prompt 未体现 gate 约束

- [ ] **步骤 3：重写 reviewer agent 目标**

```yaml
data_quality_reviewer:
  goal: >
    基于 reviewer tools 的结构化结果解释问题、指定 rerun_targets、
    判断是否需要模型升级；不要主观决定是否放行，放行由 gate policy 决定。
```

- [ ] **步骤 4：重写 writer task 的阻断约束**

```yaml
investment_report_task:
  description: >
    仅当 analysis gate 已通过时，才允许撰写正式投资备忘录。
    若 gate 未通过，不得生成正式投资建议，不得擅自补充缺失证据。
  expected_output: >
    一份高质量中文 markdown 投资备忘录；若接收到 blocked 状态，必须输出阻断说明而非正式报告。
```

- [ ] **步骤 5：运行测试验证通过**

运行：`python3 -m pytest tests/test_crew_structure.py -k gate_and_tool_outputs -v`
预期：PASS

### 任务 6：调整 CLI 成功语义，让“阻断”不是异常而是可交付状态

**文件：**
- 修改：`src/multi_agent/main.py`
- 修改：`src/multi_agent/runtime.py`
- 测试：`tests/test_main_cli.py`

- [ ] **步骤 1：编写失败的 CLI 阻断测试**

```python
def test_run_writes_blocked_outputs_when_gate_fails(monkeypatch, tmp_path) -> None:
    result = {
        "status": "blocked",
        "trust_score": 68,
        "blocking_reasons": ["evidence_coverage_ratio<0.80", "unsupported_critical_claims>0"],
    }

    monkeypatch.setattr("multi_agent.main._kickoff_workflow", lambda inputs: result)

    run()

    report = (tmp_path / "company__ticker" / "20260101_000000" / "04_investment_report.md").read_text(encoding="utf-8")
    assert "阻断" in report
    assert "evidence_coverage_ratio<0.80" in report
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_main_cli.py -k blocked_outputs_when_gate_fails -v`
预期：FAIL，当前 CLI 只接受“正式报告已生成”的成功路径

- [ ] **步骤 3：新增阻断物化逻辑**

```python
def _write_blocked_report(output_paths: RunOutputPaths, result: dict[str, object]) -> None:
    body = "\n".join(
        [
            "状态：硬门控未通过，正式投资备忘录未放行。",
            "",
            f"trust_score：{result.get('trust_score', 0)}",
            "阻断原因：",
            *[f"- {item}" for item in result.get("blocking_reasons", [])],
        ]
    )
    _write_markdown_file(output_paths.final_report_path, "投资备忘录（已阻断）", body, placeholder=False)
```

- [ ] **步骤 4：调整成功校验与结构化输出**

```python
def _validate_successful_outputs(output_paths: RunOutputPaths, *, final_status: str) -> None:
    required_paths = [
        output_paths.market_validation_path,
        output_paths.market_intelligence_path,
        output_paths.filing_review_path,
        output_paths.financial_analysis_path,
        output_paths.data_quality_review_path,
        output_paths.logic_compliance_review_path,
    ]
    if final_status == "passed":
        required_paths.append(output_paths.final_report_path)
    else:
        required_paths.append(output_paths.final_report_path)

    for path in required_paths:
        if not path.exists() or not path.read_text(encoding="utf-8").strip() or _is_placeholder_file(path):
            raise RuntimeError(f"运行结束但未生成规范输出文件：{path.name}")
```

- [ ] **步骤 5：运行测试验证通过**

运行：`python3 -m pytest tests/test_main_cli.py -v`
预期：PASS

## 落地顺序建议

1. 先做 `任务 1`
2. 再做 `任务 2`
3. 再做 `任务 3`
4. 然后做 `任务 4`
5. 最后做 `任务 5` 和 `任务 6`

原因：

- 不先定义 gate 契约，reviewer 工具没有统一消费目标。
- 不先挂 reviewer 工具，Flow 就只能消费 markdown 审查，不是真正的硬门控。
- 不先扩状态与动态路由，recurrent 只能重跑，不能伴随升级。
- CLI 成功语义必须最后改，避免在中途把所有旧测试一次性打碎。

## 风险控制

- reviewer 工具只读已有结构化产物和中间结果，不直接读最终 markdown 以外的私有上下文。
- `trust_score` 只是总览指标，真正阻断依赖显式规则，不依赖单一数字。
- 每个闭环最多重跑一次，避免无限循环。
- 当 `status="blocked"` 时，流程仍写完整 artifacts，便于人工复盘。
- `Crew` 保持 `Process.sequential` 兼容层，不在本阶段把全部 CLI 强制切到 Flow。

## 预期收益

- reviewer 从“会写问题清单”升级成“能提供可信放行依据”的控制节点。
- `fast_model` 前置更安全，因为质量不过关时可以被 gate 驳回并升级重跑。
- CLI 不再把“未放行”误当成“运行错误”，而是形成可审计的阻断结果。
- 最终报告真正具备“达标才放行”的质量门禁，而不是“只要流程跑完就输出”。
