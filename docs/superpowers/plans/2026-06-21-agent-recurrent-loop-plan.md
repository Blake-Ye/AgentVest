# Agent Recurrent Loop 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 把当前“顺序执行 + 末端审查”的 Agent 流水线升级为“分析 Agent 可被 Reviewer 驳回并定向重跑”的 recurrent 工作流。

**架构：** 保留现有 7 个 Agent 职责分工，但把执行中枢从单向 `Process.sequential` 提升为 `Flow + 结构化状态 + 驳回回流`。核心变化不是增加更多 Agent，而是为现有分析 Agent、Writer 和 Reviewer 之间补上显式的 `pass / fail / rerun_target / escalation` 回路。

**技术栈：** Python 3.10+、CrewAI Flow、Pydantic、pytest、现有 Tavily / Official SEC / MarketValidation 工具链

---

## 目标架构

### 1. Agent 分层保持不变，但执行语义改变

- 路由类：
  - `market_validation_analyst`
- 分析类：
  - `event_guidance_analyst`
  - `fundamental_analyst`
  - `quant_valuation_analyst`
- 审查类：
  - `data_quality_reviewer`
  - `logic_compliance_reviewer`
- 交付类：
  - `report_writing_analyst`

### 2. recurrent 只放在两个闭环里

- **闭环 A：分析闭环**
  - 三个分析 Agent 完成后，进入 `data_quality_reviewer`
  - Reviewer 输出：
    - `passed: true/false`
    - `rerun_targets: ["event_guidance" | "fundamental" | "quant_valuation"]`
    - `issues`
    - `requires_model_escalation`
  - 如果 `passed=false`：
    - 只重跑被点名的 Agent
    - 每个 Agent 最多重跑一次
    - 如果仍失败，降级为人工确认或终止交付

- **闭环 B：写作闭环**
  - `report_writing_analyst` 产出报告后，进入 `logic_compliance_reviewer`
  - Reviewer 输出：
    - `passed: true/false`
    - `rerun_target: "report_writer" | "upstream_analysis"`
    - `issues`
  - 如果 `passed=false`：
    - 优先只重跑 `report_writing_analyst`
    - 如果明确指出证据或分析前提有问题，再回流到对应分析 Agent
    - Writer 最多重跑一次

### 3. Agent 使用模式

- **生产型 Agent**
  - 输出结构化结果，而不是只产 markdown
  - 必须携带：
    - `facts`
    - `evidence_refs`
    - `unknowns`
    - `confidence`
- **Reviewer Agent**
  - 不负责重写内容
  - 只负责：
    - 判断是否通过
    - 指出问题类别
    - 指定回流目标
    - 指定是否升级模型
- **Writer Agent**
  - 只消费“已通过数据质量审查”的结构化结果
  - 不直接调用原始采集工具

### 4. 模型升级规则

- `fast_model`
  - 市场验证
  - 事件抽取
  - 初始结构化整理
- `deep_model`
  - 基本面综合
  - 估值综合
  - 报告写作
- `review_model`
  - 两个 Reviewer 固定使用
- **recurrent 触发升级**
  - 当 Reviewer 标记 `requires_model_escalation=true` 时：
    - 被驳回的分析 Agent 从 `fast -> deep`
    - 已经是 `deep` 的 Agent 不再升级，只允许一次重跑

### 5. 状态驱动而不是 markdown 驱动

- markdown 产物继续保留，但只作为“人类可读投影”
- 系统运行状态以结构化 state 为准，至少包含：
  - `market_validation`
  - `analysis_outputs`
  - `review_findings`
  - `rerun_budget`
  - `model_tier_overrides`
  - `final_decision`

## 文件结构

**创建：**
- `src/multi_agent/core/review_contracts.py`
- `tests/test_review_loops.py`

**修改：**
- `src/multi_agent/core/state.py`
- `src/multi_agent/core/model_routing.py`
- `src/multi_agent/flows/market_review_flow.py`
- `src/multi_agent/config/agents.yaml`
- `src/multi_agent/config/tasks.yaml`
- `src/multi_agent/crew.py`
- `tests/test_flow_routing.py`
- `tests/test_crew_structure.py`

## 修改方案

### 任务 1：把 Agent 输出契约从“文本结果”提升为“结构化结果”

**文件：**
- 修改：`src/multi_agent/core/state.py`
- 创建：`src/multi_agent/core/review_contracts.py`
- 测试：`tests/test_flow_routing.py`

- [ ] **步骤 1：编写失败的状态契约测试**

```python
def test_review_contract_can_point_to_rerun_targets():
    review = DataQualityReviewResult(
        passed=False,
        rerun_targets=["event_guidance", "fundamental"],
        requires_model_escalation=True,
        issues=["missing earnings evidence"],
    )

    assert review.passed is False
    assert review.rerun_targets == ["event_guidance", "fundamental"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_flow_routing.py -k review_contract -v`
预期：FAIL，报错缺少 `DataQualityReviewResult`

- [ ] **步骤 3：实现最少结构化契约**

```python
class DataQualityReviewResult(BaseModel):
    passed: bool
    rerun_targets: list[str] = Field(default_factory=list)
    requires_model_escalation: bool = False
    issues: list[str] = Field(default_factory=list)


class LogicComplianceReviewResult(BaseModel):
    passed: bool
    rerun_target: str = ""
    issues: list[str] = Field(default_factory=list)
```

- [ ] **步骤 4：扩展运行状态**

```python
class ResearchRunState(BaseModel):
    request_id: str
    company_name: str
    input_ticker: str = ""
    input_exchange: str = ""
    market_validation: MarketValidationResult | None = None
    evidence_ledger: list[EvidenceItem] = Field(default_factory=list)
    analysis_outputs: dict[str, dict] = Field(default_factory=dict)
    review_findings: dict[str, dict] = Field(default_factory=dict)
    rerun_budget: dict[str, int] = Field(default_factory=dict)
    model_tier_overrides: dict[str, str] = Field(default_factory=dict)
```

- [ ] **步骤 5：运行测试验证通过**

运行：`python3 -m pytest tests/test_flow_routing.py -k review_contract -v`
预期：PASS

### 任务 2：把 ModelRouter 从静态 tier 映射改成“静态默认 + 动态覆盖”

**文件：**
- 修改：`src/multi_agent/core/model_routing.py`
- 测试：`tests/test_flow_routing.py`

- [ ] **步骤 1：编写失败的模型覆盖测试**

```python
def test_model_router_prefers_agent_override():
    settings = InvestmentResearchSettings(
        fast_model="qwen-flash",
        deep_model="qwen-pro",
        review_model="qwen-review",
        company_resolver_model="qwen-flash",
        openai_api_key="k",
        openai_base_url="https://example.com",
        tavily_api_key="tvly",
        sec_api_email="analyst@example.com",
    )

    router = ModelRouter(settings)
    assert router.for_agent("event_guidance", {"event_guidance": "deep"}) == "qwen-pro"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_flow_routing.py -k model_router_prefers_agent_override -v`
预期：FAIL，报错 `for_agent` 不存在

- [ ] **步骤 3：实现最少动态路由**

```python
class ModelRouter:
    def for_tier(self, tier: str) -> str:
        mapping = {
            "fast": self.settings.fast_model,
            "deep": self.settings.deep_model,
            "review": self.settings.review_model,
        }
        return mapping[tier]

    def for_agent(self, agent_name: str, overrides: dict[str, str] | None = None) -> str:
        overrides = overrides or {}
        tier = overrides.get(agent_name, self.default_tier_for_agent(agent_name))
        return self.for_tier(tier)
```

- [ ] **步骤 4：定义 Agent 默认 tier**

```python
    def default_tier_for_agent(self, agent_name: str) -> str:
        defaults = {
            "market_validation": "fast",
            "event_guidance": "fast",
            "fundamental": "deep",
            "quant_valuation": "deep",
            "report_writer": "deep",
            "data_quality_reviewer": "review",
            "logic_compliance_reviewer": "review",
        }
        return defaults[agent_name]
```

- [ ] **步骤 5：运行测试验证通过**

运行：`python3 -m pytest tests/test_flow_routing.py -k model_router_prefers_agent_override -v`
预期：PASS

### 任务 3：把 Flow 从“壳层委托 Crew”升级为“显式回流状态机”

**文件：**
- 修改：`src/multi_agent/flows/market_review_flow.py`
- 测试：`tests/test_flow_routing.py`

- [ ] **步骤 1：编写失败的回流测试**

```python
def test_flow_reruns_only_flagged_analysis_agents():
    flow = MarketReviewFlow(...)
    result = flow.kickoff()

    assert flow.state.rerun_budget["event_guidance"] == 1
    assert flow.state.rerun_budget["fundamental"] == 1
    assert flow.state.rerun_budget["quant_valuation"] == 0
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_flow_routing.py -k reruns_only_flagged_analysis_agents -v`
预期：FAIL，当前 Flow 只会 `execute_existing_crew`

- [ ] **步骤 3：把 Flow 拆成显式阶段**

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
```

- [ ] **步骤 4：增加回流分支**

```python
    @listen(review_analysis)
    def rerun_analysis_if_needed(self, review_result):
        if review_result.passed:
            return "passed"
        if self.state.rerun_budget.get("analysis", 0) >= 1:
            return "failed"
        self.state.rerun_budget["analysis"] = self.state.rerun_budget.get("analysis", 0) + 1
        return self.run_analysis()
```

- [ ] **步骤 5：增加写作闭环**

```python
    @listen("passed")
    def write_report(self):
        ...

    @listen(write_report)
    def review_report(self, report):
        ...
```

- [ ] **步骤 6：运行测试验证通过**

运行：`python3 -m pytest tests/test_flow_routing.py -v`
预期：PASS

### 任务 4：让 Reviewer 从“意见输出”升级为“控制信号输出”

**文件：**
- 修改：`src/multi_agent/config/agents.yaml`
- 修改：`src/multi_agent/config/tasks.yaml`
- 测试：`tests/test_crew_structure.py`

- [ ] **步骤 1：编写失败的 Reviewer 契约测试**

```python
def test_reviewer_tasks_require_rerun_decision_fields():
    task_output = build_reviewer_output_contract()
    assert "passed" in task_output
    assert "issues" in task_output
    assert "rerun_targets" in task_output or "rerun_target" in task_output
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_crew_structure.py -k reviewer -v`
预期：FAIL，当前 Reviewer 只要求 markdown 审查报告

- [ ] **步骤 3：重写 Reviewer Prompt 目标**

```yaml
data_quality_reviewer:
  goal: >
    审查上游分析是否通过；若不通过，必须指出需要重跑的 Agent、
    是否需要模型升级，以及问题是否阻断最终交付。
```

- [ ] **步骤 4：重写 Reviewer Task 输出要求**

```yaml
data_quality_review_task:
  expected_output: >
    一份结构化审查结果，必须包含 passed、issues、rerun_targets、
    requires_model_escalation，并附带中文 markdown 审查说明。
```

- [ ] **步骤 5：运行测试验证通过**

运行：`python3 -m pytest tests/test_crew_structure.py -k reviewer -v`
预期：PASS

### 任务 5：把 Crew 降级为兼容层，不再承担 recurrent 编排

**文件：**
- 修改：`src/multi_agent/crew.py`
- 测试：`tests/test_crew_structure.py`

- [ ] **步骤 1：编写失败的兼容层测试**

```python
def test_crew_remains_compatibility_layer():
    crew = MultiAgent().crew()
    assert crew.process == Process.sequential
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/test_crew_structure.py -k compatibility_layer -v`
预期：FAIL，当前没有显式兼容层约束

- [ ] **步骤 3：限制 Crew 只保留线性回退用途**

```python
@crew
def crew(self) -> Crew:
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.sequential,
        cache=True,
        verbose=True,
    )
```

- [ ] **步骤 4：在代码注释中声明 Flow 才是 recurrent 主路径**

```python
# ponytail: Crew 仅保留为 CLI 回退兼容层；带回流的正式路径由 Flow 承担。
```

- [ ] **步骤 5：运行测试验证通过**

运行：`python3 -m pytest tests/test_crew_structure.py -v`
预期：PASS

## 落地顺序建议

1. 先做 `任务 1` 和 `任务 2`
2. 再做 `任务 3`
3. 最后做 `任务 4` 和 `任务 5`

原因：
- 不先补结构化状态和 Reviewer 契约，回流逻辑无从表达
- 不先补动态模型路由，recurrent 只能重跑，不能升级
- Crew 兼容层应该最后处理，避免过早影响当前 CLI 稳定性

## 风险控制

- 不把所有 markdown 中间产物删除；先保留给人类读和回归测试用
- 每个回流环只允许一次重跑，避免无限循环
- `UNRESOLVED` 市场不进入估值和最终推荐
- Reviewer 只能发“控制信号”，不能直接偷偷改写上游事实

## 预期收益

- Agent 不再是“顺序跑完就结束”，而是进入“审查 - 定向修复 - 再审查”的闭环
- `fast_model` 可以安全前置，因为被驳回后能升级到 `deep_model`
- Reviewer 从“写问题清单”变成“真正控制工作流质量门禁”的节点

