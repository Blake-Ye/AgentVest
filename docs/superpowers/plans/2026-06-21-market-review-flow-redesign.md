# 市场验证驱动投研 Flow 重构实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将当前串行 Crew 投研流程升级为基于市场验证、结构化证据、双层 Reviewer 和分层模型路由的高鲁棒 Flow。

**架构：** 入口先通过 `CompanyResolver + MarketValidationTool` 确定公司实体与市场标签，再把可用工具策略写入结构化状态。分析阶段不再直接串行消费 markdown，而是并行产出结构化结果，经 `Data Quality Reviewer` 和 `Logic & Compliance Reviewer` 两层审查后再生成最终报告与结构化 recommendation。

**技术栈：** Python 3.10+、CrewAI Flow、Pydantic、requests、Tavily API、SEC 官方端点、pytest

---

## 重新设计后的 Agent 执行逻辑

- 当前实现的问题不是 Agent 数量不足，而是执行单元定义错误：现在的“Agent=阶段”实际上仍是“Task 串行文本拼接”，见 `src/multi_agent/crew.py`。
- 新执行逻辑应采用“Flow 编排 + 直接 Agent kickoff + 结构化状态推进”，而不是继续让所有 Agent 挂在一个 `Process.sequential` 的 Crew 上。
- 建议把执行单元分成四类：
  - 路由类：`CompanyResolver`、`MarketValidationAgent`
  - 分析类：`FundamentalAnalyst`、`QuantValuationAnalyst`、`EventGuidanceAnalyst`
  - 审查类：`DataQualityReviewer`、`LogicComplianceReviewer`
  - 交付类：`ReportWriter`
- 每个分析类 Agent 使用 `agent.kickoff(..., response_format=...)` 直接产出 Pydantic 结构，不再依赖 markdown 中间产物作为下游输入。
- Flow 的执行顺序：
  1. 解析公司实体
  2. 进行市场验证并写入 `state.market_validation`
  3. 若 `UNRESOLVED`，直接降级结束并输出人工确认请求
  4. 并行运行三个分析 Agent
  5. 运行第一层 Reviewer
  6. 若失败，仅定向重跑受影响 Agent 一次
  7. 运行 Report Writer
  8. 运行第二层 Reviewer
  9. 若失败，仅重跑 Report Writer 一次
  10. 落盘 artifacts、structured report、structured recommendation、evaluation metrics
- 这意味着：
  - `Crew` 不再是核心 orchestrator，而是可选兼容层
  - `Flow` 成为事实上的运行中枢
  - Markdown 文件变为“人类可读投影”，不是系统状态源
- 模型策略必须嵌入执行逻辑：
  - 路由、抽取、结构化整理默认使用 `fast_model`
  - 事件影响路径综合、复杂冲突判断、报告写作使用 `deep_model`
  - Reviewer 固定使用 `review_model`
  - 分析 Agent 允许在发现冲突或字段缺失时从 `fast_model` 升级到 `deep_model`

## 文件结构

**创建：**
- `src/multi_agent/core/state.py`
- `src/multi_agent/core/market.py`
- `src/multi_agent/core/model_routing.py`
- `src/multi_agent/core/artifact_paths.py`
- `src/multi_agent/flows/market_review_flow.py`
- `src/multi_agent/runtime.py`
- `src/multi_agent/api.py`
- `src/multi_agent/tools/market_validation.py`
- `src/multi_agent/tools/tavily_search.py`
- `src/multi_agent/tools/official_sec.py`
- `src/multi_agent/tools/market_data.py`
- `src/multi_agent/tools/transcripts.py`
- `src/multi_agent/tools/ownership.py`
- `tests/test_market_validation.py`
- `tests/test_tavily_tool.py`
- `tests/test_official_sec_tools.py`
- `tests/test_reviewers.py`
- `tests/test_flow_routing.py`

**修改：**
- `src/multi_agent/settings.py`
- `src/multi_agent/resolver.py`
- `src/multi_agent/tools/investment_tools.py`
- `src/multi_agent/config/agents.yaml`
- `src/multi_agent/config/tasks.yaml`
- `src/multi_agent/crew.py`
- `src/multi_agent/main.py`
- `src/multi_agent/evaluation.py`
- `src/multi_agent/recommendation.py`
- `tests/test_settings.py`
- `tests/test_main_cli.py`

## 当前执行顺序（稳定版）

以下顺序覆盖原始任务 3-10 的执行优先级；原文中更靠后的详细步骤若与本节冲突，以本节为准。

### 已完成

- 任务 A：收口配置与路径模型
- 任务 B：替换搜索层为 Tavily

### 当前主线

### 任务 C：纯官方 SEC 底座

**目标：** 彻底移除 `sec-api.io` 依赖，把 resolver、filings、company facts 统一到 SEC 官方端点和本地匹配逻辑上。

**文件：**
- 创建：`src/multi_agent/tools/official_sec.py`
- 修改：`src/multi_agent/resolver.py`
- 修改：`src/multi_agent/tools/investment_tools.py`
- 修改：`src/multi_agent/tools/__init__.py`
- 测试：`tests/test_official_sec_tools.py`
- 测试：`tests/test_company_resolver.py`
- 验证：`tests/test_tools.py`

**稳定推进要求：**
- 先跑红灯：`python3 -m pytest tests/test_official_sec_tools.py tests/test_company_resolver.py -v`
- `official_sec.py` 只负责官方端点访问、ticker 目录缓存、本地名称匹配、submissions/companyfacts 规范化
- `resolver.py` 不再直接拼任何外部 URL，只依赖官方 SEC 服务
- `investment_tools.py` 先保留 façade 和财务快照逻辑，但 SEC 相关工具全部切到 `OfficialSecService`
- 保留 `sec_api_key` 构造兼容字段，但运行路径不得再读取或使用它

### 任务 D：市场验证层与工具策略

**目标：** 先用本地规则和已有官方数据推断 `US / EU / HK / UNRESOLVED`，再生成工具白名单/禁用策略。

**文件：**
- 创建：`src/multi_agent/core/market.py`
- 创建：`src/multi_agent/tools/market_validation.py`
- 测试：`tests/test_market_validation.py`

**稳定推进要求：**
- 默认采用“非 API 优先，官方源校验增强”
- `UNRESOLVED` 必须阻断高风险结论
- SEC 工具策略必须由市场标签显式控制

### 任务 E：结构化状态与模型路由

**目标：** 固化市场验证结果、证据账本、分析结果、Reviewer 结论以及模型升级条件。

**文件：**
- 创建：`src/multi_agent/core/state.py`
- 创建：`src/multi_agent/core/model_routing.py`
- 测试：`tests/test_flow_routing.py`

### 任务 F：Agent/Task 配置重写

**目标：** 把 Agent 职责、输入输出、Reviewer 节点和工具映射写回配置层，为 Flow 迁移做准备。

**文件：**
- 修改：`src/multi_agent/config/agents.yaml`
- 修改：`src/multi_agent/config/tasks.yaml`
- 测试：`tests/test_crew_structure.py`

### 任务 G：最小 Flow 迁移

**目标：** 只迁移真正需要分支、回路、状态推进的部分，保留兼容入口，避免一次性推翻所有运行路径。

**文件：**
- 创建：`src/multi_agent/flows/market_review_flow.py`
- 修改：`src/multi_agent/crew.py`
- 测试：`tests/test_flow_routing.py`

**稳定推进要求：**
- 只在任务 C-F 稳定后开始
- `crew.py` 先降级为兼容层，不立即删除

### 任务 H：runtime / CLI / API 适配

**文件：**
- 创建：`src/multi_agent/runtime.py`
- 创建：`src/multi_agent/api.py`
- 修改：`src/multi_agent/main.py`
- 测试：`tests/test_main_cli.py`

### 任务 I：评估、报告结构与 watchlist

**文件：**
- 修改：`src/multi_agent/evaluation.py`
- 修改：`src/multi_agent/recommendation.py`
- 修改：`src/multi_agent/watchlist.py`
- 测试：`tests/test_reviewers.py`
- 测试：`tests/test_main_cli.py`

### 任务 J：全量验证与文档收尾

**文件：**
- 修改：`.env.example`
- 修改：`README.md`
- 全量验证：`python3 -m pytest tests -v`
- `tests/test_crew_structure.py`

**测试：**
- `tests/test_settings.py`
- `tests/test_market_validation.py`
- `tests/test_tavily_tool.py`
- `tests/test_official_sec_tools.py`
- `tests/test_reviewers.py`
- `tests/test_flow_routing.py`
- `tests/test_main_cli.py`

### 任务 1：收口配置与路径模型

**文件：**
- 修改：`src/multi_agent/settings.py`
- 创建：`src/multi_agent/core/artifact_paths.py`
- 测试：`tests/test_settings.py`

- [ ] **步骤 1：编写失败的配置测试**

```python
def test_settings_support_model_tiers(monkeypatch):
    monkeypatch.setenv("FAST_MODEL", "qwen-v4-flash")
    monkeypatch.setenv("DEEP_MODEL", "qwen-v4-pro")
    monkeypatch.setenv("REVIEW_MODEL", "qwen-v4-pro")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")
    monkeypatch.setenv("SEC_API_EMAIL", "analyst@example.com")

    settings = InvestmentResearchSettings.from_env()

    assert settings.fast_model == "qwen-v4-flash"
    assert settings.deep_model == "qwen-v4-pro"
    assert settings.review_model == "qwen-v4-pro"
    assert settings.tavily_api_key == "tvly-key"
    assert settings.sec_api_email == "analyst@example.com"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_settings.py -k "model_tiers or tavily" -v`
预期：FAIL，报错 `InvestmentResearchSettings` 缺少新字段或仍要求 `SEC_API_KEY`

- [ ] **步骤 3：实现最少配置改造**

```python
@dataclass(frozen=True)
class InvestmentResearchSettings:
    fast_model: str
    deep_model: str
    review_model: str
    company_resolver_model: str
    openai_api_key: str
    openai_base_url: str
    tavily_api_key: str
    sec_api_email: str
    artifact_root: str = "src/multi_agent/artifacts"
    runs_dir: str = "src/multi_agent/artifacts/runs"
    latest_dir: str = "src/multi_agent/artifacts/latest"
    watchlist_path: str = "src/multi_agent/artifacts/watchlist.json"
```

- [ ] **步骤 4：新增路径解析模型**

```python
@dataclass(frozen=True)
class RunArtifactPaths:
    run_dir: Path
    market_validation_json: Path
    final_report_md: Path
    final_report_json: Path
```

- [ ] **步骤 5：运行测试验证通过**

运行：`pytest tests/test_settings.py -v`
预期：PASS

- [ ] **步骤 6：Commit**

```bash
git add src/multi_agent/settings.py src/multi_agent/core/artifact_paths.py tests/test_settings.py
git commit -m "refactor: introduce tiered model and artifact path settings"
```

### 任务 2：替换搜索层为 Tavily

**文件：**
- 创建：`src/multi_agent/tools/tavily_search.py`
- 修改：`src/multi_agent/tools/investment_tools.py`
- 测试：`tests/test_tavily_tool.py`

- [ ] **步骤 1：编写 Tavily 规范化测试**

```python
def test_tavily_tool_normalizes_results():
    tool = TavilySearchTool(settings=build_settings(), service=StubTavilyService())
    result = tool._run(query="Apple earnings guidance", topic="guidance", market_label="US")

    assert result["results"][0]["url"] == "https://example.com/apple"
    assert result["results"][0]["source_type"] == "news"
    assert result["results"][0]["published_at"] == "2026-06-20"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_tavily_tool.py -v`
预期：FAIL，提示 `TavilySearchTool` 不存在

- [ ] **步骤 3：实现 Tavily 工具和结果规范化**

```python
class TavilySearchTool(BaseTool):
    def _run(self, query: str, topic: str, market_label: str) -> dict[str, Any]:
        payload = self._service.search(query=query)
        return {"results": [normalize_tavily_result(item, topic, market_label) for item in payload]}
```

- [ ] **步骤 4：在 façade 中替换旧搜索导出**

```python
from multi_agent.tools.tavily_search import TavilySearchTool
```

- [ ] **步骤 5：运行测试验证通过**

运行：`pytest tests/test_tavily_tool.py tests/test_tools.py -v`
预期：PASS

- [ ] **步骤 6：Commit**

```bash
git add src/multi_agent/tools/tavily_search.py src/multi_agent/tools/investment_tools.py tests/test_tavily_tool.py
git commit -m "feat: replace google search integration with tavily"
```

### 任务 3：切换到纯官方 SEC 方案

**文件：**
- 创建：`src/multi_agent/tools/official_sec.py`
- 修改：`src/multi_agent/resolver.py`
- 修改：`src/multi_agent/tools/investment_tools.py`
- 测试：`tests/test_official_sec_tools.py`

- [ ] **步骤 1：编写官方 SEC 工具测试**

```python
def test_official_sec_ticker_resolver_returns_cik():
    tool = OfficialSecTickerResolverTool(settings=build_settings(), service=StubSecService())
    result = tool._run(ticker="AAPL", company_name="")

    assert result["resolved_ticker"] == "AAPL"
    assert result["cik"] == "0000320193"
    assert result["match_type"] == "ticker_exact"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_official_sec_tools.py -v`
预期：FAIL，提示工具未实现或仍使用 `sec-api.io`

- [ ] **步骤 3：实现官方端点客户端**

```python
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
```

- [ ] **步骤 4：移除 resolver 中的 sec-api.io mapping 依赖**

```python
def _resolve_by_ticker(self, ticker: str) -> CompanyResolution | None:
    item = self._official_sec_lookup_by_ticker(ticker)
    if item is None:
        return None
    return self._from_official_ticker_item(item)
```

- [ ] **步骤 5：加入非 US 市场 SEC 禁用守卫**

```python
def sec_market_guard(market_label: str) -> None:
    if market_label != "US":
        raise ValueError("SEC tools are only available for US market.")
```

- [ ] **步骤 6：运行测试验证通过**

运行：`pytest tests/test_official_sec_tools.py tests/test_company_resolver.py -v`
预期：PASS

- [ ] **步骤 7：Commit**

```bash
git add src/multi_agent/tools/official_sec.py src/multi_agent/resolver.py src/multi_agent/tools/investment_tools.py tests/test_official_sec_tools.py
git commit -m "refactor: migrate sec integration to official endpoints only"
```

### 任务 4：建立市场验证层与工具策略

**文件：**
- 创建：`src/multi_agent/core/market.py`
- 创建：`src/multi_agent/tools/market_validation.py`
- 测试：`tests/test_market_validation.py`

- [ ] **步骤 1：编写市场验证测试**

```python
def test_market_validation_blocks_sec_for_hk():
    tool = MarketValidationTool(service=StubMarketValidationService())
    result = tool._run(company_name="Tencent Holdings", ticker="0700.HK", exchange="HKEX")

    assert result["market_label"] == "HK"
    assert result["tool_policy"]["sec_allowed"] is False
    assert result["needs_human_confirmation"] is False
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_market_validation.py -v`
预期：FAIL，提示 `MarketValidationTool` 缺失

- [ ] **步骤 3：实现市场标签与工具策略工厂**

```python
def build_tool_policy(market_label: str) -> ToolPolicy:
    if market_label == "US":
        return ToolPolicy(True, True, True, True, True, [])
    if market_label == "EU":
        return ToolPolicy(False, True, True, True, False, ["SEC based fundamental conclusions"])
    if market_label == "HK":
        return ToolPolicy(False, True, True, True, False, ["SEC based fundamental conclusions"])
    return ToolPolicy(False, True, False, False, False, ["investment recommendation", "valuation conclusion"])
```

- [ ] **步骤 4：实现市场验证工具**

```python
class MarketValidationTool(BaseTool):
    def _run(self, company_name: str, ticker: str, exchange: str = "") -> dict[str, Any]:
        # exchange -> suffix -> sec exact -> tavily evidence -> unresolved
        ...
```

- [ ] **步骤 5：运行测试验证通过**

运行：`pytest tests/test_market_validation.py -v`
预期：PASS

- [ ] **步骤 6：Commit**

```bash
git add src/multi_agent/core/market.py src/multi_agent/tools/market_validation.py tests/test_market_validation.py
git commit -m "feat: add market validation layer and tool policy"
```

### 任务 5：引入结构化状态和模型路由

**文件：**
- 创建：`src/multi_agent/core/state.py`
- 创建：`src/multi_agent/core/model_routing.py`
- 测试：`tests/test_flow_routing.py`

- [ ] **步骤 1：编写状态与模型路由测试**

```python
def test_model_router_returns_review_model_for_reviewer():
    router = ModelRouter(settings=build_settings())
    assert router.for_role("review") == "qwen-v4-pro"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_flow_routing.py -k model_router -v`
预期：FAIL，提示 `ModelRouter` 或 `ResearchRunState` 不存在

- [ ] **步骤 3：实现结构化状态**

```python
class ResearchRunState(BaseModel):
    request_id: str
    company_name: str
    input_ticker: str = ""
    input_exchange: str = ""
    market_validation: MarketValidationResult | None = None
    evidence_ledger: list[EvidenceItem] = []
```

- [ ] **步骤 4：实现模型路由器**

```python
class ModelRouter:
    def for_role(self, tier: str) -> str:
        mapping = {"fast": self.settings.fast_model, "deep": self.settings.deep_model, "review": self.settings.review_model}
        return mapping[tier]
```

- [ ] **步骤 5：运行测试验证通过**

运行：`pytest tests/test_flow_routing.py -k "model_router or state" -v`
预期：PASS

- [ ] **步骤 6：Commit**

```bash
git add src/multi_agent/core/state.py src/multi_agent/core/model_routing.py tests/test_flow_routing.py
git commit -m "feat: introduce structured state and tiered model router"
```

### 任务 6：重写 Agent 配置与任务配置

**文件：**
- 修改：`src/multi_agent/config/agents.yaml`
- 修改：`src/multi_agent/config/tasks.yaml`
- 测试：`tests/test_crew_structure.py`

- [ ] **步骤 1：编写新的结构测试**

```python
def test_crew_config_contains_reviewers():
    crew = MultiAgent().crew()
    roles = [agent.role for agent in crew.agents]
    assert any("数据质量审查员" in role for role in roles)
    assert any("逻辑与合规审查员" in role for role in roles)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_crew_structure.py -v`
预期：FAIL，当前只有 3 个 agent

- [ ] **步骤 3：更新 agents.yaml**

```yaml
data_quality_reviewer:
  role: "{company_name} 数据质量审查员"
  goal: "检查来源缺失、跨市场误用工具、数据冲突和无依据结论。"
  model_tier: review
```

- [ ] **步骤 4：更新 tasks.yaml**

```yaml
data_quality_review_task:
  description: >
    审查三类分析结果是否存在来源缺失、市场边界错误或数据冲突。
  expected_output: >
    一份结构化审查结果，包含 blocking issues、repair actions 和 approved evidence ids。
  agent: data_quality_reviewer
```

- [ ] **步骤 5：运行测试验证通过**

运行：`pytest tests/test_crew_structure.py -v`
预期：PASS

- [ ] **步骤 6：Commit**

```bash
git add src/multi_agent/config/agents.yaml src/multi_agent/config/tasks.yaml tests/test_crew_structure.py
git commit -m "feat: add reviewer agents and structured task definitions"
```

### 任务 7：将执行中枢从 Crew 串行迁移到 Flow

**文件：**
- 创建：`src/multi_agent/flows/market_review_flow.py`
- 修改：`src/multi_agent/crew.py`
- 测试：`tests/test_flow_routing.py`

- [ ] **步骤 1：编写 Flow 路由测试**

```python
def test_flow_stops_when_market_unresolved():
    flow = MarketReviewFlow()
    flow.state.market_validation = MarketValidationResult(
        market_label="UNRESOLVED",
        confidence="low",
        needs_human_confirmation=True,
        resolution_basis=[],
        tool_policy=build_tool_policy("UNRESOLVED"),
    )
    assert flow.route_after_market_validation() == "manual_confirmation_required"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_flow_routing.py -k flow -v`
预期：FAIL，`MarketReviewFlow` 不存在

- [ ] **步骤 3：实现 Flow 编排骨架**

```python
class MarketReviewFlow(Flow[ResearchRunState]):
    @start()
    def resolve_company(self):
        ...

    @listen(resolve_company)
    def validate_market(self):
        ...

    @router(validate_market)
    def route_after_market_validation(self):
        if self.state.market_validation.market_label == "UNRESOLVED":
            return "manual_confirmation_required"
        return "analysis_ready"
```

- [ ] **步骤 4：实现并行分析和双层 Reviewer 回路**

```python
@listen("analysis_ready")
def run_parallel_analysis(self):
    # fundamental / valuation / event
    ...

@listen(run_parallel_analysis)
def review_data_quality(self):
    ...
```

- [ ] **步骤 5：运行测试验证通过**

运行：`pytest tests/test_flow_routing.py -v`
预期：PASS

- [ ] **步骤 6：Commit**

```bash
git add src/multi_agent/flows/market_review_flow.py src/multi_agent/crew.py tests/test_flow_routing.py
git commit -m "refactor: move orchestration from sequential crew to market review flow"
```

### 任务 8：抽离 runtime 并适配 CLI / API

**文件：**
- 创建：`src/multi_agent/runtime.py`
- 创建：`src/multi_agent/api.py`
- 修改：`src/multi_agent/main.py`
- 测试：`tests/test_main_cli.py`

- [ ] **步骤 1：编写 CLI 兼容测试**

```python
def test_run_uses_runtime_and_new_artifact_root(monkeypatch, tmp_path):
    ...
    assert (tmp_path / "src" / "multi_agent" / "artifacts").exists()
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_main_cli.py -v`
预期：FAIL，`main.run()` 仍直接控制所有运行步骤

- [ ] **步骤 3：实现 runtime 入口**

```python
def run_research(company_name: str, company_ticker: str, *, save_to_watchlist: bool = False) -> dict[str, Any]:
    state = MarketReviewFlow().kickoff(inputs={...})
    return persist_run_outputs(state)
```

- [ ] **步骤 4：让 main.py 只保留参数解析与错误展示**

```python
def run():
    args = _build_parser().parse_args()
    runtime.run_cli(args)
```

- [ ] **步骤 5：运行测试验证通过**

运行：`pytest tests/test_main_cli.py -v`
预期：PASS

- [ ] **步骤 6：Commit**

```bash
git add src/multi_agent/runtime.py src/multi_agent/api.py src/multi_agent/main.py tests/test_main_cli.py
git commit -m "refactor: introduce runtime entrypoints for cli and api"
```

### 任务 9：升级评估、报告结构与 watchlist

**文件：**
- 修改：`src/multi_agent/evaluation.py`
- 修改：`src/multi_agent/recommendation.py`
- 修改：`src/multi_agent/watchlist.py`
- 测试：`tests/test_reviewers.py`
- 测试：`tests/test_main_cli.py`

- [ ] **步骤 1：编写 Reviewer 与 recommendation 测试**

```python
def test_recommendation_is_blocked_when_market_unresolved():
    recommendation = build_structured_recommendation(...)
    assert recommendation["stance"] == "hold"
    assert recommendation["validation"]["market_label"] == "UNRESOLVED"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_reviewers.py tests/test_main_cli.py -v`
预期：FAIL，当前 recommendation 未绑定市场验证状态和 reviewer 结果

- [ ] **步骤 3：扩展 evaluation 指标**

```python
metrics["market_validation_status"] = state.market_validation.market_label
metrics["review_failures_count"] = len(blocking_issues)
metrics["evidence_count"] = len(state.evidence_ledger)
```

- [ ] **步骤 4：扩展 recommendation 结构**

```python
{
  "stance": "...",
  "trust_score": ...,
  "validation": {
    "market_label": "...",
    "data_quality_review": "pass",
    "logic_review": "pass"
  }
}
```

- [ ] **步骤 5：运行测试验证通过**

运行：`pytest tests/test_reviewers.py tests/test_main_cli.py -v`
预期：PASS

- [ ] **步骤 6：Commit**

```bash
git add src/multi_agent/evaluation.py src/multi_agent/recommendation.py src/multi_agent/watchlist.py tests/test_reviewers.py tests/test_main_cli.py
git commit -m "feat: tie evaluation and recommendation to market validation and reviewers"
```

### 任务 10：全量验证与文档收尾

**文件：**
- 修改：`.env.example`
- 修改：`README.md`
- 测试：`tests/test_settings.py`
- 测试：`tests/test_market_validation.py`
- 测试：`tests/test_tavily_tool.py`
- 测试：`tests/test_official_sec_tools.py`
- 测试：`tests/test_reviewers.py`
- 测试：`tests/test_flow_routing.py`
- 测试：`tests/test_main_cli.py`

- [ ] **步骤 1：更新 env 样例**

```dotenv
FAST_MODEL=qwen-v4-flash
DEEP_MODEL=qwen-v4-pro
REVIEW_MODEL=qwen-v4-pro
TAVILY_API_KEY=your_tavily_api_key
SEC_API_EMAIL=analyst@example.com
```

- [ ] **步骤 2：运行全量测试**

运行：`pytest tests/test_settings.py tests/test_market_validation.py tests/test_tavily_tool.py tests/test_official_sec_tools.py tests/test_reviewers.py tests/test_flow_routing.py tests/test_main_cli.py -v`
预期：PASS

- [ ] **步骤 3：运行 CLI 冒烟验证**

运行：`uv run python -m multi_agent.main --company-name "Apple Inc." --company-ticker AAPL`
预期：在 `src/multi_agent/artifacts/runs/` 下生成 run 目录及结构化产物

- [ ] **步骤 4：Commit**

```bash
git add .env.example README.md tests
git commit -m "docs: finalize env and validation for market review flow"
```
