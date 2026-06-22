# 项目架构说明

## 1. 当前定位

AgentVest 当前不是“单纯的顺序 Crew Demo”，而是一个以 CrewAI 为执行底座、
以 `MarketReviewFlow` 为控制平面的自动化投研系统。它的目标是把公司解析、
市场验证、研究分析、质量审查、最终交付收口为一条可解释、可阻断、可回流的
工程化链路。

当前对外口径应统一为：

- `7 Agent / 7 Task`
- `Flow + Gate + Recurrent`
- `结构化 summary 优先，Markdown 文案仅作为 fallback`
- `blocked` 状态下只允许输出阻断说明，不允许继续交付正式投资建议

## 2. 架构快照

```text
AgentVest/
├── src/multi_agent/
│   ├── config/
│   │   ├── agents.yaml
│   │   └── tasks.yaml
│   ├── core/
│   │   ├── artifact_paths.py
│   │   ├── confidence_gate.py
│   │   ├── market.py
│   │   ├── model_routing.py
│   │   ├── review_contracts.py
│   │   └── state.py
│   ├── flows/
│   │   └── market_review_flow.py
│   ├── tools/
│   │   ├── investment_tools.py
│   │   ├── market_validation.py
│   │   ├── official_sec.py
│   │   ├── review_tools.py
│   │   └── tavily_search.py
│   ├── api.py
│   ├── crew.py
│   ├── evaluation.py
│   ├── finance.py
│   ├── main.py
│   ├── recommendation.py
│   ├── resolver.py
│   ├── runtime.py
│   ├── settings.py
│   └── watchlist.py
├── docs/
│   ├── project_architecture.md
│   ├── project_er_diagram.mmd
│   └── project_flow_diagram.mmd
├── tests/
├── README.md
└── pyproject.toml
```

## 3. 分层职责

### 3.1 控制层

- `runtime.py`
  负责在运行时决定走 `Crew` 还是 `Flow`，并准备本地化 CrewAI 运行环境。
- `flows/market_review_flow.py`
  负责状态机编排、Gate 路由、rerun 预算控制和最终交付收口，是当前主控制面。
- `core/confidence_gate.py`
  负责把结构化审查摘要收敛为 `passed / rerun / blocked` 三态决策。
- `core/review_contracts.py`
  定义 `ReviewToolSummary` 与 `GateDecision` 契约，保证 Gate 输入输出稳定。

### 3.2 执行层

- `crew.py`
  定义 7 个 Agent、7 个 Task 和顺序执行的 Crew，是分析阶段的执行底座。
- `config/agents.yaml`
  定义角色、目标、边界和中文输出要求。
- `config/tasks.yaml`
  定义每个任务的输入约束、预期产物和交付边界。

### 3.3 能力层

- `tools/market_validation.py`
  负责市场归属判断与工具白名单边界控制。
- `tools/tavily_search.py`
  负责事件、新闻、指引相关的外部搜索。
- `tools/investment_tools.py`
  负责财务指标、官方 SEC 数据、PDF 文本提取等分析能力。
- `tools/review_tools.py`
  负责证据覆盖、字段完整度、跨源一致性、工具越界等审查能力。

### 3.4 交付层

- `main.py`
  负责 CLI 入口、运行指标、产物写入、失败态收口和结构化 JSON 输出。
- `recommendation.py`
  负责从最终 Markdown 报告提取 `summary / stance / catalysts / risks`，
  生成结构化推荐和结构化报告快照。
- `evaluation.py`
  负责任务完成记录与运行时评估辅助。

## 4. 7 Agent / 7 Task

### 4.1 Agent 编制

1. `market_validation_analyst`
   在所有分析之前先确认市场标签和工具可用边界。
2. `event_guidance_analyst`
   基于新闻、公告和管理层指引生成事件分析。
3. `fundamental_analyst`
   基于 filings 与 company facts 生成基本面分析。
4. `quant_valuation_analyst`
   基于财务指标和量化信号生成估值分析。
5. `data_quality_reviewer`
   调用 reviewer tools，对分析结果做结构化质量审查。
6. `report_writing_analyst`
   只在 analysis gate 通过后撰写正式投资备忘录。
7. `logic_compliance_reviewer`
   在最终交付前审查逻辑一致性、证据边界和合规表达。

### 4.2 Task 编排

1. `market_validation_task`
   输出 `00_market_validation.md`
2. `market_intelligence_task`
   输出 `01_market_intelligence.md`
3. `filing_review_task`
   输出 `02_filing_review.md`
4. `financial_analysis_task`
   输出 `03_financial_analysis.md`
5. `data_quality_review_task`
   输出 `08_data_quality_review.md`
6. `investment_report_task`
   输出 `04_investment_report.md`
7. `logic_compliance_review_task`
   输出 `09_logic_compliance_review.md`

这里需要注意两点：

- `Crew` 内部仍然是 `Process.sequential` 顺序执行，但整个系统的“是否继续”
  已经不只由顺序决定，而是由 `Flow` 中的 Gate 决策控制。
- `investment_report_task` 和 `logic_compliance_review_task` 虽然仍存在于 Crew 中，
  但当前对外叙事应强调：正式交付资格由 `MarketReviewFlow` 控制，而不是单靠
  Crew 顺序跑完就默认可交付。

## 5. Flow + Gate + Recurrent

### 5.1 主链路

当前 `MarketReviewFlow` 的主控制链路可以概括为：

1. `validate_market`
2. `run_analysis`
3. `review_analysis`
4. `apply_analysis_gate`
5. `route_after_analysis_gate`
6. `rerun_analysis_if_needed`（当 Gate 返回 `rerun` 且预算未耗尽时）
7. `write_report`
8. `review_report`
9. `finalize_delivery` 或 `finalize_blocked_delivery`

### 5.2 Gate 三态

`ConfidenceGatePolicy` 当前默认阈值如下：

- `evidence_coverage_ratio >= 0.80`
- `financial_coverage_score >= 0.80`
- `trust_score >= 75`

Gate 的三种结果分别代表：

- `passed`
  允许进入正式报告写作与最终交付。
- `rerun`
  说明问题理论上可修复，Flow 会在预算内触发一次定向回流。
- `blocked`
  说明存在不可接受风险，或 `rerun` 预算已经耗尽，系统必须停止正式建议交付。

### 5.3 Recurrent 回流机制

当前回流不是“重跑整条链路”，而是有预算和模型覆盖策略的定向重做：

- rerun key 为 `analysis`
- `event_guidance_analyst`
- `fundamental_analyst`
- `quant_valuation_analyst`

在 rerun 时会切换到更高模型层级，并重新执行分析阶段；如果回流后仍未达到
放行条件，Flow 会把终态收敛为 `blocked`，而不是无限循环。

## 6. 结构化 Summary 优先

这是当前架构中最关键的控制约束之一。

`MarketReviewFlow._default_analysis_gate()` 会优先从分析结果中读取
`analysis_review_summary`，并尝试反序列化为 `ReviewToolSummary`。只要结构化摘要
存在且可解析，就直接交给 `ConfidenceGatePolicy` 评估。

只有在结构化 summary 缺失时，系统才会回退到读取 `08_data_quality_review.md`
里的 Markdown 阻断标记，例如：

- `必须修复（Must Fix）`
- `Blockers`
- `P0`
- `Gate 阻断风险`

这意味着当前正确叙事应当是：

- 主控制面依据结构化审查结果做路由
- Markdown 文案只承担兜底兼容职责
- 文档、README、面试口径都不应再把“识别 reviewer Markdown 文案”描述为主机制

## 7. 结构化交付与失败态语义

除了 Markdown 产物外，系统还会在交付阶段生成面向后续产品化消费的结构化产物：

- `06_structured_recommendation.json`
  保存 `summary`、`stance`、`trust_score`、`catalysts`、`risks` 等摘要信息。
- `07_structured_report.json`
  保存更完整的结构化报告快照，包括章节拆分、引用 URL 和校验信息。
- `evaluation_summary.json`
  保存本轮运行的评估汇总。

其中 `recommendation.py` 会优先从最终报告中提取：

- `summary`
- `stance / stance_label`
- `catalysts`
- `risks`
- `trust_score / trust_level / trust_summary`

失败态必须遵守以下语义：

- `blocked` 时允许写阻断说明版报告
- `blocked` 时禁止伪装成正式投资建议
- `structured_report` 会把 `status` 与 `final_decision` 写成 `blocked`
- `structured_report["stance"]` 必须收紧为 `blocked`
- `blocking_reasons` 必须保留下来，供 CLI、前端或后续排障消费

## 8. 关键约束

1. `YAML-first`
   角色与任务定义尽量留在 YAML，Python 负责编排、能力接线和规则实现。
2. `Flow 决定交付资格`
   是否生成正式建议由 Gate 决定，不再由“Task 是否跑完”隐式决定。
3. `结构化优先`
   Gate 判断优先消费 reviewer tools 的结构化 summary，而不是 Markdown 文案。
4. `阻断态可交付但不可冒充成功`
   `blocked` 状态必须保留可审计产物，但不得输出误导性的正式建议。
5. `本地 PDF 受控暴露`
   只有在本地 PDF 路径真实存在时，相关工具才会注入到 Agent。
6. `回流预算有限`
   Recurrent 只能在预算内定向重做，预算耗尽后自动收敛为 `blocked`。

## 9. 对外讲述建议

如果要在 README、简历、面试或 Demo 中讲清楚这个项目，建议统一使用下面这句：

> AgentVest 是一个基于 CrewAI 的多 Agent 自动化投研系统，底层用 7 Agent / 7 Task
> 执行分析，顶层用 `MarketReviewFlow + ConfidenceGatePolicy` 做 `passed / rerun / blocked`
> 三态控制；Gate 优先消费结构化 reviewer summary，并通过受控 recurrent loop
> 在预算内定向回流，最终输出 Markdown 报告与结构化推荐快照。

## 10. 图文件说明

- `docs/project_flow_diagram.mmd`
  适合展示 Flow、Gate、Recurrent 和最终交付的控制链路。
- `docs/project_er_diagram.mmd`
  适合展示 Agent、Task、Tool、Artifact 和 Gate 契约之间的结构关系。

如果后续继续更新 Mermaid 图，请优先保证它们与本文档和 README 的口径同步，
避免再次出现“代码已是 7 Agent / Flow 控制，但文档仍停留在早期 Crew 版本”的漂移。
