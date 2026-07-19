# AgentVest 完整报告流水线设计

## 1. 背景与目标

AgentVest 已经具备 7 Agent / 7 Task、Flow、Gate、回流和标准产物归档，但当前 Apple 实例仍暴露出三个系统性问题：财务字段缺少期间与来源语义、终态与报告文本可能不一致、结构化报告依赖自由格式 Markdown 标题进行二次猜测。

本次改造采用“契约优先”方案，保留 CrewAI 作为 Agent 执行底座，由 `MarketReviewFlow` 继续承担控制面。完成条件不是文件存在，而是 Apple Inc. / AAPL 的真实运行达到 `passed + formal_report`，同时 Markdown、结构化推荐、结构化完整报告和审查结果一致且内容完整。

## 2. 设计原则

- 证据先于文本：数值、期间、单位和来源必须先进入结构化证据契约，Agent 才能使用。
- Gate 判断语义而非字段存在性：正式放行要求期间对齐、来源可追溯且无关键冲突。
- 终态单一真值：`final_decision.json` 是唯一终态，其他交付物只能由它投影。
- 报告结构显式：Writer 产生标准章节和结构化文档，不依赖事后猜测标题。
- 失败可诊断、回流可执行：每个失败必须给出机器可读的缺失项和修复动作。
- 不为通过而降低标准：证据不足时允许受限交付，但不得伪造或用模型猜测数据。

## 3. 核心架构

### 3.1 结构化证据层

新增 `ResearchEvidenceBundle` 作为分析阶段的唯一事实输入。它至少包含：

- 公司身份、ticker、市场、交易所和解析置信度。
- 财务事实列表，每个事实包含 `field_name`、`value`、`unit`、`period_start`、`period_end`、`fiscal_year`、`fiscal_period`、`form`、`accession`、`filed_at`、`source_url` 和 `source_tag`。
- 市场快照，包含股价、时间戳、来源和与财务分母的时间关系。
- 事件证据，包含事件时间、来源 URL、来源类型、置信度和是否独立交叉验证。
- 工具健康状态、原始 artifact 引用和数据缺口。

`EvidenceNormalizer` 负责从 SEC Company Facts、filing HTML、市场报价和 Tavily 响应构造该 bundle。任何缺少期间元数据的财务值只能作为受限证据，不能进入正式估值计算。

### 3.2 分析与审查层

分析 Agent 消费同一个 `ResearchEvidenceBundle`，输出观点、计算过程和 claim-source 映射。Reviewer 输出严格的 `ReviewContract`，包括覆盖率、关键冲突、策略违规、工具健康、交付资格、回流原因和修复动作。

`ConfidenceGatePolicy` 只消费结构化 bundle 和 review contract。Markdown 解析仅保留为旧 artifact 兼容路径，不能决定新运行的正式终态。

正式放行同时要求：

- 必需字段存在且可解析。
- 估值相关分子和分母期间兼容。
- 每个关键字段存在可审计来源。
- 无关键跨来源冲突、市场策略违规或无证据关键结论。
- Reviewer 明确允许 `formal_report`。

### 3.3 报告与交付层

Flow 在 Gate 后创建不可变的 `ReportGenerationContext`，包含 company、evidence bundle、review contract、锁定的 `report_mode` 和允许使用的 claims。

Writer 生成 `ReportDocument`，其标准章节为：

- 执行摘要
- 公司与业务概览
- 近期事件与催化剂
- 财务分析与估值
- 关键风险
- 投资结论
- 来源索引

Markdown 由 `ReportDocument` 渲染；`06_structured_recommendation.json` 和 `07_structured_report.json` 直接从同一对象生成。旧 Markdown 提取器只用于重建历史产物。

`DeliveryValidator` 在落盘前校验终态、报告模式、章节、引用、结构化字段和 stance。校验失败时本轮状态为 `failed`，不得写成成功或 passed。

## 4. 数据流与回流

1. CLI 解析公司并创建 run 目录。
2. 工具层获取原始外部数据并记录调用状态。
3. `EvidenceNormalizer` 构建、校验并持久化 evidence bundle。
4. 分析 Agent 基于 bundle 生成分析 artifact。
5. Reviewer 生成 `ReviewContract`。
6. Gate 返回 `passed`、`rerun`、`evidence_limited` 或 `blocked`。
7. `rerun` 携带缺失字段、失败来源和目标 Agent，在预算内定向补证。
8. 终态确定后，Writer 生成 `ReportDocument`。
9. Renderer 生成 Markdown 和两个结构化 JSON。
10. `DeliveryValidator` 校验全部交付物，CLI 写入最终指标与 `final_decision.json`。

## 5. 错误处理

- Tavily 单次失败可降级；来源数量或独立性不足时不得正式放行。
- SEC 权限、主体解析或 filing 获取失败时立即失败，并保留诊断产物。
- 市场报价失败可重试或转为证据受限，禁止模型猜测股价。
- 财务期间不明确、跨 artifact 数值矛盾或来源缺失时触发定向回流；预算耗尽后只能受限或阻断。
- Writer 输出与锁定 `report_mode` 不一致时由 `DeliveryValidator` 拒绝。
- 所有异常路径保留标准文件名、错误类型、服务名、HTTP 状态和可执行修复建议。

## 6. 测试策略

- 单元测试：期间选择、来源追踪、证据完整性、Gate 决策、报告模式和结构化渲染。
- 集成测试：使用固定 SEC、市场报价、Tavily 和 LLM 响应，离线验证完整 Flow；测试不得意外调用真实网络。
- 回归测试：覆盖“受限正文却标为 passed”“正文有内容但结构化章节为空”“跨期间数据被组合估值”“Reviewer 格式变化绕过 Gate”。
- 全量回归：所有现有测试通过；不得通过删除有效断言或放宽业务标准修绿。
- 真实实例：运行 Apple Inc. / AAPL，按机器诊断持续修复和重跑。

## 7. Apple 最终验收

只有以下条件全部成立，目标才算完成：

- `final_decision` 为 `passed`。
- `final_delivery_state` 为 `formal_report`。
- 所有标准 artifact 存在、非占位且状态一致。
- Markdown 包含七个标准章节，核心结论有来源绑定。
- 结构化推荐的 `summary`、`catalysts`、`risks` 非空。
- 结构化完整报告的核心 sections 非空，validation 全部通过。
- 正式财务结论的期间、单位、来源和计算过程可审计。
- Reviewer 无关键冲突、策略违规或 unsupported critical claim。
- 全量自动化测试通过。

若真实外部数据暂时不足，系统必须先增强获取或解析能力并继续重跑；不得通过伪造数据、隐藏失败或降低 Gate 标准达成 passed。

## 8. 范围边界

本次不重写 CrewAI、不引入数据库、不扩展非美国市场数据源，也不改造前端。改动仅围绕 AAPL 正式报告所需的数据语义、控制面、报告交付和验证闭环。
