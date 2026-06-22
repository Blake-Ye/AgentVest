# 结构化 Gate 驱动 Recurrent 交付设计说明

**目标**

将当前 `market_review_flow.py` 中主要依赖 Markdown 文案识别的 gate 逻辑，升级为：

- `review_tools` 产出结构化审查结果
- `ConfidenceGatePolicy` 作为唯一 gate 判定源
- `Flow` 基于结构化 `GateDecision` 决定 `passed / rerun / blocked`
- 只有 `passed` 才放行正式报告
- 若最终仍未通过，仅输出阻断说明版报告与 blocked 结构化产物

该设计只服务当前 CLI 主路径和现有 Flow 编排，不引入新的 Agent 角色、不新增外部数据源，也不做全仓库级重构。

---

## 背景

当前系统已经具备以下能力：

1. `MarketReviewFlow` 已有分析执行、analysis gate、rerun、report review 与最终交付骨架
2. `review_tools.py` 已定义证据覆盖率、财务字段完整度、跨源冲突、市场策略违规等结构化指标
3. `confidence_gate.py` 已定义阈值策略，可从 `ReviewToolSummary` 计算 `GateDecision`
4. 仓库已有 `test_flow_routing.py` 与 `test_review_gate.py`，可以支撑行为级回归验证

但当前主问题仍然存在：

- analysis gate 默认仍主要通过读取 `08_data_quality_review.md` 并匹配阻断文案来判定是否 blocked
- report gate 默认仍通过识别报告中的“阻断”字样来判断失败态
- `ConfidenceGatePolicy` 尚未成为 Flow 的唯一控制平面
- recurrent 流程可以“重跑”，但重跑是否成功、是否应放行正式报告，还没有完全由结构化 gate 控制

结果是：系统设计目标已经是“证据驱动硬门控”，但主链路里仍残留“文案识别驱动流程”的不稳定路径。

---

## 设计原则

### 1. 结构化 gate 是唯一真相源

- `passed / rerun / blocked` 必须由结构化 gate 决定
- reviewer Markdown 只负责向人解释，不再作为流程分支的主判据
- 若文案与结构化结果冲突，必须以结构化结果为准

### 2. 正式报告只能在通过 gate 后生成

- 未通过 gate 的内容不得进入正式投资建议路径
- 若 analysis gate 未通过，不得继续生成正式报告
- 若最终状态 blocked，只能产出阻断说明版报告，不输出正式投资建议语义

### 3. recurrent 的目标是提高通过率，不是掩盖失败

- recurrent 允许基于结构化失败原因回流重做
- recurrent 必须有明确预算，不能无限重试
- 若回流后仍未通过，应稳定收敛为 blocked，而不是继续冒险放行

### 4. 小步接线，优先主路径闭环

- 本次只做“gate -> recurrent -> report delivery”主链路收口
- 不在本次引入复杂多阶段多 crew 架构
- 优先复用现有 `Flow`、`state`、`review_tools` 与 `confidence_gate`

### 5. 以“实习项目基准线”作为收口标准

本次设计的完成标准不是“生产级金融系统”，而是达到以下基准线：

- 面试时可以清楚说明系统如何避免未通过审查的内容进入正式交付
- 关键行为能用自动化测试证明，而不是靠口头解释
- 失败态和成功态的产物边界清楚、可复盘、可审计

---

## 目标行为

### 场景 1：首次 analysis gate 通过

1. 执行分析阶段
2. reviewer tools 产出结构化 `ReviewToolSummary`
3. `ConfidenceGatePolicy` 计算 gate
4. gate 结果为 `passed`
5. 进入正式报告生成与最终交付

最终结果：

- 生成正式投资报告
- 生成正式 recommendation/report JSON
- 最终状态为 `passed`

### 场景 2：首次 analysis gate 未通过，但回流后通过

1. 执行分析阶段
2. gate 结果为 `rerun`
3. 系统消耗一次 rerun budget，并按预定义覆盖策略提升相关 Agent 到更高模型层级或补做分析
4. 再次产出结构化 `ReviewToolSummary`
5. gate 结果转为 `passed`
6. 进入正式报告生成与最终交付

最终结果：

- 首次结果不放行
- recurrent 完成后生成正式报告
- 最终状态为 `passed`

### 场景 3：analysis gate 多次未通过，预算耗尽

1. 执行分析阶段
2. gate 结果为 `rerun`
3. 回流一次或多次后仍未达到阈值
4. rerun budget 耗尽
5. 系统直接收敛为 `blocked`

最终结果：

- 不生成正式投资建议
- 输出阻断说明版报告
- 输出 blocked recommendation/report JSON
- 最终状态为 `blocked`

### 场景 4：analysis gate 通过，但 report gate 未通过

1. analysis gate 为 `passed`
2. 生成正式报告草稿
3. report review 发现缺少风险披露、逻辑不一致或合规问题
4. report gate 返回 `blocked`

最终结果：

- 不放行正式投资建议
- 输出阻断说明版报告或 blocked 报告态
- 最终状态为 `blocked`

---

## 控制面设计

### 单一 gate 判定契约

当前 `GateDecision` 只支持：

- `passed`
- `blocked`

为了支撑 recurrent，本次设计建议将控制面语义扩展为：

- `passed`
- `rerun`
- `blocked`

约束：

- `passed`：允许进入下一阶段
- `rerun`：允许回流重做，但不能放行正式报告
- `blocked`：禁止继续正式交付，直接进入阻断收口

建议修改：

- `GateDecision.final_decision` 从二元值扩展为三元控制值
- `ResearchRunState.final_decision` 仍保持最终交付语义：`passed | blocked`
- `rerun` 只作为流程中间控制状态，不作为最终交付状态

### review summary 与 gate policy

`ReviewToolSummary` 继续承载结构化审查指标，包括：

- `evidence_coverage_ratio`
- `financial_coverage_score`
- `critical_conflict_count`
- `market_policy_violations`
- `unsupported_critical_claims`
- `blocking_reasons`

`ConfidenceGatePolicy` 根据阈值输出 `GateDecision`。

本次建议的行为是：

- 明确区分“可回流修复”和“必须直接阻断”的问题
- 对以下情形优先返回 `blocked`：
  - `critical_conflict_count > 0`
  - `market_policy_violations > 0`
  - 明确存在不可放行的结构化阻断原因
- 对以下情形可返回 `rerun`：
  - 证据覆盖率不足
  - 财务字段完整度不足
  - 信任分不足但仍具备补做空间

这样可以把 recurrent 绑定到“可修复的质量缺口”，而不是一律失败或一律阻断。

---

## Flow 路由设计

### 当前问题

当前 `MarketReviewFlow` 的 analysis gate 路由本质上是：

1. `analysis_result`
2. 读取物化 review Markdown
3. 识别阻断文案
4. `passed -> write_report`
5. `blocked + 有预算 -> rerun`
6. `blocked + 无预算 -> finalize`

这使得 gate 的根因是文案格式，而不是结构化证据。

### 新路由

建议将 Flow 调整为：

1. `run_analysis`
2. `collect_review_summary`
3. `apply_analysis_gate`
4. `route_after_analysis_gate`
5. `passed -> write_formal_report`
6. `rerun -> rerun_analysis_if_needed`
7. `blocked -> finalize_blocked_delivery`

对于 report 阶段：

1. `write_formal_report`
2. `collect_report_review_summary`
3. `apply_report_gate`
4. `passed -> finalize_passed_delivery`
5. `blocked -> finalize_blocked_delivery`

### recurrent 策略

analysis recurrent 继续保留现有预算控制：

- `rerun_budget["analysis"]`

首次 rerun 可复用当前策略：

- `event_guidance_analyst -> deep`
- `fundamental_analyst -> deep`
- `quant_valuation_analyst -> deep`

本次不扩展复杂多轮差异化回流策略，只保证：

- 有预算时才 rerun
- rerun 后重新做结构化 gate
- budget 耗尽后强制 blocked

---

## 产物策略

### 正式报告路径

只有同时满足以下条件才允许正式交付：

1. analysis gate `passed`
2. report gate `passed`

正式交付包括：

- `04_investment_report.md` 正式投资备忘录
- `06_structured_recommendation.json` 正式 recommendation
- `07_structured_report.json` 正式 report snapshot

### 阻断报告路径

当最终状态为 `blocked` 时：

- 不输出正式投资建议语义
- 必须输出阻断说明版 Markdown 报告
- 必须输出 blocked 状态的 recommendation/report JSON
- `latest_run_metrics.json` 中最终状态必须与 Markdown/JSON 一致

### 文案的角色

本次设计保留阻断说明版文案，但它的角色变更为：

- 对人类用户解释为什么 blocked
- 提供审计与复盘线索

而不是：

- 反向决定系统是否 blocked

---

## 模块分工

### `src/multi_agent/core/review_contracts.py`

职责：

- 定义 `ReviewToolSummary`
- 定义 `GateDecision`
- 明确控制面字段和允许取值

不负责：

- 具体阈值计算
- Flow 路由

### `src/multi_agent/core/confidence_gate.py`

职责：

- 根据 `ReviewToolSummary` 计算 gate 结果
- 明确 `rerun` 与 `blocked` 的判定边界

不负责：

- 从文件读取 Markdown
- 写入最终报告

### `src/multi_agent/flows/market_review_flow.py`

职责：

- 编排 analysis / review / gate / rerun / report / finalize
- 只消费结构化 gate 结果
- 保证正式报告路径只从 `passed` 分支进入

不负责：

- 自己解析 Markdown 决定 gate
- 计算 review 指标

### `src/multi_agent/main.py`

职责：

- 根据 Flow 最终状态决定生成正式报告还是阻断说明版报告
- 统一收口 Markdown / JSON / metrics 的最终语义

不负责：

- 具体 gate 策略计算
- recurrent 路由决策

---

## 测试设计

至少覆盖以下行为级测试：

1. 结构化 summary 达标时，analysis gate 返回 `passed`
2. 结构化 summary 仅存在可修复缺口时，analysis gate 返回 `rerun`
3. 存在关键冲突或策略违规时，analysis gate 直接返回 `blocked`
4. `rerun_budget > 0` 且 gate=`rerun` 时，Flow 必须进入 rerun 分支
5. rerun 后通过时，最终生成正式报告
6. rerun 后仍失败且预算耗尽时，只生成阻断报告
7. report gate blocked 时，不放行正式 recommendation
8. reviewer Markdown 文案变化但结构化 summary 不变时，gate 结果保持不变

建议重点修改或新增：

- `tests/test_review_gate.py`
- `tests/test_flow_routing.py`
- 与最终产物生成条件相关的 CLI/主入口测试

---

## 失败与降级策略

### reviewer tools 结果缺失

- 如果结构化 review summary 缺失，不能默认放行
- 应视为 gate 失败，并进入 `rerun` 或 `blocked`

### Flow 中途异常

- 不能伪装成通过
- 必须保留 blocked 或 failed 语义，并输出可读的失败说明

### 文案与结构化结果不一致

- 一律以结构化 gate 为准
- Markdown 仅作展示，不允许反向覆盖 gate 判定

---

## 范围边界

本次不做：

- 新增 Agent 角色
- 全量多轮精细回流编排
- 新外部数据源接入
- 完整服务化/API 化重写
- 面向生产的高级调度与持久化恢复

本次只做：

- 结构化 gate 成为 Flow 的唯一控制源
- recurrent 只在结构化 gate 明确要求时触发
- formal report 与 blocked report 的边界收紧
- 自动化测试覆盖这条主链路

---

## 最终决策

本设计确认采用以下行为：

1. `review_tools -> ReviewToolSummary -> ConfidenceGatePolicy -> GateDecision` 成为唯一 gate 主链路
2. `GateDecision` 支持 `passed / rerun / blocked` 三种流程控制状态
3. 只有 `passed` 才允许进入正式报告交付
4. `rerun` 仅作为中间控制状态，受 `rerun_budget` 限制
5. 预算耗尽或存在关键阻断项时，最终收敛为 `blocked`
6. `blocked` 状态下只输出阻断说明版报告与 blocked 结构化产物
7. reviewer Markdown 文案不再决定系统是否放行
