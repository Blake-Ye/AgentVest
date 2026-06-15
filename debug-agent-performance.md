# [OPEN] agent-performance

## 目标

使用 `cProfile`、`py-spy` 等工具分析当前 Agent 系统的运行时性能瓶颈，判断主要耗时来自编排、网络请求、LLM 推理还是本地后处理。

## 现象

- 用户希望定位现有 Agent 系统的性能瓶颈。
- 当前项目是多 Agent 顺序工作流，存在明显外部 API 与 LLM 调用。

## 初始假设

1. 顺序执行的 Task 链路导致总耗时接近各阶段耗时之和。
2. 搜索、SEC、Company Facts 等外部 HTTP 请求占主要墙钟时间。
3. 最终报告生成阶段的 LLM 推理和长上下文处理占主要 CPU / 等待时间。
4. `HTTP_TIMEOUT_SECONDS` 与 `MAX_HTTP_RETRIES` 放大了慢请求的尾延迟。
5. 本地 JSON / Markdown 写文件与 recommendation 后处理并不是主要瓶颈。

## 证据计划

- 阅读入口和运行链路，确认最合适的剖析点。
- 用 `cProfile` 跑一次代表性入口，获取函数级累计耗时。
- 用 `py-spy` 获取采样视图，区分 CPU 热点与 I/O 等待。
- 结合已有 `latest_run_metrics.json` 对比任务级耗时。

## 当前状态

- 已完成运行时证据收集，尚未修改业务逻辑。

## 已收集证据

### 1. cProfile：真实主流程

- 运行入口：`python -m cProfile -o artifacts/profiles/apple-run.prof -m multi_agent.main --company-name "Apple Inc." --company-ticker AAPL`
- 采样结果：
  - 总耗时约 `237.9s`
  - `openai chat completions.create` 累计约 `216.1s`
  - `crewai.utilities.agent_utils.get_llm_response` 累计约 `216.4s`
  - `task.execute_sync / agent.execute_task` 基本都包裹在上述等待时间内
  - 大量时间显示在 `threading.wait / lock.acquire`，说明主线程主要在等待异步/网络结果而不是本地 CPU 计算

### 2. cProfile：watchlist 重建路径

- 运行入口：`python -m cProfile -o artifacts/profiles/rebuild.prof -m multi_agent.main --watchlist-rebuild`
- 采样结果：
  - 总耗时约 `1.9s`
  - 主要耗时在 import：`crewai`、`investment_tools.py`、`resolver.py`
  - 实际业务逻辑与 JSON 重建本身只占很小一部分

### 3. 任务级运行指标

- Apple 真实运行：`233.995s`
  - `market_intelligence_task`: `49.504s`
  - `filing_review_task`: `56.744s`
  - `financial_analysis_task`: `53.284s`
  - `investment_report_task`: `74.441s`
- Tencent 历史运行：`216.292s`
  - `market_intelligence_task`: `73.988s`
  - `filing_review_task`: `34.228s`
  - `financial_analysis_task`: `48.419s`
  - `investment_report_task`: `59.629s`

### 4. py-spy 状态

- `py-spy 0.4.2` 已确认存在于 `MultiAgent` 环境。
- 在 macOS 上执行 `py-spy record ...` 时收到错误：`This program requires root on OSX.`
- 结论：当前会话无法在无提升权限的情况下拿到 `py-spy` 火焰图。

## 假设结论

1. **确认**：顺序执行链路是总耗时线性累加的重要原因。
2. **部分确认**：外部请求确实拖慢流程，但从 `cProfile` 看更大的总时间沉淀在 LLM 调用等待。
3. **确认**：LLM 推理 / 远程 completion 调用是主瓶颈。
4. **确认**：失败率和重试会放大尾延迟，Google Search 失败率高达 `50%`。
5. **确认**：本地文件写入与 recommendation 后处理不是主要瓶颈。

## 当前结论

- 主流程慢的核心不是 Python 本地计算，而是：
  1. 顺序编排
  2. 远程 LLM completion
  3. 外部搜索/SEC 请求及失败重试
- 本地“结果重建”路径已经足够快，优化重点不应放在 JSON/Markdown 落盘上。
