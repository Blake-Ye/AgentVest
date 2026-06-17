# 项目架构说明

## 1. 项目目标

本项目是一个基于 CrewAI 的自动化投研工作流。用户输入公司名称和股票代码后，
系统会按顺序完成市场情报收集、SEC 文件复核、财务分析和最终投资备忘录撰写。

## 2. 核心目录

```text
multi_agent/
├── src/multi_agent/
│   ├── config/
│   │   ├── agents.yaml
│   │   └── tasks.yaml
│   ├── tools/
│   │   ├── __init__.py
│   │   └── investment_tools.py
│   ├── crew.py
│   ├── finance.py
│   ├── main.py
│   └── settings.py
├── tests/
├── docs/
│   ├── architecture/
│   │   ├── project_architecture.md
│   │   ├── project_er_diagram.mmd
│   │   └── project_flow_diagram.mmd
│   ├── evaluation/
│   ├── operations/
│   ├── guides/
│   └── archive/
├── outputs/
│   ├── artifacts/
│   ├── benchmark/
│   ├── profiles/
│   └── watchlist/
├── README.md
├── AGENTS.md
├── .env.example
└── pyproject.toml
```

## 3. 分层职责

- `main.py`
  负责 CLI 入口、运行时环境准备、工作流入参组装、统一错误退出。
- `settings.py`
  负责 `.env` 加载和配置收口，是所有工具与 Crew 的单一配置来源。
- `crew.py`
  负责定义 3 个 Agent、4 个 Task，以及它们的顺序编排关系。
- `runtime.py`
  负责准备本地 CrewAI 运行时目录，避免把 SQLite 和运行时缓存写入系统用户目录。
- `config/agents.yaml`
  负责角色、目标、背景设定，强调中文输出和职责边界。
- `config/tasks.yaml`
  负责任务描述、预期产出和工作流约束。
- `tools/investment_tools.py`
  负责搜索、SEC、财务指标、PDF 提取、文件写入等外部能力。
- `finance.py`
  负责纯计算逻辑，降低 Prompt 中硬算指标的风险。
- `tests/`
  负责验证配置加载、工具行为、Crew 结构、CLI 和财务逻辑。

## 4. 运行时主链路

1. `main.py` 从命令行读取公司名和股票代码。
2. `settings.py` 从 `.env` 组装模型、搜索、SEC 等配置。
3. `main.py` 把参数传给 `MultiAgent().crew().kickoff(...)`。
4. `crew.py` 按 `Process.sequential` 顺序执行四个任务。
5. 每个 Task 使用自己的 Agent 和工具集生成中间产出。
6. 最终由报告撰写 Agent 输出投资备忘录。

## 5. 当前 Agent 与 Task 关系

- `information_gathering_analyst`
  负责 `market_intelligence_task` 和 `filing_review_task`。
- `financial_statement_analyst`
  负责 `financial_analysis_task`。
- `report_writing_analyst`
  负责 `investment_report_task`。
- `investment_report_task`
  依赖前三个任务的 context，不再读取未明确提供的本地文件。

## 6. 关键设计约束

- 采用 YAML-first 方式定义 Agent 和 Task，减少业务逻辑散落在 Python 代码中。
- 最终报告基于前序任务 context 聚合，避免模型自行猜测本地文件路径。
- 本地 PDF 工具只在 `LOCAL_FILING_PDF_PATH` 真实存在时暴露。
- `401/403/429` 等外部服务错误会直接终止流程并给出中文提示。

## 7. 图文件说明

- `docs/architecture/project_flow_diagram.mmd`
  适合展示工作流和模块调用链。
- `docs/architecture/project_er_diagram.mmd`
  适合展示配置、Crew、Agent、Task、Tool、Artifact 的结构关系。

如果你要放进 Figma / FigJam，直接把这两个 Mermaid 文件内容复制到 Mermaid 插件中即可。

## 8. 测试工程师建议先验什么

如果目标是快速判断工程是否可交付、可定位、可回归，建议优先验证：

1. `tests/` 是否全绿
2. `outputs/artifacts/<company>__<ticker>/<timestamp>/` 是否按规范生成
3. `04_investment_report.md`、`06_structured_recommendation.json`、`07_structured_report.json` 是否齐全
4. `latest_run_metrics.json` 是否能清楚表达成功/失败状态

更具体的检查项见 `docs/guides/testing_engineer_quickcheck_zh.md`。
