# Agent-Driven Investment Research System

基于 `CrewAI 1.14.6` 构建的自动化投研系统。输入公司名即可运行；如果你知道股票代码，也可以额外提供。系统会自动完成：

- 市场情报搜集
- SEC 财报与监管文件检索
- 财务指标计算
- 投资报告撰写
- 可信度评分与结构化建议输出
- watchlist 持久化管理
- 中间产物归档

这个项目的目标不是做一个“能演示”的 Demo，而是做一个**新人能读懂、面试能讲清楚、后续能扩展**的多 Agent 项目。

推荐配合阅读：

- `docs/project_architecture.md`
- `docs/hr_pitch_zh.md`
- `docs/evaluation_plan_zh.md`
- `docs/PROJECT_BREAKDOWN_ROADMAP_ZH.md`
- `docs/PERFORMANCE_PROFILING_ZH.md`

## 1. 业务场景

你可以把这个系统理解成一个自动化投研小组：

- `Information Gathering Analyst`
  负责联网搜索、公司动态、竞争格局、事件催化剂和主要风险。
- `Financial Statement Analyst`
  负责 SEC 公司财务事实、关键财务指标、财报文本解释。
- `Report Writing Analyst`
  负责把前两位分析师的中间结果整合成结构化投资备忘录。

适合展示的简历场景：

- Multi-Agent 协作完成复杂投研工作流
- 将人工搜集、财报解读、报告撰写串成顺序自动化流程
- 为分析师显著减少重复检索与整理时间

你在简历里可以写成：

> 基于 CrewAI 构建 Agent 驱动的自动化投研系统，集成 Google Search、SEC API、PDF 解析与财务指标计算工具，自动生成结构化投资备忘录，实现从公司研究资料搜集到初稿输出的流程自动化。

## 2. 系统架构

### Agents

- `information_gathering_analyst`
  - 负责外部市场情报和 SEC 文件索引
- `financial_statement_analyst`
  - 负责财务事实抽取、指标计算和财报解读
- `report_writing_analyst`
  - 负责最终投资报告生成

### Tasks

- `market_intelligence_task`
  - 输出 `artifacts/01_market_intelligence.md`
- `filing_review_task`
  - 输出 `artifacts/02_filing_review.md`
- `financial_analysis_task`
  - 输出 `artifacts/03_financial_analysis.md`
- `investment_report_task`
  - 输出 `report.md`

### Tools

当前系统实际接入了 4 个默认工具，外加 1 个条件工具：

- `GoogleSearchTool`
  - 通过 Serper 或 SerpApi 调用 Google 搜索
- `SecFilingSearchTool`
  - 调用 SEC API 查询 10-K / 10-Q 等监管文件
- `SecCompanyFactsTool`
  - 调用 SEC XBRL company facts API 获取标准化财务事实
- `FinancialMetricsTool`
  - 计算毛利率、营业利润率、净利率、流动比率、债务资产比、自由现金流
- `PDFTextExtractTool`
  - 仅当本地 PDF 路径真实存在时才会启用

## 3. 目录说明

```text
multi_agent/
├── .env.example
├── README.md
├── INTERVIEW_PREP_ZH.md
├── tests/
├── src/multi_agent/
│   ├── config/
│   │   ├── agents.yaml
│   │   └── tasks.yaml
│   ├── tools/
│   │   ├── __init__.py
│   │   └── investment_tools.py
│   ├── crew.py
│   ├── evaluation.py
│   ├── finance.py
│   ├── main.py
│   └── settings.py
└── report.md
```

新人最值得先看的 8 个文件：

- `src/multi_agent/settings.py`
  - 所有运行参数和 API Key 从这里读入
- `src/multi_agent/tools/investment_tools.py`
  - 所有工具都在这里定义
- `src/multi_agent/finance.py`
  - 财务指标计算逻辑在这里
- `src/multi_agent/crew.py`
  - Agent、Task、Crew 的编排入口
- `src/multi_agent/main.py`
  - 命令行输入、运行入口、artifact 目录准备
- `src/multi_agent/evaluation.py`
  - 单次运行指标、成功率、API 失败率、可信度评分与财务字段提取状态统计
- `src/multi_agent/recommendation.py`
  - 可信度评分和结构化投资建议生成逻辑
- `src/multi_agent/watchlist.py`
  - watchlist 的本地持久化与去重更新

## 4. 代码怎么读

如果你是新手，建议按下面顺序看代码：

1. 先看 `main.py`
   - 明白输入参数怎么进来
2. 再看 `settings.py`
   - 明白 API Key 和配置怎么加载
3. 再看 `crew.py`
   - 明白 3 个 Agent 和 4 个 Task 怎么串起来
4. 再看 `agents.yaml` 和 `tasks.yaml`
   - 明白提示词层面的角色和任务描述
5. 最后看 `investment_tools.py`
   - 明白 Agent 什么时候会调用真实外部能力

一句话总结代码分层：

- `main.py` 负责“接输入”
- `evaluation.py` 负责“记指标”
- `crew.py` 负责“编排流程”
- `tools/` 负责“接外部世界”
- `finance.py` 负责“纯业务计算”
- `config/*.yaml` 负责“角色和任务描述”

## 5. 环境准备

### Python

- `>=3.10,<3.14`

### 推荐环境

当前项目已经验证过在 conda 环境 `MultiAgent` 中运行。

```bash
conda activate MultiAgent
```

### 安装依赖

如果你要和仓库依赖声明保持一致：

```bash
pip install -e ".[dev]"
```

或者在已有环境里至少保证这些包存在：

- `crewai[tools]==1.14.6`
- `requests`
- `pypdf`
- `pytest`

## 6. `.env` 配置

先复制模板：

```bash
cp .env.example .env
```

然后填写如下配置：

```env
MODEL=qwen-plus
COMPANY_RESOLVER_MODEL=qwen-plus
MARKET_IDENTIFIER_MODEL=deepseek-v4-flash
OPENAI_API_KEY=your_dashscope_or_openai_compatible_key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

SEARCH_PROVIDER=auto
SERPER_API_KEY=your_serper_api_key
SERPAPI_API_KEY=your_serpapi_api_key
SEC_API_KEY=your_sec_api_key
SEC_API_EMAIL=analyst@example.com
LOCAL_FILING_PDF_PATH=

DEFAULT_COMPANY_NAME=Apple Inc.
DEFAULT_COMPANY_TICKER=
MAX_SEARCH_RESULTS=5
HTTP_TIMEOUT_SECONDS=20
MAX_HTTP_RETRIES=3
ARTIFACTS_DIR=artifacts
FINAL_REPORT_PATH=report.md
WATCHLIST_PATH=artifacts/watchlist.json
```

运行入口会自动读取当前工作目录下的 `.env`；如果当前目录没有，再回退读取仓库根目录的 `.env`。

### 这些变量分别干什么

- `MODEL`
  - CrewAI 里使用的 LLM 名称
- `COMPANY_RESOLVER_MODEL`
  - 仅用于公司名称解析兜底的小模型；建议使用更便宜、更快的模型
- `MARKET_IDENTIFIER_MODEL`
  - 用于市场识别层的轻量模型配置；当前实现以规则识别为主，这个模型位预留给后续 flash 兜底判断
- `OPENAI_API_KEY`
  - OpenAI 兼容接口的密钥，这里可接阿里 DashScope
- `OPENAI_BASE_URL`
  - OpenAI 兼容接口地址
- `SEARCH_PROVIDER`
  - 搜索 provider，支持 `auto`、`serper`、`serpapi`
- `SERPER_API_KEY`
  - Google 搜索代理服务 Serper 的密钥
- `SERPAPI_API_KEY`
  - Google 搜索代理服务 SerpApi 的密钥
- `SEC_API_KEY`
  - `sec-api.io` 的搜索密钥
- `SEC_API_EMAIL`
  - 访问官方 SEC Company Facts API 时的联系邮箱，用于 `User-Agent`
- `LOCAL_FILING_PDF_PATH`
  - 可选，本地 PDF 财报路径；只有当这个路径真实存在时，系统才会启用 PDF 文本提取工具
- `WATCHLIST_PATH`
  - watchlist 本地存储路径，默认是 `artifacts/watchlist.json`

## 7. 运行方式

### 市场识别层

- 系统现在会先识别 ticker 所属市场，再决定是否适用 SEC 数据源
- 当前内置规则支持：
  - `AAPL` / `MSFT` 这类无后缀 ticker 识别为 `us_sec`
  - `1810.HK` 识别为 `hkex`
  - `600519.SH` / `000001.SZ` 识别为 `cn_a_share`
  - `ASML.AS` / `SAP.DE` 等识别为 `eu_listed`
- 对非 SEC ticker，`SEC Filing Search`、`SEC Company Facts`、`Financial Metrics Calculator` 不再整链路报错，而是返回结构化 `degraded` 结果
- 显式传入 `1810.HK`、`600519.SH`、`ASML.AS` 这类 ticker 时，解析层会直接接受该市场代码，不再强依赖 SEC mapping API
- 新增 `Market Profile` 工具，返回 `issuer_profile`、provider 能力边界和官方披露查询链接
- 欧洲市场不再统一折叠到一个入口；`AS/BR/PA` 走 Euronext，`DE` 走 Deutsche Boerse，`L` 走 LSE，`SW` 走 SIX
- 新增 `Official Disclosure Search` 工具，为港股/A 股/欧洲市场返回对应官方披露入口、查询提示，以及尽可能直接可用的查询链接或入口页
- `information_gathering_analyst` 默认优先使用 `MARKET_IDENTIFIER_MODEL`，适合挂接 `deepseek-v4-flash`

### 推荐：直接指定公司运行

```bash
conda run -n MultiAgent env PYTHONPATH=src python -m multi_agent.main \
  --company-name "Apple Inc."
```

如果你知道 ticker，也可以显式传入：

```bash
conda run -n MultiAgent env PYTHONPATH=src python -m multi_agent.main \
  --company-name "Apple Inc." \
  --company-ticker AAPL
```

如果希望在运行成功后，把这次结构化建议直接写入 watchlist：

```bash
conda run -n MultiAgent env PYTHONPATH=src python -m multi_agent.main \
  --company-name "Alibaba Group Holding Ltd" \
  --company-ticker BABA \
  --save-to-watchlist
```

只查看当前 watchlist：

```bash
conda run -n MultiAgent env PYTHONPATH=src python -m multi_agent.main \
  --watchlist-list
```

如果你已经有历史运行结果，想在**不重新调 API** 的情况下重建结构化建议和 watchlist：

```bash
conda run -n MultiAgent env PYTHONPATH=src python -m multi_agent.main \
  --watchlist-rebuild
```

这个命令会扫描 `artifacts/*/*/` 下已有运行目录，基于 `04_investment_report.md` 和
`latest_run_metrics.json` 重新生成 `06_structured_recommendation.json`，并同步重建
`artifacts/watchlist.json`。

### 使用 CrewAI CLI

```bash
conda run -n MultiAgent crewai run
```

如果你只是想快速试运行，也可以依赖 `.env` 中的默认公司配置。

## 8. 运行后会产出什么

系统会按照顺序工作流生成：

- `artifacts/<company_slug>__<ticker_slug>/<timestamp>/README.md`
- `artifacts/<company_slug>__<ticker_slug>/<timestamp>/01_market_intelligence.md`
- `artifacts/<company_slug>__<ticker_slug>/<timestamp>/02_filing_review.md`
- `artifacts/<company_slug>__<ticker_slug>/<timestamp>/03_financial_analysis.md`
- `artifacts/<company_slug>__<ticker_slug>/<timestamp>/04_investment_report.md`
- `artifacts/<company_slug>__<ticker_slug>/<timestamp>/05_runtime.txt`
- `artifacts/<company_slug>__<ticker_slug>/<timestamp>/06_structured_recommendation.json`
- `artifacts/<company_slug>__<ticker_slug>/<timestamp>/07_structured_report.json`
- `artifacts/<company_slug>__<ticker_slug>/<timestamp>/latest_run_metrics.json`
- `artifacts/<company_slug>__<ticker_slug>/<timestamp>/evaluation_summary.json`
- `artifacts/watchlist.json`

无论运行成功、失败还是中断，目录内都会保留上述固定文件名，便于排查和归档：

- 成功运行时，`01-04` 会保存真实产出内容。
- 成功运行时，`06_structured_recommendation.json` 会保存结构化投资建议、风险、催化剂和可信度评分。
- 成功运行时，`07_structured_report.json` 会保存更完整的结构化报告快照，适合前端、API 和数据库消费。
- 失败或中断时，`01-04` 会保留失败说明，`06_structured_recommendation.json` 会写入失败状态，`latest_run_metrics.json` 会写明 `status` 与错误原因。

这就是“每一步中间产出”的落地方式。

例如：

- `artifacts/apple_inc__aapl/20260615_103045/`
- `artifacts/shenzhen_inovance_technology_co_ltd__300124_sz/20260615_090807/`

默认情况下，Agent 和 Task 都使用中文提示词，最终报告也要求输出为中文。

如果你没有准备本地 PDF 财报文件，也不用额外处理。当前版本会自动关闭 PDF 提取工具，
避免 Agent 在运行时猜测文件名并触发“文件不存在”。

`latest_run_metrics.json` 中会包含这些客观指标：

- 单次运行总耗时
- 每个 task 耗时
- API 失败率
- 报告是否完整生成
- 引用条数
- 中间产物是否齐全
- 财务字段是否成功提取，以及规范化后的值
- 可信度评分 `trust_score` 及其拆分项

`evaluation_summary.json` 会累计统计：

- 总运行次数
- 成功率
- 累计 API 调用次数与失败率

`06_structured_recommendation.json` 会提供一个更适合后续产品化消费的结构：

- `stance` / `stance_label`
  - 标准化投资立场，例如 `buy`、`hold`、`sell`、`watch`
- `trust_score` / `trust_level`
  - 基于报告完整性、引用数、中间产物完整性、财务字段覆盖率、API 稳定性计算
- `summary`
  - 从执行摘要中提取的核心结论
- `catalysts` / `risks`
  - 从报告中提取出的关键催化剂与风险点

`07_structured_report.json` 会进一步提供：

- `sections`
  - 对业务概览、近期动态、财务分析、投资建议等章节做结构化快照
- `citation_urls`
  - 从报告中抽取出的引用链接
- `validation`
  - 对摘要、催化剂、风险、投资建议和引用数量做完整性检查

`artifacts/watchlist.json` 会保存被加入观察池的公司，便于后续继续做：

- 定期重跑投研
- 跟踪催化剂兑现
- 接事件提醒或 paper trading

## 9. 为什么这个实现更接近工业级

### 一个可交付的 MVP

当前版本已经可以被视为一个完成度较高的 MVP，因为它同时具备：

- 端到端工作流
- 中间产物沉淀
- 结构化建议输出
- 结构化完整报告输出
- watchlist 重建能力
- 基础评估指标与测试覆盖

如果后续再补多市场数据源、数据库和人工审核节点，它就会从 MVP 进一步升级成更强的作品集项目。

### 配置集中管理

- 所有关键配置统一收敛到 `settings.py`
- API Key 全部来自 `.env`
- 不把外部依赖硬编码进业务逻辑

### 工具层和业务层分离

- `investment_tools.py` 负责联网和 I/O
- `finance.py` 负责纯财务计算
- 这样测试时可以用 stub，避免真实调用外部 API

### 异常处理和重试

- 外部 HTTP 请求统一走带重试的 `requests.Session`
- 对 `429/5xx` 做自动重试
- Tool 层在异常时返回明确错误信息，避免任务直接崩掉
- Agent 级别设置了 `max_retry_limit=3`

### 可测试

项目已经加入测试，覆盖：

- 配置加载
- 财务指标计算
- Tool 输出格式
- Crew 结构是否满足 3 Agent / 4 Task
- 运行评估文件生成
- 连续多次运行下的累计统计稳定性

运行测试：

```bash
conda run -n MultiAgent env PYTHONPATH=src python -m pytest tests -q
```

## 10. 工作流说明

### 第一步：市场情报

`Information Gathering Analyst` 使用：

- Google 搜索

目标：

- 建立公司外部事实基础
- 找近期催化剂、风险、竞争变化

### 第二步：SEC 文件梳理

还是由 `Information Gathering Analyst` 负责：

- 检索最近的 10-K / 10-Q
- 记录 filing URL、日期、优先阅读项

### 第三步：财务分析

`Financial Statement Analyst` 使用：

- SEC company facts
- 财务指标计算
- PDF 文本抽取

目标：

- 把“原始财务事实”变成“可解释的分析观点”

### 第四步：报告撰写

`Report Writing Analyst` 使用：

- 前三步的中间产物
- 文件读写

输出：

- 最终投资备忘录 `report.md`

## 11. 新人最容易问的 5 个问题

### 1. 为什么不是一个 Agent 全做？

因为投研任务天然适合分工：

- 搜集信息
- 解读财务
- 写报告

拆分后更容易：

- 控制 prompt
- 管理工具权限
- 调试哪一步出错
- 向面试官说明系统设计

### 2. 为什么 `finance.py` 单独拆出来？

因为财务指标计算本质是确定性逻辑，不应该混在 prompt 或 Agent 里。

### 3. 为什么工具要单独封装？

因为联网搜索、SEC API、PDF 解析都属于“外部能力”，这部分最容易变化，也最需要单测。

### 4. 为什么中间产物要写文件？

因为工业项目里可观察性很重要。你不能只看最终报告，要能回溯每一步输出。

### 5. 为什么要用 `.env`？

因为：

- API Key 不能写死在代码里
- 开发、测试、生产环境配置不同
- 方便团队协作和部署

## 12. 面试时你可以怎么讲

推荐讲法：

1. 先讲业务问题
   - 分析师做投研初稿很耗时，尤其卡在搜集信息和整理材料
2. 再讲架构
   - 用 3 个 Agent 分工协作
3. 再讲工具
   - Google Search、SEC API、PDF、财务计算、文件系统
4. 再讲稳定性
   - 重试、异常处理、中间产物记录、可测试
5. 最后讲结果
   - 初稿产出更快、更可复盘，也更便于团队复用

## 13. 后续可扩展方向

- 接入真实新闻 API 或研报数据库
- 把最终报告改成结构化 JSON + Markdown 双输出
- 引入 `output_pydantic` 做更强的下游消费
- 把工作流升级为 Flow，实现人工审核节点
- 加入缓存层，减少重复查询消耗
- 引入数据库保存多家公司历史研究结果
- 在 watchlist 基础上扩展事件提醒、估值阈值和 paper trading

## 14. 常见错误排查

### 报缺少 API Key

先检查 `.env` 是否补全：

- `SERPER_API_KEY`
- `SEC_API_KEY`
- `SEC_API_EMAIL`
- `OPENAI_API_KEY`

如果缺失，程序会直接报出类似下面的错误，帮助你一次性补全：

```text
Missing required environment variables: SERPER_API_KEY, SEC_API_KEY, SEC_API_EMAIL.
Please fill them in your .env file.
```

### 报公司查不到

优先检查：

- 公司名是否足够完整，例如用 `Alibaba Group` 而不是过短缩写
- 如果是业务线或品牌，是否应改为母公司，例如 `Sony Group Corporation`
- 如果是未上市公司，当前工作流不会自动编造 ticker

### 报搜索失败

优先检查：

- `SEARCH_PROVIDER` 是否与实际 key 来源一致
- `SERPER_API_KEY` 或 `SERPAPI_API_KEY` 是否有效
- 是否达到配额

### Serper 和 SerpApi 不要混用

这两个名字很像，但它们是不同服务：

- `Serper`
  - 常见接口是 `https://google.serper.dev/search`
  - 对应 `SERPER_API_KEY`
- `SerpApi`
  - 常见接口是 `https://serpapi.com/search.json`
  - 对应 `SERPAPI_API_KEY`

当前项目已经兼容两者：

- 当 `SEARCH_PROVIDER=serper` 时，强制走 Serper
- 当 `SEARCH_PROVIDER=serpapi` 时，强制走 SerpApi
- 当 `SEARCH_PROVIDER=auto` 时，会优先尝试 Serper；如果鉴权失败且你实际上填的是 SerpApi 的 key，会自动回退到 SerpApi

如果你不想记 provider 差异，推荐直接配置：

```env
SEARCH_PROVIDER=serpapi
SERPAPI_API_KEY=your_real_serpapi_key
```

### 遇到 401 / 403 / 429 后程序直接退出

这是当前项目的故意设计，不是异常行为：

- `401 / 403`
  - 通常表示 API Key 无效、权限不足或接口未开通
- `429`
  - 通常表示请求过快或额度耗尽

为了避免 Agent 在错误凭证上继续反复重试，工具层会把这些错误识别为致命错误，
主入口收到后会直接结束运行，并用中文提示你检查对应的 API Key 或权限配置。

### 报财务数据为空

优先检查：

- 公司是否有 SEC 披露
- 该公司是否使用了不同的财务标签

## 15. 相关阅读

- [CrewAI Docs](https://docs.crewai.com)
- [CrewAI Quickstart](https://docs.crewai.com/en/quickstart)
- [Agents Concept](https://docs.crewai.com/en/concepts/agents)
- [Tasks Concept](https://docs.crewai.com/en/concepts/tasks)
- [Tools Concept](https://docs.crewai.com/en/concepts/tools)
