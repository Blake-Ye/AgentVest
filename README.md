# Multi-Agent Investment Research

本文档只描述三件事：

- 当前工作区里真实存在的入口
- 当前代码稳定输出的文件与字段
- benchmark 可以承诺的稳定边界

如果你要看更细的检查项，继续读 `docs/README.md`。

## 当前真实入口

### 主工作流

- 安装后脚本：`uv run multi_agent --company-name "Apple Inc." --company-ticker AAPL`
- 模块入口：`PYTHONPATH=src python3 -m multi_agent.main --company-name "Apple Inc." --company-ticker AAPL`
- 触发器入口：`uv run run_with_trigger '{"company_name":"Apple Inc.","company_ticker":"AAPL"}'`

主工作流由 `src/multi_agent/main.py` 暴露，负责：

- 解析参数
- 解析公司名称和 ticker
- 创建本次运行目录
- 运行 Crew
- 落盘结构化结果与评估指标

### Benchmark

- 安装后脚本：`uv run benchmark --dataset benchmark_samples/example_benchmark.jsonl --output-dir outputs/benchmark/demo --factual-only`
- 模块入口：`PYTHONPATH=src python3 -m multi_agent.benchmark --dataset benchmark_samples/example_benchmark.jsonl --output-dir outputs/benchmark/demo --factual-only`

当前 benchmark CLI 只稳定支持以下参数：

- `--dataset`
- `--output-dir`
- `--judge-model`
- `--factual-only`

### 辅助入口

- `uv run profile_agent ...` 对应 `src/multi_agent/perf.py`
- `uv run api_server` 对应 `src/multi_agent/api.py`

其中：

- `profile_agent` 用于性能分析，不参与主输出契约，也不参与 benchmark 契约
- `api_server` 提供最小展示版 FastAPI + 后台任务 + dashboard 单页，复用现有主工作流
- `api_server` 默认把任务状态持久化到 `outputs/api/jobs.db`，也可以通过 `MULTI_AGENT_DATA_DIR` 改写

### API 展示层

- 启动：`uv run api_server`
- 打开：`http://127.0.0.1:8000/`
- 当前稳定接口：
  - `POST /api/jobs`
  - `GET /api/jobs`
  - `GET /api/jobs/{job_id}`
  - `GET /api/jobs/{job_id}/artifacts`
  - `GET /api/jobs/{job_id}/artifacts/{artifact_name}`

dashboard 当前支持：

- 提交投研任务
- 查看最近任务状态
- 查看任务详情
- 查看稳定 artifact 列表
- 打开或下载 `04_investment_report.md`、结构化 JSON、metrics 等稳定产物

### Render 云部署

- 仓库根目录已提供 `render.yaml`
- Render 启动命令使用：`python -m uvicorn multi_agent.api:create_app --factory --host 0.0.0.0 --port $PORT`
- `PORT` 由平台注入，不需要手动设置
- Blueprint 默认把运行数据写到 `/tmp/multi-agent/...`
- 首次部署时至少需要在 Render 控制台补齐：
  - `OPENAI_API_KEY`
  - `SERPER_API_KEY`
  - `SEC_API_EMAIL`
- `SEC_API_EMAIL` 仅用于给 SEC 官方免费接口提供合规 `User-Agent` 联系方式，不再需要额外 SEC 密钥
- 非密钥变量已经在 `render.yaml` 中给出默认值，包括：
  - `MODEL=qwen-plus`
  - `OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
  - `MULTI_AGENT_DATA_DIR=/tmp/multi-agent/api`
  - `ARTIFACTS_DIR=/tmp/multi-agent/artifacts`
  - `WATCHLIST_PATH=/tmp/multi-agent/watchlist/watchlist.json`
  - `FINAL_REPORT_PATH=/tmp/multi-agent/report.md`
  - `BENCHMARK_OUTPUT_DIR=/tmp/multi-agent/benchmark`

Render 部署步骤：

1. 把当前分支推到 GitHub/GitLab/Bitbucket
2. 打开 Render，选择 `New -> Blueprint`
3. 连接本仓库并选择根目录 `render.yaml`
4. 在创建页面填入上面的 3 个环境变量
5. 部署完成后访问 Render 分配的公开域名

如果你只是临时给手机查看页面，也可以本地启动后再用公网隧道暴露：

```bash
PYTHONPATH=src python3 -m uvicorn multi_agent.api:create_app --factory --host 0.0.0.0 --port 8000
```

然后任选一个隧道工具：

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

或：

```bash
ngrok http 8000
```

## 最小环境变量

```bash
cp .env.example .env
uv sync --extra dev
```

主工作流当前要求以下环境变量可用：

```env
MODEL=qwen-plus
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
SERPER_API_KEY=your_serper_key
# 或者改用 SERPAPI_API_KEY
# SEC 官方免费接口不需要额外 SEC 密钥
SEC_API_EMAIL=analyst@example.com
ARTIFACTS_DIR=outputs/artifacts
FINAL_REPORT_PATH=outputs/report.md
WATCHLIST_PATH=outputs/watchlist/watchlist.json
BENCHMARK_OUTPUT_DIR=outputs/benchmark
```

回归测试入口：

```bash
PYTHONPATH=src python3 -m pytest tests -q
```

## 主工作流输出契约

### 目录结构

每次运行会创建一个目录：

```text
outputs/artifacts/<company_slug>__<ticker_slug>/<YYYYMMDD_HHMMSS>/
```

其中稳定可依赖的文件是：

- `README.md`
- `01_market_intelligence.md`
- `02_filing_review.md`
- `03_financial_analysis.md`
- `04_investment_report.md`
- `06_structured_recommendation.json`
- `07_structured_report.json`
- `latest_run_metrics.json`
- `evaluation_summary.json`

不要把 `05_runtime.txt` 当成当前稳定契约。代码里保留了这个路径名，但当前主流程不会稳定生成该文件。

### 运行中状态

主流程开始后会先写入占位文件：

- `01` 到 `04` 先写占位 Markdown
- `06_structured_recommendation.json` 和 `07_structured_report.json` 先写 `status: "running"`
- `latest_run_metrics.json` 先写 `status: "running"`

因此看到占位内容不代表失败，只代表流程尚未完成。

### 成功状态

成功时可以稳定依赖以下结果：

- `04_investment_report.md` 是最终报告的稳定文件位置
- `06_structured_recommendation.json` 与 `07_structured_report.json` 当前改为 JSON-first 生成路径，不再依赖对 Markdown 标题/表格的反解析
- `06_structured_recommendation.json` 至少包含：
  - `generated_at`
  - `company_name`
  - `company_ticker`
  - `stance`
  - `stance_label`
  - `trust_score`
  - `trust_level`
  - `trust_summary`
  - `summary`
  - `catalysts`
  - `risks`
  - `next_actions`
  - `source_report_path`
- `07_structured_report.json` 至少包含：
  - `generated_at`
  - `company_name`
  - `company_ticker`
  - `summary`
  - `stance`
  - `stance_label`
  - `trust_score`
  - `trust_level`
  - `trust_summary`
  - `catalysts`
  - `risks`
  - `next_actions`
  - `sections`
  - `citation_urls`
  - `validation`
  - `source_report_path`
- `latest_run_metrics.json` 至少包含：
  - `started_at`
  - `finished_at`
  - `status`
  - `success`
  - `error_message`
  - `company_name`
  - `company_ticker`
  - `total_runtime_seconds`
  - `task_durations_seconds`
  - `api_calls`
  - `report_generated`
  - `report_complete`
  - `citation_count`
  - `intermediate_artifacts_complete`
  - `artifacts`
  - `financial_fields`
  - `financial_fields_success_rate`
  - `trust_score`

### 失败状态

失败时当前代码仍会尽量保留可定位信息：

- `README.md` 仍然存在
- `01` 到 `04` 会改写为中文失败说明
- `06_structured_recommendation.json` 和 `07_structured_report.json` 会写入失败 JSON
- `latest_run_metrics.json` 会写入：
  - `status: "failed"`
  - `success: false`
  - `error_message`

`FINAL_REPORT_PATH` 只是兼容输入参数；对下游最稳定的最终报告位置仍然是本次运行目录里的 `04_investment_report.md`。

## Benchmark 稳定性边界

### 稳定文件与目录

一次 benchmark 运行当前稳定生成：

- `benchmark_summary.json`
- `benchmark_results.json`
- `details/<sample_id>.json`
- `artifacts/`

其中 `artifacts/` 目录用于复用主工作流输出，不应被当成 benchmark 汇总 API 的唯一来源。

### 稳定的 summary 结构

`benchmark_summary.json` 当前稳定提供以下顶层字段：

- `generated_at`
- `sample_count`
- `completed_count`
- `failed_count`
- `completion_rate`
- `judge_completed_count`
- `judge_skipped_count`
- `judge_execution_rate`
- `metrics`
- `detail_files`
- `output_dir`

`metrics` 当前稳定包含以下指标桶：

- `factual_accuracy`
- `risk_recall`
- `catalyst_recall`
- `hallucination_score`
- `overall_quality_score`

每个指标桶当前稳定包含：

- `mean`
- `count`
- `stddev`
- `min`
- `max`

### 稳定的 detail 边界

`details/<sample_id>.json` 与 `benchmark_results.json.samples[*]` 在当前代码里有两层边界：

- 所有样本至少稳定包含：
  - `sample_id`
  - `company_name`
  - `ticker`
  - `status`
  - `expected_facts`
  - `expected_key_points`
  - `metrics`
  - `factual_check`
  - `llm_judge`
- 失败样本稳定包含：
  - `error_message`
- 仅在样本成功完成主工作流时，才应依赖：
  - `generated_artifacts`
  - `workflow_metrics`

### 不能承诺的部分

以下内容不属于当前 benchmark 的稳定契约：

- 具体分数高低，因为它依赖外部 LLM、搜索和 SEC 数据可用性
- 错误文案全文，因为它直接透传外部调用失败信息
- `detail_files`、`output_dir`、`generated_artifacts.*`、`source_report_path` 里的绝对路径字符串
- `risk_recall`、`catalyst_recall`、`hallucination_score` 在 `--factual-only` 或 judge 不可用时的非空值

当前实现会在以下情况下自动降级：

- 显式传入 `--factual-only`
- judge 客户端配置不可用
- judge 调用过程中发生异常

降级后：

- `llm_judge.status` 会是 `skipped` 或 `unavailable`
- 与 judge 相关的指标可能为 `null`
- `overall_quality_score` 在 factual-only 路径下退化为 `factual_accuracy`

## 相关文档

- `docs/README.md`
- `docs/guides/testing_engineer_quickcheck_zh.md`
- `docs/evaluation/benchmark_evaluation_zh.md`
- `outputs/README.md`
