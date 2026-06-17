# Outputs

这里只说明当前工作区里可稳定查找的输出位置，不描述尚未稳定的运行日志或临时文件。

## 主工作流

主工作流输出目录：

```text
artifacts/<company_slug>__<ticker_slug>/<YYYYMMDD_HHMMSS>/
```

当前稳定可依赖的文件名：

- `README.md`
- `01_market_intelligence.md`
- `02_filing_review.md`
- `03_financial_analysis.md`
- `04_investment_report.md`
- `06_structured_recommendation.json`
- `07_structured_report.json`
- `latest_run_metrics.json`
- `evaluation_summary.json`

当前不要依赖 `05_runtime.txt`。代码里保留了这个路径名，但主流程并不会稳定写出该文件。

最常用查找路径：

- 一次主流程完整结果：`artifacts/<company_slug>__<ticker_slug>/<timestamp>/`
- 最终报告：`artifacts/<company_slug>__<ticker_slug>/<timestamp>/04_investment_report.md`
- 结构化建议：`artifacts/<company_slug>__<ticker_slug>/<timestamp>/06_structured_recommendation.json`
- 结构化完整报告：`artifacts/<company_slug>__<ticker_slug>/<timestamp>/07_structured_report.json`
- 单次运行指标：`artifacts/<company_slug>__<ticker_slug>/<timestamp>/latest_run_metrics.json`

## Benchmark

benchmark 输出目录：

```text
benchmark/<run_name>/
```

当前稳定可依赖的文件名：

- `benchmark_summary.json`
- `benchmark_results.json`
- `details/<sample_id>.json`

`artifacts/` 子目录用于承接 benchmark 过程中复用的主工作流产物，但不应被当作 benchmark 汇总契约本身。

## 其他目录

- `profiles/`
  - `profile_agent` 生成的性能分析文件。
- `watchlist/`
  - `WATCHLIST_PATH` 默认指向的持久化 watchlist 文件。

## 使用建议

- 主流程运行后，终端会打印“本次输出目录”和“最终报告路径”。
- 如果只是找最终研究结论，优先打开 `04_investment_report.md`。
- 如果是机器消费，优先读取 `06_structured_recommendation.json`、`07_structured_report.json` 和 `latest_run_metrics.json`。
