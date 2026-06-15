# 性能分析使用说明

## 1. 目标

本工具用于对当前 Agent 系统做两类性能分析：

- `cProfile`
  - 获取函数级累计耗时，适合找“哪里慢”
- `py-spy`
  - 获取采样火焰图，适合区分 CPU 热点和 I/O 等待

工具入口在：

- `src/multi_agent/perf.py`
- CLI 脚本：`profile_agent`

## 2. 支持的模式

### workflow

对完整主流程做性能分析，例如：

- 公司解析
- Agent 编排
- 外部 API 请求
- LLM 调用
- 结构化结果落盘

### watchlist-rebuild

对历史 artifacts 重建路径做性能分析，例如：

- recommendation 重建
- structured report 重建
- watchlist 重建

## 3. 打印命令

先查看工具将执行什么命令：

```bash
conda run -n MultiAgent env PYTHONPATH=src python -m multi_agent.perf print-commands \
  --mode workflow \
  --company-name "Apple Inc." \
  --company-ticker AAPL \
  --output artifacts/profiles/apple-run
```

## 4. 运行 cProfile

### 主流程

```bash
conda run -n MultiAgent env PYTHONPATH=src python -m multi_agent.perf run-cprofile \
  --mode workflow \
  --company-name "Apple Inc." \
  --company-ticker AAPL \
  --output artifacts/profiles/apple-run.prof
```

### watchlist 重建

```bash
conda run -n MultiAgent env PYTHONPATH=src python -m multi_agent.perf run-cprofile \
  --mode watchlist-rebuild \
  --output artifacts/profiles/rebuild.prof
```

## 5. 汇总 .prof 结果

```bash
conda run -n MultiAgent env PYTHONPATH=src python -m multi_agent.perf summarize \
  --profile artifacts/profiles/apple-run.prof \
  --top-n 20
```

输出为 JSON，便于：

- 保存结果
- 做前后版本对比
- 接入后续可视化

## 6. 运行 py-spy

```bash
conda run -n MultiAgent env PYTHONPATH=src python -m multi_agent.perf run-py-spy \
  --mode watchlist-rebuild \
  --output artifacts/profiles/rebuild.svg
```

### macOS 注意事项

在 macOS 上，`py-spy` 常常需要更高权限才能采样 Python 进程。

如果出现类似报错：

```text
This program requires root on OSX.
```

可以改为手动执行提升权限版本：

```bash
sudo $(conda run -n MultiAgent which py-spy) record \
  -o artifacts/profiles/rebuild.svg \
  -- env PYTHONPATH=src python -m multi_agent.main --watchlist-rebuild
```

## 7. 推荐分析顺序

1. 先跑 `workflow` 的 `cProfile`
   - 看总耗时主要压在哪条调用链
2. 再跑 `watchlist-rebuild` 的 `cProfile`
   - 区分主流程慢还是后处理慢
3. 最后用 `py-spy`
   - 判断是 CPU 计算热点还是远程等待

## 8. 如何解读结果

### 如果 `openai` / `completion.create` 很高

- 主瓶颈在 LLM 推理或远程等待

### 如果 `threading.wait` / `lock.acquire` 很高

- 说明程序大多在等待异步任务、网络请求或模型返回

### 如果 import 很高

- 说明冷启动开销明显
- 这种情况多出现在 `watchlist-rebuild` 这种轻路径里

### 如果本地模块函数很高

- 才说明真正存在 Python 侧计算热点

## 9. 建议保留的产物

- `artifacts/profiles/*.prof`
- `artifacts/profiles/*.svg`
- 本次运行对应的 `latest_run_metrics.json`

这样后续你可以对比：

- 不同公司
- 不同模型
- 不同 prompt
- 不同缓存策略
- 优化前后版本
