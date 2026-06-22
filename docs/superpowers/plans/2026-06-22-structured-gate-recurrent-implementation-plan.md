# 结构化 Gate Recurrent 与展示收口实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 清理低价值历史产物，将真实 reviewer tools 输出真正接入 Flow 的 gate 主链路，并补齐端到端测试与 README/架构文档，使项目达到“可投实习、可讲清楚、可验证”的基准线。

**架构：** 保持现有 `ReviewToolSummary -> ConfidenceGatePolicy -> GateDecision` 控制面，继续把 analysis gate 从 Markdown 文案识别迁移到真实结构化 summary 驱动；同时保留受控 fallback，避免在主链路尚未完全接线前破坏运行稳定性。对外展示层则同步更新 README 和架构文档，让仓库叙述与当前代码行为一致。

**技术栈：** Python 3.13, Pydantic, CrewAI Flow, pytest, Markdown 文档

---

## 文件结构

- 删除：`/Users/yeziqing/Projects/AgentVest/Users`
  - 清理之前绝对路径写错导致的误产物目录。
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/<old-run>`
  - 删除历史运行目录，仅保留最新 1 次样例。
- 修改：`/Users/yeziqing/Projects/AgentVest/src/multi_agent/core/review_contracts.py`
  - 保持 gate 契约与结构化 summary 字段一致，必要时补充字段说明。
- 修改：`/Users/yeziqing/Projects/AgentVest/src/multi_agent/core/confidence_gate.py`
  - 维持 `passed / rerun / blocked` 策略边界，并与真实 reviewer tool 输出对齐。
- 修改：`/Users/yeziqing/Projects/AgentVest/src/multi_agent/flows/market_review_flow.py`
  - 让默认 analysis gate 优先消费真实 `analysis_review_summary`，仅在缺失时回退到 Markdown fallback。
- 修改：`/Users/yeziqing/Projects/AgentVest/src/multi_agent/main.py`
  - 保持 blocked 产物语义收口，并确保 README 中描述的行为与 CLI 一致。
- 修改：`/Users/yeziqing/Projects/AgentVest/tests/test_review_gate.py`
  - 验证 gate 三态仍与真实 summary 行为一致。
- 修改：`/Users/yeziqing/Projects/AgentVest/tests/test_flow_routing.py`
  - 验证真实 summary 优先级、rerun 路由、budget 耗尽和 blocked 收口。
- 修改：`/Users/yeziqing/Projects/AgentVest/tests/test_main_cli.py`
  - 验证 blocked recommendation 不再暴露正式建议语义。
- 修改：`/Users/yeziqing/Projects/AgentVest/README.md`
  - 对外说明当前 7 Agent、gate/recurrent、正式/阻断报告边界与测试入口。
- 修改：`/Users/yeziqing/Projects/AgentVest/docs/project_architecture.md`
  - 把旧的 3 Agent/4 Task 描述更新为当前实际架构。

### 任务 1：清理低价值历史产物

**文件：**
- 删除：`/Users/yeziqing/Projects/AgentVest/Users`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_014906`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_015149`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_020250`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_020613`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_021623`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_125807`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_130236`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_130343`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_131708`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_132925`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_134155`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_135343`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_135627`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_140735`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_141437`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_142933`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_144441`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_151040`
- 删除：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/20260622_152935`
- 保留：`/Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/<latest-run>`

- [ ] **步骤 1：确认最新运行目录**

运行：

```bash
ls -1 /Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft
```

预期：拿到按时间排序的目录列表，并确认保留最后一项。

- [ ] **步骤 2：删除误产物目录 `Users/`**

执行：

```text
DeleteFile:
- /Users/yeziqing/Projects/AgentVest/Users
```

预期：删除成功，`git status --short` 中不再出现 `?? Users/`。

- [ ] **步骤 3：删除旧的运行目录，仅保留最新 1 次**

执行：

```text
DeleteFile:
- /Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft/<all-old-runs>
```

预期：`var/runs/microsoft_corporation__msft/` 下只剩最新 1 个目录。

- [ ] **步骤 4：验证清理结果**

运行：

```bash
ls -R /Users/yeziqing/Projects/AgentVest/var/runs/microsoft_corporation__msft
git status --short
```

预期：历史样例目录显著减少，误产物 `Users/` 已删除，且不会误删 `docs/`、`src/`、`tests/`。

- [ ] **步骤 5：Commit**

```bash
git add -A /Users/yeziqing/Projects/AgentVest/var /Users/yeziqing/Projects/AgentVest/Users
git commit -m "chore: remove stale run artifacts"
```

### 任务 2：把真实 reviewer tools 输出接进 Flow

**文件：**
- 修改：`/Users/yeziqing/Projects/AgentVest/src/multi_agent/flows/market_review_flow.py`
- 修改：`/Users/yeziqing/Projects/AgentVest/src/multi_agent/core/review_contracts.py`
- 修改：`/Users/yeziqing/Projects/AgentVest/src/multi_agent/core/confidence_gate.py`
- 测试：`/Users/yeziqing/Projects/AgentVest/tests/test_flow_routing.py`
- 测试：`/Users/yeziqing/Projects/AgentVest/tests/test_review_gate.py`

- [ ] **步骤 1：先写失败测试，锁定“真实 summary 优先于 Markdown 文案”行为**

```python
def test_default_flow_analysis_gate_prefers_structured_review_summary_over_markdown_markers(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "08_data_quality_review.md"
    review_path.write_text(
        "# 审查报告\n\n## 必须修复（阻断级）\n- 旧文案阻断\n",
        encoding="utf-8",
    )

    flow = MarketReviewFlow(
        analysis_executor=lambda _inputs: {
            "analysis_markdown": "analysis complete",
            "analysis_review_summary": {
                "evidence_coverage_ratio": 1.0,
                "financial_coverage_score": 1.0,
                "critical_conflict_count": 0,
                "market_policy_violations": [],
                "unsupported_critical_claims": [],
                "blocking_reasons": [],
            },
        },
        report_writer=lambda _result: "# investment report",
        report_reviewer=lambda _report: GateDecision(
            passed=True,
            final_decision="passed",
            trust_score=90,
        ),
        initial_state=MarketReviewFlowState(
            request_id="run-011",
            company_name="Microsoft Corporation",
            artifacts_dir=str(tmp_path),
        ),
    )

    result = flow.kickoff()

    assert result["status"] == "passed"
```

- [ ] **步骤 2：运行测试验证当前默认 gate 还会被 Markdown 误导**

运行：`pytest /Users/yeziqing/Projects/AgentVest/tests/test_flow_routing.py -q -k structured_review_summary`
预期：FAIL，失败原因是当前默认 analysis gate 仍优先用文案阻断结果。

- [ ] **步骤 3：在 Flow 中实现结构化 summary 优先解析**

```python
@staticmethod
def _structured_review_summary(result: Any, key: str) -> ReviewToolSummary | None:
    if not isinstance(result, dict):
        return None
    raw_summary = result.get(key)
    if not isinstance(raw_summary, dict):
        return None
    return ReviewToolSummary.model_validate(raw_summary)


def _default_analysis_gate(self, analysis_result: Any) -> GateDecision:
    summary = self._structured_review_summary(analysis_result, "analysis_review_summary")
    if summary is not None:
        return ConfidenceGatePolicy.default().evaluate(summary)
    review_text = self._materialized_analysis_review_content()
    ...
```

- [ ] **步骤 4：保持 fallback，但把它降级为兜底路径**

```python
if summary is not None:
    return ConfidenceGatePolicy.default().evaluate(summary)

review_text = self._materialized_analysis_review_content()
if self._contains_blocked_review_marker(review_text):
    return GateDecision(
        passed=False,
        final_decision="blocked",
        trust_score=0,
        blocking_reasons=["analysis_review_marked_blocked"],
    )
return self._allow_stage_to_continue(analysis_result)
```

- [ ] **步骤 5：运行测试验证真实 summary 主链路通过**

运行：

```bash
pytest /Users/yeziqing/Projects/AgentVest/tests/test_review_gate.py -q
pytest /Users/yeziqing/Projects/AgentVest/tests/test_flow_routing.py -q
```

预期：gate 三态测试与 Flow 路由测试全部 PASS。

- [ ] **步骤 6：Commit**

```bash
git add \
  /Users/yeziqing/Projects/AgentVest/src/multi_agent/core/review_contracts.py \
  /Users/yeziqing/Projects/AgentVest/src/multi_agent/core/confidence_gate.py \
  /Users/yeziqing/Projects/AgentVest/src/multi_agent/flows/market_review_flow.py \
  /Users/yeziqing/Projects/AgentVest/tests/test_review_gate.py \
  /Users/yeziqing/Projects/AgentVest/tests/test_flow_routing.py
git commit -m "feat: prioritize structured reviewer summaries in flow gate"
```

### 任务 3：补端到端测试，锁定正式/阻断交付边界

**文件：**
- 修改：`/Users/yeziqing/Projects/AgentVest/tests/test_main_cli.py`
- 可能修改：`/Users/yeziqing/Projects/AgentVest/src/multi_agent/main.py`

- [ ] **步骤 1：补 blocked recommendation 语义测试**

```python
def test_run_writes_blocked_outputs_when_gate_fails(...) -> None:
    ...
    assert recommendation["status"] == "blocked"
    assert recommendation["stance"] == "blocked"
    assert recommendation["stance_label"] == "阻断"
```

- [ ] **步骤 2：补“recurrent 后通过”端到端测试**

```python
def test_run_allows_formal_report_after_rerun_passes(...) -> None:
    final_result = {
        "status": "passed",
        "trust_score": 88,
        "blocking_reasons": [],
        "analysis_result": {"analysis_markdown": "rerun recovered"},
        "report_result": "# 投资备忘录\n\n## 投资建议\n建议持有",
    }
    ...
    assert recommendation["status"] == "passed"
    assert recommendation["stance"] in {"hold", "buy", "sell", "watch"}
```

- [ ] **步骤 3：运行测试验证红灯或行为未锁死**

运行：`pytest /Users/yeziqing/Projects/AgentVest/tests/test_main_cli.py -q`
预期：在新增端到端断言前至少出现 1 个 FAIL 或缺失断言场景。

- [ ] **步骤 4：必要时微调主入口收口逻辑**

```python
if final_status == "blocked":
    recommendation["stance"] = "blocked"
    recommendation["stance_label"] = "阻断"
    structured_report["stance"] = "blocked"
    structured_report["stance_label"] = "阻断"
```

- [ ] **步骤 5：运行完整 CLI 回归**

运行：

```bash
pytest /Users/yeziqing/Projects/AgentVest/tests/test_main_cli.py -q
```

预期：CLI 测试全部 PASS，blocked 与 passed 产物边界都成立。

- [ ] **步骤 6：Commit**

```bash
git add \
  /Users/yeziqing/Projects/AgentVest/src/multi_agent/main.py \
  /Users/yeziqing/Projects/AgentVest/tests/test_main_cli.py
git commit -m "test: cover blocked and rerun delivery paths"
```

### 任务 4：更新 README，让项目可展示

**文件：**
- 修改：`/Users/yeziqing/Projects/AgentVest/README.md`

- [ ] **步骤 1：写失败检查，找出 README 与现状不一致的地方**

运行：

```bash
grep -n "3 个 Agent\\|4 个 Task\\|Flow\\|gate\\|blocked\\|recurrent" /Users/yeziqing/Projects/AgentVest/README.md
```

预期：发现 README 尚未明确写出 `passed / rerun / blocked` 控制流、结构化 gate 优先级和 blocked recommendation 边界。

- [ ] **步骤 2：更新 README 的系统定位**

```markdown
- 新增：结构化 gate 与 recurrent 说明
- 新增：正式报告与阻断报告的差异
- 新增：最小测试命令
- 新增：为何这个项目更像“可讲清楚的实习项目”而不是单纯 Demo
```

- [ ] **步骤 3：补“当前关键模块”一节**

```markdown
- `src/multi_agent/flows/market_review_flow.py`
- `src/multi_agent/core/confidence_gate.py`
- `src/multi_agent/tools/review_tools.py`
- `tests/test_flow_routing.py`
```

- [ ] **步骤 4：验证 README 可读性**

运行：

```bash
grep -n "rerun\\|blocked\\|ConfidenceGatePolicy\\|test_flow_routing.py" /Users/yeziqing/Projects/AgentVest/README.md
```

预期：README 能准确反映当前 gate/recurrent 行为，且不再只是静态架构介绍。

- [ ] **步骤 5：Commit**

```bash
git add /Users/yeziqing/Projects/AgentVest/README.md
git commit -m "docs: refresh readme for gate and recurrent workflow"
```

### 任务 5：更新架构文档，让仓库叙述与代码一致

**文件：**
- 修改：`/Users/yeziqing/Projects/AgentVest/docs/project_architecture.md`

- [ ] **步骤 1：写失败检查，确认旧文档仍停留在 3 Agent/4 Task**

运行：

```bash
grep -n "3 个 Agent\\|4 个 Task\\|information_gathering_analyst\\|financial_statement_analyst" /Users/yeziqing/Projects/AgentVest/docs/project_architecture.md
```

预期：命中旧描述，证明文档需要更新。

- [ ] **步骤 2：按当前实现重写架构说明**

```markdown
- 7 Agent / 7 Task
- `Flow + Gate + Recurrent`
- `passed / rerun / blocked`
- 结构化 summary 优先，Markdown 仅 fallback
- blocked 报告与正式报告的产物边界
```

- [ ] **步骤 3：在主链路中明确 Flow 阶段**

```markdown
1. validate_market
2. run_analysis
3. apply_analysis_gate
4. rerun_analysis_if_needed（条件触发）
5. write_report
6. review_report
7. finalize_delivery
```

- [ ] **步骤 4：验证文档与代码一致**

运行：

```bash
grep -n "7 个 Agent\\|7 个 Task\\|rerun\\|blocked\\|Flow" /Users/yeziqing/Projects/AgentVest/docs/project_architecture.md
```

预期：旧的 3 Agent/4 Task 叙述消失，文档可直接用于面试讲解。

- [ ] **步骤 5：Commit**

```bash
git add /Users/yeziqing/Projects/AgentVest/docs/project_architecture.md
git commit -m "docs: align architecture doc with current flow design"
```

### 任务 6：做最终回归并整理交付说明

**文件：**
- 修改：`/Users/yeziqing/Projects/AgentVest/docs/superpowers/plans/2026-06-22-structured-gate-recurrent-implementation-plan.md`
- 测试：`/Users/yeziqing/Projects/AgentVest/tests/test_review_gate.py`
- 测试：`/Users/yeziqing/Projects/AgentVest/tests/test_flow_routing.py`
- 测试：`/Users/yeziqing/Projects/AgentVest/tests/test_main_cli.py`

- [ ] **步骤 1：运行最终测试集**

运行：

```bash
pytest /Users/yeziqing/Projects/AgentVest/tests/test_review_gate.py \
       /Users/yeziqing/Projects/AgentVest/tests/test_flow_routing.py \
       /Users/yeziqing/Projects/AgentVest/tests/test_main_cli.py -q
```

预期：全部 PASS。

- [ ] **步骤 2：运行诊断检查**

执行：

```text
GetDiagnostics:
- src/multi_agent/core/review_contracts.py
- src/multi_agent/core/confidence_gate.py
- src/multi_agent/flows/market_review_flow.py
- src/multi_agent/main.py
- tests/test_review_gate.py
- tests/test_flow_routing.py
- tests/test_main_cli.py
```

预期：无新的 diagnostics。

- [ ] **步骤 3：记录实际验证结果到计划尾部**

```markdown
## 执行结果
- reviewer gate 三态：PASS
- Flow rerun 路由：PASS
- blocked recommendation 语义：PASS
- README / 架构文档一致性：PASS
```

- [ ] **步骤 4：Commit**

```bash
git add \
  /Users/yeziqing/Projects/AgentVest/docs/superpowers/plans/2026-06-22-structured-gate-recurrent-implementation-plan.md
git commit -m "docs: record final verification for gate recurrent plan"
```
