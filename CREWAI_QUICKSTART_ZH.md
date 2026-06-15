# CrewAI Quickstart To Advanced

这份文档基于当前项目的实际配置整理，目标是让你从“能跑”快速过渡到“能改、能扩展、能调试”。

当前项目关键信息：

- CrewAI 版本：`1.14.6`
- Python 版本要求：`>=3.10,<3.14`
- 当前模型接入方式：阿里 DashScope 的 OpenAI 兼容接口
- 当前默认模型：`qwen-plus`
- 推荐入口：`crewai run`
- 可选入口：`poetry run multi_agent`

## 1. 先理解 4 个核心概念

### Agent

Agent 是“角色化的智能体”，负责思考和决策。

它通常定义：

- `role`：角色
- `goal`：目标
- `backstory`：背景设定
- `llm`：使用哪个模型
- `tools`：它能调用哪些工具

### Task

Task 是“要完成的具体工作”。

它通常定义：

- `description`：任务描述
- `expected_output`：期望输出
- `agent`：由哪个 agent 执行
- `context`：依赖哪些前置任务输出

### Crew

Crew 是“多个 agent + 多个 task 的编排器”。

它通常定义：

- `agents`
- `tasks`
- `process`
- `verbose`

### Flow

Flow 是更高一级的“事件驱动工作流”，适合：

- 多阶段流程
- 条件分支
- 状态管理
- 多个 crew 串联

如果你只是做一个 2 到 5 步的协作任务，先用 Crew 即可；如果你要做完整产品流程，再上 Flow。

## 2. 当前项目结构

```text
multi_agent/
├── .env
├── pyproject.toml
├── src/multi_agent/
│   ├── crew.py
│   ├── main.py
│   ├── config/
│   │   ├── agents.yaml
│   │   └── tasks.yaml
│   └── tools/
│       └── custom_tool.py
└── report.md
```

你最常改的 5 个文件：

- `src/multi_agent/config/agents.yaml`
- `src/multi_agent/config/tasks.yaml`
- `src/multi_agent/crew.py`
- `src/multi_agent/main.py`
- `src/multi_agent/tools/custom_tool.py`

## 3. Quickstart：先跑通

### 环境准备

如果你继续使用当前 conda 环境：

```bash
conda activate MultiAgent
```

### 当前 `.env`

项目当前走阿里兼容 OpenAI 接口：

```env
MODEL=qwen-plus
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

如果你是国际站 key，改成：

```env
OPENAI_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
```

### 推荐运行方式

官方 CrewAI 入口：

```bash
conda run -n MultiAgent crewai run
```

或者先激活环境后运行：

```bash
crewai run
```

### Poetry 方式

如果你要走 Poetry 入口：

```bash
poetry run multi_agent
```

当前项目已经额外处理了 CrewAI 的本地存储目录，所以在受限环境下也能运行。

### 成功标志

成功后会在项目根目录生成：

```text
report.md
```

## 4. YAML 驱动开发

CrewAI 模板最重要的原则之一：先改 YAML，再改 Python。

### `agents.yaml`

这里写“谁来做”。

示例：

```yaml
researcher:
  role: >
    {topic} Senior Data Researcher
  goal: >
    Uncover cutting-edge developments in {topic}
  backstory: >
    You're a seasoned researcher...
```

### `tasks.yaml`

这里写“要做什么”。

示例：

```yaml
research_task:
  description: >
    Conduct a thorough research about {topic}
  expected_output: >
    A list with 10 bullet points...
  agent: researcher
```

### 变量插值

YAML 里的 `{topic}`、`{current_year}` 来自 `main.py` 的 `inputs`：

```python
inputs = {
    "topic": "AI LLMs",
    "current_year": "2026",
}
```

所以你改输入变量时，要同步检查：

- `main.py`
- `agents.yaml`
- `tasks.yaml`

## 5. `crew.py` 怎么看

`crew.py` 是把 YAML 配置变成可运行对象的地方。

典型结构：

```python
@CrewBase
class MultiAgent:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def _llm(self) -> LLM:
        return LLM(...)

    @agent
    def researcher(self) -> Agent:
        return Agent(config=self.agents_config["researcher"], llm=self._llm())

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, process=Process.sequential)
```

你可以把它理解成三层：

- `@agent`：把 YAML 里的 agent 变成 Python 对象
- `@task`：把 YAML 里的 task 变成 Python 对象
- `@crew`：把前两者编排起来

## 6. 最短开发路径

如果你要快速做一个新功能，建议按这个顺序：

1. 先改 `main.py` 输入
2. 再改 `agents.yaml`
3. 再改 `tasks.yaml`
4. 最后只在 `crew.py` 补工具、模型、流程控制
5. 跑 `crewai run`

这条路径的好处是：

- 改动小
- 调试快
- 不容易把 Python 逻辑写乱

## 7. Tool 怎么接进来

### Tool 的作用

Tool 是 agent 的“外接能力”，比如：

- 搜索网页
- 调接口
- 读文件
- 算法计算
- 调用数据库

### 自定义 Tool

当前项目已有示例：

`src/multi_agent/tools/custom_tool.py`

典型写法：

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(..., description="Search query")

class MyTool(BaseTool):
    name: str = "custom_search"
    description: str = "Search custom information."
    args_schema = SearchInput

    def _run(self, query: str) -> str:
        return f"Results for {query}"
```

### 挂到 Agent 上

```python
from multi_agent.tools.custom_tool import MyTool

@agent
def researcher(self) -> Agent:
    return Agent(
        config=self.agents_config["researcher"],  # type: ignore[index]
        llm=self._llm(),
        tools=[MyTool()],
        verbose=True,
    )
```

### 什么时候写 Tool

只有当 prompt 已经不够时再写 tool。

优先级建议：

1. 先只靠 prompt + task
2. 不够再加 tool
3. 再不够再上 Flow

## 8. LLM 配置的 3 种方式

CrewAI 常见有 3 种模型配置方式：

### 方式 1：`.env`

最简单，适合起步：

```env
MODEL=qwen-plus
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
```

### 方式 2：YAML 中指定

```yaml
researcher:
  llm: openai/gpt-4o
```

### 方式 3：Python 中直接 `LLM(...)`

适合你需要细控参数时：

```python
llm = LLM(
    model="qwen-plus",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.5,
)
```

当前项目已经用了方式 3。

## 9. 什么时候用 Crew，什么时候用 Flow

### 用 Crew

适合：

- 研究 -> 总结
- 生成 -> 审核
- 收集 -> 分析 -> 输出

也就是：任务链基本线性，前后依赖明确。

### 用 Flow

适合：

- 多阶段状态流转
- 条件分支
- 人工审核节点
- 多个 crew 串联
- 需要持久化和恢复执行

### 快速判断

- 3 个以内 agent，线性任务：用 Crew
- 需要 if/else、状态机、并发触发：用 Flow

## 10. 进阶能力地图

### Structured Output

当 task 输出要给后续代码消费时，优先结构化输出。

你可以用 `output_pydantic` 或 `response_format`。

适合：

- 提取字段
- 多任务传递 JSON
- 稳定接口输出

### Guardrails

当输出必须满足规范时，用 guardrail。

适合：

- 限制字数
- 校验 JSON 格式
- 检查是否缺字段

### Memory

适合长链任务或跨轮上下文增强，但不是所有项目都该一开始就开。

建议：

- 第一版先别开
- 确认有长期记忆需求再加

### Planning / Reasoning

适合复杂任务，但成本会升高。

如果只是模板研究报告，默认不开也能跑。

### Hierarchical Process

只有当你真的需要一个 manager agent 做分发时再用：

```python
process=Process.hierarchical
```

这时必须配置 `manager_llm` 或 `manager_agent`。

## 11. 最常见报错与排查顺序

### 1. `NameError` / decorator 未定义

先看 `crew.py`：

- 是否导入了对应装饰器
- 是否用了过时写法
- 方法名和 YAML key 是否一致

### 2. `ModuleNotFoundError`

先看：

- 当前是不是正确虚拟环境
- `crewai` 是否真的安装在这个环境里
- 用的是 `poetry` 还是 `conda`

### 3. API 调用失败

先查：

- `.env` 是否被加载
- `OPENAI_API_KEY` 是否正确
- `OPENAI_BASE_URL` 是否跟 key 所在地域匹配
- `MODEL` 名称是否有效

### 4. YAML 变量报错

先查：

- `main.py` 的 `inputs`
- YAML 里的 `{变量名}`
- 拼写是否一致

### 5. Task 不执行或输出异常

先查：

- `agent:` 名称是否和 `@agent` 方法名一致
- `expected_output` 是否过于模糊
- `verbose=True` 下日志里是否出现 tool 调用失败

## 12. 当前项目的推荐开发工作流

### 日常开发

```bash
conda activate MultiAgent
crewai run
```

### 如果你要测试 Poetry 入口

```bash
poetry run multi_agent
```

### 改 prompt / 改角色

优先改：

- `agents.yaml`
- `tasks.yaml`

### 改模型参数 / 加工具 / 改流程

优先改：

- `crew.py`

### 改输入变量

优先改：

- `main.py`

## 13. 一个推荐学习顺序

如果你想从 quickstart 一路学到能快速开发，我建议：

1. 跑通当前模板项目
2. 只改 `topic`
3. 改 `agents.yaml` 和 `tasks.yaml`
4. 给 researcher 加一个真实 tool
5. 让 task 输出结构化 JSON
6. 再学习 Flow
7. 最后再学 memory / guardrails / hierarchical

## 14. 你现在最值得记住的 5 条

- 先用 YAML 驱动，后用 Python 补逻辑
- 先跑通 Crew，再考虑 Flow
- 先不用 memory，除非你真的需要
- 先别写太多 tool，只补最缺的能力
- 入口优先用 `crewai run`，项目级脚本用 `poetry run multi_agent`

## 15. 下一步建议

如果你要快速进入实战，最适合的下一步是下面三选一：

1. 把 `topic` 改成命令行参数
2. 给 `researcher` 接一个真实搜索工具
3. 把输出改成结构化 JSON + Markdown 双输出
