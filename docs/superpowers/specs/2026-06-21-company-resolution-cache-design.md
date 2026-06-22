# 公司解析缓存设计说明

**目标**

把当前 `resolver.py` 中以 Python 常量维护的公司别名解析逻辑，升级为：

- 极小 seed 映射
- `SQLite` 主缓存
- `LLM` 兜底解析
- `CLI` 人工确认后写回缓存

该设计只服务当前 CLI 主路径，不引入后端服务，不改变投研 Agent 主流程。

---

## 背景

当前公司解析链路是：

1. 先查 `CompanyResolver._ALIASES`
2. 再查官方 SEC 名称匹配
3. 最后用 `OpenAICompanyResolver` 兜底

这有两个主要问题：

- `_ALIASES` 作为 Python 常量扩展性差，每增加一个中文简称都要改代码
- `LLM` 虽然能提高泛化能力，但如果没有持久化和人工确认，解析成功也无法沉淀成后续可复用知识

因此需要把“公司简称 -> 标准上市实体”的成功映射，从代码常量迁移到本地持久化缓存中。

---

## 设计原则

### 1. 解析知识属于运行态数据，不属于业务代码

- 高频简称和历史成功映射应存到本地 SQLite
- Python 代码只保留极小 seed，用于冷启动和测试稳定性

### 2. LLM 结果不能直接入库

- `LLM` 解析结果只有在用户人工确认后，才能继续本次运行并写入缓存
- 未确认的 LLM 结果不得进入 SQLite

### 3. CLI 优先，非交互模式保守失败

- 当前项目阶段只保障 CLI 可正常运行
- 因此“人工确认”只在 CLI 交互路径启用
- 对非交互路径，若只能依赖未确认的 LLM 结果，应直接失败，不自动接受

### 4. 不引入复杂抽象

- 不新增 ORM
- 不引入外部数据库依赖
- 只增加一个很薄的 SQLite 存储层

---

## 目标行为

### 交互式 CLI

当用户输入例如 `微软`：

1. seed 未命中
2. SQLite 未命中
3. 官方源未直接匹配
4. `LLM` 给出候选：
   - `Microsoft Corporation`
   - `MSFT`
   - `NASDAQ`
5. CLI 向用户展示候选结果并请求确认
6. 若用户确认：
   - 本次运行继续
   - 写入 SQLite
7. 若用户拒绝：
   - 本次运行终止
   - 提示用户使用更完整公司名或 ticker

### 已缓存输入

同样的用户再次输入 `微软`：

1. 直接命中 SQLite
2. 不再询问人工确认
3. 直接进入后续投研工作流

### 非交互模式

如果后续触发器或批处理路径无法进行人工确认：

- 若命中 seed / SQLite / 官方源，可继续
- 若只能依赖未确认的 LLM 结果，必须失败

---

## 架构方案

### 解析顺序

新的 `CompanyResolver` 采用以下顺序：

1. seed 映射
2. SQLite 缓存
3. 官方源解析
4. LLM 兜底
5. 人工确认
6. 写回 SQLite

### 模块分工

#### `resolver.py`

职责：

- 编排解析顺序
- 统一返回 `CompanyResolution`
- 不再承担大规模别名字典维护

不负责：

- SQLite 底层读写细节
- CLI 输入交互

#### `company_resolution_store.py`

职责：

- SQLite 读写
- 根据 `input_name` / `ticker` 查询已有映射
- 将确认后的解析结果落库

不负责：

- 公司解析
- 调用 LLM

#### `main.py`

职责：

- 在 CLI 交互模式下展示 LLM 候选解析结果
- 收集用户确认
- 把确认结果传回 resolver / store

不负责：

- 解析逻辑本身
- 缓存读写策略

---

## SQLite 存储设计

### 文件位置

SQLite 文件放在 `artifacts` 体系下：

- 推荐路径：`src/multi_agent/artifacts/company_resolution_cache.sqlite3`

这样可以与当前 CLI 产物、watchlist 保持一致的运行态数据管理方式。

### 表设计

表名：`company_resolution_cache`

建议字段：

- `input_name` TEXT PRIMARY KEY
- `normalized_name` TEXT NOT NULL
- `ticker` TEXT NOT NULL
- `exchange` TEXT NOT NULL DEFAULT ''
- `entity_type` TEXT NOT NULL
- `parent_company` TEXT NOT NULL
- `confidence` REAL NOT NULL
- `source` TEXT NOT NULL
- `confirmed_by_user` INTEGER NOT NULL DEFAULT 0
- `created_at` TEXT NOT NULL
- `updated_at` TEXT NOT NULL

### source 取值

- `seed`
- `official_sec`
- `llm_confirmed`
- `manual`

### 入库规则

- `seed` 可直接初始化
- `official_sec` 成功结果可直接入库
- `llm_confirmed` 必须在用户确认后入库
- 不允许未确认的 LLM 结果入库

---

## seed 策略

seed 仍然保留，但只保留极小集合。

用途：

- 冷启动体验
- 测试稳定性
- 覆盖最高频、最确定的中文简称

约束：

- 不再把 seed 当作长期知识库
- seed 数量应控制在 5-10 条左右

---

## 用户确认策略

### 确认提示

建议 CLI 展示：

```text
解析候选：
- 输入：微软
- 标准公司：Microsoft Corporation
- ticker：MSFT
- exchange：NASDAQ
- 类型：public_company
- 置信度：0.96

是否确认使用该映射，并写入本地缓存？ [Y/n]
```

### 用户确认后

- 返回该 `CompanyResolution`
- 将结果写入 SQLite
- 本次运行继续

### 用户拒绝后

- 抛出用户可理解错误
- 不写入 SQLite

---

## 接口建议

### `CompanyResolutionStore`

建议提供的最小接口：

- `lookup(input_name: str) -> CompanyResolution | None`
- `lookup_by_ticker(ticker: str) -> CompanyResolution | None`
- `upsert_resolution(resolution: CompanyResolution, source: str, confirmed_by_user: bool) -> None`

不额外增加搜索、分页、淘汰等复杂能力。

### `CompanyResolver`

建议新增能力：

- 接受可选的 `resolution_store`
- 接受可选的 `confirmation_callback`

其中：

- `confirmation_callback` 返回 `True / False`
- CLI 路径由 `main.py` 注入交互确认实现
- 非交互路径可传入固定拒绝实现

---

## 错误处理

### LLM 成功但用户拒绝

- 报错信息应明确提示：
  - 当前解析结果未被确认
  - 请提供更完整公司名或 ticker

### SQLite 不可用

- 不应阻塞整个 CLI
- 应降级为：
  - seed
  - 官方源
  - LLM + 人工确认
- 但不给持久化能力

### 非交互模式下遇到 LLM 候选

- 必须失败
- 不允许默认接受

---

## 测试范围

至少覆盖以下场景：

1. seed 命中直接返回
2. SQLite 命中直接返回
3. 官方源命中并写入 SQLite
4. LLM 给出候选，用户确认后继续并写入 SQLite
5. LLM 给出候选，用户拒绝后失败
6. 非交互模式下 LLM 候选直接失败
7. SQLite 损坏或不可写时降级运行

---

## 范围边界

本次不做：

- 后端服务化
- 共享远程数据库
- 自动同步团队公共映射库
- 复杂别名搜索引擎
- 模糊匹配排名系统

本次只做：

- 本地 SQLite 主缓存
- LLM 结果人工确认
- CLI 路径稳定可用

---

## 最终决策

本设计确认采用以下行为：

1. `LLM` 结果必须先人工确认，确认后才能继续本次运行
2. 已确认的结果写入 SQLite，后续命中不再重复确认
3. Python 文件中只保留极小 seed，SQLite 才是公司解析映射的主存储

