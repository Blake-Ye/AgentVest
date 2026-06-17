# Docs Navigation

本目录当前只把“真实入口、输出契约、benchmark 稳定边界”相关材料聚在一起。

如果某份文档没有直接回答下面三个问题，它就不应作为日常交接的主文档：

- 从哪里启动
- 稳定会产出什么
- benchmark 哪些字段和行为可以依赖

## 推荐阅读顺序

1. `../README.md`
2. `../outputs/README.md`
3. `evaluation/benchmark_evaluation_zh.md`
4. `guides/testing_engineer_quickcheck_zh.md`

## 文档职责

- `../README.md`
  - 根入口文档，描述真实 CLI/API 入口、主输出契约、artifact API 与 benchmark 边界。
- `../outputs/README.md`
  - 只解释如何查找主工作流与 benchmark 的输出目录，以及哪些文件名可依赖。
- `evaluation/benchmark_evaluation_zh.md`
  - 只解释 benchmark 的输入、输出、降级路径和稳定性边界。
- `guides/testing_engineer_quickcheck_zh.md`
  - 面向测试工程师的最小验收清单。

## 非主线文档

- `architecture/`
  - 保留架构背景材料，不作为当前输出契约的唯一依据。
- `operations/`
  - 保留性能分析和运行排障材料。
- `archive/`
  - 历史资料，不参与当前交接基线。
