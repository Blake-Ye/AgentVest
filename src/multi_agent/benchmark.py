from __future__ import annotations

import argparse

from pathlib import Path
from typing import Sequence

from multi_agent.benchmark_runner import run_benchmark
from multi_agent.settings import BenchmarkSettings


def build_parser() -> argparse.ArgumentParser:
    """构建 benchmark CLI，默认值统一从 BenchmarkSettings 读取。"""

    settings = BenchmarkSettings.from_env()
    parser = argparse.ArgumentParser(description="运行自动化投研内容质量 Benchmark。")
    parser.add_argument(
        "--dataset",
        required=True,
        help="golden dataset JSONL 文件路径。",
    )
    parser.add_argument(
        "--output-dir",
        default=settings.output_dir,
        help="Benchmark 输出目录，默认读取 BENCHMARK_OUTPUT_DIR。",
    )
    parser.add_argument(
        "--judge-model",
        default=settings.judge_model,
        help="LLM judge 使用的模型名。",
    )
    parser.add_argument(
        "--factual-only",
        action="store_true",
        help="只运行 factual check，不调用 LLM judge。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    """命令行入口：解析参数、执行 benchmark、打印结果文件位置。"""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = run_benchmark(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        judge_model=str(args.judge_model).strip(),
        factual_only=bool(args.factual_only),
    )
    print(f"Benchmark 完成，汇总结果：{result['summary_path']}")
    if "results_path" in result:
        print(f"详细结果：{result['results_path']}")
    return result


if __name__ == "__main__":
    main()
