from __future__ import annotations

import argparse
import json
import os
import pstats
import subprocess
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PerfMode = Literal["workflow", "watchlist-rebuild"]


@dataclass(frozen=True)
class PerfRunConfig:
    mode: PerfMode
    company_name: str = ""
    company_ticker: str = ""


def build_profile_target_args(config: PerfRunConfig) -> list[str]:
    if config.mode == "watchlist-rebuild":
        return ["-m", "multi_agent.main", "--watchlist-rebuild"]

    args = ["-m", "multi_agent.main"]
    if config.company_name.strip():
        args.extend(["--company-name", config.company_name])
    if config.company_ticker.strip():
        args.extend(["--company-ticker", config.company_ticker])
    return args


def build_cprofile_command(
    *,
    python_executable: str,
    output_path: Path,
    config: PerfRunConfig,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "cProfile",
        "-o",
        str(output_path),
        *build_profile_target_args(config),
    ]


def build_py_spy_command(
    *,
    py_spy_executable: str,
    python_executable: str,
    output_path: Path,
    config: PerfRunConfig,
) -> list[str]:
    return [
        py_spy_executable,
        "record",
        "-o",
        str(output_path),
        "--",
        "env",
        "PYTHONPATH=src",
        python_executable,
        *build_profile_target_args(config),
    ]


def summarize_profile(profile_path: Path, *, top_n: int = 20) -> dict[str, object]:
    stats = pstats.Stats(str(profile_path))
    ranked_items = sorted(
        stats.stats.items(),
        key=lambda item: item[1][3],
        reverse=True,
    )[:top_n]
    top_functions = [
        {
            "function": f"{func[0]}:{func[1]}({func[2]})",
            "call_count": int(values[0]),
            "primitive_call_count": int(values[1]),
            "total_seconds": round(float(values[2]), 6),
            "cumulative_seconds": round(float(values[3]), 6),
        }
        for func, values in ranked_items
    ]
    return {
        "profile_path": str(profile_path),
        "top_n": top_n,
        "total_calls": int(stats.total_calls),
        "primitive_calls": int(stats.prim_calls),
        "total_seconds": round(float(stats.total_tt), 6),
        "top_functions": top_functions,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="性能分析辅助工具。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name in ("print-commands", "run-cprofile", "run-py-spy"):
        subparser = subparsers.add_parser(command_name)
        subparser.add_argument(
            "--mode",
            choices=["workflow", "watchlist-rebuild"],
            default="workflow",
        )
        subparser.add_argument("--company-name", default="Apple Inc.")
        subparser.add_argument("--company-ticker", default="")
        subparser.add_argument(
            "--output",
            default="artifacts/profiles/profile.out",
            help="输出文件路径。",
        )

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--profile", required=True, help=".prof 文件路径。")
    summarize_parser.add_argument("--top-n", type=int, default=20)

    return parser


def _config_from_args(args: argparse.Namespace) -> PerfRunConfig:
    return PerfRunConfig(
        mode=args.mode,
        company_name=args.company_name,
        company_ticker=args.company_ticker,
    )


def _run_command(command: list[str]) -> int:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", "src")
    completed = subprocess.run(command, env=env)
    return int(completed.returncode)


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "summarize":
        summary = summarize_profile(Path(args.profile), top_n=args.top_n)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    config = _config_from_args(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.command == "print-commands":
        payload = {
            "cprofile": build_cprofile_command(
                python_executable=sys.executable,
                output_path=output_path.with_suffix(".prof"),
                config=config,
            ),
            "py_spy": build_py_spy_command(
                py_spy_executable="py-spy",
                python_executable=sys.executable,
                output_path=output_path.with_suffix(".svg"),
                config=config,
            ),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if args.command == "run-cprofile":
        command = build_cprofile_command(
            python_executable=sys.executable,
            output_path=output_path,
            config=config,
        )
        raise SystemExit(_run_command(command))

    if args.command == "run-py-spy":
        command = build_py_spy_command(
            py_spy_executable="py-spy",
            python_executable=sys.executable,
            output_path=output_path,
            config=config,
        )
        raise SystemExit(_run_command(command))

    raise SystemExit(f"不支持的命令：{args.command}")


if __name__ == "__main__":
    main()
