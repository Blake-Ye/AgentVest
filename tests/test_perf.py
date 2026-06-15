import cProfile
from pathlib import Path

from multi_agent.perf import (
    PerfRunConfig,
    build_cprofile_command,
    build_profile_target_args,
    build_py_spy_command,
    summarize_profile,
)


def _sample_workload() -> int:
    total = 0
    for value in range(2000):
        total += value * value
    return total


def test_build_profile_target_args_for_workflow_mode() -> None:
    config = PerfRunConfig(
        mode="workflow",
        company_name="Apple Inc.",
        company_ticker="AAPL",
    )

    args = build_profile_target_args(config)

    assert args == [
        "-m",
        "multi_agent.main",
        "--company-name",
        "Apple Inc.",
        "--company-ticker",
        "AAPL",
    ]


def test_build_profile_target_args_for_rebuild_mode() -> None:
    config = PerfRunConfig(mode="watchlist-rebuild")

    args = build_profile_target_args(config)

    assert args == ["-m", "multi_agent.main", "--watchlist-rebuild"]


def test_build_cprofile_command_wraps_module_execution() -> None:
    config = PerfRunConfig(
        mode="workflow",
        company_name="Alibaba Group Holding Ltd",
        company_ticker="BABA",
    )

    command = build_cprofile_command(
        python_executable="python",
        output_path=Path("artifacts/profiles/run.prof"),
        config=config,
    )

    assert command == [
        "python",
        "-m",
        "cProfile",
        "-o",
        "artifacts/profiles/run.prof",
        "-m",
        "multi_agent.main",
        "--company-name",
        "Alibaba Group Holding Ltd",
        "--company-ticker",
        "BABA",
    ]


def test_build_py_spy_command_records_svg() -> None:
    config = PerfRunConfig(mode="watchlist-rebuild")

    command = build_py_spy_command(
        py_spy_executable="py-spy",
        python_executable="python",
        output_path=Path("artifacts/profiles/rebuild.svg"),
        config=config,
    )

    assert command == [
        "py-spy",
        "record",
        "-o",
        "artifacts/profiles/rebuild.svg",
        "--",
        "env",
        "PYTHONPATH=src",
        "python",
        "-m",
        "multi_agent.main",
        "--watchlist-rebuild",
    ]


def test_summarize_profile_returns_top_functions(tmp_path: Path) -> None:
    profile_path = tmp_path / "sample.prof"
    cProfile.runctx("_sample_workload()", globals(), locals(), str(profile_path))

    summary = summarize_profile(profile_path, top_n=5)

    assert summary["profile_path"] == str(profile_path)
    assert summary["top_n"] == 5
    assert summary["total_calls"] > 0
    assert summary["primitive_calls"] > 0
    assert summary["total_seconds"] >= 0.0
    assert any("_sample_workload" in item["function"] for item in summary["top_functions"])
