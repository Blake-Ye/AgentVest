import json
from pathlib import Path

from multi_agent import benchmark


def test_benchmark_cli_passes_key_options_to_runner(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "benchmark.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "company_name": "Apple Inc.",
                "ticker": "AAPL",
                "expected_facts": {"form_type": "10-K"},
                "expected_key_points": {"risks": [], "catalysts": [], "thesis": []},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"
    captured: dict[str, object] = {}

    def _stub_run_benchmark(*, dataset_path, output_dir, judge_model, factual_only):
        captured["dataset_path"] = dataset_path
        captured["output_dir"] = output_dir
        captured["judge_model"] = judge_model
        captured["factual_only"] = factual_only
        return {"summary_path": str(output_dir / "benchmark_summary.json")}

    monkeypatch.setattr(benchmark, "run_benchmark", _stub_run_benchmark)

    benchmark.main(
        [
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--judge-model",
            "openai/gpt-4o-mini",
            "--factual-only",
        ]
    )

    assert captured["dataset_path"] == dataset_path
    assert captured["output_dir"] == output_dir
    assert captured["judge_model"] == "openai/gpt-4o-mini"
    assert captured["factual_only"] is True
