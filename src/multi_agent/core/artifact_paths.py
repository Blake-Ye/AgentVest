from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_ARTIFACT_ROOT = "src/multi_agent/artifacts"
DEFAULT_RUNS_DIR = str(Path(DEFAULT_ARTIFACT_ROOT) / "runs")
DEFAULT_LATEST_DIR = str(Path(DEFAULT_ARTIFACT_ROOT) / "latest")
DEFAULT_WATCHLIST_PATH = str(Path(DEFAULT_ARTIFACT_ROOT) / "watchlist.json")
DEFAULT_FINAL_REPORT_PATH = str(Path(DEFAULT_RUNS_DIR) / "04_investment_report.md")


@dataclass(frozen=True)
class RunArtifactPaths:
    company_dir: Path
    run_dir: Path
    market_intelligence_path: Path
    filing_review_path: Path
    financial_analysis_path: Path
    final_report_path: Path
    runtime_log_path: Path
    structured_recommendation_path: Path
    structured_report_path: Path
    latest_metrics_path: Path
    evaluation_summary_path: Path
    readme_path: Path
    evidence_bundle_path: Path
    report_document_path: Path
    final_decision_path: Path

    @property
    def market_validation_json(self) -> Path:
        return self.market_intelligence_path

    @property
    def final_report_md(self) -> Path:
        return self.final_report_path

    @property
    def final_report_json(self) -> Path:
        return self.structured_report_path


def build_default_runs_dir(artifact_root: str = DEFAULT_ARTIFACT_ROOT) -> str:
    return str(Path(artifact_root) / "runs")


def build_default_latest_dir(artifact_root: str = DEFAULT_ARTIFACT_ROOT) -> str:
    return str(Path(artifact_root) / "latest")


def build_default_watchlist_path(artifact_root: str = DEFAULT_ARTIFACT_ROOT) -> str:
    return str(Path(artifact_root) / "watchlist.json")


def build_run_artifact_paths(run_dir: Path) -> RunArtifactPaths:
    return RunArtifactPaths(
        company_dir=run_dir.parent,
        run_dir=run_dir,
        market_intelligence_path=run_dir / "01_market_intelligence.md",
        filing_review_path=run_dir / "02_filing_review.md",
        financial_analysis_path=run_dir / "03_financial_analysis.md",
        final_report_path=run_dir / "04_investment_report.md",
        runtime_log_path=run_dir / "05_runtime.txt",
        structured_recommendation_path=run_dir / "06_structured_recommendation.json",
        structured_report_path=run_dir / "07_structured_report.json",
        latest_metrics_path=run_dir / "latest_run_metrics.json",
        evaluation_summary_path=run_dir / "evaluation_summary.json",
        readme_path=run_dir / "README.md",
        evidence_bundle_path=run_dir / "10_research_evidence.json",
        report_document_path=run_dir / "11_report_document.json",
        final_decision_path=run_dir / "final_decision.json",
    )
