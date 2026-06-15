from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompanyFinancialSnapshot:
    """用于关键财务比率计算的标准化财务快照。"""

    revenue: float
    gross_profit: float
    operating_income: float
    net_income: float
    current_assets: float
    current_liabilities: float
    total_assets: float
    total_liabilities: float
    operating_cash_flow: float
    capital_expenditure: float


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_key_metrics(snapshot: CompanyFinancialSnapshot) -> dict[str, float]:
    """计算适合投研与面试讲解的核心财务指标。"""

    free_cash_flow = snapshot.operating_cash_flow + snapshot.capital_expenditure
    return {
        "gross_margin": _safe_divide(snapshot.gross_profit, snapshot.revenue),
        "operating_margin": _safe_divide(snapshot.operating_income, snapshot.revenue),
        "net_margin": _safe_divide(snapshot.net_income, snapshot.revenue),
        "current_ratio": _safe_divide(snapshot.current_assets, snapshot.current_liabilities),
        "debt_to_assets": _safe_divide(snapshot.total_liabilities, snapshot.total_assets),
        "free_cash_flow": free_cash_flow,
    }
