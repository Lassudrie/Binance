from __future__ import annotations

from typing import Any

import numpy as np
from quantflow.backtest.portfolio import TradeRecord
from quantflow.data_model.events import FillEvent


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def compute_summary_metrics(
    *,
    equity_curve: list[dict[str, float | str]],
    trades: list[TradeRecord],
    fills: list[FillEvent],
    initial_cash: float,
    bar_seconds: float,
) -> dict[str, Any]:
    equities = np.array([float(point["equity"]) for point in equity_curve], dtype=float)
    if equities.size == 0:
        raise ValueError("Equity curve is empty")

    running_max = np.maximum.accumulate(equities)
    drawdowns = (equities / running_max) - 1.0
    returns = np.diff(equities) / np.maximum(equities[:-1], 1e-12)

    total_fees = sum(fill.fee for fill in fills)
    turnover = sum(abs(fill.notional) for fill in fills)
    avg_gross_exposure = float(np.mean([float(point["gross_exposure"]) for point in equity_curve]))
    avg_equity = float(np.mean(equities))
    exposure_ratio = _safe_ratio(avg_gross_exposure, avg_equity)

    trade_pnls = (
        np.array([trade.net_pnl for trade in trades], dtype=float) if trades else np.array([])
    )
    positive = trade_pnls[trade_pnls > 0.0]
    negative = trade_pnls[trade_pnls < 0.0]
    long_pnl = sum(trade.net_pnl for trade in trades if trade.side == "long")
    short_pnl = sum(trade.net_pnl for trade in trades if trade.side == "short")

    sharpe = None
    if returns.size > 1 and float(np.std(returns, ddof=1)) > 0.0:
        annualization = np.sqrt((86_400.0 * 365.0) / bar_seconds)
        sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * annualization)

    return {
        "initial_cash": initial_cash,
        "final_equity": float(equities[-1]),
        "net_pnl": float(equities[-1] - initial_cash),
        "gross_pnl": float(equities[-1] - initial_cash + total_fees),
        "total_fees": float(total_fees),
        "turnover_notional": float(turnover),
        "turnover_multiple": _safe_ratio(float(turnover), initial_cash),
        "max_drawdown": float(drawdowns.min()),
        "average_gross_exposure_ratio": exposure_ratio,
        "trade_count": len(trades),
        "fill_count": len(fills),
        "hit_ratio": None if trade_pnls.size == 0 else float(np.mean(trade_pnls > 0.0)),
        "expectancy": None if trade_pnls.size == 0 else float(np.mean(trade_pnls)),
        "profit_factor": None
        if negative.size == 0
        else float(positive.sum() / abs(negative.sum()))
        if positive.size > 0
        else 0.0,
        "average_holding_seconds": None
        if not trades
        else float(np.mean([trade.holding_seconds for trade in trades])),
        "long_pnl": float(long_pnl),
        "short_pnl": float(short_pnl),
        "sharpe_intraday": sharpe,
        "sharpe_warning": "Intraday Sharpe is unstable and should not be used alone.",
    }
