from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import tempfile
from typing import Any

import pandas as pd
import polars as pl

from ofbot.config import AppConfig
from ofbot.engine import PaperEngine
from ofbot.replay.player import iter_replay_events_from_frame


@dataclass(slots=True)
class ReplayBacktestResult:
    strategy_name: str
    params: dict[str, Any]
    split_name: str
    metrics: dict[str, Any]


def build_single_strategy_config(
    *,
    base_config: AppConfig,
    strategy_name: str,
    params: dict[str, Any],
    execution_overrides: dict[str, Any] | None = None,
    disable_learning: bool = True,
) -> AppConfig:
    config = base_config.model_copy(deep=True)
    config.strategy_library.default = strategy_name  # type: ignore[assignment]
    for variant_name in ("continuation", "exhaustion", "hybrid"):
        variant = getattr(config.strategy_library, variant_name)
        variant.enabled = variant_name == strategy_name
        if variant_name == strategy_name:
            merged = dict(variant.params)
            merged.update(params)
            variant.params = merged
    if disable_learning:
        config.learning.enabled = False
        config.learning.enable_promotion = False
    config.paper_local_record_raw = False
    if execution_overrides:
        for key, value in execution_overrides.items():
            setattr(config.execution, key, value)
    return config


def run_replay_backtest(
    *,
    base_config: AppConfig,
    strategy_name: str,
    params: dict[str, Any],
    frame: pl.DataFrame,
    split_name: str,
    execution_overrides: dict[str, Any] | None = None,
) -> ReplayBacktestResult:
    events = iter_replay_events_from_frame(frame)
    runtime_config = build_single_strategy_config(
        base_config=base_config,
        strategy_name=strategy_name,
        params=params,
        execution_overrides=execution_overrides,
    )
    with tempfile.TemporaryDirectory(prefix="ofbot_research_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        runtime_config.paths.raw_dir = temp_dir / "raw"
        runtime_config.paths.reports_dir = temp_dir / "reports"
        runtime_config.paths.memory_dir = temp_dir / "memory"
        runtime_config.learning.duckdb_path = temp_dir / "memory" / "learning.duckdb"

        engine = PaperEngine(config=runtime_config, depth_snapshot_client=None, runtime_clock="event")
        try:
            engine.process_events(events)
            trades = engine.store.run_query_df(
                """
                SELECT *
                FROM trade_contexts
                WHERE run_id=?
                ORDER BY exit_time ASC
                """,
                [engine.run_id],
            )
            journal = engine.store.run_query_df(
                """
                SELECT *
                FROM event_journal
                WHERE run_id=?
                ORDER BY event_time ASC, id ASC
                """,
                [engine.run_id],
            )
            metrics = compute_replay_metrics(trades=trades, journal=journal)
        finally:
            engine.close()
    return ReplayBacktestResult(
        strategy_name=strategy_name,
        params=dict(params),
        split_name=split_name,
        metrics=metrics,
    )


def compute_replay_metrics(
    *,
    trades: pd.DataFrame,
    journal: pd.DataFrame,
) -> dict[str, Any]:
    if trades.empty:
        return {
            "trade_count": 0,
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "total_fees": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "sharpe_like": 0.0,
            "average_holding_seconds": 0.0,
            "expected_edge_blocks": _count_matches(journal, "reason", "risk_expected_net_edge"),
            "signal_submit_count": _count_matches(journal, "order_intent", "submit"),
        }

    realized = pd.to_numeric(trades.get("realized_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    gross = pd.to_numeric(trades.get("gross_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    fees = pd.to_numeric(trades.get("fees", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    holdings = pd.to_numeric(trades.get("holding_seconds", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    entry_notional = pd.to_numeric(
        trades.get("entry_notional", trades.get("entry_price", pd.Series(dtype=float)) * trades.get("qty", 0.0)),
        errors="coerce",
    ).replace(0.0, pd.NA)
    trade_returns = (realized / entry_notional).replace([pd.NA, math.inf, -math.inf], pd.NA).dropna()

    winners = realized[realized > 0.0]
    losers = realized[realized < 0.0]
    negative_abs = abs(float(losers.sum()))
    if negative_abs <= 1e-12:
        profit_factor = float("inf") if float(winners.sum()) > 0.0 else 0.0
    else:
        profit_factor = float(winners.sum()) / negative_abs

    sharpe_like = 0.0
    if len(trade_returns.index) > 1 and float(trade_returns.std(ddof=1)) > 0.0:
        sharpe_like = float(trade_returns.mean() / trade_returns.std(ddof=1) * math.sqrt(len(trade_returns.index)))

    return {
        "trade_count": int(len(trades.index)),
        "net_pnl": float(realized.sum()),
        "gross_pnl": float(gross.sum()),
        "total_fees": float(fees.sum()),
        "max_drawdown": _max_drawdown(realized),
        "win_rate": float((realized > 0.0).mean()),
        "profit_factor": float(profit_factor),
        "expectancy": float(realized.mean()),
        "sharpe_like": float(sharpe_like),
        "average_holding_seconds": float(holdings.mean()) if len(holdings.index) > 0 else 0.0,
        "expected_edge_blocks": _count_matches(journal, "reason", "risk_expected_net_edge"),
        "signal_submit_count": _count_matches(journal, "order_intent", "submit"),
    }


def _max_drawdown(realized: pd.Series) -> float:
    cumulative = realized.cumsum()
    peak = cumulative.cummax()
    drawdown = peak - cumulative
    if drawdown.empty:
        return 0.0
    return float(drawdown.max())


def _count_matches(frame: pd.DataFrame, column: str, expected: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int((frame[column].fillna("") == expected).sum())
