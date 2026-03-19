from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl
from quantflow.backtest.engine import run_backtest
from quantflow.core.config import AppConfig
from quantflow.strategy.base import Strategy
from quantflow.validation.splits import slice_frame_by_time_window, walk_forward_time_splits


@dataclass(slots=True)
class WalkForwardResult:
    windows: list[dict[str, Any]]
    aggregate: dict[str, Any]


def run_walk_forward(
    features: pl.DataFrame,
    config: AppConfig,
    strategy_factory: Callable[[], Strategy],
) -> WalkForwardResult:
    windows = walk_forward_time_splits(
        features,
        train_bars=config.validation.train_bars,
        test_bars=config.validation.test_bars,
        step_bars=config.validation.step_bars,
        embargo_bars=config.validation.embargo_bars,
    )
    if not windows:
        raise ValueError("No walk-forward windows available with the current configuration")

    window_results: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        train_frame = slice_frame_by_time_window(
            features,
            start_time=window.train_start_time,
            end_time=window.train_end_time,
        )
        test_frame = slice_frame_by_time_window(
            features,
            start_time=window.test_start_time,
            end_time=window.test_end_time,
        )
        train_result = run_backtest(train_frame, strategy_factory(), config)
        test_result = run_backtest(test_frame, strategy_factory(), config)
        window_results.append(
            {
                "window_index": index,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "train_start_time": window.train_start_time.isoformat(),
                "train_end_time": window.train_end_time.isoformat(),
                "test_start_time": window.test_start_time.isoformat(),
                "test_end_time": window.test_end_time.isoformat(),
                "train_net_pnl": train_result.metrics["net_pnl"],
                "test_net_pnl": test_result.metrics["net_pnl"],
                "train_max_drawdown": train_result.metrics["max_drawdown"],
                "test_max_drawdown": test_result.metrics["max_drawdown"],
                "train_trade_count": train_result.metrics["trade_count"],
                "test_trade_count": test_result.metrics["trade_count"],
            }
        )

    positive_tests = sum(1 for item in window_results if float(item["test_net_pnl"]) > 0.0)
    aggregate = {
        "window_count": len(window_results),
        "mean_test_net_pnl": sum(float(item["test_net_pnl"]) for item in window_results)
        / len(window_results),
        "median_test_net_pnl": sorted(float(item["test_net_pnl"]) for item in window_results)[
            len(window_results) // 2
        ],
        "positive_test_window_ratio": positive_tests / len(window_results),
    }
    return WalkForwardResult(windows=window_results, aggregate=aggregate)
