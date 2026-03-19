from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from quantflow.backtest.engine import run_backtest
from quantflow.core.config import AppConfig
from quantflow.strategy.base import Strategy
from quantflow.validation.splits import slice_frame_by_time_window, walk_forward_time_splits


@dataclass(slots=True)
class BootstrapResult:
    windows: list[dict[str, Any]]
    aggregate: dict[str, Any]
    samples: list[dict[str, float]]


def _equity_returns(equity_curve: list[dict[str, float | str]]) -> np.ndarray:
    if len(equity_curve) < 2:
        return np.array([], dtype=float)
    equities = np.array([float(point["equity"]) for point in equity_curve], dtype=float)
    previous = np.maximum(equities[:-1], 1e-12)
    return np.diff(equities) / previous


def _block_bootstrap_samples(
    returns: np.ndarray,
    *,
    initial_cash: float,
    resamples: int,
    block_size: int,
) -> list[dict[str, float]]:
    if returns.size == 0:
        return [{"net_pnl": 0.0, "mean_bar_return": 0.0}] * resamples

    rng = np.random.default_rng(7)
    series_size = returns.size
    sample_records: list[dict[str, float]] = []

    for _ in range(resamples):
        sampled: list[float] = []
        while len(sampled) < series_size:
            start = int(rng.integers(0, series_size))
            for offset in range(block_size):
                sampled.append(float(returns[(start + offset) % series_size]))
                if len(sampled) >= series_size:
                    break
        array = np.array(sampled, dtype=float)
        compounded = float(np.prod(1.0 + array) - 1.0)
        sample_records.append(
            {
                "net_pnl": initial_cash * compounded,
                "mean_bar_return": float(np.mean(array)),
            }
        )

    return sample_records


def run_bootstrap_validation(
    features: pl.DataFrame,
    config: AppConfig,
    *,
    strategy_factory: Callable[[], Strategy],
) -> BootstrapResult:
    windows = walk_forward_time_splits(
        features,
        train_bars=config.validation.train_bars,
        test_bars=config.validation.test_bars,
        step_bars=config.validation.step_bars,
        embargo_bars=config.validation.embargo_bars,
    )
    if not windows:
        raise ValueError("No walk-forward windows available with the current configuration")

    window_records: list[dict[str, Any]] = []
    oos_returns: list[float] = []
    observed_net_pnl = 0.0

    for index, window in enumerate(windows):
        test_frame = slice_frame_by_time_window(
            features,
            start_time=window.test_start_time,
            end_time=window.test_end_time,
        )
        result = run_backtest(test_frame, strategy_factory(), config)
        observed_net_pnl += float(result.metrics["net_pnl"])
        returns = _equity_returns(result.equity_curve)
        oos_returns.extend(float(value) for value in returns)
        window_records.append(
            {
                "window_index": index,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "test_start_time": window.test_start_time.isoformat(),
                "test_end_time": window.test_end_time.isoformat(),
                "test_net_pnl": float(result.metrics["net_pnl"]),
                "test_trade_count": int(result.metrics["trade_count"]),
                "test_return_count": int(returns.size),
            }
        )

    returns_array = np.array(oos_returns, dtype=float)
    sample_records = _block_bootstrap_samples(
        returns_array,
        initial_cash=config.portfolio.initial_cash,
        resamples=config.validation.bootstrap_resamples,
        block_size=max(config.validation.bootstrap_block_bars, 1),
    )
    sample_net_pnls = np.array([record["net_pnl"] for record in sample_records], dtype=float)
    sample_mean_returns = np.array(
        [record["mean_bar_return"] for record in sample_records],
        dtype=float,
    )

    aggregate = {
        "window_count": len(window_records),
        "resample_count": len(sample_records),
        "observed_net_pnl": observed_net_pnl,
        "observed_mean_bar_return": (
            0.0 if returns_array.size == 0 else float(np.mean(returns_array))
        ),
        "ci_95_net_pnl": [
            float(np.quantile(sample_net_pnls, 0.025)),
            float(np.quantile(sample_net_pnls, 0.975)),
        ],
        "ci_95_mean_bar_return": [
            float(np.quantile(sample_mean_returns, 0.025)),
            float(np.quantile(sample_mean_returns, 0.975)),
        ],
        "probability_net_pnl_positive": float(np.mean(sample_net_pnls > 0.0)),
    }
    return BootstrapResult(windows=window_records, aggregate=aggregate, samples=sample_records)
