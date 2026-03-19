from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import product
from typing import Any

import polars as pl
from quantflow.backtest.engine import run_backtest
from quantflow.core.config import AppConfig
from quantflow.strategy.base import Strategy


@dataclass(slots=True)
class SensitivityResult:
    scenarios: list[dict[str, Any]]
    aggregate: dict[str, Any]


def _scaled_values(base_value: float, multipliers: tuple[float, ...]) -> list[float]:
    return sorted({round(max(0.0, base_value * multiplier), 6) for multiplier in multipliers})


def _latency_values(base_latency: int) -> list[int]:
    return sorted({max(1, base_latency), max(1, base_latency + 1), max(1, base_latency + 2)})


def run_execution_sensitivity(
    features: pl.DataFrame,
    config: AppConfig,
    strategy_factory: Callable[[], Strategy],
) -> SensitivityResult:
    taker_fees = _scaled_values(config.execution.taker_fee_bps, (0.5, 1.0, 1.5))
    slippages = _scaled_values(config.execution.slippage_bps, (0.5, 1.0, 1.5))
    latencies = _latency_values(config.execution.latency_bars)

    scenarios: list[dict[str, Any]] = []
    baseline_key = (
        round(config.execution.taker_fee_bps, 6),
        round(config.execution.slippage_bps, 6),
        config.execution.latency_bars,
    )

    baseline_net_pnl: float | None = None
    for taker_fee_bps, slippage_bps, latency_bars in product(taker_fees, slippages, latencies):
        scenario_execution = replace(
            config.execution,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
            latency_bars=latency_bars,
        )
        scenario_config = replace(config, execution=scenario_execution)
        backtest = run_backtest(features, strategy_factory(), scenario_config)
        scenario_key = (
            round(taker_fee_bps, 6),
            round(slippage_bps, 6),
            latency_bars,
        )
        if scenario_key == baseline_key:
            baseline_net_pnl = float(backtest.metrics["net_pnl"])

        scenarios.append(
            {
                "scenario_name": (
                    f"fee_{taker_fee_bps:g}_slip_{slippage_bps:g}_lat_{latency_bars}"
                ),
                "taker_fee_bps": taker_fee_bps,
                "slippage_bps": slippage_bps,
                "latency_bars": latency_bars,
                "net_pnl": float(backtest.metrics["net_pnl"]),
                "gross_pnl": float(backtest.metrics["gross_pnl"]),
                "total_fees": float(backtest.metrics["total_fees"]),
                "trade_count": int(backtest.metrics["trade_count"]),
                "max_drawdown": float(backtest.metrics["max_drawdown"]),
            }
        )

    if baseline_net_pnl is None:
        raise ValueError("Baseline execution scenario was not evaluated")

    scenario_net_pnls = sorted(float(item["net_pnl"]) for item in scenarios)
    stressed = min(scenarios, key=lambda item: float(item["net_pnl"]))
    resilient = max(scenarios, key=lambda item: float(item["net_pnl"]))

    aggregate = {
        "scenario_count": len(scenarios),
        "baseline_net_pnl": baseline_net_pnl,
        "worst_case_net_pnl": float(stressed["net_pnl"]),
        "best_case_net_pnl": float(resilient["net_pnl"]),
        "mean_net_pnl": sum(scenario_net_pnls) / len(scenario_net_pnls),
        "median_net_pnl": scenario_net_pnls[len(scenario_net_pnls) // 2],
        "baseline_to_worst_case_delta": float(stressed["net_pnl"]) - baseline_net_pnl,
        "baseline_to_best_case_delta": float(resilient["net_pnl"]) - baseline_net_pnl,
        "positive_scenario_ratio": (
            sum(1 for item in scenario_net_pnls if item > 0.0) / len(scenario_net_pnls)
        ),
        "most_stressed_scenario": dict(stressed),
    }
    return SensitivityResult(scenarios=scenarios, aggregate=aggregate)
