from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl
from quantflow.backtest.engine import BacktestResult, run_backtest
from quantflow.core.config import AppConfig
from quantflow.strategy.base import Strategy
from quantflow.validation.splits import slice_frame_by_time_window, walk_forward_time_splits


@dataclass(slots=True)
class GridSearchResult:
    parameter_grid: list[dict[str, Any]]
    windows: list[dict[str, Any]]
    aggregate: dict[str, Any]


def _coerce_cli_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." not in value and "e" not in lowered:
            return int(value)
        return float(value)
    except ValueError:
        return value.strip()


def parse_param_grid_specs(specs: list[str]) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid param-grid spec: {spec}")
        key, raw_values = spec.split("=", 1)
        values = [_coerce_cli_value(item) for item in raw_values.split(",") if item.strip()]
        if not values:
            raise ValueError(f"Param-grid spec has no values: {spec}")
        grid[key.strip()] = values
    return grid


def expand_param_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    items = sorted(grid.items())
    combinations: list[dict[str, Any]] = [{}]
    for key, values in items:
        next_combinations: list[dict[str, Any]] = []
        for combination in combinations:
            for value in values:
                next_combinations.append({**combination, key: value})
        combinations = next_combinations
    return combinations


def _score_result(result: BacktestResult, objective: str) -> float:
    value = result.metrics.get(objective)
    if value is None:
        raise ValueError(f"Objective metric not found or null: {objective}")
    return float(value)


def _param_signature(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True)


def run_walk_forward_grid_search(
    features: pl.DataFrame,
    config: AppConfig,
    parameter_grid: list[dict[str, Any]],
    strategy_builder: Callable[[dict[str, Any]], Strategy],
    objective: str = "net_pnl",
) -> GridSearchResult:
    windows = walk_forward_time_splits(
        features,
        train_bars=config.validation.train_bars,
        test_bars=config.validation.test_bars,
        step_bars=config.validation.step_bars,
        embargo_bars=config.validation.embargo_bars,
    )
    if not windows:
        raise ValueError("No walk-forward windows available with the current configuration")
    if not parameter_grid:
        raise ValueError("Parameter grid is empty")

    window_results: list[dict[str, Any]] = []
    selected_signatures: list[str] = []

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

        ranked: list[tuple[float, dict[str, Any], BacktestResult]] = []
        for params in parameter_grid:
            train_result = run_backtest(train_frame, strategy_builder(params), config)
            score = _score_result(train_result, objective)
            ranked.append((score, params, train_result))
        ranked.sort(key=lambda item: item[0], reverse=True)
        best_score, best_params, best_train_result = ranked[0]
        best_test_result = run_backtest(test_frame, strategy_builder(best_params), config)
        selected_signatures.append(_param_signature(best_params))

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
                "selected_params": dict(best_params),
                "train_objective": best_score,
                "test_objective": _score_result(best_test_result, objective),
                "train_net_pnl": float(best_train_result.metrics["net_pnl"]),
                "test_net_pnl": float(best_test_result.metrics["net_pnl"]),
                "train_max_drawdown": float(best_train_result.metrics["max_drawdown"]),
                "test_max_drawdown": float(best_test_result.metrics["max_drawdown"]),
                "train_trade_count": int(best_train_result.metrics["trade_count"]),
                "test_trade_count": int(best_test_result.metrics["trade_count"]),
            }
        )

    counts = Counter(selected_signatures)
    test_objectives = sorted(float(item["test_objective"]) for item in window_results)
    most_common_signature, most_common_count = counts.most_common(1)[0]

    aggregate = {
        "window_count": len(window_results),
        "combo_count": len(parameter_grid),
        "objective": objective,
        "mean_test_objective": sum(test_objectives) / len(test_objectives),
        "median_test_objective": test_objectives[len(test_objectives) // 2],
        "positive_test_window_ratio": (
            sum(1 for item in window_results if float(item["test_net_pnl"]) > 0.0)
            / len(window_results)
        ),
        "most_selected_params": json.loads(most_common_signature),
        "most_selected_count": most_common_count,
        "parameter_stability_ratio": most_common_count / len(window_results),
    }
    return GridSearchResult(
        parameter_grid=parameter_grid,
        windows=window_results,
        aggregate=aggregate,
    )
