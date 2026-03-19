from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import polars as pl
from quantflow.backtest.engine import run_backtest
from quantflow.backtest.portfolio import TradeRecord
from quantflow.core.config import AppConfig
from quantflow.strategy import build_strategy
from quantflow.validation.optimization import expand_param_grid, run_walk_forward_grid_search
from quantflow.validation.walkforward import run_walk_forward

FEATURE_PROFILES = {
    "5s": "order_flow_suite_5s_v1",
    "15s": "order_flow_suite_15s_v1",
}
EXECUTION_MODES = ("market", "passive")


@dataclass(frozen=True, slots=True)
class OrderFlowStrategySpec:
    name: str
    base_params: dict[str, Any]
    grid: dict[str, list[Any]]


@dataclass(slots=True)
class OrderFlowSuiteResult:
    cells: list[dict[str, Any]]
    symbol_rows: list[dict[str, Any]]
    ranked_candidates: list[dict[str, Any]]
    aggregate: dict[str, Any]


def _strategy_specs() -> list[OrderFlowStrategySpec]:
    vol_grid = [0.8, 1.0, 1.2]
    return [
        OrderFlowStrategySpec(
            name="cumdelta_reversion_v1",
            base_params={
                "entry_z": 1.5,
                "exit_z": 0.25,
                "vol_regime_min": 1.0,
            },
            grid={
                "entry_z": [1.0, 1.5, 2.0, 2.5],
                "exit_z": [0.1, 0.25, 0.5],
                "vol_regime_min": vol_grid,
            },
        ),
        OrderFlowStrategySpec(
            name="delta_impulse_continuation_v1",
            base_params={
                "source": "delta_rz",
                "entry_z": 1.0,
                "exit_z": 0.25,
                "vol_regime_min": 1.0,
            },
            grid={
                "source": ["delta_rz", "quote_delta_rz"],
                "entry_z": [0.75, 1.0, 1.25, 1.5],
                "exit_z": [0.0, 0.25, 0.5],
                "vol_regime_min": vol_grid,
            },
        ),
        OrderFlowStrategySpec(
            name="imbalance_burst_exhaustion_v1",
            base_params={
                "imbalance_z": 1.0,
                "burst_z": 1.0,
                "exit_z": 0.25,
                "vol_regime_min": 1.0,
            },
            grid={
                "imbalance_z": [0.75, 1.0, 1.25, 1.5],
                "burst_z": [0.5, 1.0, 1.5],
                "exit_z": [0.0, 0.25, 0.5],
                "vol_regime_min": vol_grid,
            },
        ),
    ]


def _scenario_config(
    config: AppConfig,
    *,
    bar_size: str,
    execution_mode: str,
) -> AppConfig:
    feature_profile = FEATURE_PROFILES[bar_size]
    execution = config.execution
    if execution_mode == "market":
        execution = replace(
            config.execution,
            default_order_type="market",
        )
    elif execution_mode == "passive":
        execution = replace(
            config.execution,
            default_order_type="limit",
            allow_limit_orders=True,
        )
    else:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")

    return replace(
        config,
        features=replace(
            config.features,
            bar_size=bar_size,
            feature_profile=feature_profile,
        ),
        execution=execution,
        portfolio=replace(config.portfolio, allow_short=False),
    )


def _symbol_summary_rows(
    trades: list[TradeRecord],
    *,
    strategy_name: str,
    bar_size: str,
    execution_mode: str,
) -> list[dict[str, Any]]:
    if not trades:
        return []

    frame = pl.DataFrame([trade.to_dict() for trade in trades]).with_columns(
        [
            pl.when(pl.col("net_pnl") > 0.0)
            .then(pl.col("net_pnl"))
            .otherwise(pl.lit(0.0))
            .alias("positive_net_pnl"),
            pl.when(pl.col("net_pnl") < 0.0)
            .then(pl.col("net_pnl").abs())
            .otherwise(pl.lit(0.0))
            .alias("negative_net_pnl_abs"),
        ]
    )
    grouped = (
        frame.group_by("symbol")
        .agg(
            [
                pl.len().alias("trade_count"),
                pl.col("net_pnl").sum().alias("net_pnl"),
                pl.col("net_pnl").mean().alias("expectancy"),
                (pl.col("net_pnl") > 0.0).mean().alias("hit_ratio"),
                pl.when(pl.col("negative_net_pnl_abs").sum() > 0.0)
                .then(pl.col("positive_net_pnl").sum() / pl.col("negative_net_pnl_abs").sum())
                .otherwise(None)
                .alias("profit_factor"),
            ]
        )
        .sort("symbol")
    )
    rows = grouped.to_dicts()
    for row in rows:
        row.update(
            {
                "strategy_name": strategy_name,
                "bar_size": bar_size,
                "execution_mode": execution_mode,
            }
        )
    return rows


def _build_strategy_config(
    config: AppConfig,
    spec: OrderFlowStrategySpec,
    params: dict[str, Any] | None = None,
) -> Any:
    return replace(
        config.strategy,
        name=spec.name,
        params={**spec.base_params, **(params or {})},
    )


def _strategy_from_params(
    scenario_config: AppConfig,
    strategy_spec: OrderFlowStrategySpec,
    params: dict[str, Any] | None = None,
) -> Any:
    return build_strategy(_build_strategy_config(scenario_config, strategy_spec, params))


def _candidate_row(
    *,
    strategy_name: str,
    bar_size: str,
    execution_mode: str,
    grid_point: str,
    params: dict[str, Any],
    walkforward_mean_test_net_pnl: float,
    positive_test_window_ratio: float,
    turnover_multiple: float | None,
    trade_count: int,
    execution_fragile: bool,
) -> dict[str, Any]:
    return {
        "strategy_name": strategy_name,
        "bar_size": bar_size,
        "execution_mode": execution_mode,
        "grid_point": grid_point,
        "params": dict(params),
        "walkforward_mean_test_net_pnl": walkforward_mean_test_net_pnl,
        "positive_test_window_ratio": positive_test_window_ratio,
        "turnover_multiple": turnover_multiple,
        "trade_count": trade_count,
        "execution_fragile": execution_fragile,
    }


def _parameter_grid_for_spec(
    config: AppConfig,
    strategy_spec: OrderFlowStrategySpec,
) -> list[dict[str, Any]]:
    grid = expand_param_grid(strategy_spec.grid)
    limit = config.validation.grid_search_max_points
    if limit > 0:
        return grid[:limit]
    return grid


def run_order_flow_suite(
    *,
    feature_frames: dict[str, pl.DataFrame],
    config: AppConfig,
) -> OrderFlowSuiteResult:
    missing_bar_sizes = sorted(set(FEATURE_PROFILES) - set(feature_frames))
    if missing_bar_sizes:
        raise ValueError(f"Missing feature frames for bar sizes: {missing_bar_sizes}")

    cells: list[dict[str, Any]] = []
    symbol_rows: list[dict[str, Any]] = []
    ranked_candidates: list[dict[str, Any]] = []

    for bar_size in FEATURE_PROFILES:
        spec_frame = feature_frames[bar_size]
        for execution_mode in EXECUTION_MODES:
            scenario_config = _scenario_config(
                config,
                bar_size=bar_size,
                execution_mode=execution_mode,
            )
            for strategy_spec in _strategy_specs():
                strategy_config = _build_strategy_config(scenario_config, strategy_spec)

                def _suite_strategy_builder(
                    params: dict[str, Any],
                    *,
                    scenario_config: AppConfig = scenario_config,
                    strategy_spec: OrderFlowStrategySpec = strategy_spec,
                ) -> Any:
                    return _strategy_from_params(
                        scenario_config,
                        strategy_spec,
                        params,
                    )

                backtest = run_backtest(
                    spec_frame,
                    _strategy_from_params(scenario_config, strategy_spec),
                    scenario_config,
                )
                walkforward = run_walk_forward(
                    spec_frame,
                    scenario_config,
                    strategy_factory=lambda strategy_config=strategy_config: build_strategy(
                        strategy_config
                    ),
                )
                parameter_grid = _parameter_grid_for_spec(scenario_config, strategy_spec)
                grid_search = run_walk_forward_grid_search(
                    spec_frame,
                    scenario_config,
                    parameter_grid=parameter_grid,
                    strategy_builder=_suite_strategy_builder,
                    objective="net_pnl",
                )

                cell = {
                    "strategy_name": strategy_spec.name,
                    "bar_size": bar_size,
                    "feature_profile": FEATURE_PROFILES[bar_size],
                    "execution_mode": execution_mode,
                    "filter_name": "vol_regime_min",
                    "filter_value": float(strategy_spec.base_params["vol_regime_min"]),
                    "default_params": dict(strategy_spec.base_params),
                    "full_sample_net_pnl": float(backtest.metrics["net_pnl"]),
                    "full_sample_trade_count": int(backtest.metrics["trade_count"]),
                    "full_sample_turnover_multiple": backtest.metrics["turnover_multiple"],
                    "walkforward_mean_test_net_pnl": float(
                        walkforward.aggregate["mean_test_net_pnl"]
                    ),
                    "walkforward_positive_test_window_ratio": float(
                        walkforward.aggregate["positive_test_window_ratio"]
                    ),
                    "walkforward_window_count": int(walkforward.aggregate["window_count"]),
                    "grid_mean_test_net_pnl": float(grid_search.aggregate["mean_test_objective"]),
                    "grid_positive_test_window_ratio": float(
                        grid_search.aggregate["positive_test_window_ratio"]
                    ),
                    "grid_parameter_stability_ratio": float(
                        grid_search.aggregate["parameter_stability_ratio"]
                    ),
                    "best_grid_params": dict(grid_search.aggregate["most_selected_params"]),
                    "execution_fragile": False,
                }
                cells.append(cell)
                symbol_rows.extend(
                    _symbol_summary_rows(
                        backtest.trades,
                        strategy_name=strategy_spec.name,
                        bar_size=bar_size,
                        execution_mode=execution_mode,
                    )
                )

    keyed_cells: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for cell in cells:
        key = (str(cell["strategy_name"]), str(cell["bar_size"]))
        keyed_cells.setdefault(key, {})[str(cell["execution_mode"])] = cell

    for cell_group in keyed_cells.values():
        market_cell = cell_group.get("market")
        passive_cell = cell_group.get("passive")
        execution_fragile = (
            market_cell is not None
            and passive_cell is not None
            and float(market_cell["walkforward_mean_test_net_pnl"]) < 0.0
            and float(passive_cell["walkforward_mean_test_net_pnl"]) > 0.0
        )
        for cell in cell_group.values():
            cell["execution_fragile"] = execution_fragile

    for cell in cells:
        ranked_candidates.append(
            _candidate_row(
                strategy_name=str(cell["strategy_name"]),
                bar_size=str(cell["bar_size"]),
                execution_mode=str(cell["execution_mode"]),
                grid_point="default",
                params=dict(cell["default_params"]),
                walkforward_mean_test_net_pnl=float(cell["walkforward_mean_test_net_pnl"]),
                positive_test_window_ratio=float(cell["walkforward_positive_test_window_ratio"]),
                turnover_multiple=_optional_float(cell["full_sample_turnover_multiple"]),
                trade_count=int(cell["full_sample_trade_count"]),
                execution_fragile=bool(cell["execution_fragile"]),
            )
        )
        ranked_candidates.append(
            _candidate_row(
                strategy_name=str(cell["strategy_name"]),
                bar_size=str(cell["bar_size"]),
                execution_mode=str(cell["execution_mode"]),
                grid_point="best_walkforward_grid",
                params=dict(cell["best_grid_params"]),
                walkforward_mean_test_net_pnl=float(cell["grid_mean_test_net_pnl"]),
                positive_test_window_ratio=float(cell["grid_positive_test_window_ratio"]),
                turnover_multiple=_optional_float(cell["full_sample_turnover_multiple"]),
                trade_count=int(cell["full_sample_trade_count"]),
                execution_fragile=bool(cell["execution_fragile"]),
            )
        )

    ranked_candidates.sort(
        key=lambda row: (
            float(row["walkforward_mean_test_net_pnl"]),
            float(row["positive_test_window_ratio"]),
            -float("inf")
            if row["turnover_multiple"] is None
            else -float(row["turnover_multiple"]),
        ),
        reverse=True,
    )
    for index, row in enumerate(ranked_candidates, start=1):
        row["rank"] = index

    top_default = next((row for row in ranked_candidates if row["grid_point"] == "default"), None)
    top_candidate = ranked_candidates[0] if ranked_candidates else None
    aggregate = {
        "cell_count": len(cells),
        "strategy_count": len(_strategy_specs()),
        "bar_size_count": len(FEATURE_PROFILES),
        "execution_mode_count": len(EXECUTION_MODES),
        "top_default_strategy_name": None if top_default is None else top_default["strategy_name"],
        "top_default_bar_size": None if top_default is None else top_default["bar_size"],
        "top_default_execution_mode": (
            None if top_default is None else top_default["execution_mode"]
        ),
        "top_default_walkforward_mean_test_net_pnl": (
            None if top_default is None else top_default["walkforward_mean_test_net_pnl"]
        ),
        "top_candidate_strategy_name": (
            None if top_candidate is None else top_candidate["strategy_name"]
        ),
        "top_candidate_bar_size": None if top_candidate is None else top_candidate["bar_size"],
        "top_candidate_execution_mode": (
            None if top_candidate is None else top_candidate["execution_mode"]
        ),
        "top_candidate_grid_point": None if top_candidate is None else top_candidate["grid_point"],
        "top_candidate_walkforward_mean_test_net_pnl": (
            None if top_candidate is None else top_candidate["walkforward_mean_test_net_pnl"]
        ),
        "execution_fragile_count": sum(1 for cell in cells if bool(cell["execution_fragile"])),
    }
    return OrderFlowSuiteResult(
        cells=cells,
        symbol_rows=symbol_rows,
        ranked_candidates=ranked_candidates,
        aggregate=aggregate,
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
