from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Any

import polars as pl
from quantflow.backtest.engine import BacktestResult, run_backtest
from quantflow.backtest.portfolio import TradeRecord
from quantflow.core.config import AppConfig, StrategyConfig
from quantflow.strategy import build_strategy
from quantflow.validation.bootstrap import BootstrapResult, run_bootstrap_validation
from quantflow.validation.walkforward import WalkForwardResult, run_walk_forward


@dataclass(slots=True)
class PassiveLimitSensitivityResult:
    scenarios: list[dict[str, Any]]
    aggregate: dict[str, Any]


@dataclass(slots=True)
class CandidateDeepDiveResult:
    phase1_scenarios: list[dict[str, Any]]
    phase2_results: list[dict[str, Any]]
    best_scenario: dict[str, Any]
    best_symbol_rows: list[dict[str, Any]]
    best_backtest: BacktestResult
    best_walkforward: WalkForwardResult
    best_bootstrap: BootstrapResult
    passive_sensitivity: PassiveLimitSensitivityResult
    market_sanity_check: dict[str, Any]
    aggregate: dict[str, Any]


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


def _session_label(allowed_sessions: tuple[str, ...]) -> str:
    if not allowed_sessions:
        return "all"
    return "+".join(allowed_sessions)


def _filter_family(vol_regime_min: float, allowed_sessions: tuple[str, ...]) -> str:
    if vol_regime_min <= 0.0 and not allowed_sessions:
        return "baseline"
    if allowed_sessions and vol_regime_min <= 0.0:
        return "session_only"
    if vol_regime_min > 0.0 and not allowed_sessions:
        return "vol_only"
    return "combined"


def _scenario_name(vol_regime_min: float, allowed_sessions: tuple[str, ...]) -> str:
    family = _filter_family(vol_regime_min, allowed_sessions)
    session_label = _session_label(allowed_sessions)
    if family == "baseline":
        return "baseline"
    if family == "vol_only":
        return f"vol_only__vol_{vol_regime_min:g}"
    if family == "session_only":
        return f"session_only__{session_label}"
    return f"combined__vol_{vol_regime_min:g}__{session_label}"


def _phase1_filter_scenarios() -> list[dict[str, Any]]:
    vol_values = [0.8, 1.0, 1.2]
    session_values = [("europe", "us"), ("us",)]

    scenarios: list[dict[str, Any]] = [
        {
            "scenario_name": "baseline",
            "filter_family": "baseline",
            "vol_regime_min": 0.0,
            "allowed_sessions_tuple": (),
            "allowed_sessions": "all",
        }
    ]

    for vol_regime_min in vol_values:
        scenarios.append(
            {
                "scenario_name": _scenario_name(vol_regime_min, ()),
                "filter_family": "vol_only",
                "vol_regime_min": vol_regime_min,
                "allowed_sessions_tuple": (),
                "allowed_sessions": "all",
            }
        )

    for allowed_sessions in session_values:
        scenarios.append(
            {
                "scenario_name": _scenario_name(0.0, allowed_sessions),
                "filter_family": "session_only",
                "vol_regime_min": 0.0,
                "allowed_sessions_tuple": allowed_sessions,
                "allowed_sessions": _session_label(allowed_sessions),
            }
        )

    for vol_regime_min, allowed_sessions in product(vol_values, session_values):
        scenarios.append(
            {
                "scenario_name": _scenario_name(vol_regime_min, allowed_sessions),
                "filter_family": "combined",
                "vol_regime_min": vol_regime_min,
                "allowed_sessions_tuple": allowed_sessions,
                "allowed_sessions": _session_label(allowed_sessions),
            }
        )
    return scenarios


def _execution_config(config: AppConfig, execution_mode: str) -> AppConfig:
    if execution_mode == "passive":
        execution = replace(
            config.execution,
            default_order_type="limit",
            allow_limit_orders=True,
        )
    elif execution_mode == "market":
        execution = replace(
            config.execution,
            default_order_type="market",
        )
    else:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")

    return replace(
        config,
        execution=execution,
        portfolio=replace(config.portfolio, allow_short=False),
    )


def _candidate_strategy_config(config: AppConfig, params: dict[str, Any]) -> StrategyConfig:
    return replace(
        config.strategy,
        name="cumdelta_reversion_v1",
        params=dict(params),
    )


def _candidate_strategy(config: AppConfig, params: dict[str, Any]) -> Any:
    return build_strategy(_candidate_strategy_config(config, params))


def _scenario_row(
    *,
    phase: str,
    phase1_scenario_name: str | None,
    execution_mode: str,
    scenario_name: str,
    filter_family: str,
    params: dict[str, Any],
    backtest: BacktestResult,
    walkforward: WalkForwardResult,
) -> dict[str, Any]:
    allowed_sessions = tuple(params.get("allowed_sessions") or ())
    return {
        "phase": phase,
        "phase1_scenario_name": phase1_scenario_name,
        "execution_mode": execution_mode,
        "scenario_name": scenario_name,
        "filter_family": filter_family,
        "entry_z": float(params["entry_z"]),
        "exit_z": float(params["exit_z"]),
        "vol_regime_min": float(params["vol_regime_min"]),
        "allowed_sessions": _session_label(allowed_sessions),
        "full_sample_net_pnl": float(backtest.metrics["net_pnl"]),
        "full_sample_trade_count": int(backtest.metrics["trade_count"]),
        "full_sample_turnover_multiple": _optional_float(backtest.metrics["turnover_multiple"]),
        "walkforward_mean_test_net_pnl": float(walkforward.aggregate["mean_test_net_pnl"]),
        "positive_test_window_ratio": float(walkforward.aggregate["positive_test_window_ratio"]),
        "walkforward_window_count": int(walkforward.aggregate["window_count"]),
    }


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["walkforward_mean_test_net_pnl"]),
            -float(row["positive_test_window_ratio"]),
            float("inf")
            if row["full_sample_turnover_multiple"] is None
            else float(row["full_sample_turnover_multiple"]),
        ),
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def _symbol_summary_rows(
    trades: list[TradeRecord],
    symbols: list[str],
) -> list[dict[str, Any]]:
    if not trades:
        return [
            {
                "symbol": symbol,
                "trade_count": 0,
                "net_pnl": 0.0,
                "expectancy": None,
                "hit_ratio": None,
                "profit_factor": None,
            }
            for symbol in symbols
        ]

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
        .to_dicts()
    )
    by_symbol = {str(row["symbol"]): row for row in grouped}
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        row = by_symbol.get(symbol)
        if row is None:
            rows.append(
                {
                    "symbol": symbol,
                    "trade_count": 0,
                    "net_pnl": 0.0,
                    "expectancy": None,
                    "hit_ratio": None,
                    "profit_factor": None,
                }
            )
            continue
        rows.append(
            {
                "symbol": symbol,
                "trade_count": int(row["trade_count"]),
                "net_pnl": float(row["net_pnl"]),
                "expectancy": _optional_float(row["expectancy"]),
                "hit_ratio": _optional_float(row["hit_ratio"]),
                "profit_factor": _optional_float(row["profit_factor"]),
            }
        )
    return rows


def _run_passive_execution_sensitivity(
    features: pl.DataFrame,
    config: AppConfig,
    params: dict[str, Any],
) -> PassiveLimitSensitivityResult:
    maker_fees = [0.5, 1.0, 2.0]
    limit_offsets = [0.5, 1.0, 2.0]
    lifetimes = [1, 3, 5]
    latencies = [1, 2]

    scenarios: list[dict[str, Any]] = []
    baseline_key = (
        round(config.execution.maker_fee_bps, 6),
        round(config.execution.limit_offset_bps, 6),
        config.execution.max_order_lifetime_bars,
        config.execution.latency_bars,
    )
    baseline_net_pnl: float | None = None

    for maker_fee_bps, limit_offset_bps, max_order_lifetime_bars, latency_bars in product(
        maker_fees,
        limit_offsets,
        lifetimes,
        latencies,
    ):
        execution = replace(
            config.execution,
            maker_fee_bps=maker_fee_bps,
            limit_offset_bps=limit_offset_bps,
            max_order_lifetime_bars=max_order_lifetime_bars,
            latency_bars=latency_bars,
            default_order_type="limit",
            allow_limit_orders=True,
        )
        scenario_config = replace(config, execution=execution)
        backtest = run_backtest(
            features,
            _candidate_strategy(scenario_config, params),
            scenario_config,
        )

        row = {
            "maker_fee_bps": maker_fee_bps,
            "limit_offset_bps": limit_offset_bps,
            "max_order_lifetime_bars": max_order_lifetime_bars,
            "latency_bars": latency_bars,
            "net_pnl": float(backtest.metrics["net_pnl"]),
            "gross_pnl": float(backtest.metrics["gross_pnl"]),
            "total_fees": float(backtest.metrics["total_fees"]),
            "trade_count": int(backtest.metrics["trade_count"]),
            "max_drawdown": float(backtest.metrics["max_drawdown"]),
        }
        scenarios.append(row)

        scenario_key = (
            round(maker_fee_bps, 6),
            round(limit_offset_bps, 6),
            max_order_lifetime_bars,
            latency_bars,
        )
        if scenario_key == baseline_key:
            baseline_net_pnl = float(backtest.metrics["net_pnl"])

    sorted_net_pnls = sorted(float(row["net_pnl"]) for row in scenarios)
    best = max(scenarios, key=lambda row: float(row["net_pnl"]))
    worst = min(scenarios, key=lambda row: float(row["net_pnl"]))
    aggregate = {
        "scenario_count": len(scenarios),
        "baseline_net_pnl": baseline_net_pnl,
        "best_case_net_pnl": float(best["net_pnl"]),
        "worst_case_net_pnl": float(worst["net_pnl"]),
        "mean_net_pnl": sum(sorted_net_pnls) / len(sorted_net_pnls),
        "median_net_pnl": sorted_net_pnls[len(sorted_net_pnls) // 2],
        "positive_scenario_ratio": (
            sum(1 for row in scenarios if float(row["net_pnl"]) > 0.0) / len(scenarios)
        ),
        "best_case_scenario": dict(best),
        "worst_case_scenario": dict(worst),
    }
    return PassiveLimitSensitivityResult(scenarios=scenarios, aggregate=aggregate)


def _market_sanity_check(
    features: pl.DataFrame,
    config: AppConfig,
    params: dict[str, Any],
    passive_row: dict[str, Any],
) -> dict[str, Any]:
    market_config = _execution_config(config, "market")
    backtest = run_backtest(features, _candidate_strategy(market_config, params), market_config)
    walkforward = run_walk_forward(
        features,
        market_config,
        strategy_factory=lambda market_config=market_config, params=params: _candidate_strategy(
            market_config,
            params,
        ),
    )
    market_walkforward_mean = float(walkforward.aggregate["mean_test_net_pnl"])
    passive_walkforward_mean = float(passive_row["walkforward_mean_test_net_pnl"])
    return {
        "params": dict(params),
        "backtest_metrics": dict(backtest.metrics),
        "walkforward_aggregate": dict(walkforward.aggregate),
        "execution_fragile": passive_walkforward_mean > 0.0 and market_walkforward_mean <= 0.0,
    }


def _execution_verdict(passive_row: dict[str, Any], market_sanity_check: dict[str, Any]) -> str:
    passive_mean = float(passive_row["walkforward_mean_test_net_pnl"])
    market_mean = float(market_sanity_check["walkforward_aggregate"]["mean_test_net_pnl"])
    if passive_mean > 0.0 and market_mean <= 0.0:
        return "execution_fragile"
    if passive_mean > 0.0:
        return "passive_survives"
    return "passive_not_positive"


def _final_verdict(
    passive_row: dict[str, Any],
    market_sanity_check: dict[str, Any],
    symbol_rows: list[dict[str, Any]],
    bootstrap: BootstrapResult,
) -> str:
    passive_mean = float(passive_row["walkforward_mean_test_net_pnl"])
    passive_positive_ratio = float(passive_row["positive_test_window_ratio"])
    market_mean = float(market_sanity_check["walkforward_aggregate"]["mean_test_net_pnl"])
    bootstrap_positive_probability = float(bootstrap.aggregate["probability_net_pnl_positive"])
    symbols_positive = bool(symbol_rows) and all(float(row["net_pnl"]) > 0.0 for row in symbol_rows)

    if (
        passive_mean > 0.0
        and passive_positive_ratio >= 0.6
        and market_mean > 0.0
        and bootstrap_positive_probability >= 0.6
        and symbols_positive
    ):
        return "validated_edge"
    return "research_candidate"


def run_candidate_deep_dive(
    *,
    features: pl.DataFrame,
    config: AppConfig,
) -> CandidateDeepDiveResult:
    if config.strategy.name != "cumdelta_reversion_v1":
        raise ValueError("Candidate deep dive requires strategy.name = cumdelta_reversion_v1")

    passive_config = _execution_config(config, "passive")
    base_params = {
        "entry_z": float(config.strategy.params.get("entry_z", 1.0)),
        "exit_z": float(config.strategy.params.get("exit_z", 0.1)),
        "vol_regime_min": 0.0,
        "allowed_sessions": [],
    }

    phase1_rows: list[dict[str, Any]] = []
    for scenario in _phase1_filter_scenarios():
        params = {
            **base_params,
            "vol_regime_min": float(scenario["vol_regime_min"]),
            "allowed_sessions": list(scenario["allowed_sessions_tuple"]),
        }
        backtest = run_backtest(
            features,
            _candidate_strategy(passive_config, params),
            passive_config,
        )
        walkforward = run_walk_forward(
            features,
            passive_config,
            strategy_factory=(
                lambda passive_config=passive_config, params=params: _candidate_strategy(
                    passive_config,
                    params,
                )
            ),
        )
        phase1_rows.append(
            _scenario_row(
                phase="phase1",
                phase1_scenario_name=None,
                execution_mode="passive",
                scenario_name=str(scenario["scenario_name"]),
                filter_family=str(scenario["filter_family"]),
                params=params,
                backtest=backtest,
                walkforward=walkforward,
            )
        )

    ranked_phase1 = _rank_rows(phase1_rows)
    top_phase1 = ranked_phase1[:3]

    phase2_rows: list[dict[str, Any]] = []
    for phase1_row in top_phase1:
        filter_params = {
            "vol_regime_min": float(phase1_row["vol_regime_min"]),
            "allowed_sessions": []
            if phase1_row["allowed_sessions"] == "all"
            else str(phase1_row["allowed_sessions"]).split("+"),
        }
        for entry_z, exit_z in product([1.0, 1.25, 1.5], [0.05, 0.1, 0.25]):
            params = {
                **base_params,
                **filter_params,
                "entry_z": float(entry_z),
                "exit_z": float(exit_z),
            }
            scenario_name = (
                f"{phase1_row['scenario_name']}__entry_{entry_z:g}__exit_{exit_z:g}"
            )
            backtest = run_backtest(
                features,
                _candidate_strategy(passive_config, params),
                passive_config,
            )
            walkforward = run_walk_forward(
                features,
                passive_config,
                strategy_factory=(
                    lambda passive_config=passive_config, params=params: _candidate_strategy(
                        passive_config,
                        params,
                    )
                ),
            )
            phase2_rows.append(
                _scenario_row(
                    phase="phase2",
                    phase1_scenario_name=str(phase1_row["scenario_name"]),
                    execution_mode="passive",
                    scenario_name=scenario_name,
                    filter_family=str(phase1_row["filter_family"]),
                    params=params,
                    backtest=backtest,
                    walkforward=walkforward,
                )
            )

    ranked_phase2 = _rank_rows(phase2_rows)
    if not ranked_phase2:
        raise ValueError("Candidate deep dive produced no phase 2 results")

    best_scenario = dict(ranked_phase2[0])
    best_params = {
        "entry_z": float(best_scenario["entry_z"]),
        "exit_z": float(best_scenario["exit_z"]),
        "vol_regime_min": float(best_scenario["vol_regime_min"]),
        "allowed_sessions": []
        if best_scenario["allowed_sessions"] == "all"
        else str(best_scenario["allowed_sessions"]).split("+"),
    }

    best_backtest = run_backtest(
        features,
        _candidate_strategy(passive_config, best_params),
        passive_config,
    )
    best_walkforward = run_walk_forward(
        features,
        passive_config,
        strategy_factory=(
            lambda passive_config=passive_config, best_params=best_params: _candidate_strategy(
                passive_config,
                best_params,
            )
        ),
    )
    best_bootstrap = run_bootstrap_validation(
        features,
        passive_config,
        strategy_factory=(
            lambda passive_config=passive_config, best_params=best_params: _candidate_strategy(
                passive_config,
                best_params,
            )
        ),
    )
    best_symbol_rows = _symbol_summary_rows(best_backtest.trades, list(config.dataset.symbols))
    passive_sensitivity = _run_passive_execution_sensitivity(features, passive_config, best_params)
    market_sanity_check = _market_sanity_check(features, config, best_params, best_scenario)

    execution_verdict = _execution_verdict(best_scenario, market_sanity_check)
    final_verdict = _final_verdict(
        best_scenario,
        market_sanity_check,
        best_symbol_rows,
        best_bootstrap,
    )
    aggregate = {
        "phase1_scenario_count": len(ranked_phase1),
        "phase2_grid_count": len(ranked_phase2),
        "top_phase1_scenario_name": top_phase1[0]["scenario_name"] if top_phase1 else None,
        "best_scenario_name": best_scenario["scenario_name"],
        "best_phase1_scenario_name": best_scenario["phase1_scenario_name"],
        "best_walkforward_mean_test_net_pnl": best_walkforward.aggregate["mean_test_net_pnl"],
        "best_positive_test_window_ratio": best_walkforward.aggregate["positive_test_window_ratio"],
        "market_walkforward_mean_test_net_pnl": market_sanity_check["walkforward_aggregate"][
            "mean_test_net_pnl"
        ],
        "execution_verdict": execution_verdict,
        "final_verdict": final_verdict,
        "bootstrap_probability_net_pnl_positive": best_bootstrap.aggregate[
            "probability_net_pnl_positive"
        ],
    }

    return CandidateDeepDiveResult(
        phase1_scenarios=ranked_phase1,
        phase2_results=ranked_phase2,
        best_scenario=best_scenario,
        best_symbol_rows=best_symbol_rows,
        best_backtest=best_backtest,
        best_walkforward=best_walkforward,
        best_bootstrap=best_bootstrap,
        passive_sensitivity=passive_sensitivity,
        market_sanity_check=market_sanity_check,
        aggregate=aggregate,
    )
