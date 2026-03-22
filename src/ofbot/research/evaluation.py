from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Callable

from ofbot.research.backtest import ReplayBacktestResult
from ofbot.research.config import CompositeScoreConfig, DatasetConfig, RobustnessConfig
from ofbot.research.dataset import build_walk_forward_windows, split_subperiods


@dataclass(slots=True)
class ScoreCard:
    score: float
    components: dict[str, float]
    gates: dict[str, bool]
    accepted: bool
    reasons: list[str]


@dataclass(slots=True)
class RobustnessReport:
    subperiod_positive_ratio: float
    walk_forward_positive_ratio: float
    stressed_positive_ratio: float
    neighbor_positive_ratio: float
    details: dict[str, Any]


BacktestRunner = Callable[[dict[str, Any], Any, str, dict[str, Any] | None], ReplayBacktestResult]


def build_score_card(
    *,
    metrics: dict[str, Any],
    config: CompositeScoreConfig,
    subperiod_positive_ratio: float,
) -> ScoreCard:
    components = {
        "net_pnl": _tanh(metrics.get("net_pnl", 0.0), config.pnl_scale),
        "max_drawdown": -_tanh(abs(float(metrics.get("max_drawdown", 0.0) or 0.0)), config.drawdown_scale),
        "sharpe_like": math.tanh(float(metrics.get("sharpe_like", 0.0) or 0.0) / 2.0),
        "win_rate": max(-1.0, min(1.0, 2.0 * float(metrics.get("win_rate", 0.0) or 0.0) - 1.0)),
        "profit_factor": math.tanh((float(metrics.get("profit_factor", 0.0) or 0.0) - 1.0) / 1.0),
        "expectancy": _tanh(float(metrics.get("expectancy", 0.0) or 0.0), config.expectancy_scale),
        "stability": max(-1.0, min(1.0, 2.0 * subperiod_positive_ratio - 1.0)),
    }
    weights = config.weights.model_dump()
    weight_sum = sum(float(value) for value in weights.values())
    score = sum(components[key] * float(weights[key]) for key in components) / max(weight_sum, 1e-12)

    trade_count = int(metrics.get("trade_count", 0) or 0)
    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
    max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)

    gates = {
        "min_trades": trade_count >= config.min_trades,
        "max_drawdown_abs": abs(max_drawdown) <= config.max_drawdown_abs,
        "min_profit_factor": profit_factor >= config.min_profit_factor,
        "min_win_rate": win_rate >= config.min_win_rate,
    }
    reasons = [name for name, passed in gates.items() if not passed]
    return ScoreCard(
        score=float(score),
        components=components,
        gates=gates,
        accepted=all(gates.values()),
        reasons=reasons,
    )


def run_robustness_suite(
    *,
    params: dict[str, Any],
    validation_frame,
    train_validation_frame,
    dataset_config: DatasetConfig,
    robustness_config: RobustnessConfig,
    param_spaces: dict[str, list[Any]],
    run_backtest: BacktestRunner,
    split_label_prefix: str,
) -> RobustnessReport:
    subperiod_results = []
    for period in split_subperiods(
        validation_frame,
        count=dataset_config.stability_subperiods,
        timestamp_column=dataset_config.timestamp_column,
    ):
        subperiod_results.append(
            run_backtest(params, period.frame, f"{split_label_prefix}_{period.name}", None)
        )
    subperiod_positive_ratio = _positive_ratio(subperiod_results)

    walk_forward_results = []
    for window in build_walk_forward_windows(
        train_validation_frame,
        timestamp_column=dataset_config.timestamp_column,
        train_ratio=dataset_config.walk_forward_train_ratio,
        window_count=dataset_config.walk_forward_windows,
    ):
        walk_forward_results.append(
            run_backtest(params, window.test.frame, f"{split_label_prefix}_wf_{window.index + 1}", None)
        )
    walk_forward_positive_ratio = _positive_ratio(walk_forward_results)

    stressed_results = []
    for maker_mult, taker_mult, impact_mult, latency_add in itertools.product(
        robustness_config.stressed_maker_fee_multipliers,
        robustness_config.stressed_taker_fee_multipliers,
        robustness_config.stressed_impact_multipliers,
        robustness_config.stressed_latency_ms_add,
    ):
        stressed_results.append(
            run_backtest(
                params,
                validation_frame,
                split_label_prefix,
                {
                    "maker_fee_bps": ("mul", maker_mult),
                    "taker_fee_bps": ("mul", taker_mult),
                    "impact_bps_per_unit_participation": ("mul", impact_mult),
                    "latency_ms": ("add", latency_add),
                },
            )
        )
    stressed_positive_ratio = _positive_ratio(stressed_results)

    neighbor_results = []
    for neighbor in neighbor_param_sets(params=params, param_spaces=param_spaces):
        neighbor_results.append(run_backtest(neighbor, validation_frame, split_label_prefix, None))
    neighbor_positive_ratio = _positive_ratio(neighbor_results) if neighbor_results else 1.0

    return RobustnessReport(
        subperiod_positive_ratio=subperiod_positive_ratio,
        walk_forward_positive_ratio=walk_forward_positive_ratio,
        stressed_positive_ratio=stressed_positive_ratio,
        neighbor_positive_ratio=neighbor_positive_ratio,
        details={
            "subperiod_results": [result.metrics for result in subperiod_results],
            "walk_forward_results": [result.metrics for result in walk_forward_results],
            "stressed_results": [result.metrics for result in stressed_results],
            "neighbor_results": [result.metrics for result in neighbor_results],
        },
    )


def robustness_gate_failures(
    *,
    report: RobustnessReport,
    config: RobustnessConfig,
) -> list[str]:
    failures: list[str] = []
    if report.subperiod_positive_ratio < config.min_subperiod_positive_ratio:
        failures.append("subperiod_stability")
    if report.walk_forward_positive_ratio < config.min_walk_forward_positive_ratio:
        failures.append("walk_forward_stability")
    if report.stressed_positive_ratio < config.min_stressed_positive_ratio:
        failures.append("stressed_costs")
    if report.neighbor_positive_ratio < config.min_neighbor_positive_ratio:
        failures.append("parameter_fragility")
    return failures


def neighbor_param_sets(
    *,
    params: dict[str, Any],
    param_spaces: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    neighbors: list[dict[str, Any]] = []
    for key, values in sorted(param_spaces.items()):
        if key not in params or params[key] not in values:
            continue
        index = values.index(params[key])
        for offset in (-1, 1):
            neighbor_index = index + offset
            if neighbor_index < 0 or neighbor_index >= len(values):
                continue
            updated = dict(params)
            updated[key] = values[neighbor_index]
            neighbors.append(updated)
    return neighbors


def _positive_ratio(results: list[ReplayBacktestResult]) -> float:
    if not results:
        return 0.0
    positive = sum(1 for result in results if float(result.metrics.get("net_pnl", 0.0) or 0.0) > 0.0)
    return positive / len(results)


def _tanh(value: float | int | None, scale: float) -> float:
    if value is None or scale <= 0.0:
        return 0.0
    return math.tanh(float(value) / scale)
