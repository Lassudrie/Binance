from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

import numpy as np
import polars as pl
from quantflow.core.config import AppConfig
from quantflow.validation.splits import slice_frame_by_time_window, walk_forward_time_splits

CORE_COLUMNS = {
    "symbol",
    "bar_start",
    "bar_end",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "buy_count",
    "sell_count",
    "hour_of_day",
    "session",
    "news_regime",
}
DEFAULT_CANDIDATE_FEATURES = [
    "delta",
    "quote_delta",
    "cumulative_delta",
    "cumulative_delta_rz",
    "rolling_delta",
    "rolling_delta_rz",
    "trade_count_imbalance",
    "trade_count_imbalance_rz",
    "average_trade_size",
    "tick_direction_persistence",
    "burst_intensity",
    "burst_intensity_rz",
    "realized_volatility",
    "realized_volatility_rz",
    "micro_momentum",
    "micro_momentum_rz",
    "micro_range_breakout",
    "vol_regime",
]
MIN_SAMPLE_COUNT = 8


@dataclass(slots=True)
class AlphaScanResult:
    features: list[str]
    horizons: list[int]
    windows: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    aggregate: dict[str, Any]


def _prepare_forward_returns(features: pl.DataFrame, horizons: list[int]) -> pl.DataFrame:
    prepared = features.sort(["symbol", "bar_end"])
    for horizon in horizons:
        prepared = prepared.with_columns(
            (
                pl.col("close").shift(-horizon).over("symbol") / pl.col("close") - 1.0
            ).alias(f"forward_return_{horizon}")
        )
    return prepared


def _candidate_features(features: pl.DataFrame, requested: list[str] | None) -> list[str]:
    if requested:
        missing = [column for column in requested if column not in features.columns]
        if missing:
            raise ValueError(f"Requested feature columns not found: {missing}")
        return requested

    available = set(features.columns)
    selected = [column for column in DEFAULT_CANDIDATE_FEATURES if column in available]
    if selected:
        return selected

    inferred: list[str] = []
    for column, dtype in features.schema.items():
        if column in CORE_COLUMNS or column.startswith("forward_return_"):
            continue
        if dtype.is_numeric():
            inferred.append(column)
    if not inferred:
        raise ValueError("No numeric candidate features available for alpha scan")
    return inferred


def _extract_arrays(
    frame: pl.DataFrame,
    feature_name: str,
    return_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    subset = frame.select([feature_name, return_column]).drop_nulls()
    if subset.is_empty():
        return np.array([], dtype=float), np.array([], dtype=float)
    x = subset.get_column(feature_name).to_numpy().astype(float, copy=False)
    y = subset.get_column(return_column).to_numpy().astype(float, copy=False)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < MIN_SAMPLE_COUNT or len(y) < MIN_SAMPLE_COUNT:
        return None
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def _quantile_spread(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    lower_quantile: float = 0.2,
    upper_quantile: float = 0.8,
) -> dict[str, float | int | None] | None:
    if len(train_x) < MIN_SAMPLE_COUNT or len(test_x) < MIN_SAMPLE_COUNT:
        return None

    lower_threshold = float(np.quantile(train_x, lower_quantile))
    upper_threshold = float(np.quantile(train_x, upper_quantile))
    if lower_threshold >= upper_threshold:
        return None

    train_bottom = train_y[train_x <= lower_threshold]
    train_top = train_y[train_x >= upper_threshold]
    test_bottom = test_y[test_x <= lower_threshold]
    test_top = test_y[test_x >= upper_threshold]
    if (
        len(train_bottom) < MIN_SAMPLE_COUNT // 2
        or len(train_top) < MIN_SAMPLE_COUNT // 2
        or len(test_bottom) < MIN_SAMPLE_COUNT // 2
        or len(test_top) < MIN_SAMPLE_COUNT // 2
    ):
        return None

    train_spread = float(np.mean(train_top) - np.mean(train_bottom))
    test_spread = float(np.mean(test_top) - np.mean(test_bottom))
    direction = 0 if train_spread == 0.0 else (1 if train_spread > 0.0 else -1)
    directional_test_spread = None if direction == 0 else float(direction * test_spread)
    return {
        "train_spread": train_spread,
        "test_spread": test_spread,
        "directional_test_spread": directional_test_spread,
        "lower_threshold": lower_threshold,
        "upper_threshold": upper_threshold,
        "train_bottom_count": int(len(train_bottom)),
        "train_top_count": int(len(train_top)),
        "test_bottom_count": int(len(test_bottom)),
        "test_top_count": int(len(test_top)),
    }


def run_alpha_scan(
    features: pl.DataFrame,
    config: AppConfig,
    *,
    feature_names: list[str] | None = None,
    horizons: list[int] | None = None,
) -> AlphaScanResult:
    resolved_horizons = sorted(set(horizons or [1, 3, 6, 12]))
    if not resolved_horizons:
        raise ValueError("Alpha scan requires at least one horizon")

    prepared = _prepare_forward_returns(features, resolved_horizons)
    candidates = _candidate_features(prepared, feature_names)
    windows = walk_forward_time_splits(
        prepared,
        train_bars=config.validation.train_bars,
        test_bars=config.validation.test_bars,
        step_bars=config.validation.step_bars,
        embargo_bars=config.validation.embargo_bars,
    )
    if not windows:
        raise ValueError("No walk-forward windows available with the current configuration")

    window_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []

    for feature_name in candidates:
        for horizon in resolved_horizons:
            return_column = f"forward_return_{horizon}"
            metrics_for_candidate: list[dict[str, Any]] = []

            for index, window in enumerate(windows):
                train_frame = slice_frame_by_time_window(
                    prepared,
                    start_time=window.train_start_time,
                    end_time=window.train_end_time,
                )
                test_frame = slice_frame_by_time_window(
                    prepared,
                    start_time=window.test_start_time,
                    end_time=window.test_end_time,
                )
                train_x, train_y = _extract_arrays(train_frame, feature_name, return_column)
                test_x, test_y = _extract_arrays(test_frame, feature_name, return_column)
                spread_metrics = _quantile_spread(train_x, train_y, test_x, test_y)
                if spread_metrics is None:
                    continue

                train_ic = _safe_corr(train_x, train_y)
                test_ic = _safe_corr(test_x, test_y)
                inferred_side = (
                    "buy_high_sell_low"
                    if float(spread_metrics["train_spread"]) >= 0.0
                    else "buy_low_sell_high"
                )
                record = {
                    "feature_name": feature_name,
                    "horizon_bars": horizon,
                    "window_index": index,
                    "train_start": window.train_start,
                    "train_end": window.train_end,
                    "test_start": window.test_start,
                    "test_end": window.test_end,
                    "train_start_time": window.train_start_time.isoformat(),
                    "train_end_time": window.train_end_time.isoformat(),
                    "test_start_time": window.test_start_time.isoformat(),
                    "test_end_time": window.test_end_time.isoformat(),
                    "train_observations": int(len(train_x)),
                    "test_observations": int(len(test_x)),
                    "train_ic": train_ic,
                    "test_ic": test_ic,
                    "inferred_side": inferred_side,
                    **spread_metrics,
                }
                metrics_for_candidate.append(record)
                window_records.append(record)

            if not metrics_for_candidate:
                continue

            directional_spreads = [
                float(item["directional_test_spread"])
                for item in metrics_for_candidate
                if item["directional_test_spread"] is not None
            ]
            test_spreads = [float(item["test_spread"]) for item in metrics_for_candidate]
            train_spreads = [float(item["train_spread"]) for item in metrics_for_candidate]
            mean_train_spread = _safe_mean(train_spreads)
            train_ics = [
                float(item["train_ic"])
                for item in metrics_for_candidate
                if item["train_ic"] is not None
            ]
            test_ics = [
                float(item["test_ic"])
                for item in metrics_for_candidate
                if item["test_ic"] is not None
            ]
            sign_matches = sum(
                1
                for item in metrics_for_candidate
                if float(item["train_spread"]) * float(item["test_spread"]) > 0.0
            )
            positive_directional = sum(value > 0.0 for value in directional_spreads)
            inferred_side = (
                "buy_high_sell_low"
                if float(mean_train_spread or 0.0) >= 0.0
                else "buy_low_sell_high"
            )
            candidate_records.append(
                {
                    "feature_name": feature_name,
                    "horizon_bars": horizon,
                    "window_count": len(metrics_for_candidate),
                    "inferred_side": inferred_side,
                    "mean_train_ic": _safe_mean(train_ics),
                    "mean_test_ic": _safe_mean(test_ics),
                    "mean_train_spread": mean_train_spread,
                    "mean_test_spread": _safe_mean(test_spreads),
                    "mean_directional_test_spread": _safe_mean(directional_spreads),
                    "median_directional_test_spread": _safe_median(directional_spreads),
                    "positive_directional_test_ratio": (
                        positive_directional / len(directional_spreads)
                    ),
                    "sign_stability_ratio": sign_matches / len(metrics_for_candidate),
                    "mean_test_top_count": _safe_mean(
                        [float(item["test_top_count"]) for item in metrics_for_candidate]
                    ),
                    "mean_test_bottom_count": _safe_mean(
                        [float(item["test_bottom_count"]) for item in metrics_for_candidate]
                    ),
                }
            )

    if not candidate_records:
        raise ValueError("Alpha scan produced no valid candidate metrics")

    candidate_records.sort(
        key=lambda item: (
            float(item["mean_directional_test_spread"] or float("-inf")),
            float(item["sign_stability_ratio"]),
            float(abs(item["mean_test_ic"] or 0.0)),
        ),
        reverse=True,
    )
    top_candidate = candidate_records[0]
    aggregate = {
        "candidate_count": len(candidate_records),
        "feature_count": len(candidates),
        "horizons": resolved_horizons,
        "window_count": len(windows),
        "top_feature_name": top_candidate["feature_name"],
        "top_horizon_bars": top_candidate["horizon_bars"],
        "top_inferred_side": top_candidate["inferred_side"],
        "top_mean_directional_test_spread": top_candidate["mean_directional_test_spread"],
        "top_sign_stability_ratio": top_candidate["sign_stability_ratio"],
    }
    return AlphaScanResult(
        features=candidates,
        horizons=resolved_horizons,
        windows=window_records,
        candidates=candidate_records,
        aggregate=aggregate,
    )
