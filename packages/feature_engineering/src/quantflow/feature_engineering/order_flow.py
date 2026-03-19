from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import polars as pl
from quantflow.core.config import AppConfig
from quantflow.data_ingestion.pipeline import load_silver_dataset
from quantflow.utils.io import ensure_directory

LOGGER = logging.getLogger(__name__)
EPSILON = 1e-9
ATR_WINDOW = 14
RSI_WINDOW = 14
EMA_FAST_WINDOW = 20
EMA_TREND_WINDOW = 200


def _bar_size_to_timedelta(bar_size: str) -> timedelta:
    value = int(bar_size[:-1])
    unit = bar_size[-1]
    if unit == "s":
        return timedelta(seconds=value)
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    raise ValueError(f"Unsupported bar size: {bar_size}")


def _add_robust_zscores(frame: pl.DataFrame, columns: list[str], window: int) -> pl.DataFrame:
    result = frame
    helper_columns: list[str] = []

    for column in columns:
        median_column = f"__{column}_median"
        mad_column = f"__{column}_mad"
        helper_columns.extend([median_column, mad_column])
        result = result.with_columns(
            pl.col(column)
            .shift(1)
            .rolling_median(window_size=window, min_samples=1)
            .over("symbol")
            .alias(median_column)
        )
        result = result.with_columns(
            (pl.col(column).shift(1) - pl.col(median_column))
            .abs()
            .rolling_median(window_size=window, min_samples=1)
            .over("symbol")
            .alias(mad_column)
        )
        result = result.with_columns(
            pl.when(pl.col(mad_column).abs() > EPSILON)
            .then((pl.col(column) - pl.col(median_column)) / pl.col(mad_column))
            .otherwise(pl.lit(None))
            .alias(f"{column}_rz")
        )

    return result.drop(helper_columns)


def build_trade_features(trades: pl.DataFrame, config: AppConfig) -> pl.DataFrame:
    bar_delta = _bar_size_to_timedelta(config.features.bar_size)
    rolling_window = config.features.rolling_window_bars
    vol_window = config.features.volatility_window_bars
    range_window = config.features.range_window_bars
    norm_window = config.features.normalization_window_bars

    bars = (
        trades.sort(["symbol", "event_time", "trade_id"])
        .group_by_dynamic(
            index_column="event_time",
            every=config.features.bar_size,
            period=config.features.bar_size,
            group_by="symbol",
            label="right",
            closed="right",
        )
        .agg(
            [
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("qty").sum().alias("volume"),
                pl.col("quote_qty").sum().alias("quote_volume"),
                pl.len().alias("trade_count"),
                pl.col("signed_qty").sum().alias("delta"),
                pl.col("signed_quote_qty").sum().alias("quote_delta"),
                pl.col("side_sign").eq(1).sum().alias("buy_count"),
                pl.col("side_sign").eq(-1).sum().alias("sell_count"),
                pl.when(pl.col("side_sign") == 1)
                .then(pl.col("qty"))
                .otherwise(pl.lit(0.0))
                .sum()
                .alias("buy_aggressor_volume"),
                pl.when(pl.col("side_sign") == -1)
                .then(pl.col("qty"))
                .otherwise(pl.lit(0.0))
                .sum()
                .alias("sell_aggressor_volume"),
            ]
        )
        .rename({"event_time": "bar_end"})
        .sort(["symbol", "bar_end"])
    )

    if bars.is_empty():
        raise ValueError("No trade bars were produced from the silver trades dataset")

    bars = bars.with_columns(
        [
            (pl.col("bar_end") - pl.lit(bar_delta)).alias("bar_start"),
            (pl.col("quote_volume") / (pl.col("volume") + EPSILON)).alias("vwap"),
            (pl.col("volume") / (pl.col("trade_count") + EPSILON)).alias("average_trade_size"),
            (pl.col("buy_aggressor_volume") / (pl.col("volume") + EPSILON)).alias(
                "taker_buy_ratio"
            ),
            (
                (pl.col("buy_count") - pl.col("sell_count")) / (pl.col("trade_count") + EPSILON)
            ).alias("trade_count_imbalance"),
            (
                (pl.col("buy_count") - pl.col("sell_count")).abs()
                / (pl.col("trade_count") + EPSILON)
            ).alias("tick_direction_persistence"),
            pl.col("delta").cum_sum().over("symbol").alias("cumulative_delta"),
            pl.col("delta")
            .rolling_sum(window_size=rolling_window, min_samples=1)
            .over("symbol")
            .alias("rolling_delta"),
            pl.col("close").cum_count().over("symbol").alias("bar_number"),
        ]
    )

    bars = bars.with_columns(
        [
            (
                pl.col("trade_count")
                / (
                    pl.col("trade_count")
                    .shift(1)
                    .rolling_mean(window_size=rolling_window, min_samples=1)
                    .over("symbol")
                    + EPSILON
                )
            ).alias("burst_intensity"),
            (
                pl.col("close")
                .log()
                .diff()
                .rolling_std(window_size=vol_window, min_samples=1)
                .over("symbol")
            ).alias("realized_volatility"),
            pl.col("close").pct_change(range_window).over("symbol").alias("micro_momentum"),
            pl.col("high")
            .shift(1)
            .rolling_max(window_size=range_window, min_samples=1)
            .over("symbol")
            .alias("range_high_prev"),
            pl.col("low")
            .shift(1)
            .rolling_min(window_size=range_window, min_samples=1)
            .over("symbol")
            .alias("range_low_prev"),
            pl.col("bar_end").dt.hour().alias("hour_of_day"),
        ]
    )

    bars = bars.with_columns(
        [
            pl.col("close").shift(1).over("symbol").alias("prev_close"),
            pl.col("close").diff().over("symbol").alias("close_change"),
            pl.when(pl.col("hour_of_day") < 7)
            .then(pl.lit("asia"))
            .when(pl.col("hour_of_day") < 13)
            .then(pl.lit("europe"))
            .when(pl.col("hour_of_day") < 21)
            .then(pl.lit("us"))
            .otherwise(pl.lit("off_hours"))
            .alias("session"),
            pl.when((pl.col("range_high_prev") - pl.col("range_low_prev")).abs() > EPSILON)
            .then(
                (pl.col("close") - pl.col("range_low_prev"))
                / ((pl.col("range_high_prev") - pl.col("range_low_prev")) + EPSILON)
            )
            .otherwise(pl.lit(None))
            .alias("micro_range_breakout"),
            (
                pl.col("realized_volatility")
                / (
                    pl.col("realized_volatility")
                    .shift(1)
                    .rolling_median(window_size=vol_window, min_samples=1)
                    .over("symbol")
                    + EPSILON
                )
            ).alias("vol_regime"),
            pl.lit("unknown").alias("news_regime"),
        ]
    )

    bars = bars.with_columns(
        [
            pl.max_horizontal(
                [
                    (pl.col("high") - pl.col("low")),
                    (pl.col("high") - pl.col("prev_close")).abs(),
                    (pl.col("low") - pl.col("prev_close")).abs(),
                ]
            ).alias("true_range"),
            pl.when(pl.col("close_change") > 0.0)
            .then(pl.col("close_change"))
            .otherwise(pl.lit(0.0))
            .alias("rsi_gain"),
            pl.when(pl.col("close_change") < 0.0)
            .then(-pl.col("close_change"))
            .otherwise(pl.lit(0.0))
            .alias("rsi_loss"),
        ]
    )

    bars = bars.with_columns(
        [
            pl.col("close")
            .ewm_mean(span=EMA_FAST_WINDOW, adjust=False, min_samples=EMA_FAST_WINDOW)
            .over("symbol")
            .alias("ema_fast_20"),
            pl.col("close")
            .ewm_mean(span=EMA_TREND_WINDOW, adjust=False, min_samples=EMA_TREND_WINDOW)
            .over("symbol")
            .alias("ema_trend_200"),
            pl.col("true_range")
            .rolling_mean(window_size=ATR_WINDOW, min_samples=ATR_WINDOW)
            .over("symbol")
            .alias("atr_14"),
            pl.col("rsi_gain")
            .rolling_mean(window_size=RSI_WINDOW, min_samples=RSI_WINDOW)
            .over("symbol")
            .alias("rsi_avg_gain"),
            pl.col("rsi_loss")
            .rolling_mean(window_size=RSI_WINDOW, min_samples=RSI_WINDOW)
            .over("symbol")
            .alias("rsi_avg_loss"),
        ]
    )

    bars = bars.with_columns(
        [
            (pl.col("atr_14") / (pl.col("close") + EPSILON)).alias("atr_ratio"),
            pl.when(pl.col("rsi_avg_loss") > EPSILON)
            .then(100.0 - (100.0 / (1.0 + (pl.col("rsi_avg_gain") / pl.col("rsi_avg_loss")))))
            .when(pl.col("rsi_avg_gain") > EPSILON)
            .then(pl.lit(100.0))
            .otherwise(pl.lit(50.0))
            .alias("rsi_14"),
        ]
    )

    bars = _add_robust_zscores(
        bars,
        columns=[
            "delta",
            "quote_delta",
            "cumulative_delta",
            "rolling_delta",
            "trade_count_imbalance",
            "burst_intensity",
            "realized_volatility",
            "micro_momentum",
        ],
        window=norm_window,
    )

    return bars.drop(
        [
            "prev_close",
            "close_change",
            "true_range",
            "rsi_gain",
            "rsi_loss",
            "rsi_avg_gain",
            "rsi_avg_loss",
        ]
    )


def _gold_output_path(config: AppConfig, symbol: str, dataset_type: str) -> Path:
    output_dir = (
        config.paths.gold_dir
        / "features"
        / f"dataset_type={dataset_type}"
        / f"bar_size={config.features.bar_size}"
        / f"feature_profile={config.features.feature_profile}"
        / f"symbol={symbol}"
    )
    ensure_directory(output_dir)
    return output_dir / "part-0.parquet"


def build_feature_dataset(
    config: AppConfig,
    symbol: str,
    dataset_type: str | None = None,
) -> Path:
    resolved_dataset_type = dataset_type or config.dataset.dataset_type
    silver = load_silver_dataset(config, symbol, dataset_type=resolved_dataset_type)
    features = build_trade_features(silver, config)
    output_path = _gold_output_path(config, symbol, resolved_dataset_type)
    features.write_parquet(output_path)
    LOGGER.info(
        "Built gold features for %s dataset=%s at %s",
        symbol,
        resolved_dataset_type,
        output_path,
    )
    return output_path
