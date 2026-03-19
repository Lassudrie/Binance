from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import polars as pl
from quantflow.validation.alpha import run_alpha_scan


def _alpha_frame(size: int = 36) -> pl.DataFrame:
    start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    predictive_feature = [1.0 if index % 2 == 0 else -1.0 for index in range(size)]
    noise_feature = [float(((index * 7) % 5) - 2) for index in range(size)]

    closes = [100.0]
    for index in range(size - 1):
        forward_return = 0.003 * predictive_feature[index]
        closes.append(closes[-1] * (1.0 + forward_return))

    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * size,
            "bar_start": [start + timedelta(seconds=index) for index in range(size)],
            "bar_end": [start + timedelta(seconds=index) for index in range(size)],
            "open": closes,
            "high": [value + 0.1 for value in closes],
            "low": [value - 0.1 for value in closes],
            "close": closes,
            "volume": [10.0] * size,
            "quote_volume": [value * 10.0 for value in closes],
            "trade_count": [5] * size,
            "predictive_feature": predictive_feature,
            "noise_feature": noise_feature,
        }
    )


def test_alpha_scan_ranks_predictive_feature_first(app_config) -> None:
    config = replace(
        app_config,
        validation=replace(
            app_config.validation,
            train_bars=12,
            test_bars=8,
            step_bars=8,
            embargo_bars=0,
        ),
    )
    result = run_alpha_scan(
        _alpha_frame(),
        config,
        feature_names=["predictive_feature", "noise_feature"],
        horizons=[1],
    )

    assert result.aggregate["top_feature_name"] == "predictive_feature"
    assert result.candidates[0]["mean_directional_test_spread"] > 0.0
    assert result.candidates[0]["sign_stability_ratio"] == 1.0
