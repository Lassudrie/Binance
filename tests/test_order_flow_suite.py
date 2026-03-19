from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import polars as pl
from quantflow.validation.order_flow_suite import run_order_flow_suite


def _suite_frame(size: int = 8) -> pl.DataFrame:
    start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    cumulative = [-2.0, -1.8, -0.1, 1.2, -1.7, -0.1, -1.6, -0.1]
    delta = [1.2, 0.1, 0.0, 1.1, 0.2, 0.1, 1.3, 0.1]
    imbalance = [-1.4, -0.2, -0.1, -1.2, -0.2, -0.1, -1.1, -0.2]
    burst = [1.3, 0.5, 0.4, 1.2, 0.5, 0.4, 1.1, 0.5]
    prices = [100.0, 100.4, 100.6, 100.2, 100.8, 101.0, 100.7, 101.1]

    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * size,
            "bar_start": [start + timedelta(seconds=index) for index in range(size)],
            "bar_end": [start + timedelta(seconds=index) for index in range(size)],
            "open": prices,
            "high": [price + 0.2 for price in prices],
            "low": [price - 0.2 for price in prices],
            "close": prices,
            "volume": [50.0] * size,
            "trade_count": [10] * size,
            "session": ["asia"] * size,
            "hour_of_day": [0] * size,
            "vol_regime": [1.1] * size,
            "cumulative_delta_rz": cumulative,
            "rolling_delta_rz": cumulative,
            "delta_rz": delta,
            "quote_delta_rz": delta,
            "trade_count_imbalance_rz": imbalance,
            "burst_intensity_rz": burst,
        }
    )


def test_order_flow_suite_produces_expected_cells(app_config) -> None:
    config = replace(
        app_config,
        portfolio=replace(app_config.portfolio, allow_short=False),
        validation=replace(app_config.validation, train_bars=3, test_bars=2, step_bars=1),
    )
    feature_frames = {
        "5s": _suite_frame(),
        "15s": _suite_frame(),
    }

    result = run_order_flow_suite(feature_frames=feature_frames, config=config)

    assert len(result.cells) == 12
    assert len(result.ranked_candidates) == 24
    assert {cell["bar_size"] for cell in result.cells} == {"5s", "15s"}
    assert {cell["execution_mode"] for cell in result.cells} == {"market", "passive"}
    assert {cell["strategy_name"] for cell in result.cells} == {
        "cumdelta_reversion_v1",
        "delta_impulse_continuation_v1",
        "imbalance_burst_exhaustion_v1",
    }
    assert result.aggregate["cell_count"] == 12
