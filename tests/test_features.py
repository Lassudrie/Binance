from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from quantflow.feature_engineering.order_flow import build_trade_features


def _silver_trades() -> pl.DataFrame:
    event_times = [
        datetime(2025, 1, 1, 0, 0, 0, 100_000, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 0, 1, 100_000, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 0, 2, 100_000, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 0, 3, 100_000, tzinfo=UTC),
    ]
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 4,
            "market": ["spot"] * 4,
            "dataset_type": ["trades"] * 4,
            "source_date": [datetime(2025, 1, 1, tzinfo=UTC).date()] * 4,
            "event_date": [datetime(2025, 1, 1, tzinfo=UTC).date()] * 4,
            "event_time": event_times,
            "trade_id": [1, 2, 3, 4],
            "price": [100.0, 101.0, 99.0, 120.0],
            "qty": [1.0, 0.5, 1.5, 0.2],
            "quote_qty": [100.0, 50.5, 148.5, 24.0],
            "is_buyer_maker": [False, True, False, False],
            "is_best_match": [True, True, True, True],
            "side_sign": [1, -1, 1, 1],
            "signed_qty": [1.0, -0.5, 1.5, 0.2],
            "signed_quote_qty": [100.0, -50.5, 148.5, 24.0],
        }
    )


def test_cumulative_delta_and_signed_volume(app_config) -> None:
    features = build_trade_features(_silver_trades(), app_config)
    cumulative = features["cumulative_delta"].to_list()
    assert cumulative[0] == 1.0
    assert cumulative[1] == 0.5
    assert cumulative[2] == 2.0
    ratios = features["taker_buy_ratio"].to_list()[:3]
    assert ratios == pytest.approx([1.0, 0.0, 1.0])
    assert "delta_rz" in features.columns
    assert "quote_delta_rz" in features.columns


def test_range_features_do_not_look_into_future(app_config) -> None:
    features = build_trade_features(_silver_trades(), app_config)
    second_bar = features.row(1, named=True)
    assert second_bar["range_high_prev"] == 100.0
    assert second_bar["range_low_prev"] == 100.0


def test_regime_indicators_exist_after_warmup(app_config) -> None:
    base_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    rows = 240
    prices = [100.0 + (0.05 * index) for index in range(rows)]
    silver = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * rows,
            "market": ["spot"] * rows,
            "dataset_type": ["trades"] * rows,
            "source_date": [base_time.date()] * rows,
            "event_date": [base_time.date()] * rows,
            "event_time": [
                base_time.replace(second=index % 60, minute=index // 60)
                for index in range(rows)
            ],
            "trade_id": list(range(rows)),
            "price": prices,
            "qty": [1.0] * rows,
            "quote_qty": prices,
            "is_buyer_maker": [False if index % 2 == 0 else True for index in range(rows)],
            "is_best_match": [True] * rows,
            "side_sign": [1 if index % 2 == 0 else -1 for index in range(rows)],
            "signed_qty": [1.0 if index % 2 == 0 else -1.0 for index in range(rows)],
            "signed_quote_qty": [
                price if index % 2 == 0 else -price
                for index, price in enumerate(prices)
            ],
        }
    )

    features = build_trade_features(silver, app_config)
    last_bar = features.row(-1, named=True)

    assert last_bar["bar_number"] == rows
    assert last_bar["ema_fast_20"] is not None
    assert last_bar["ema_trend_200"] is not None
    assert last_bar["atr_14"] is not None
    assert last_bar["rsi_14"] is not None
    assert last_bar["atr_ratio"] is not None
