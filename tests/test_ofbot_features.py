from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ofbot.market.features import FeatureEngine
from ofbot.market.orderbook import OrderBookState
from ofbot.market.trades import parse_agg_trade, parse_book_ticker, parse_depth


def _time_millis(base: datetime, seconds: int, millis: int = 0) -> int:
    event_time = base + timedelta(seconds=seconds, milliseconds=millis)
    return int(event_time.timestamp() * 1000)


def _time(base: datetime, seconds: int, millis: int = 0) -> datetime:
    return base + timedelta(seconds=seconds, milliseconds=millis)


def test_parse_sign_from_agg_trade_m_flag() -> None:
    buy_aggressor = parse_agg_trade(
        {
            "s": "BTCUSDT",
            "E": 1,
            "a": 10,
            "f": 1,
            "l": 2,
            "p": "100.0",
            "q": "0.4",
            "m": False,
        }
    )
    sell_aggressor = parse_agg_trade(
        {
            "s": "BTCUSDT",
            "E": 2,
            "a": 11,
            "f": 3,
            "l": 4,
            "p": "101.0",
            "q": "0.2",
            "m": True,
        }
    )

    assert buy_aggressor.aggressor_side == 1
    assert buy_aggressor.signed_qty == 0.4
    assert sell_aggressor.aggressor_side == -1
    assert sell_aggressor.signed_qty == -0.2


def test_feature_engine_builds_horizon_features() -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    engine = FeatureEngine(
        horizons_sec=[1, 5],
        feature_window_sec=300,
        large_trade_quantile=0.7,
        volatility_window_sec=60,
        zscore_window=30,
    )
    book = OrderBookState(symbol="BTCUSDT")
    book.apply_book_ticker(
        parse_book_ticker(
            {
                "s": "BTCUSDT",
                "E": _time_millis(base, 0),
                "b": "100.0",
                "B": "2.0",
                "a": "100.2",
                "A": "1.0",
            }
        )
    )

    engine.on_orderbook(_time(base, 0), symbol="BTCUSDT", book=book)
    event_specs = [
        (False, 0.4, 100.00),
        (False, 0.5, 100.05),
        (True, 0.1, 100.10),
        (False, 0.2, 100.15),
        (True, 0.6, 100.20),
    ]
    for offset, (aggressor_is_buyer, quantity, price) in enumerate(event_specs):
        event = parse_agg_trade(
            {
                "s": "BTCUSDT",
                "E": _time_millis(base, 1 + offset * 1),
                "a": 100 + offset,
                "f": 100 + offset,
                "l": 100 + offset,
                "p": f"{price}",
                "q": f"{quantity}",
                "m": not aggressor_is_buyer,
            }
        )
        engine.on_agg_trade(event)

    book.apply_depth(
        parse_depth(
            {
                "s": "BTCUSDT",
                "E": _time_millis(base, 2),
                "U": 1,
                "u": 1,
                "b": [["99.8", "2.0"]],
                "a": [["100.3", "1.0"]],
            }
        )
    )

    snap = engine.snapshot("BTCUSDT", event_time=_time(base, 2), book=book)
    f = snap.features

    assert f["cvd_base_1s"] < 0.0
    assert "burst_intensity_1s" in f
    assert "large_trade_count_1s" in f
    assert "large_trade_signed_flow_1s" in f
    assert "trade_count_intensity_1s" in f
    assert "distance_vwap_bps_1s" in f
    assert f["queue_imbalance"] > 0.0
    assert f["regime_volatility"] in {"low", "high", "neutral"}
    assert "cvd_base_1s_z" in f
    assert "microprice_drift_bps_5s" in f
