from __future__ import annotations

from datetime import UTC, datetime, timedelta

from quantflow.data_model.events import BarEvent, PortfolioState, SignalSide
from quantflow.strategy.orderflow import Spot10mRegimePullbackStrategy


def _event(*, taker_buy_ratio: float | None = 0.6) -> BarEvent:
    end_time = datetime(2025, 1, 1, 0, 10, tzinfo=UTC)
    return BarEvent(
        symbol="BTCUSDT",
        start_time=end_time - timedelta(minutes=10),
        end_time=end_time,
        open_price=100.0,
        high_price=101.0,
        low_price=99.9,
        close_price=100.8,
        volume=100.0,
        trade_count=20,
        features={
            "bar_number": 250,
            "ema_fast_20": 100.5,
            "ema_trend_200": 99.5,
            "atr_14": 0.8,
            "atr_ratio": 0.008,
            "rsi_14": 58.0,
            "taker_buy_ratio": taker_buy_ratio,
            "session": "us",
            "hour_of_day": 0,
            "vol_regime": 1.0,
        },
    )


def _state() -> PortfolioState:
    return PortfolioState(
        event_time=datetime(2025, 1, 1, 0, 10, tzinfo=UTC),
        cash=10_000.0,
        equity=10_000.0,
        positions={},
        gross_exposure=0.0,
        net_exposure=0.0,
    )


def test_spot_regime_pullback_generates_long_signal() -> None:
    strategy = Spot10mRegimePullbackStrategy()
    signal = strategy.on_event(_event(), _state())

    assert signal is not None
    assert signal.side == SignalSide.LONG
    assert signal.target_position == 1
    assert signal.context["stop_distance"] == 0.8
    assert signal.context["target_distance"] == 0.96
    assert signal.context["time_stop_bars"] == 6


def test_spot_regime_pullback_respects_taker_buy_ratio_filter() -> None:
    strategy = Spot10mRegimePullbackStrategy(
        use_taker_buy_ratio_filter=True,
        taker_buy_ratio_min=0.55,
    )
    signal = strategy.on_event(_event(taker_buy_ratio=0.51), _state())

    assert signal is None
