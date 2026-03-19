from __future__ import annotations

from datetime import UTC, datetime

from quantflow.data_model.events import BarEvent, PortfolioState
from quantflow.strategy.baselines import NoTradeBaselineStrategy, RandomBaselineStrategy
from quantflow.strategy.orderflow import (
    CumDeltaReversionV1Strategy,
    CumulativeDeltaMeanReversionStrategy,
    DeltaContinuationStrategy,
    DeltaImpulseContinuationV1Strategy,
    ImbalanceBurstExhaustionV1Strategy,
)


def _bar_event() -> BarEvent:
    return BarEvent(
        symbol="BTCUSDT",
        start_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        end_time=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
        open_price=100.0,
        high_price=100.5,
        low_price=99.5,
        close_price=100.2,
        volume=1.0,
        trade_count=1,
        features={"session": "asia", "hour_of_day": 0},
    )


def _state() -> PortfolioState:
    return PortfolioState(
        event_time=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
        cash=10_000.0,
        equity=10_000.0,
        positions={},
        gross_exposure=0.0,
        net_exposure=0.0,
    )


def test_no_trade_baseline_never_emits_signal() -> None:
    strategy = NoTradeBaselineStrategy()
    assert strategy.on_event(_bar_event(), _state()) is None


def test_random_baseline_is_reproducible_after_reset() -> None:
    strategy = RandomBaselineStrategy(enter_probability=1.0, flat_probability=0.0, seed=11)
    first = strategy.on_event(_bar_event(), _state())
    strategy.reset()
    second = strategy.on_event(_bar_event(), _state())
    assert first is not None
    assert second is not None
    assert first.target_position == second.target_position


def test_cumulative_delta_mean_reversion_goes_long_on_capitulation() -> None:
    strategy = CumulativeDeltaMeanReversionStrategy(
        cumulative_delta_threshold=1.0,
        price_extension_threshold=0.3,
        exit_zscore_threshold=0.2,
    )
    event = BarEvent(
        symbol="BTCUSDT",
        start_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        end_time=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
        open_price=100.0,
        high_price=100.5,
        low_price=99.5,
        close_price=99.6,
        volume=1.0,
        trade_count=1,
        features={
            "session": "asia",
            "hour_of_day": 0,
            "cumulative_delta_rz": -1.5,
            "micro_range_breakout": 0.1,
        },
    )

    signal = strategy.on_event(event, _state())

    assert signal is not None
    assert signal.target_position == 1
    assert signal.reason == "cumulative_delta_mean_reversion_long"


def test_delta_continuation_goes_short_on_negative_flow() -> None:
    strategy = DeltaContinuationStrategy(
        delta_threshold=0.8,
        imbalance_threshold=0.1,
        exit_delta_threshold=0.2,
    )
    event = BarEvent(
        symbol="BTCUSDT",
        start_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        end_time=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
        open_price=100.0,
        high_price=100.5,
        low_price=99.5,
        close_price=99.8,
        volume=1.0,
        trade_count=1,
        features={
            "session": "asia",
            "hour_of_day": 0,
            "delta_rz": -1.0,
            "trade_count_imbalance": -0.3,
        },
    )

    signal = strategy.on_event(event, _state())

    assert signal is not None
    assert signal.target_position == -1
    assert signal.reason == "delta_continuation_short"


def test_cumdelta_reversion_v1_requires_vol_filter() -> None:
    strategy = CumDeltaReversionV1Strategy(entry_z=1.5, exit_z=0.25, vol_regime_min=1.0)
    event = BarEvent(
        symbol="BTCUSDT",
        start_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        end_time=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
        open_price=100.0,
        high_price=100.5,
        low_price=99.5,
        close_price=99.8,
        volume=1.0,
        trade_count=1,
        features={
            "session": "asia",
            "hour_of_day": 0,
            "cumulative_delta_rz": -2.0,
            "vol_regime": 0.8,
        },
    )

    assert strategy.on_event(event, _state()) is None


def test_cumdelta_reversion_v1_respects_allowed_sessions() -> None:
    strategy = CumDeltaReversionV1Strategy(
        entry_z=1.5,
        exit_z=0.25,
        vol_regime_min=0.0,
        allowed_sessions=["us"],
    )
    event = BarEvent(
        symbol="BTCUSDT",
        start_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        end_time=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
        open_price=100.0,
        high_price=100.5,
        low_price=99.5,
        close_price=99.8,
        volume=1.0,
        trade_count=1,
        features={
            "session": "asia",
            "hour_of_day": 0,
            "cumulative_delta_rz": -2.0,
            "vol_regime": 0.7,
        },
    )

    assert strategy.on_event(event, _state()) is None


def test_delta_impulse_continuation_v1_goes_long_on_positive_flow() -> None:
    strategy = DeltaImpulseContinuationV1Strategy(
        source="delta_rz",
        entry_z=1.0,
        exit_z=0.25,
        vol_regime_min=1.0,
    )
    event = BarEvent(
        symbol="BTCUSDT",
        start_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        end_time=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
        open_price=100.0,
        high_price=100.5,
        low_price=99.5,
        close_price=100.4,
        volume=1.0,
        trade_count=1,
        features={
            "session": "asia",
            "hour_of_day": 0,
            "delta_rz": 1.2,
            "vol_regime": 1.2,
        },
    )

    signal = strategy.on_event(event, _state())

    assert signal is not None
    assert signal.target_position == 1
    assert signal.reason == "delta_impulse_continuation_v1_long"


def test_imbalance_burst_exhaustion_v1_goes_long_on_capitulation_burst() -> None:
    strategy = ImbalanceBurstExhaustionV1Strategy(
        imbalance_z=1.0,
        burst_z=1.0,
        exit_z=0.25,
        vol_regime_min=1.0,
    )
    event = BarEvent(
        symbol="BTCUSDT",
        start_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        end_time=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
        open_price=100.0,
        high_price=100.5,
        low_price=99.5,
        close_price=99.7,
        volume=1.0,
        trade_count=1,
        features={
            "session": "asia",
            "hour_of_day": 0,
            "trade_count_imbalance_rz": -1.5,
            "burst_intensity_rz": 1.3,
            "vol_regime": 1.1,
        },
    )

    signal = strategy.on_event(event, _state())

    assert signal is not None
    assert signal.target_position == 1
    assert signal.reason == "imbalance_burst_exhaustion_v1_long"
