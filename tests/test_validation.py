from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
from quantflow.data_model.events import BarEvent, PortfolioState, SignalEvent, SignalSide
from quantflow.strategy.base import Strategy
from quantflow.validation.bootstrap import run_bootstrap_validation
from quantflow.validation.sensitivity import run_execution_sensitivity


class FastRoundTripStrategy(Strategy):
    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        if event.end_time.second == 0:
            return SignalEvent(
                symbol=event.symbol,
                event_time=event.end_time,
                side=SignalSide.LONG,
                target_position=1,
                reason="enter",
                context={},
            )
        if event.end_time.second == 1:
            return SignalEvent(
                symbol=event.symbol,
                event_time=event.end_time,
                side=SignalSide.FLAT,
                target_position=0,
                reason="exit",
                context={},
            )
        return None

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, object]:
        return {}


def test_execution_sensitivity_penalizes_higher_cost_scenarios(app_config) -> None:
    start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    features = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 4,
            "bar_start": [start + timedelta(seconds=i) for i in range(4)],
            "bar_end": [start + timedelta(seconds=i) for i in range(4)],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [10.0, 10.0, 10.0, 10.0],
            "trade_count": [5, 5, 5, 5],
            "session": ["asia"] * 4,
            "hour_of_day": [0] * 4,
            "vol_regime": [1.0] * 4,
        }
    )

    result = run_execution_sensitivity(
        features,
        app_config,
        strategy_factory=FastRoundTripStrategy,
    )

    by_cost = {
        (item["taker_fee_bps"], item["slippage_bps"], item["latency_bars"]): item
        for item in result.scenarios
    }
    low_cost = by_cost[(2.5, 1.0, 1)]
    high_cost = by_cost[(7.5, 3.0, 1)]

    assert result.aggregate["scenario_count"] == 27
    assert float(high_cost["net_pnl"]) <= float(low_cost["net_pnl"])


def test_bootstrap_validation_produces_expected_summary(app_config) -> None:
    start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    features = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 8,
            "bar_start": [start + timedelta(seconds=i) for i in range(8)],
            "bar_end": [start + timedelta(seconds=i) for i in range(8)],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5],
            "volume": [10.0] * 8,
            "trade_count": [5] * 8,
            "session": ["asia"] * 8,
            "hour_of_day": [0] * 8,
            "vol_regime": [1.0] * 8,
        }
    )

    result = run_bootstrap_validation(
        features,
        app_config,
        strategy_factory=FastRoundTripStrategy,
    )

    assert result.aggregate["window_count"] >= 1
    assert result.aggregate["resample_count"] == app_config.validation.bootstrap_resamples
    assert len(result.aggregate["ci_95_net_pnl"]) == 2
    assert "probability_net_pnl_positive" in result.aggregate
