from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import polars as pl
from quantflow.backtest.engine import run_backtest
from quantflow.data_model.events import BarEvent, PortfolioState, SignalEvent, SignalSide
from quantflow.strategy.base import Strategy


class OneShotLongStrategy(Strategy):
    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        if event.end_time.second == 0:
            return SignalEvent(
                symbol=event.symbol,
                event_time=event.end_time,
                side=SignalSide.LONG,
                target_position=1,
                reason="enter_long",
                context={"session": "asia", "hour_of_day": 0, "vol_regime": 1.0},
            )
        if event.end_time.second == 2:
            return SignalEvent(
                symbol=event.symbol,
                event_time=event.end_time,
                side=SignalSide.FLAT,
                target_position=0,
                reason="exit_long",
                context={"session": "asia", "hour_of_day": 0, "vol_regime": 1.0},
            )
        return None

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, object]:
        return {}


class RepeatingLongStrategy(Strategy):
    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        return SignalEvent(
            symbol=event.symbol,
            event_time=event.end_time,
            side=SignalSide.LONG,
            target_position=1,
            reason="reenter_long",
            context={"session": "asia", "hour_of_day": 0, "vol_regime": 1.0},
        )

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, object]:
        return {}


class BracketOneShotStrategy(Strategy):
    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        if event.end_time.second != 0:
            return None
        return SignalEvent(
            symbol=event.symbol,
            event_time=event.end_time,
            side=SignalSide.LONG,
            target_position=1,
            reason="enter_with_bracket",
            context={
                "session": "asia",
                "hour_of_day": 0,
                "vol_regime": 1.0,
                "atr_ratio": 0.01,
                "stop_distance": 1.0,
                "target_distance": 1.0,
                "time_stop_bars": 6,
                "entry_reason": "enter_with_bracket",
                "atr_entry": 1.0,
            },
        )

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, object]:
        return {}


class RankedEntryStrategy(Strategy):
    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        if event.end_time.second != 0:
            return None
        atr_ratio = 0.02 if event.symbol == "ETHUSDT" else 0.01
        return SignalEvent(
            symbol=event.symbol,
            event_time=event.end_time,
            side=SignalSide.LONG,
            target_position=1,
            reason="ranked_entry",
            context={"atr_ratio": atr_ratio},
        )

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, object]:
        return {}


class DailyLossLimitStrategy(Strategy):
    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        if event.end_time.second not in {0, 2}:
            return None
        return SignalEvent(
            symbol=event.symbol,
            event_time=event.end_time,
            side=SignalSide.LONG,
            target_position=1,
            reason="loss_limited_entry",
            context={
                "atr_ratio": 0.01,
                "stop_distance": 1.0,
                "target_distance": 10.0,
                "time_stop_bars": 6,
                "entry_reason": "loss_limited_entry",
            },
        )

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, object]:
        return {}


def test_backtest_pnl_is_consistent_with_costs(app_config) -> None:
    start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 4,
            "bar_start": [start + timedelta(seconds=i) for i in range(4)],
            "bar_end": [start + timedelta(seconds=i) for i in range(4)],
            "open": [100.0, 101.0, 103.0, 104.0],
            "high": [101.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 102.0, 103.0],
            "close": [100.5, 102.5, 103.5, 104.5],
            "volume": [10.0, 10.0, 10.0, 10.0],
            "trade_count": [5, 5, 5, 5],
            "session": ["asia"] * 4,
            "hour_of_day": [0] * 4,
            "vol_regime": [1.0] * 4,
        }
    )

    result = run_backtest(frame, OneShotLongStrategy(), app_config)

    assert result.metrics["trade_count"] == 1
    assert result.metrics["gross_pnl"] > result.metrics["net_pnl"]
    assert result.metrics["net_pnl"] > 0.0


def test_backtest_does_not_stack_duplicate_pending_limit_orders(app_config) -> None:
    config = replace(
        app_config,
        execution=replace(
            app_config.execution,
            default_order_type="limit",
            limit_offset_bps=5.0,
            max_order_lifetime_bars=4,
        ),
    )
    start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 4,
            "bar_start": [start + timedelta(seconds=i) for i in range(4)],
            "bar_end": [start + timedelta(seconds=i) for i in range(4)],
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.2, 100.2, 100.2, 100.2],
            "low": [100.0, 100.0, 99.9, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [10.0, 10.0, 10.0, 10.0],
            "trade_count": [5, 5, 5, 5],
            "session": ["asia"] * 4,
            "hour_of_day": [0] * 4,
            "vol_regime": [1.0] * 4,
        }
    )

    result = run_backtest(frame, RepeatingLongStrategy(), config)

    assert len(result.fills) == 1
    assert result.fills[0].quantity == 1.0


def test_backtest_bracket_stop_wins_when_stop_and_target_hit_same_bar(app_config) -> None:
    start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 3,
            "bar_start": [start + timedelta(seconds=i) for i in range(3)],
            "bar_end": [start + timedelta(seconds=i) for i in range(3)],
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 101.5, 100.0],
            "low": [100.0, 98.5, 100.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [10.0, 10.0, 10.0],
            "trade_count": [5, 5, 5],
            "session": ["asia"] * 3,
            "hour_of_day": [0] * 3,
            "vol_regime": [1.0] * 3,
        }
    )

    result = run_backtest(frame, BracketOneShotStrategy(), app_config)

    assert result.metrics["trade_count"] == 1
    assert result.trades[0].exit_reason == "bracket_stop_loss"
    assert result.trades[0].net_pnl < 0.0


def test_backtest_selects_highest_atr_ratio_when_slots_are_limited(app_config) -> None:
    config = replace(
        app_config,
        portfolio=replace(app_config.portfolio, allow_short=False),
        risk=replace(app_config.risk, max_open_positions=1, risk_per_trade_fraction=0.0),
    )
    start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    timestamps = [start, start, start + timedelta(seconds=1), start + timedelta(seconds=1)]
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT", "ETHUSDT"],
            "bar_start": timestamps,
            "bar_end": timestamps,
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [10.0, 10.0, 10.0, 10.0],
            "trade_count": [5, 5, 5, 5],
            "session": ["asia"] * 4,
            "hour_of_day": [0] * 4,
            "vol_regime": [1.0] * 4,
            "atr_ratio": [0.01, 0.02, 0.01, 0.02],
        }
    )

    result = run_backtest(frame, RankedEntryStrategy(), config)

    buy_fills = [fill for fill in result.fills if fill.side.value == "buy"]
    assert len(buy_fills) == 1
    assert buy_fills[0].symbol == "ETHUSDT"


def test_backtest_blocks_new_entries_after_daily_loss_limit(app_config) -> None:
    config = replace(
        app_config,
        risk=replace(
            app_config.risk,
            risk_per_trade_fraction=0.0025,
            daily_loss_limit_r=0.5,
            max_open_positions=1,
        ),
    )
    start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 5,
            "bar_start": [start + timedelta(seconds=i) for i in range(5)],
            "bar_end": [start + timedelta(seconds=i) for i in range(5)],
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 98.0, 100.0, 100.0, 100.0],
            "close": [100.0, 99.0, 100.0, 100.0, 100.0],
            "volume": [10_000.0] * 5,
            "trade_count": [5] * 5,
            "session": ["asia"] * 5,
            "hour_of_day": [0] * 5,
            "vol_regime": [1.0] * 5,
        }
    )

    result = run_backtest(frame, DailyLossLimitStrategy(), config)

    buy_fills = [fill for fill in result.fills if fill.side.value == "buy"]
    assert len(buy_fills) == 1
    assert result.metrics["trade_count"] == 1
