from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import polars as pl
from quantflow.backtest.engine import run_backtest
from quantflow.data_model.events import BarEvent, OrderSide, PortfolioState, SignalEvent, SignalSide
from quantflow.strategy.base import Strategy


class EnterAndHoldStrategy(Strategy):
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
        return None

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, object]:
        return {}


class AlwaysLongStrategy(Strategy):
    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        if self.current_target == 1:
            return None
        return SignalEvent(
            symbol=event.symbol,
            event_time=event.end_time,
            side=SignalSide.LONG,
            target_position=1,
            reason="always_long",
            context={"session": "asia", "hour_of_day": 0, "vol_regime": 1.0},
        )

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, object]:
        return {}


def _frame(closes: list[float]) -> pl.DataFrame:
    start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * len(closes),
            "bar_start": [start + timedelta(seconds=i) for i in range(len(closes))],
            "bar_end": [start + timedelta(seconds=i) for i in range(len(closes))],
            "open": closes,
            "high": [value + 0.5 for value in closes],
            "low": [value - 0.5 for value in closes],
            "close": closes,
            "volume": [10.0] * len(closes),
            "trade_count": [5] * len(closes),
            "session": ["asia"] * len(closes),
            "hour_of_day": [0] * len(closes),
            "vol_regime": [1.0] * len(closes),
        }
    )


def test_risk_manager_exits_on_max_holding(app_config) -> None:
    config = replace(
        app_config,
        risk=replace(
            app_config.risk,
            max_holding_bars=2,
            stop_loss_bps=0.0,
            take_profit_bps=0.0,
            trailing_stop_bps=0.0,
            cooldown_bars_after_stop=0,
        ),
    )
    result = run_backtest(
        _frame([100.0, 101.0, 102.0, 103.0, 104.0]),
        EnterAndHoldStrategy(),
        config,
    )

    assert result.metrics["trade_count"] == 1
    assert result.trades[0].signal_reason == "risk_max_holding_bars"


def test_risk_manager_exits_on_stop_loss(app_config) -> None:
    config = replace(
        app_config,
        risk=replace(
            app_config.risk,
            max_holding_bars=0,
            stop_loss_bps=50.0,
            take_profit_bps=0.0,
            trailing_stop_bps=0.0,
            cooldown_bars_after_stop=0,
        ),
    )
    result = run_backtest(_frame([100.0, 100.0, 98.0, 97.0, 97.0]), EnterAndHoldStrategy(), config)

    assert result.metrics["trade_count"] == 1
    assert result.trades[0].signal_reason == "risk_stop_loss"


def test_risk_manager_exits_on_trailing_stop(app_config) -> None:
    config = replace(
        app_config,
        risk=replace(
            app_config.risk,
            max_holding_bars=0,
            stop_loss_bps=0.0,
            take_profit_bps=0.0,
            trailing_stop_bps=50.0,
            cooldown_bars_after_stop=0,
        ),
    )
    result = run_backtest(
        _frame([100.0, 100.0, 102.0, 101.0, 101.0, 101.0]),
        EnterAndHoldStrategy(),
        config,
    )

    assert result.metrics["trade_count"] == 1
    assert result.trades[0].signal_reason == "risk_trailing_stop"


def test_risk_manager_blocks_reentry_during_cooldown(app_config) -> None:
    config = replace(
        app_config,
        risk=replace(
            app_config.risk,
            max_holding_bars=0,
            stop_loss_bps=50.0,
            take_profit_bps=0.0,
            trailing_stop_bps=0.0,
            cooldown_bars_after_stop=2,
        ),
    )
    result = run_backtest(
        _frame([100.0, 100.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0]),
        AlwaysLongStrategy(),
        config,
    )

    buy_fill_seconds = [
        fill.event_time.second for fill in result.fills if fill.side == OrderSide.BUY
    ]

    assert result.metrics["trade_count"] == 1
    assert result.trades[0].signal_reason == "risk_stop_loss"
    assert buy_fill_seconds == [1, 6]
