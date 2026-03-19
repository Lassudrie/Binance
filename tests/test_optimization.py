from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
from quantflow.data_model.events import BarEvent, PortfolioState, SignalEvent, SignalSide
from quantflow.strategy.base import Strategy
from quantflow.validation.optimization import (
    expand_param_grid,
    parse_param_grid_specs,
    run_walk_forward_grid_search,
)


class DirectionStrategy(Strategy):
    def __init__(self, direction: int) -> None:
        super().__init__()
        self.direction = direction

    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        if self.direction == 0:
            return None
        if self.current_target == self.direction:
            return None
        return SignalEvent(
            symbol=event.symbol,
            event_time=event.end_time,
            side=SignalSide.LONG if self.direction > 0 else SignalSide.SHORT,
            target_position=self.direction,
            reason="direction",
            context={},
        )

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, object]:
        return {"direction": self.direction}


def _features() -> pl.DataFrame:
    start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    prices = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0]
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * len(prices),
            "bar_start": [start + timedelta(seconds=i) for i in range(len(prices))],
            "bar_end": [start + timedelta(seconds=i) for i in range(len(prices))],
            "open": prices,
            "high": [price + 1.0 for price in prices],
            "low": [price - 1.0 for price in prices],
            "close": [price + 0.5 for price in prices],
            "volume": [10.0] * len(prices),
            "trade_count": [5] * len(prices),
            "session": ["asia"] * len(prices),
            "hour_of_day": [0] * len(prices),
            "vol_regime": [1.0] * len(prices),
        }
    )


def test_parse_param_grid_specs_and_expand() -> None:
    grid = parse_param_grid_specs(
        [
            "threshold=0.5,1.0",
            "enabled=true,false",
            "label=fast,slow",
        ]
    )
    combos = expand_param_grid(grid)

    assert grid["threshold"] == [0.5, 1.0]
    assert grid["enabled"] == [True, False]
    assert len(combos) == 8


def test_grid_search_selects_profitable_direction(app_config) -> None:
    result = run_walk_forward_grid_search(
        _features(),
        app_config,
        parameter_grid=[{"direction": 0}, {"direction": 1}],
        strategy_builder=lambda params: DirectionStrategy(direction=int(params["direction"])),
        objective="net_pnl",
    )

    assert result.aggregate["combo_count"] == 2
    assert result.aggregate["most_selected_params"] == {"direction": 1}
    assert result.aggregate["parameter_stability_ratio"] == 1.0
