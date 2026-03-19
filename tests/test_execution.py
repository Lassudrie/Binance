from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from quantflow.backtest.transforms import SignalToOrderTranslator
from quantflow.data_model.events import (
    BarEvent,
    OrderEvent,
    OrderSide,
    OrderType,
    PortfolioState,
    SignalEvent,
    SignalSide,
)
from quantflow.execution.simulator import ExecutionSimulator


def test_market_execution_applies_fees_and_slippage(app_config) -> None:
    simulator = ExecutionSimulator(app_config.execution)
    bar = BarEvent(
        symbol="BTCUSDT",
        start_time=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
        end_time=datetime(2025, 1, 1, 0, 0, 2, tzinfo=UTC),
        open_price=100.0,
        high_price=101.0,
        low_price=99.0,
        close_price=100.5,
        volume=10.0,
        trade_count=5,
        features={},
    )
    order = OrderEvent(
        symbol="BTCUSDT",
        created_time=bar.start_time,
        activation_time=bar.start_time,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1.0,
        target_position=1,
    )
    simulator.submit_orders([order], current_index=0)
    fills = simulator.process_bar(bar, current_index=1)

    assert len(fills) == 1
    fill = fills[0]
    assert round(fill.price, 6) == round(100.0 * 1.0004, 6)
    assert round(fill.fee, 6) == round(fill.price * 0.0005, 6)


def test_market_execution_can_partial_fill_across_bars(app_config) -> None:
    execution_config = replace(
        app_config.execution,
        market_participation_cap=0.2,
        max_order_lifetime_bars=3,
    )
    simulator = ExecutionSimulator(execution_config)
    order = OrderEvent(
        symbol="BTCUSDT",
        created_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        activation_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=3.0,
        target_position=1,
    )
    simulator.submit_orders([order], current_index=0)

    first_bar = BarEvent(
        symbol="BTCUSDT",
        start_time=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
        end_time=datetime(2025, 1, 1, 0, 0, 2, tzinfo=UTC),
        open_price=100.0,
        high_price=101.0,
        low_price=99.0,
        close_price=100.5,
        volume=10.0,
        trade_count=5,
        features={},
    )
    second_bar = BarEvent(
        symbol="BTCUSDT",
        start_time=datetime(2025, 1, 1, 0, 0, 2, tzinfo=UTC),
        end_time=datetime(2025, 1, 1, 0, 0, 3, tzinfo=UTC),
        open_price=101.0,
        high_price=102.0,
        low_price=100.0,
        close_price=101.5,
        volume=10.0,
        trade_count=5,
        features={},
    )

    first_fills = simulator.process_bar(first_bar, current_index=1)
    second_fills = simulator.process_bar(second_bar, current_index=2)

    assert [fill.quantity for fill in first_fills] == [2.0]
    assert [fill.quantity for fill in second_fills] == [1.0]
    assert simulator.pending_orders == []


def test_limit_order_expires_when_not_touched(app_config) -> None:
    execution_config = replace(
        app_config.execution,
        max_order_lifetime_bars=2,
    )
    simulator = ExecutionSimulator(execution_config)
    order = OrderEvent(
        symbol="BTCUSDT",
        created_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        activation_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1.0,
        target_position=1,
        limit_price=95.0,
    )
    simulator.submit_orders([order], current_index=0)

    for index in (1, 2, 3):
        bar = BarEvent(
            symbol="BTCUSDT",
            start_time=datetime(2025, 1, 1, 0, 0, index, tzinfo=UTC),
            end_time=datetime(2025, 1, 1, 0, 0, index + 1, tzinfo=UTC),
            open_price=100.0,
            high_price=101.0,
            low_price=99.0,
            close_price=100.0,
            volume=10.0,
            trade_count=5,
            features={},
        )
        fills = simulator.process_bar(bar, current_index=index)
        assert fills == []

    assert simulator.pending_orders == []


def test_translator_builds_limit_order_with_offset(app_config) -> None:
    translator = SignalToOrderTranslator(
        position_size=1.0,
        allow_short=True,
        use_limit_orders=True,
        limit_offset_bps=5.0,
    )
    state = PortfolioState(
        event_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        cash=10_000.0,
        equity=10_000.0,
        positions={},
        gross_exposure=0.0,
        net_exposure=0.0,
    )
    signal = SignalEvent(
        symbol="BTCUSDT",
        event_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        side=SignalSide.LONG,
        target_position=1,
        reason="enter",
        context={},
    )

    orders = translator.from_signal(signal, state, reference_price=100.0)

    assert len(orders) == 1
    assert orders[0].order_type == OrderType.LIMIT
    assert orders[0].limit_price == 99.95
