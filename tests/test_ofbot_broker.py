from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ofbot.execution.paper_broker import PaperBroker
from ofbot.execution.portfolio import PaperPortfolio
from ofbot.market.orderbook import OrderBookState
from ofbot.market.trades import parse_book_ticker, parse_depth_snapshot
from ofbot.config import ExecutionConfig
from ofbot.strategy.base import StrategyDecision


def _book() -> OrderBookState:
    book = OrderBookState(symbol="BTCUSDT")
    book.apply_book_ticker(
        parse_book_ticker(
            {
                "s": "BTCUSDT",
                "E": 1,
                "b": "100.0",
                "B": "10.0",
                "a": "101.0",
                "A": "10.0",
            }
        )
    )
    return book


def _soft_exit_decision(*, order_type: str = "limit") -> StrategyDecision:
    return StrategyDecision(
        symbol="BTCUSDT",
        side=0,
        target_qty=0.0,
        reason="risk_max_holding",
        confidence=1.0,
        order_type=order_type,
        entry_context={
            "decision_reason": "risk_max_holding",
            "preferred_exit_order_type": order_type,
            "strategy": "risk",
        },
    )


def _depth_book() -> OrderBookState:
    book = OrderBookState(symbol="BTCUSDT", require_depth_sync=True, book_ticker_divergence_bps=25.0)
    book.apply_book_ticker(
        parse_book_ticker(
            {
                "s": "BTCUSDT",
                "E": 1,
                "b": "100.0",
                "B": "10.0",
                "a": "100.1",
                "A": "10.0",
            }
        )
    )
    book.apply_depth_snapshot(
        parse_depth_snapshot(
            {
                "s": "BTCUSDT",
                "lastUpdateId": 100,
                "bids": [["100.0", "10.0"], ["99.9", "8.0"]],
                "asks": [["100.1", "0.3"], ["100.2", "0.4"], ["100.3", "1.0"]],
            },
            fallback_event_time=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    return book


def test_market_broker_fill_updates_cash_and_position() -> None:
    broker = PaperBroker(
        ExecutionConfig(
            allow_limit_orders=True,
            max_order_lifetime_s=5,
            latency_ms=0,
            taker_fee_bps=10.0,
            fallback_half_spread_bps=1.0,
            impact_bps_per_unit_participation=5.0,
            market_participation_cap=1.0,
        )
    )
    portfolio = PaperPortfolio(initial_cash=10_000.0)
    event_time = datetime(2026, 1, 1, tzinfo=UTC)
    book = _book()

    decision = StrategyDecision(
        symbol="BTCUSDT",
        side=1,
        target_qty=1.0,
        reason="test_entry",
        confidence=1.0,
        order_type="market",
        max_holding_time_s=60,
    )
    orders = broker.queue_from_decision(
        decision=decision,
        symbol="BTCUSDT",
        current_qty=0.0,
        reference_price=100.5,
        event_time=event_time,
        strategy_id="test",
    )
    assert len(orders) == 1
    broker.submit(orders[0])

    fills = broker.process(symbol="BTCUSDT", now=event_time, book=book)
    assert len(fills) == 1
    fill = fills[0]
    assert fill.fill_qty == 1.0
    assert fill.side == 1

    trades = portfolio.apply_fill(fill)
    assert trades == []
    state = portfolio.snapshot("BTCUSDT", mark_time=event_time)
    assert state.net_position == 1.0
    assert state.avg_entry_price == fill.fill_price
    assert portfolio.cash < 10_000.0
    assert portfolio.fees_paid["BTCUSDT"] > 0.0


def test_paper_broker_avoids_duplicate_orders() -> None:
    broker = PaperBroker(
        ExecutionConfig(
            allow_limit_orders=False,
            max_order_lifetime_s=5,
            latency_ms=0,
            taker_fee_bps=8.0,
            fallback_half_spread_bps=1.0,
            impact_bps_per_unit_participation=10.0,
            market_participation_cap=1.0,
        )
    )
    event_time = datetime(2026, 1, 1, tzinfo=UTC)
    decision = StrategyDecision(
        symbol="BTCUSDT",
        side=1,
        target_qty=1.0,
        reason="test_entry",
        order_type="market",
    )
    orders = broker.queue_from_decision(
        decision=decision,
        symbol="BTCUSDT",
        current_qty=0.0,
        reference_price=100.0,
        event_time=event_time,
        strategy_id="dup",
    )
    assert len(orders) == 1
    orders_duplicate = broker.queue_from_decision(
        decision=decision,
        symbol="BTCUSDT",
        current_qty=0.0,
        reference_price=100.0,
        event_time=event_time,
        strategy_id="dup",
    )
    assert orders_duplicate == []


def test_partial_fill_cap() -> None:
    broker = PaperBroker(
        ExecutionConfig(
            allow_limit_orders=True,
            max_order_lifetime_s=5,
            latency_ms=0,
            taker_fee_bps=5.0,
            maker_fee_bps=1.0,
            fallback_half_spread_bps=1.0,
            impact_bps_per_unit_participation=10.0,
            market_participation_cap=0.2,
        )
    )
    book = _book()
    book.ask_qty = 2.0
    event_time = datetime(2026, 1, 1, tzinfo=UTC)
    decision = StrategyDecision(
        symbol="BTCUSDT",
        side=1,
        target_qty=1.0,
        reason="part",
        order_type="limit",
    )
    orders = broker.queue_from_decision(
        decision=decision,
        symbol="BTCUSDT",
        current_qty=0.0,
        reference_price=101.1,
        event_time=event_time,
        strategy_id="partial",
    )
    assert len(orders) == 1
    order = orders[0]
    broker.submit(order)

    fills_step_one = broker.process(symbol="BTCUSDT", now=event_time, book=book)
    fills_step_two = broker.process(symbol="BTCUSDT", now=event_time, book=book)
    fills_step_three = broker.process(symbol="BTCUSDT", now=event_time, book=book)

    filled_qty = sum(fill.fill_qty for fill in fills_step_one + fills_step_two + fills_step_three)
    assert abs(filled_qty - 1.0) < 1e-12
    assert len(fills_step_three) > 0


def test_market_order_sweeps_visible_depth_immediately() -> None:
    broker = PaperBroker(
        ExecutionConfig(
            allow_limit_orders=True,
            max_order_lifetime_s=5,
            latency_ms=0,
            taker_fee_bps=8.0,
            fallback_half_spread_bps=1.0,
            impact_bps_per_unit_participation=8.0,
            market_participation_cap=0.1,
            market_sweep_depth_bps=25.0,
        )
    )
    event_time = datetime(2026, 1, 1, tzinfo=UTC)
    book = _depth_book()
    decision = StrategyDecision(
        symbol="BTCUSDT",
        side=1,
        target_qty=0.6,
        reason="sweep",
        order_type="market",
    )

    orders = broker.queue_from_decision(
        decision=decision,
        symbol="BTCUSDT",
        current_qty=0.0,
        reference_price=100.05,
        event_time=event_time,
        strategy_id="sweep",
    )
    assert len(orders) == 1
    broker.submit(orders[0])

    fills = broker.process(symbol="BTCUSDT", now=event_time, book=book)

    assert len(fills) == 1
    fill = fills[0]
    expected_price = ((0.3 * 100.1) + (0.3 * 100.2)) / 0.6
    assert fill.fill_qty == 0.6
    assert abs(fill.fill_price - expected_price) < 1e-12
    assert fill.is_maker is False


def test_broker_tracks_order_lifecycle_from_pending_to_fill() -> None:
    broker = PaperBroker(
        ExecutionConfig(
            allow_limit_orders=True,
            max_order_lifetime_s=5,
            latency_ms=250,
            taker_fee_bps=8.0,
            fallback_half_spread_bps=1.0,
            impact_bps_per_unit_participation=8.0,
            market_sweep_depth_bps=25.0,
        )
    )
    event_time = datetime(2026, 1, 1, tzinfo=UTC)
    book = _depth_book()
    decision = StrategyDecision(
        symbol="BTCUSDT",
        side=1,
        target_qty=0.4,
        reason="lifecycle",
        order_type="market",
    )

    orders = broker.queue_from_decision(
        decision=decision,
        symbol="BTCUSDT",
        current_qty=0.0,
        reference_price=100.05,
        event_time=event_time,
        strategy_id="lifecycle",
    )
    broker.submit(orders[0])

    fills_before_activation = broker.process(symbol="BTCUSDT", now=event_time, book=book)
    fills_after_activation = broker.process(
        symbol="BTCUSDT",
        now=event_time.replace(microsecond=300_000),
        book=book,
    )

    assert fills_before_activation == []
    assert len(fills_after_activation) == 1
    assert broker.pending_order_count == 0
    assert broker.lifecycle_stage_counts["submitted"] == 1
    assert broker.lifecycle_stage_counts["pending"] >= 1
    assert broker.lifecycle_stage_counts["filled"] == 1
    assert broker.lifecycle_reason_counts["activation_wait"] >= 1
    assert broker.last_lifecycle_event is not None
    assert broker.last_lifecycle_event["stage"] == "filled"


def test_broker_marks_expired_orders_with_reason() -> None:
    broker = PaperBroker(
        ExecutionConfig(
            allow_limit_orders=True,
            max_order_lifetime_s=1,
            latency_ms=0,
            taker_fee_bps=8.0,
            fallback_half_spread_bps=1.0,
            impact_bps_per_unit_participation=8.0,
        )
    )
    event_time = datetime(2026, 1, 1, tzinfo=UTC)
    decision = StrategyDecision(
        symbol="BTCUSDT",
        side=1,
        target_qty=0.4,
        reason="expire",
        order_type="market",
    )
    orders = broker.queue_from_decision(
        decision=decision,
        symbol="BTCUSDT",
        current_qty=0.0,
        reference_price=100.0,
        event_time=event_time,
        strategy_id="expire",
    )
    broker.submit(orders[0])

    fills = broker.process(
        symbol="BTCUSDT",
        now=event_time + timedelta(seconds=2),
        book=_depth_book(),
    )

    assert fills == []
    assert broker.pending_order_count == 0
    assert broker.lifecycle_stage_counts["dropped"] == 1
    assert broker.lifecycle_reason_counts["expired"] == 1
    assert broker.last_lifecycle_event is not None
    assert broker.last_lifecycle_event["reason"] == "expired"


def test_broker_can_drop_pending_orders_with_custom_reason() -> None:
    broker = PaperBroker(
        ExecutionConfig(
            allow_limit_orders=True,
            max_order_lifetime_s=5,
            latency_ms=250,
            taker_fee_bps=8.0,
            fallback_half_spread_bps=1.0,
            impact_bps_per_unit_participation=8.0,
        )
    )
    event_time = datetime(2026, 1, 1, tzinfo=UTC)
    decision = StrategyDecision(
        symbol="BTCUSDT",
        side=1,
        target_qty=0.4,
        reason="invalidate",
        order_type="market",
    )
    orders = broker.queue_from_decision(
        decision=decision,
        symbol="BTCUSDT",
        current_qty=0.0,
        reference_price=100.0,
        event_time=event_time,
        strategy_id="invalidate",
    )
    broker.submit(orders[0])

    dropped = broker.drop_pending_orders(
        symbol="BTCUSDT",
        now=event_time,
        reason="signal_invalidated",
    )

    assert dropped == 1
    assert broker.pending_order_count == 0
    assert broker.lifecycle_stage_counts["dropped"] == 1
    assert broker.lifecycle_reason_counts["signal_invalidated"] == 1
    assert broker.last_lifecycle_event is not None
    assert broker.last_lifecycle_event["reason"] == "signal_invalidated"


def test_broker_escalates_soft_exit_limit_to_market_after_ttl_and_blocks_duplicate_exit() -> None:
    broker = PaperBroker(
        ExecutionConfig(
            allow_limit_orders=True,
            max_order_lifetime_s=5,
            latency_ms=0,
            taker_fee_bps=8.0,
            maker_fee_bps=2.0,
            fallback_half_spread_bps=1.0,
            impact_bps_per_unit_participation=8.0,
            soft_exit_limit_ttl_s=1.0,
            soft_exit_limit_adverse_bps=0.0,
        )
    )
    event_time = datetime(2026, 1, 1, tzinfo=UTC)
    decision = _soft_exit_decision(order_type="limit")

    orders = broker.queue_from_decision(
        decision=decision,
        symbol="BTCUSDT",
        current_qty=1.0,
        reference_price=100.0,
        event_time=event_time,
        strategy_id="risk",
    )
    assert len(orders) == 1
    broker.submit(orders[0])

    fills = broker.process(symbol="BTCUSDT", now=event_time + timedelta(seconds=2), book=None)

    assert fills == []
    pending = broker.pending_orders("BTCUSDT")
    assert len(pending) == 1
    assert pending[0].order_type == "market"
    assert broker.lifecycle_reason_counts["soft_exit_limit_ttl_escalation"] == 1

    duplicate = broker.queue_from_decision(
        decision=decision,
        symbol="BTCUSDT",
        current_qty=1.0,
        reference_price=100.0,
        event_time=event_time + timedelta(seconds=2),
        strategy_id="risk",
    )
    assert duplicate == []


def test_broker_escalates_soft_exit_limit_to_market_on_adverse_move() -> None:
    broker = PaperBroker(
        ExecutionConfig(
            allow_limit_orders=True,
            max_order_lifetime_s=5,
            latency_ms=0,
            taker_fee_bps=8.0,
            maker_fee_bps=2.0,
            fallback_half_spread_bps=1.0,
            impact_bps_per_unit_participation=8.0,
            soft_exit_limit_ttl_s=0.0,
            soft_exit_limit_adverse_bps=50.0,
        )
    )
    event_time = datetime(2026, 1, 1, tzinfo=UTC)
    decision = _soft_exit_decision(order_type="limit")

    orders = broker.queue_from_decision(
        decision=decision,
        symbol="BTCUSDT",
        current_qty=1.0,
        reference_price=100.0,
        event_time=event_time,
        strategy_id="risk",
    )
    assert len(orders) == 1
    broker.submit(orders[0])

    book = _book()
    book.bid_price = 99.0
    book.ask_price = 100.0

    fills = broker.process(symbol="BTCUSDT", now=event_time + timedelta(seconds=1), book=book)

    assert len(fills) == 1
    assert fills[0].is_maker is False
    assert fills[0].side == -1
    assert broker.pending_order_count == 0
    assert broker.lifecycle_reason_counts["soft_exit_limit_adverse_escalation"] == 1
