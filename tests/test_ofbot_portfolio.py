from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ofbot.execution.fills import FillEvent
from ofbot.execution.portfolio import PaperPortfolio


def _fill(
    *,
    side: int,
    qty: float,
    price: float,
    fee: float,
    event_time: datetime,
    context: dict[str, float | int | str | None] | None = None,
) -> FillEvent:
    return FillEvent(
        symbol="BTCUSDT",
        event_time=event_time,
        side=side,
        fill_qty=qty,
        fill_price=price,
        fee=fee,
        slippage_bps=1.0,
        is_maker=False,
        notional=qty * price,
        strategy_id="continuation",
        context=context or {"regime": "low|tight|trend_up|high"},
        order_id=f"order-{side}-{qty}-{price}",
        spread_bps=2.0,
    )


def test_portfolio_aggregates_multi_fill_trade_accounting() -> None:
    portfolio = PaperPortfolio(initial_cash=10_000.0)
    base = datetime(2026, 1, 1, tzinfo=UTC)

    assert portfolio.apply_fill(_fill(side=1, qty=0.4, price=100.0, fee=0.04, event_time=base)) == []
    assert portfolio.apply_fill(
        _fill(side=1, qty=0.6, price=101.0, fee=0.06, event_time=base + timedelta(seconds=1))
    ) == []
    assert portfolio.apply_fill(
        _fill(side=-1, qty=0.5, price=102.0, fee=0.05, event_time=base + timedelta(seconds=2))
    ) == []

    trades = portfolio.apply_fill(
        _fill(side=-1, qty=0.5, price=103.0, fee=0.05, event_time=base + timedelta(seconds=3))
    )

    assert len(trades) == 1
    trade = trades[0]
    assert trade.qty == 1.0
    assert trade.closed_qty == 1.0
    assert trade.entry_fill_count == 2
    assert trade.exit_fill_count == 2
    assert trade.entry_notional == pytest.approx(100.6)
    assert trade.exit_notional == pytest.approx(102.5)
    assert trade.entry_price == pytest.approx(100.6)
    assert trade.exit_price == pytest.approx(102.5)
    assert trade.gross_pnl == pytest.approx(1.9)
    assert trade.entry_fees == pytest.approx(0.10)
    assert trade.exit_fees == pytest.approx(0.10)
    assert trade.fees == pytest.approx(0.20)
    assert trade.realized_pnl == pytest.approx(1.7)
    assert portfolio.realized_pnl["BTCUSDT"] == pytest.approx(1.7)


def test_portfolio_preserves_fill_context_when_opening_position() -> None:
    portfolio = PaperPortfolio(initial_cash=10_000.0)
    base = datetime(2026, 1, 1, tzinfo=UTC)

    portfolio.apply_fill(
        _fill(
            side=1,
            qty=1.0,
            price=100.0,
            fee=0.10,
            event_time=base,
            context={
                "regime": "low|tight|trend_up|high",
                "preferred_exit_order_type": "limit",
                "entry_tag": "maker_bias",
            },
        )
    )

    snapshot = portfolio.snapshot("BTCUSDT")
    assert snapshot.risk_context["preferred_exit_order_type"] == "limit"
    assert snapshot.risk_context["entry_tag"] == "maker_bias"
    assert snapshot.risk_context["strategy"] == "continuation"


def test_portfolio_replaces_context_with_new_fill_context_on_reversal() -> None:
    portfolio = PaperPortfolio(initial_cash=10_000.0)
    base = datetime(2026, 1, 1, tzinfo=UTC)

    portfolio.apply_fill(
        _fill(
            side=1,
            qty=1.0,
            price=100.0,
            fee=0.10,
            event_time=base,
            context={
                "regime": "low|tight|trend_up|high",
                "preferred_exit_order_type": "limit",
                "entry_tag": "long_entry",
            },
        )
    )

    trades = portfolio.apply_fill(
        _fill(
            side=-1,
            qty=2.0,
            price=101.0,
            fee=0.20,
            event_time=base + timedelta(seconds=5),
            context={
                "regime": "high|wide|trend_down|high",
                "preferred_exit_order_type": "market",
                "entry_tag": "short_reversal",
            },
        )
    )

    assert len(trades) == 1
    snapshot = portfolio.snapshot("BTCUSDT")
    assert snapshot.net_position == pytest.approx(-1.0)
    assert snapshot.risk_context["preferred_exit_order_type"] == "market"
    assert snapshot.risk_context["entry_tag"] == "short_reversal"
    assert snapshot.risk_context["regime"] == "high|wide|trend_down|high"
