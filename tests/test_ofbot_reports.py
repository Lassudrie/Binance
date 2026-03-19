from __future__ import annotations

from datetime import UTC, datetime

from ofbot.memory.reports import build_run_report
from ofbot.memory.store import MemoryStore


def test_build_run_report_is_scoped_to_run_id(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory" / "learning.duckdb")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    try:
        store.insert_event_journal(
            run_id="run_a",
            event_time=now,
            symbol="BTCUSDT",
            action="signal",
            strategy="continuation",
            params={"estimated_entry_cost_bps": 11.0, "expected_net_edge_bps": 5.5},
            order_intent="submit",
            reason="continuation_long",
        )
        store.insert_event_journal(
            run_id="run_b",
            event_time=now,
            symbol="ETHUSDT",
            action="signal",
            strategy="continuation",
            order_intent="submit",
        )
        store.insert_event_journal(
            run_id="run_a",
            event_time=now,
            symbol="BTCUSDT",
            action="signal",
            strategy="continuation",
            params={"expected_net_edge_bps": -1.0},
            order_intent="risk_expected_net_edge",
            reason="risk_gate",
        )
        store.insert_event_journal(
            run_id="run_a",
            event_time=now,
            symbol="BTCUSDT",
            action="signal",
            strategy="risk",
            order_intent="submit",
            reason="risk_stop_loss",
        )
        store.insert_trade_context(
            run_id="run_a",
            symbol="BTCUSDT",
            strategy="continuation",
            regime="low|tight|trend_up|balanced",
            context={"source": "test"},
            entry_time=now,
            exit_time=now,
            direction=1,
            qty=1.0,
            entry_price=100.0,
            exit_price=100.2,
            realized_pnl=1.25,
            gross_pnl=1.50,
            fees=0.25,
            mae_bps=0.0,
            mfe_bps=5.0,
            spread_entry_bps=2.0,
            spread_exit_bps=2.0,
            holding_seconds=4.0,
        )
        store.insert_trade_context(
            run_id="run_b",
            symbol="ETHUSDT",
            strategy="continuation",
            regime="low|tight|trend_up|balanced",
            context={"source": "test"},
            entry_time=now,
            exit_time=now,
            direction=1,
            qty=1.0,
            entry_price=200.0,
            exit_price=199.8,
            realized_pnl=-0.75,
            gross_pnl=-0.50,
            fees=0.25,
            mae_bps=-5.0,
            mfe_bps=0.0,
            spread_entry_bps=2.0,
            spread_exit_bps=2.0,
            holding_seconds=3.0,
        )

        report_path = build_run_report(store, "run_a", tmp_path / "reports")
    finally:
        store.close()

    text = report_path.read_text()
    assert "Run ID: `run_a`" in text
    assert "Closed trades: `1`" in text
    assert "Net PnL: `1.250000`" in text
    assert "Mean expected net edge (bps): `5.500`" in text
    assert "Expected-edge blocked entries: `1`" in text
    assert "- risk_stop_loss: 1" in text
