from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging
import math

from ofbot.engine import PaperEngine
from ofbot.execution.fills import PaperOrder
from ofbot.execution.portfolio import PositionSnapshot
from ofbot.market.features import FeatureSnapshot
from ofbot.market.regimes import RegimeLabel
from ofbot.market.trades import BookTickerEvent
from ofbot.strategy.base import StrategyDecision
from ofbot.utils.manual_orders import write_manual_order_request
from tests.ofbot_helpers import build_test_config


def test_engine_pre_entry_gate_uses_projected_notional(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.risk.max_notional = 50.0

    engine = PaperEngine(config=config, depth_snapshot_client=None, runtime_clock="event")
    try:
        event_time = datetime(2026, 1, 1, tzinfo=UTC)
        engine._last_ws_event_time["BTCUSDT"] = event_time
        snapshot = FeatureSnapshot(
            symbol="BTCUSDT",
            event_time=event_time,
            features={"realized_volatility": 10.0},
            regime=RegimeLabel(volatility="low", spread="tight", trend="range", flow="neutral"),
            spread_bps=2.0,
            mid_price=100.0,
            microprice=100.0,
        )
        state = PositionSnapshot(
            symbol="BTCUSDT",
            net_position=0.0,
            avg_entry_price=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            fees_paid=0.0,
            last_price=None,
            position_open_time=None,
            mae_bps=0.0,
            mfe_bps=0.0,
            risk_context={},
        )

        reason = engine._risk_pre_entry_context(snapshot, state, projected_qty=1.0, now=event_time)
    finally:
        engine.close()

    assert reason == "risk_notional_limit"


def test_engine_blocks_entry_when_expected_net_edge_is_too_small(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.strategy_library.continuation.params["min_expected_net_edge_bps"] = 3.0

    engine = PaperEngine(config=config, depth_snapshot_client=None, runtime_clock="event")
    try:
        engine.order_books["BTCUSDT"].apply_book_ticker(
            BookTickerEvent(
                symbol="BTCUSDT",
                event_time=datetime(2026, 1, 1, tzinfo=UTC),
                event_type="bookTicker",
                raw={},
                bid_price=99.995,
                ask_price=100.005,
                bid_qty=20.0,
                ask_qty=20.0,
            )
        )
        snapshot = FeatureSnapshot(
            symbol="BTCUSDT",
            event_time=datetime(2026, 1, 1, tzinfo=UTC),
            features={
                "microprice_drift_bps_1s": 5.0,
                "momentum_bps_1s": 6.0,
                "short_return_bps_5s": 6.0,
                "microprice_drift_bps_15s": 4.0,
                "short_return_bps_15s": 4.0,
                "microprice_drift_bps_30s": 4.0,
                "short_return_bps_30s": 4.0,
                "cvd_base_1s_z": 1.4,
                "queue_imbalance": 0.995,
            },
            regime=RegimeLabel(volatility="low", spread="tight", trend="trend_up", flow="high"),
            spread_bps=2.0,
            mid_price=100.0,
            microprice=100.0,
        )
        decision = StrategyDecision(
            symbol="BTCUSDT",
            side=1,
            target_qty=1.0,
            reason="continuation_long",
            confidence=0.6,
        )

        context, reason = engine._prepare_entry_context(
            snapshot=snapshot,
            decision=decision,
            strategy_name="continuation",
        )
    finally:
        engine.close()

    assert reason == "risk_expected_net_edge"
    assert float(context["expected_net_edge_bps"] or 0.0) < 3.0


def test_engine_entry_context_uses_longer_horizon_edge_and_conviction_bonus(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.strategy_library.continuation.params.update(
        {
            "min_expected_net_edge_bps": 2.0,
            "edge_fast_horizon_s": 15,
            "edge_slow_horizon_s": 30,
            "edge_fast_weight": 0.6,
            "edge_slow_weight": 0.4,
            "edge_cvd_bonus_bps": 1.5,
            "edge_queue_bonus_bps": 4.0,
            "edge_bonus_cap_bps": 12.0,
            "no_trade_z": 1.2,
            "queue_imbalance_threshold": 0.99,
        }
    )

    engine = PaperEngine(config=config, depth_snapshot_client=None, runtime_clock="event")
    try:
        engine.order_books["BTCUSDT"].apply_book_ticker(
            BookTickerEvent(
                symbol="BTCUSDT",
                event_time=datetime(2026, 1, 1, tzinfo=UTC),
                event_type="bookTicker",
                raw={},
                bid_price=99.995,
                ask_price=100.005,
                bid_qty=20.0,
                ask_qty=20.0,
            )
        )
        snapshot = FeatureSnapshot(
            symbol="BTCUSDT",
            event_time=datetime(2026, 1, 1, tzinfo=UTC),
                features={
                    "microprice_drift_bps_1s": 1.0,
                    "momentum_bps_1s": 1.2,
                    "microprice_drift_bps_15s": 15.0,
                    "short_return_bps_15s": 14.0,
                    "microprice_drift_bps_30s": 24.0,
                    "short_return_bps_30s": 22.0,
                    "cvd_base_1s_z": 4.0,
                    "queue_imbalance": 0.9995,
                },
            regime=RegimeLabel(volatility="low", spread="tight", trend="trend_up", flow="high"),
            spread_bps=0.5,
            mid_price=100.0,
            microprice=100.0,
        )
        decision = StrategyDecision(
            symbol="BTCUSDT",
            side=1,
            target_qty=1.0,
            reason="continuation_long",
            confidence=0.8,
        )

        context, reason = engine._prepare_entry_context(
            snapshot=snapshot,
            decision=decision,
            strategy_name="continuation",
        )
    finally:
        engine.close()

    assert reason is None
    assert float(context["expected_fast_component_bps"]) == 15.0
    assert float(context["expected_slow_component_bps"]) == 24.0
    assert math.isclose(float(context["expected_trend_component_bps"]), 18.6)
    assert math.isclose(float(context["expected_cvd_bonus_bps"]), 4.2)
    assert float(context["expected_queue_bonus_bps"]) > 0.0
    assert float(context["expected_conviction_bonus_bps"]) > 4.2
    assert float(context["expected_net_edge_bps"]) >= 2.0


def test_engine_resolves_entry_sizing_from_notional_bounds(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.strategy_library.continuation.params["min_expected_net_edge_bps"] = 3.0
    config.execution.default_target_notional_usd = 500.0
    config.execution.min_target_notional_usd = 250.0
    config.execution.max_target_notional_usd = 1000.0

    engine = PaperEngine(config=config, depth_snapshot_client=None, runtime_clock="event")
    try:
        engine.order_books["BTCUSDT"].apply_book_ticker(
            BookTickerEvent(
                symbol="BTCUSDT",
                event_time=datetime(2026, 1, 1, tzinfo=UTC),
                event_type="bookTicker",
                raw={},
                bid_price=1_999.95,
                ask_price=2_000.05,
                bid_qty=20.0,
                ask_qty=20.0,
            )
        )
        snapshot = FeatureSnapshot(
            symbol="BTCUSDT",
            event_time=datetime(2026, 1, 1, tzinfo=UTC),
            features={
                "microprice_drift_bps_15s": 55.0,
                "short_return_bps_15s": 70.0,
                "microprice_drift_bps_30s": 60.0,
                "short_return_bps_30s": 65.0,
                "cvd_base_1s_z": 2.5,
                "queue_imbalance": 0.999,
            },
            regime=RegimeLabel(volatility="low", spread="tight", trend="trend_up", flow="high"),
            spread_bps=0.5,
            mid_price=2_000.0,
            microprice=2_000.0,
        )
        decision = StrategyDecision(
            symbol="BTCUSDT",
            side=1,
            target_qty=1.0,
            reason="continuation_long",
            confidence=0.9,
        )

        context, reason = engine._prepare_entry_context(
            snapshot=snapshot,
            decision=decision,
            strategy_name="continuation",
        )
    finally:
        engine.close()

    assert reason is None
    assert 250.0 <= float(context["desired_notional_usd"] or 0.0) <= 1_000.0
    assert float(context["resolved_target_qty"] or 0.0) == float(context["desired_notional_usd"] or 0.0) / 2_000.0


def test_engine_live_telemetry_snapshot_reports_runtime_state(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")

    engine = PaperEngine(config=config, depth_snapshot_client=None, runtime_clock="event")
    try:
        event_time = datetime(2026, 1, 1, tzinfo=UTC)
        engine._live_counters.update(
            {
                "processed_events": 42,
                "submits": 3,
                "fills": 2,
                "trades_closed": 1,
                "risk_gates": 5,
            }
        )
        engine._event_type_counts.update({"aggTrade": 20, "depth": 12, "bookTicker": 10})
        engine._risk_gate_counts.update({"risk_expected_net_edge": 4, "risk_notional_limit": 1})
        engine._submit_counts_by_strategy.update({"continuation": 3})
        engine._submit_counts_by_symbol.update({"BTCUSDT": 3})
        engine.portfolio.positions["BTCUSDT"] = 0.25
        engine.portfolio.avg_price["BTCUSDT"] = 100.0
        engine.portfolio.last_price["BTCUSDT"] = 101.0
        engine.portfolio.unrealized_pnl["BTCUSDT"] = 0.25
        engine.portfolio.realized_pnl["BTCUSDT"] = 1.5
        engine.portfolio.fees_paid["BTCUSDT"] = 0.4
        engine.portfolio.cash = 9_900.0
        engine.broker._pending["BTCUSDT"] = [
            PaperOrder(
                order_id="pending-1",
                symbol="BTCUSDT",
                event_time=event_time,
                side=1,
                order_type="market",
                requested_qty=0.1,
                remaining_qty=0.1,
                reference_price=100.0,
                target_qty=0.35,
                context={},
            )
        ]
        engine._last_submit = {"symbol": "BTCUSDT", "reason": "continuation_long"}
        engine._last_fill = {"symbol": "BTCUSDT", "qty": 0.1}
        engine._last_trade = {"symbol": "BTCUSDT", "realized_pnl": 1.5}

        snapshot = engine.get_live_telemetry_snapshot(emitted_at=event_time)
    finally:
        engine.close()

    assert snapshot["processed_events"] == 42
    assert snapshot["submits"] == 3
    assert snapshot["fills"] == 2
    assert snapshot["trades_closed"] == 1
    assert snapshot["risk_gate_counts"][0] == ("risk_expected_net_edge", 4)
    assert snapshot["open_position_count"] == 1
    assert snapshot["pending_order_count"] == 1
    assert snapshot["pending_orders_by_symbol"] == {"BTCUSDT": 1}
    assert snapshot["open_positions"][0]["symbol"] == "BTCUSDT"
    assert snapshot["last_submit"] == {"symbol": "BTCUSDT", "reason": "continuation_long"}


def test_engine_fills_pending_market_order_on_unsynced_l1(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.execution.latency_ms = 0
    config.execution.max_order_lifetime_s = 5

    engine = PaperEngine(config=config, depth_snapshot_client=None, runtime_clock="event")
    try:
        event_time = datetime(2026, 1, 1, tzinfo=UTC)
        book = engine.order_books["BTCUSDT"]
        book.depth_mode_active = True
        book.sync_state = "unsynced"

        decision = StrategyDecision(
            symbol="BTCUSDT",
            side=1,
            target_qty=0.5,
            reason="pending_entry",
            confidence=1.0,
            order_type="market",
        )
        orders = engine.broker.queue_from_decision(
            decision=decision,
            symbol="BTCUSDT",
            current_qty=0.0,
            reference_price=100.0,
            event_time=event_time,
            strategy_id="test",
        )
        assert len(orders) == 1
        engine.broker.submit(orders[0])

        engine.process_event(
            BookTickerEvent(
                symbol="BTCUSDT",
                event_time=event_time + timedelta(seconds=1),
                event_type="bookTicker",
                raw={},
                bid_price=99.99,
                ask_price=100.01,
                bid_qty=20.0,
                ask_qty=20.0,
            )
        )
    finally:
        engine.close()

    assert engine._live_counters["fills"] == 1
    assert engine.broker.pending_order_count == 0
    assert engine.portfolio.positions["BTCUSDT"] == 0.5


def test_engine_consumes_manual_order_request_and_fills(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.execution.latency_ms = 0
    config.execution.max_order_lifetime_s = 5

    engine = PaperEngine(config=config, depth_snapshot_client=None, runtime_clock="event")
    try:
        write_manual_order_request(
            config.paths.memory_dir,
            symbol="BTCUSDT",
            side=1,
            qty=0.4,
            order_type="market",
            reason="manual_test_fill",
            force=False,
        )
        engine.process_event(
            BookTickerEvent(
                symbol="BTCUSDT",
                event_time=datetime(2026, 1, 1, tzinfo=UTC),
                event_type="bookTicker",
                raw={},
                bid_price=99.99,
                ask_price=100.01,
                bid_qty=20.0,
                ask_qty=20.0,
            )
        )
    finally:
        engine.close()

    assert engine._live_counters["submits"] == 1
    assert engine._live_counters["fills"] == 1
    assert engine.portfolio.positions["BTCUSDT"] == 0.4
    assert engine._last_submit is not None
    assert engine._last_submit["strategy"] == "manual"


def test_engine_emit_live_telemetry_logs_summary(tmp_path, caplog) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")

    engine = PaperEngine(config=config, depth_snapshot_client=None)
    try:
        engine._live_counters.update({"processed_events": 7, "submits": 1, "fills": 1})
        engine._risk_gate_counts.update({"risk_expected_net_edge": 2})
        caplog.set_level(logging.INFO)

        snapshot = engine.emit_live_telemetry(force=True, emitted_at=datetime(2026, 1, 1, tzinfo=UTC))
    finally:
        engine.close()

    assert snapshot is not None
    assert snapshot["processed_events"] == 7
    assert "live_telemetry" in caplog.text
    assert "processed=7" in caplog.text
    assert "submits=1" in caplog.text
