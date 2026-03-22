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


def test_engine_live_runtime_uses_received_time_for_freshness(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")

    engine = PaperEngine(config=config, depth_snapshot_client=None, runtime_clock="wall")
    try:
        market_time = datetime(2026, 1, 1, tzinfo=UTC)
        received_at = market_time + timedelta(seconds=30)
        assert engine._freshness_timestamp(market_time, now=received_at) == received_at
    finally:
        engine.close()


def test_engine_event_runtime_preserves_market_time_for_freshness(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")

    engine = PaperEngine(config=config, depth_snapshot_client=None, runtime_clock="event")
    try:
        market_time = datetime(2026, 1, 1, tzinfo=UTC)
        received_at = market_time + timedelta(seconds=30)
        assert engine._freshness_timestamp(market_time, now=received_at) == market_time
    finally:
        engine.close()


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


def test_engine_entry_context_applies_expected_move_discount(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.strategy_library.continuation.params.update(
        {
            "min_expected_net_edge_bps": 2.0,
            "expected_move_discount": 0.5,
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

    assert reason == "risk_expected_net_edge"
    assert math.isclose(float(context["raw_expected_move_proxy_bps"]), 22.838)
    assert math.isclose(float(context["expected_move_discount"]), 0.5)
    assert math.isclose(float(context["expected_move_proxy_bps"]), 11.419)
    assert float(context["expected_net_edge_bps"]) < 2.0


def test_engine_drops_pending_entry_when_signal_invalidates_before_fill(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.execution.allow_limit_orders = True
    config.execution.limit_order_book_depth_bps = 1.0
    config.execution.latency_ms = 0
    config.execution.max_order_lifetime_s = 30
    config.strategy_library.continuation.params.update(
        {
            "order_type": "limit",
            "revalidate_on_fill": True,
            "require_trend_up_regime": True,
            "queue_imbalance_threshold": 0.99,
            "microprice_drift_threshold_bps": 0.2,
            "no_trade_z": 1.2,
        }
    )

    engine = PaperEngine(config=config, depth_snapshot_client=None, runtime_clock="event")
    try:
        event_time = datetime(2026, 1, 1, tzinfo=UTC)
        engine.order_books["BTCUSDT"].apply_book_ticker(
            BookTickerEvent(
                symbol="BTCUSDT",
                event_time=event_time,
                event_type="bookTicker",
                raw={},
                bid_price=99.99,
                ask_price=100.01,
                bid_qty=20.0,
                ask_qty=20.0,
            )
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
        submit_snapshot = FeatureSnapshot(
            symbol="BTCUSDT",
            event_time=event_time,
            features={
                "realized_volatility": 10.0,
                "cvd_base_1s": 2.0,
                "cvd_base_1s_z": 2.0,
                "queue_imbalance": 0.995,
                "microprice_drift_bps_1s": 0.35,
                "microprice_drift_bps": 0.35,
                "momentum_bps_1s": 1.0,
                "short_return_bps_1s": 1.0,
                "momentum_bps_5s": 1.0,
                "short_return_bps_5s": 1.0,
            },
            regime=RegimeLabel(volatility="low", spread="low", trend="trend_up", flow="high"),
            spread_bps=0.05,
            mid_price=100.0,
            microprice=100.0,
        )

        decision = engine.strategies["continuation"].decide(submit_snapshot, state, long_only=True)
        assert decision is not None

        engine._submit_decision(
            decision,
            submit_snapshot,
            state,
            active_strategy="continuation",
            decision_time=event_time,
        )

        assert engine.broker.pending_order_count == 1

        invalid_snapshot = FeatureSnapshot(
            symbol="BTCUSDT",
            event_time=event_time + timedelta(seconds=1),
            features={
                "realized_volatility": 10.0,
                "cvd_base_1s": 0.0,
                "cvd_base_1s_z": 0.1,
                "queue_imbalance": 0.1,
                "microprice_drift_bps_1s": -0.1,
                "microprice_drift_bps": -0.1,
                "momentum_bps_1s": -0.1,
                "short_return_bps_1s": -0.1,
                "momentum_bps_5s": -0.1,
                "short_return_bps_5s": -0.1,
            },
            regime=RegimeLabel(volatility="low", spread="low", trend="range", flow="low"),
            spread_bps=0.05,
            mid_price=100.0,
            microprice=100.0,
        )

        engine._drop_invalidated_pending_entries(
            snapshot=invalid_snapshot,
            state=state,
            processing_time=invalid_snapshot.event_time,
        )
    finally:
        engine.close()

    assert engine.broker.pending_order_count == 0
    assert engine.broker.lifecycle_reason_counts["signal_invalidated"] == 1


def test_engine_keeps_pending_entry_during_revalidation_grace_period(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.execution.allow_limit_orders = True
    config.execution.limit_order_book_depth_bps = 1.0
    config.execution.latency_ms = 0
    config.execution.max_order_lifetime_s = 30
    config.strategy_library.continuation.params.update(
        {
            "order_type": "limit",
            "revalidate_on_fill": True,
            "revalidate_grace_period_s": 2.0,
            "require_trend_up_regime": True,
            "queue_imbalance_threshold": 0.99,
            "microprice_drift_threshold_bps": 0.2,
            "no_trade_z": 1.2,
        }
    )

    engine = PaperEngine(config=config, depth_snapshot_client=None, runtime_clock="event")
    try:
        event_time = datetime(2026, 1, 1, tzinfo=UTC)
        engine.order_books["BTCUSDT"].apply_book_ticker(
            BookTickerEvent(
                symbol="BTCUSDT",
                event_time=event_time,
                event_type="bookTicker",
                raw={},
                bid_price=99.99,
                ask_price=100.01,
                bid_qty=20.0,
                ask_qty=20.0,
            )
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
        submit_snapshot = FeatureSnapshot(
            symbol="BTCUSDT",
            event_time=event_time,
            features={
                "realized_volatility": 10.0,
                "cvd_base_1s": 2.0,
                "cvd_base_1s_z": 2.0,
                "queue_imbalance": 0.995,
                "microprice_drift_bps_1s": 0.35,
                "microprice_drift_bps": 0.35,
                "momentum_bps_1s": 1.0,
                "short_return_bps_1s": 1.0,
                "momentum_bps_5s": 1.0,
                "short_return_bps_5s": 1.0,
            },
            regime=RegimeLabel(volatility="low", spread="low", trend="trend_up", flow="high"),
            spread_bps=0.05,
            mid_price=100.0,
            microprice=100.0,
        )

        decision = engine.strategies["continuation"].decide(submit_snapshot, state, long_only=True)
        assert decision is not None

        engine._submit_decision(
            decision,
            submit_snapshot,
            state,
            active_strategy="continuation",
            decision_time=event_time,
        )

        invalid_snapshot = FeatureSnapshot(
            symbol="BTCUSDT",
            event_time=event_time + timedelta(milliseconds=500),
            features={
                "realized_volatility": 10.0,
                "cvd_base_1s": 0.0,
                "cvd_base_1s_z": 0.1,
                "queue_imbalance": 0.1,
                "microprice_drift_bps_1s": -0.1,
                "microprice_drift_bps": -0.1,
                "momentum_bps_1s": -0.1,
                "short_return_bps_1s": -0.1,
                "momentum_bps_5s": -0.1,
                "short_return_bps_5s": -0.1,
            },
            regime=RegimeLabel(volatility="low", spread="low", trend="range", flow="low"),
            spread_bps=0.05,
            mid_price=100.0,
            microprice=100.0,
        )

        engine._drop_invalidated_pending_entries(
            snapshot=invalid_snapshot,
            state=state,
            processing_time=invalid_snapshot.event_time,
        )
    finally:
        engine.close()

    assert engine.broker.pending_order_count == 1


def test_engine_keeps_pending_entry_on_neutral_signal_when_configured(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.execution.allow_limit_orders = True
    config.execution.limit_order_book_depth_bps = 1.0
    config.execution.latency_ms = 0
    config.execution.max_order_lifetime_s = 30
    config.strategy_library.continuation.params.update(
        {
            "order_type": "limit",
            "revalidate_on_fill": True,
            "revalidate_grace_period_s": 0.0,
            "revalidate_drop_on_neutral": False,
            "require_trend_up_regime": True,
            "queue_imbalance_threshold": 0.99,
            "microprice_drift_threshold_bps": 0.2,
            "no_trade_z": 1.2,
        }
    )

    engine = PaperEngine(config=config, depth_snapshot_client=None, runtime_clock="event")
    try:
        event_time = datetime(2026, 1, 1, tzinfo=UTC)
        engine.order_books["BTCUSDT"].apply_book_ticker(
            BookTickerEvent(
                symbol="BTCUSDT",
                event_time=event_time,
                event_type="bookTicker",
                raw={},
                bid_price=99.99,
                ask_price=100.01,
                bid_qty=20.0,
                ask_qty=20.0,
            )
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
        submit_snapshot = FeatureSnapshot(
            symbol="BTCUSDT",
            event_time=event_time,
            features={
                "realized_volatility": 10.0,
                "cvd_base_1s": 2.0,
                "cvd_base_1s_z": 2.0,
                "queue_imbalance": 0.995,
                "microprice_drift_bps_1s": 0.35,
                "microprice_drift_bps": 0.35,
                "momentum_bps_1s": 1.0,
                "short_return_bps_1s": 1.0,
                "momentum_bps_5s": 1.0,
                "short_return_bps_5s": 1.0,
            },
            regime=RegimeLabel(volatility="low", spread="low", trend="trend_up", flow="high"),
            spread_bps=0.05,
            mid_price=100.0,
            microprice=100.0,
        )

        decision = engine.strategies["continuation"].decide(submit_snapshot, state, long_only=True)
        assert decision is not None

        engine._submit_decision(
            decision,
            submit_snapshot,
            state,
            active_strategy="continuation",
            decision_time=event_time,
        )

        neutral_snapshot = FeatureSnapshot(
            symbol="BTCUSDT",
            event_time=event_time + timedelta(seconds=1),
            features={
                "realized_volatility": 10.0,
                "cvd_base_1s": 0.0,
                "cvd_base_1s_z": 0.1,
                "queue_imbalance": 0.2,
                "microprice_drift_bps_1s": 0.0,
                "microprice_drift_bps": 0.0,
                "momentum_bps_1s": 0.0,
                "short_return_bps_1s": 0.0,
                "momentum_bps_5s": 0.0,
                "short_return_bps_5s": 0.0,
            },
            regime=RegimeLabel(volatility="low", spread="low", trend="trend_up", flow="neutral"),
            spread_bps=0.05,
            mid_price=100.0,
            microprice=100.0,
        )

        engine._drop_invalidated_pending_entries(
            snapshot=neutral_snapshot,
            state=state,
            processing_time=neutral_snapshot.event_time,
        )
    finally:
        engine.close()

    assert engine.broker.pending_order_count == 1


def test_engine_anchors_limit_reference_to_touch_side_quote(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.execution.allow_limit_orders = True
    config.execution.limit_order_book_depth_bps = 0.0

    engine = PaperEngine(config=config, depth_snapshot_client=None, runtime_clock="event")
    try:
        event_time = datetime(2026, 1, 1, tzinfo=UTC)
        engine.order_books["BTCUSDT"].apply_book_ticker(
            BookTickerEvent(
                symbol="BTCUSDT",
                event_time=event_time,
                event_type="bookTicker",
                raw={},
                bid_price=99.99,
                ask_price=100.01,
                bid_qty=20.0,
                ask_qty=20.0,
            )
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
        snapshot = FeatureSnapshot(
            symbol="BTCUSDT",
            event_time=event_time,
            features={},
            regime=RegimeLabel(volatility="low", spread="low", trend="trend_up", flow="high"),
            spread_bps=2.0,
            mid_price=100.0,
            microprice=100.0,
        )
        decision = StrategyDecision(
            symbol="BTCUSDT",
            side=1,
            target_qty=0.5,
            reason="limit_anchor",
            order_type="limit",
            entry_context={},
        )

        engine._submit_decision(
            decision,
            snapshot,
            state,
            active_strategy="continuation",
            decision_time=event_time,
        )
        pending = engine.broker.pending_orders("BTCUSDT")
        assert len(pending) == 1
        assert abs(pending[0].reference_price - 100.01) < 1e-12
    finally:
        engine.close()


def test_engine_limit_entry_context_uses_lower_cost_than_market(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")

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
                "microprice_drift_bps_15s": 7.0,
                "short_return_bps_15s": 7.0,
                "microprice_drift_bps_30s": 8.0,
                "short_return_bps_30s": 8.0,
                "cvd_base_1s_z": 1.8,
                "queue_imbalance": 0.996,
            },
            regime=RegimeLabel(volatility="low", spread="tight", trend="trend_up", flow="high"),
            spread_bps=1.0,
            mid_price=100.0,
            microprice=100.0,
        )
        market_decision = StrategyDecision(
            symbol="BTCUSDT",
            side=1,
            target_qty=1.0,
            reason="continuation_long",
            confidence=0.8,
            order_type="market",
        )
        limit_decision = StrategyDecision(
            symbol="BTCUSDT",
            side=1,
            target_qty=1.0,
            reason="continuation_long",
            confidence=0.8,
            order_type="limit",
        )

        market_context, _ = engine._prepare_entry_context(
            snapshot=snapshot,
            decision=market_decision,
            strategy_name="continuation",
        )
        limit_context, _ = engine._prepare_entry_context(
            snapshot=snapshot,
            decision=limit_decision,
            strategy_name="continuation",
        )
    finally:
        engine.close()

    assert float(limit_context["roundtrip_cost_est_bps"]) < float(market_context["roundtrip_cost_est_bps"])
    assert limit_context["entry_order_type"] == "limit"
    assert limit_context["assumed_exit_order_type"] == "market"
    assert float(limit_context["estimated_exit_cost_bps"]) > float(limit_context["estimated_entry_cost_bps"])
    assert market_context["entry_order_type"] == "market"


def test_engine_blocks_entry_when_edge_cost_ratio_is_too_small(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.strategy_library.continuation.params.update(
        {
            "min_expected_net_edge_bps": 0.5,
            "min_edge_cost_ratio": 2.0,
            "expected_move_discount": 1.0,
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
                "microprice_drift_bps_15s": 7.0,
                "short_return_bps_15s": 7.0,
                "microprice_drift_bps_30s": 8.0,
                "short_return_bps_30s": 8.0,
                "cvd_base_1s_z": 1.8,
                "queue_imbalance": 0.996,
            },
            regime=RegimeLabel(volatility="low", spread="tight", trend="trend_up", flow="high"),
            spread_bps=1.0,
            mid_price=100.0,
            microprice=100.0,
        )
        decision = StrategyDecision(
            symbol="BTCUSDT",
            side=1,
            target_qty=1.0,
            reason="continuation_long",
            confidence=0.8,
            order_type="limit",
        )
        context, reason = engine._prepare_entry_context(
            snapshot=snapshot,
            decision=decision,
            strategy_name="continuation",
        )
    finally:
        engine.close()

    assert reason == "risk_expected_net_edge"
    assert float(context["expected_net_edge_bps"]) >= 0.5
    assert float(context["edge_cost_ratio"]) < 2.0


def test_engine_blocks_entry_when_fee_coverage_ratio_is_too_small(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.execution.taker_fee_bps = 8.0
    config.execution.maker_fee_bps = 2.0
    config.strategy_library.continuation.params.update(
        {
            "min_expected_net_edge_bps": -100.0,
            "min_edge_cost_ratio": 0.0,
            "min_fee_coverage_ratio": 1.5,
            "expected_move_discount": 1.0,
            "edge_fast_horizon_s": 15,
            "edge_slow_horizon_s": 30,
            "edge_fast_weight": 0.6,
            "edge_slow_weight": 0.4,
            "edge_cvd_bonus_bps": 1.5,
            "edge_queue_bonus_bps": 4.0,
            "edge_bonus_cap_bps": 12.0,
            "no_trade_z": 1.2,
            "queue_imbalance_threshold": 0.99,
            "order_type": "limit",
            "exit_cost_order_type": "market",
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
                "microprice_drift_bps_15s": 10.0,
                "short_return_bps_15s": 10.0,
                "microprice_drift_bps_30s": 9.0,
                "short_return_bps_30s": 9.0,
                "cvd_base_1s_z": 1.8,
                "queue_imbalance": 0.996,
            },
            regime=RegimeLabel(volatility="low", spread="tight", trend="trend_up", flow="high"),
            spread_bps=1.0,
            mid_price=100.0,
            microprice=100.0,
        )
        decision = StrategyDecision(
            symbol="BTCUSDT",
            side=1,
            target_qty=1.0,
            reason="continuation_long",
            confidence=0.8,
            order_type="limit",
        )
        context, reason = engine._prepare_entry_context(
            snapshot=snapshot,
            decision=decision,
            strategy_name="continuation",
        )
    finally:
        engine.close()

    assert reason == "risk_expected_net_edge"
    assert float(context["fee_coverage_ratio"]) < 1.5
    assert float(context["roundtrip_fee_bps"]) == 10.0


def test_engine_applies_strong_signal_edge_overrides_for_high_conviction_trend_up(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.strategy_library.continuation.params.update(
        {
            "min_expected_net_edge_bps": 3.0,
            "min_edge_cost_ratio": 1.5,
            "expected_move_discount": 0.65,
            "strong_signal_expected_move_discount": 0.85,
            "strong_signal_min_edge_cost_ratio": 1.25,
            "strong_signal_min_momentum_5s_bps": 10.0,
            "strong_signal_min_cvd_z": 3.0,
            "edge_fast_horizon_s": 15,
            "edge_slow_horizon_s": 30,
            "edge_fast_weight": 0.6,
            "edge_slow_weight": 0.4,
            "edge_cvd_bonus_bps": 1.5,
            "edge_queue_bonus_bps": 4.0,
            "edge_bonus_cap_bps": 12.0,
            "no_trade_z": 1.35,
            "queue_imbalance_threshold": 0.99,
            "order_type": "limit",
            "exit_cost_order_type": "market",
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
                "microprice_drift_bps_1s": 7.8,
                "momentum_bps_1s": 7.8,
                "momentum_bps_5s": 10.44,
                "short_return_bps_5s": 10.44,
                "microprice_drift_bps_15s": 7.8,
                "short_return_bps_15s": 7.8,
                "microprice_drift_bps_30s": 10.5,
                "short_return_bps_30s": 10.5,
                "cvd_base_1s_z": 4.0,
                "queue_imbalance": 0.9985,
            },
            regime=RegimeLabel(volatility="high", spread="high", trend="trend_up", flow="high"),
            spread_bps=0.05,
            mid_price=100.0,
            microprice=100.0,
        )
        decision = StrategyDecision(
            symbol="BTCUSDT",
            side=1,
            target_qty=1.0,
            reason="continuation_long",
            confidence=1.0,
            order_type="limit",
        )
        context, reason = engine._prepare_entry_context(
            snapshot=snapshot,
            decision=decision,
            strategy_name="continuation",
        )
    finally:
        engine.close()

    assert reason is None
    assert int(context["strong_signal_applied"] or 0) == 1
    assert math.isclose(float(context["expected_move_discount"]), 0.85)
    assert float(context["expected_net_edge_bps"]) >= 3.0
    assert float(context["edge_cost_ratio"]) >= 1.25
    assert float(context["raw_expected_move_proxy_bps"]) * 0.65 - float(context["roundtrip_cost_est_bps"]) < 3.0


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


def test_engine_finalize_skips_learning_cycle_when_learning_disabled(tmp_path, monkeypatch) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.learning.enabled = False
    config.housekeeping.enabled = False

    def _unexpected_learning_cycle(*args, **kwargs):
        raise AssertionError("advance_learning_cycle should not run when learning is disabled")

    monkeypatch.setattr("ofbot.engine.learning_mod.advance_learning_cycle", _unexpected_learning_cycle)

    engine = PaperEngine(config=config, depth_snapshot_client=None, runtime_clock="event")
    try:
        report_path = engine.finalize()
    finally:
        if not engine._closed:
            engine.close()

    assert report_path is not None
    assert report_path.exists()
