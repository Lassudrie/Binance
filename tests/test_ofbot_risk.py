from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ofbot.execution.portfolio import PositionSnapshot
from ofbot.execution.risk import RiskManager
from ofbot.market.features import FeatureSnapshot
from ofbot.market.regimes import RegimeLabel


def _snapshot(at_time: datetime) -> FeatureSnapshot:
    return FeatureSnapshot(
        symbol="ETHUSDT",
        event_time=at_time,
        features={"realized_volatility": 12.0},
        regime=RegimeLabel(volatility="low", spread="low", trend="trend_up", flow="high"),
        spread_bps=0.1,
        mid_price=100.0,
        microprice=100.0,
    )


def _state(*, last_price: float, opened_at: datetime) -> PositionSnapshot:
    return PositionSnapshot(
        symbol="ETHUSDT",
        net_position=1.0,
        avg_entry_price=100.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        fees_paid=0.0,
        last_price=last_price,
        position_open_time=opened_at,
        mae_bps=0.0,
        mfe_bps=0.0,
        risk_context={},
    )


def test_risk_manager_waits_before_soft_exit() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    risk = RiskManager(
        stale_data_s=6.0,
        broken_ws_s=12.0,
        abnormal_spread_bps=250.0,
        abnormal_volatility_bps=900.0,
        max_position_size=8.0,
        max_notional=50_000.0,
        max_concurrent_exposure=2,
        cooldown_after_loss_s=300,
        daily_loss_limit=0.02,
        max_holding_time_s=900,
        catastrophic_stop_loss_bps=45.0,
        min_hold_before_soft_exit_s=20,
        stop_loss_bps=20.0,
        take_profit_bps=35.0,
        trailing_stop_bps=12.0,
    )

    decision = risk.evaluate_position_exit(
        "ETHUSDT",
        _snapshot(now),
        _state(last_price=99.70, opened_at=now - timedelta(seconds=5)),
        now,
        estimated_exit_cost_bps=10.0,
    )

    assert decision is None


def test_risk_manager_triggers_soft_stop_after_min_hold() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    risk = RiskManager(
        stale_data_s=6.0,
        broken_ws_s=12.0,
        abnormal_spread_bps=250.0,
        abnormal_volatility_bps=900.0,
        max_position_size=8.0,
        max_notional=50_000.0,
        max_concurrent_exposure=2,
        cooldown_after_loss_s=300,
        daily_loss_limit=0.02,
        max_holding_time_s=900,
        catastrophic_stop_loss_bps=45.0,
        min_hold_before_soft_exit_s=20,
        stop_loss_bps=20.0,
        take_profit_bps=35.0,
        trailing_stop_bps=12.0,
    )

    decision = risk.evaluate_position_exit(
        "ETHUSDT",
        _snapshot(now),
        _state(last_price=99.70, opened_at=now - timedelta(seconds=25)),
        now,
        estimated_exit_cost_bps=10.0,
    )

    assert decision is not None
    assert decision.reason == "risk_stop_loss"


def test_risk_manager_triggers_catastrophic_stop_immediately() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    risk = RiskManager(
        stale_data_s=6.0,
        broken_ws_s=12.0,
        abnormal_spread_bps=250.0,
        abnormal_volatility_bps=900.0,
        max_position_size=8.0,
        max_notional=50_000.0,
        max_concurrent_exposure=2,
        cooldown_after_loss_s=300,
        daily_loss_limit=0.02,
        max_holding_time_s=900,
        catastrophic_stop_loss_bps=45.0,
        min_hold_before_soft_exit_s=20,
        stop_loss_bps=20.0,
        take_profit_bps=35.0,
        trailing_stop_bps=12.0,
    )

    decision = risk.evaluate_position_exit(
        "ETHUSDT",
        _snapshot(now),
        _state(last_price=99.50, opened_at=now - timedelta(seconds=3)),
        now,
        estimated_exit_cost_bps=10.0,
    )

    assert decision is not None
    assert decision.reason == "risk_catastrophic_stop"
