from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ofbot.execution.portfolio import PositionSnapshot
from ofbot.market.features import FeatureSnapshot
from ofbot.market.regimes import RegimeLabel
from ofbot.strategy.continuation import ContinuationStrategy


def _state(
    *,
    net_position: float = 0.0,
    opened_at: datetime | None = None,
    avg_entry_price: float = 0.0,
    last_price: float | None = None,
) -> PositionSnapshot:
    return PositionSnapshot(
        symbol="ETHUSDT",
        net_position=net_position,
        avg_entry_price=avg_entry_price,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        fees_paid=0.0,
        last_price=last_price,
        position_open_time=opened_at,
        mae_bps=0.0,
        mfe_bps=0.0,
        risk_context={},
    )


def _snapshot(
    *,
    at_time: datetime | None = None,
    micro_drift_bps_1s: float,
    momentum_bps_1s: float = 1.0,
    momentum_bps_5s: float = 1.0,
    cvd_base_1s: float = 2.0,
    cvd_base_1s_z: float = 2.0,
    queue_imbalance: float = 0.995,
    microprice_1s: float = 2163.855,
    trend: str = "trend_up",
) -> FeatureSnapshot:
    return FeatureSnapshot(
        symbol="ETHUSDT",
        event_time=at_time or datetime(2026, 1, 1, tzinfo=UTC),
        features={
            "realized_volatility": 10.0,
            "cvd_base_1s": cvd_base_1s,
            "cvd_base_1s_z": cvd_base_1s_z,
            "queue_imbalance": queue_imbalance,
            "microprice_drift_bps_1s": micro_drift_bps_1s,
            "microprice_drift_bps": micro_drift_bps_1s,
            "microprice_1s": microprice_1s,
            "microprice": microprice_1s,
            "momentum_bps_1s": momentum_bps_1s,
            "short_return_bps_1s": momentum_bps_1s,
            "momentum_bps_5s": momentum_bps_5s,
            "short_return_bps_5s": momentum_bps_5s,
        },
        regime=RegimeLabel(volatility="low", spread="low", trend=trend, flow="high"),
        spread_bps=0.05,
        mid_price=2163.855,
        microprice=microprice_1s,
    )


def test_continuation_uses_microprice_drift_bps_not_microprice_level() -> None:
    strategy = ContinuationStrategy(
        trend_alignment_threshold=0.20,
        queue_imbalance_threshold=0.99,
        microprice_drift_threshold_bps=0.20,
        no_trade_z=1.20,
        spread_bps_max=1.0,
        volatility_bps_max=700.0,
    )

    signal = strategy.decide(
        _snapshot(micro_drift_bps_1s=0.05, microprice_1s=2163.855),
        _state(),
        long_only=True,
    )

    assert signal is None


def test_continuation_enters_when_microprice_drift_bps_clears_gate() -> None:
    strategy = ContinuationStrategy(
        trend_alignment_threshold=0.20,
        queue_imbalance_threshold=0.99,
        microprice_drift_threshold_bps=0.20,
        no_trade_z=1.20,
        spread_bps_max=1.0,
        volatility_bps_max=700.0,
    )

    signal = strategy.decide(
        _snapshot(micro_drift_bps_1s=0.35, microprice_1s=2163.855),
        _state(),
        long_only=True,
    )

    assert signal is not None
    assert signal.reason == "continuation_long"


def test_continuation_blocks_range_regime_when_trend_regime_required() -> None:
    strategy = ContinuationStrategy(
        trend_alignment_threshold=0.20,
        queue_imbalance_threshold=0.99,
        microprice_drift_threshold_bps=0.20,
        no_trade_z=1.20,
        require_trend_up_regime=True,
        spread_bps_max=1.0,
        volatility_bps_max=700.0,
    )

    signal = strategy.decide(
        _snapshot(
            micro_drift_bps_1s=0.35,
            microprice_1s=2163.855,
            trend="range",
        ),
        _state(),
        long_only=True,
    )

    assert signal is None


def test_continuation_blocks_short_entry_outside_trend_down_when_required() -> None:
    strategy = ContinuationStrategy(
        require_trend_down_regime_for_short=True,
        reverse_exit_cvd_z=1.0,
        reverse_exit_queue_imbalance=0.85,
        reverse_exit_microprice_drift_bps=0.1,
    )

    signal = strategy.decide(
        _snapshot(
            micro_drift_bps_1s=-0.35,
            momentum_bps_1s=-0.5,
            momentum_bps_5s=-1.0,
            cvd_base_1s=-2.0,
            cvd_base_1s_z=-1.5,
            queue_imbalance=-0.95,
            trend="range",
        ),
        _state(),
        long_only=False,
    )

    assert signal is None


def test_continuation_does_not_exit_on_reverse_before_min_hold() -> None:
    now = datetime(2026, 1, 1, 0, 0, 30, tzinfo=UTC)
    strategy = ContinuationStrategy(
        min_hold_before_discretionary_exit_s=20,
        reverse_exit_cvd_z=1.0,
        reverse_exit_queue_imbalance=0.85,
        reverse_exit_microprice_drift_bps=0.1,
    )

    signal = strategy.decide(
        _snapshot(
            at_time=now,
            micro_drift_bps_1s=-0.25,
            momentum_bps_1s=-0.5,
            momentum_bps_5s=-1.0,
            cvd_base_1s=-2.0,
            cvd_base_1s_z=-1.5,
            queue_imbalance=-0.95,
        ),
        _state(net_position=1.0, opened_at=now - timedelta(seconds=10)),
        long_only=True,
    )

    assert signal is None


def test_continuation_exits_on_reverse_after_min_hold() -> None:
    now = datetime(2026, 1, 1, 0, 1, 0, tzinfo=UTC)
    strategy = ContinuationStrategy(
        min_hold_before_discretionary_exit_s=20,
        reverse_exit_cvd_z=1.0,
        reverse_exit_queue_imbalance=0.85,
        reverse_exit_microprice_drift_bps=0.1,
    )

    signal = strategy.decide(
        _snapshot(
            at_time=now,
            micro_drift_bps_1s=-0.25,
            momentum_bps_1s=-0.5,
            momentum_bps_5s=-1.0,
            cvd_base_1s=-2.0,
            cvd_base_1s_z=-1.5,
            queue_imbalance=-0.95,
        ),
        _state(net_position=1.0, opened_at=now - timedelta(seconds=30)),
        long_only=True,
    )

    assert signal is not None
    assert signal.reason == "continuation_reverse_exit"


def test_continuation_waits_for_meaningful_move_before_reverse_exit() -> None:
    now = datetime(2026, 1, 1, 0, 1, 0, tzinfo=UTC)
    strategy = ContinuationStrategy(
        min_hold_before_discretionary_exit_s=20,
        reverse_exit_cvd_z=1.0,
        reverse_exit_queue_imbalance=0.85,
        reverse_exit_microprice_drift_bps=0.1,
        reverse_exit_min_abs_pnl_bps=4.0,
    )

    signal = strategy.decide(
        _snapshot(
            at_time=now,
            micro_drift_bps_1s=-0.25,
            momentum_bps_1s=-0.5,
            momentum_bps_5s=-1.0,
            cvd_base_1s=-2.0,
            cvd_base_1s_z=-1.5,
            queue_imbalance=-0.95,
        ),
        _state(
            net_position=1.0,
            opened_at=now - timedelta(seconds=30),
            avg_entry_price=2163.20,
            last_price=2163.855,
        ),
        long_only=True,
    )

    assert signal is None


def test_continuation_exports_revalidate_drop_on_neutral_parameter() -> None:
    strategy = ContinuationStrategy(
        revalidate_on_fill=True,
        revalidate_grace_period_s=2.0,
        revalidate_drop_on_neutral=False,
    )

    params = strategy.get_parameters()

    assert params["revalidate_on_fill"] is True
    assert params["revalidate_grace_period_s"] == 2.0
    assert params["revalidate_drop_on_neutral"] is False


def test_continuation_uses_configured_exit_order_type_for_flat_decisions() -> None:
    now = datetime(2026, 1, 1, 0, 1, 0, tzinfo=UTC)
    strategy = ContinuationStrategy(
        exit_order_type="limit",
        min_hold_before_discretionary_exit_s=20,
        reverse_exit_cvd_z=1.0,
        reverse_exit_queue_imbalance=0.85,
        reverse_exit_microprice_drift_bps=0.1,
    )

    signal = strategy.decide(
        _snapshot(
            at_time=now,
            micro_drift_bps_1s=-0.25,
            momentum_bps_1s=-0.5,
            momentum_bps_5s=-1.0,
            cvd_base_1s=-2.0,
            cvd_base_1s_z=-1.5,
            queue_imbalance=-0.95,
        ),
        _state(net_position=1.0, opened_at=now - timedelta(seconds=30)),
        long_only=True,
    )

    assert signal is not None
    assert signal.order_type == "limit"
