from __future__ import annotations

import math
from typing import Any

from quantflow.data_model.events import BarEvent, PortfolioState, SignalEvent, SignalSide
from quantflow.strategy.base import Strategy


def _feature_float(event: BarEvent, key: str) -> float | None:
    value = event.features.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isnan(number):
            return None
        return number
    return None


def _context(event: BarEvent) -> dict[str, float | int | str | None]:
    return {
        "hour_of_day": event.features.get("hour_of_day"),
        "session": event.features.get("session"),
        "vol_regime": event.features.get("vol_regime"),
        "atr_ratio": event.features.get("atr_ratio"),
    }


def _first_feature_float(event: BarEvent, keys: list[str]) -> float | None:
    for key in keys:
        value = _feature_float(event, key)
        if value is not None:
            return value
    return None


def _signal_if_changed(
    strategy: Strategy,
    event: BarEvent,
    target: int,
    reason: str,
) -> SignalEvent | None:
    if target == strategy.current_target:
        return None
    side = SignalSide.LONG if target > 0 else SignalSide.SHORT if target < 0 else SignalSide.FLAT
    return SignalEvent(
        symbol=event.symbol,
        event_time=event.end_time,
        side=side,
        target_position=target,
        reason=reason,
        context=_context(event),
    )


def _vol_filter_passed(event: BarEvent, vol_regime_min: float) -> bool:
    if vol_regime_min <= 0.0:
        return True
    vol_regime = _feature_float(event, "vol_regime")
    if vol_regime is None:
        return False
    return vol_regime >= vol_regime_min


def _session_filter_passed(event: BarEvent, allowed_sessions: tuple[str, ...]) -> bool:
    if not allowed_sessions:
        return True
    session = event.features.get("session")
    return isinstance(session, str) and session in allowed_sessions


def _update_signal_context(
    signal: SignalEvent | None,
    context: dict[str, float | int | str | None],
) -> SignalEvent | None:
    if signal is None:
        return None
    signal.context.update(context)
    return signal


class CumulativeDeltaBreakoutStrategy(Strategy):
    def __init__(
        self,
        cumulative_delta_threshold: float = 1.0,
        breakout_lookback_bars: int = 12,
        exit_zscore_threshold: float = 0.2,
    ) -> None:
        super().__init__()
        self.cumulative_delta_threshold = cumulative_delta_threshold
        self.breakout_lookback_bars = breakout_lookback_bars
        self.exit_zscore_threshold = exit_zscore_threshold

    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        cumulative_delta_z = _feature_float(event, "cumulative_delta_rz")
        range_high_prev = _feature_float(event, "range_high_prev")
        range_low_prev = _feature_float(event, "range_low_prev")

        if cumulative_delta_z is None or range_high_prev is None or range_low_prev is None:
            return None

        if (
            cumulative_delta_z >= self.cumulative_delta_threshold
            and event.close_price > range_high_prev
        ):
            return _signal_if_changed(self, event, 1, "cumulative_delta_breakout_long")
        if (
            cumulative_delta_z <= -self.cumulative_delta_threshold
            and event.close_price < range_low_prev
        ):
            return _signal_if_changed(self, event, -1, "cumulative_delta_breakout_short")
        if abs(cumulative_delta_z) <= self.exit_zscore_threshold:
            return _signal_if_changed(self, event, 0, "cumulative_delta_reversion_exit")
        return None

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, Any]:
        return {
            "cumulative_delta_threshold": self.cumulative_delta_threshold,
            "breakout_lookback_bars": self.breakout_lookback_bars,
            "exit_zscore_threshold": self.exit_zscore_threshold,
        }


class Spot10mRegimePullbackStrategy(Strategy):
    def __init__(
        self,
        ema_fast_bars: int = 20,
        ema_trend_bars: int = 200,
        atr_bars: int = 14,
        rsi_bars: int = 14,
        rsi_min: float = 50.0,
        rsi_max: float = 70.0,
        min_atr_ratio: float = 0.0015,
        use_taker_buy_ratio_filter: bool = False,
        taker_buy_ratio_min: float = 0.55,
        stop_atr_multiple: float = 1.0,
        target_atr_multiple: float = 1.2,
        time_stop_bars: int = 6,
    ) -> None:
        super().__init__()
        self.ema_fast_bars = ema_fast_bars
        self.ema_trend_bars = ema_trend_bars
        self.atr_bars = atr_bars
        self.rsi_bars = rsi_bars
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.min_atr_ratio = min_atr_ratio
        self.use_taker_buy_ratio_filter = use_taker_buy_ratio_filter
        self.taker_buy_ratio_min = taker_buy_ratio_min
        self.stop_atr_multiple = stop_atr_multiple
        self.target_atr_multiple = target_atr_multiple
        self.time_stop_bars = time_stop_bars

    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        bar_number = _feature_float(event, "bar_number")
        ema_fast = _feature_float(event, "ema_fast_20")
        ema_trend = _feature_float(event, "ema_trend_200")
        atr_value = _feature_float(event, "atr_14")
        atr_ratio = _feature_float(event, "atr_ratio")
        rsi_value = _feature_float(event, "rsi_14")
        taker_buy_ratio = _feature_float(event, "taker_buy_ratio")

        if (
            bar_number is None
            or ema_fast is None
            or ema_trend is None
            or atr_value is None
            or atr_ratio is None
            or rsi_value is None
        ):
            return None

        warmup_bars = max(self.ema_trend_bars, self.atr_bars, self.rsi_bars)
        if bar_number < warmup_bars:
            return None

        if event.close_price <= ema_trend:
            return _signal_if_changed(self, event, 0, "spot_10m_regime_filter_exit")
        if event.close_price <= ema_fast:
            return None
        if event.low_price > ema_fast:
            return None
        if rsi_value < self.rsi_min or rsi_value > self.rsi_max:
            return None
        if atr_ratio < self.min_atr_ratio:
            return None
        if self.use_taker_buy_ratio_filter and (
            taker_buy_ratio is None or taker_buy_ratio < self.taker_buy_ratio_min
        ):
            return None

        signal = _signal_if_changed(self, event, 1, "spot_10m_regime_pullback_long")
        if signal is None:
            return None
        signal.context.update(
            {
                "entry_reason": signal.reason,
                "atr_entry": atr_value,
                "stop_distance": atr_value * self.stop_atr_multiple,
                "target_distance": atr_value * self.target_atr_multiple,
                "time_stop_bars": self.time_stop_bars,
                "taker_buy_ratio": taker_buy_ratio,
            }
        )
        return signal

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, Any]:
        return {
            "ema_fast_bars": self.ema_fast_bars,
            "ema_trend_bars": self.ema_trend_bars,
            "atr_bars": self.atr_bars,
            "rsi_bars": self.rsi_bars,
            "rsi_min": self.rsi_min,
            "rsi_max": self.rsi_max,
            "min_atr_ratio": self.min_atr_ratio,
            "use_taker_buy_ratio_filter": self.use_taker_buy_ratio_filter,
            "taker_buy_ratio_min": self.taker_buy_ratio_min,
            "stop_atr_multiple": self.stop_atr_multiple,
            "target_atr_multiple": self.target_atr_multiple,
            "time_stop_bars": self.time_stop_bars,
        }


class CumDeltaReversionV1Strategy(Strategy):
    def __init__(
        self,
        entry_z: float = 1.5,
        exit_z: float = 0.25,
        vol_regime_min: float = 1.0,
        allowed_sessions: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.vol_regime_min = vol_regime_min
        self.allowed_sessions = tuple(allowed_sessions or ())

    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        cumulative_delta = _first_feature_float(
            event,
            ["cumulative_delta_rz", "rolling_delta_rz"],
        )
        if cumulative_delta is None:
            return None

        if cumulative_delta >= -self.exit_z:
            return _signal_if_changed(self, event, 0, "cumdelta_reversion_v1_exit")
        if cumulative_delta > -self.entry_z:
            return None
        if not _vol_filter_passed(event, self.vol_regime_min):
            return None
        if not _session_filter_passed(event, self.allowed_sessions):
            return None
        return _update_signal_context(
            _signal_if_changed(self, event, 1, "cumdelta_reversion_v1_long"),
            {
                "entry_reason": "cumdelta_reversion_v1_long",
                "signal_feature": "cumulative_delta_rz",
                "signal_value": cumulative_delta,
                "vol_regime_min": self.vol_regime_min,
                "allowed_sessions": (
                    ",".join(self.allowed_sessions) if self.allowed_sessions else "all"
                ),
            },
        )

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, Any]:
        return {
            "entry_z": self.entry_z,
            "exit_z": self.exit_z,
            "vol_regime_min": self.vol_regime_min,
            "allowed_sessions": list(self.allowed_sessions),
        }


class DeltaImpulseContinuationV1Strategy(Strategy):
    def __init__(
        self,
        source: str = "delta_rz",
        entry_z: float = 1.0,
        exit_z: float = 0.25,
        vol_regime_min: float = 1.0,
    ) -> None:
        super().__init__()
        self.source = source
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.vol_regime_min = vol_regime_min

    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        feature_name = "quote_delta_rz" if self.source == "quote_delta_rz" else "delta_rz"
        impulse = _first_feature_float(event, [feature_name, feature_name.removesuffix("_rz")])
        if impulse is None:
            return None

        if impulse <= self.exit_z:
            return _signal_if_changed(self, event, 0, "delta_impulse_continuation_v1_exit")
        if impulse < self.entry_z:
            return None
        if not _vol_filter_passed(event, self.vol_regime_min):
            return None
        return _update_signal_context(
            _signal_if_changed(self, event, 1, "delta_impulse_continuation_v1_long"),
            {
                "entry_reason": "delta_impulse_continuation_v1_long",
                "signal_feature": feature_name,
                "signal_value": impulse,
                "vol_regime_min": self.vol_regime_min,
            },
        )

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "entry_z": self.entry_z,
            "exit_z": self.exit_z,
            "vol_regime_min": self.vol_regime_min,
        }


class ImbalanceBurstExhaustionV1Strategy(Strategy):
    def __init__(
        self,
        imbalance_z: float = 1.0,
        burst_z: float = 1.0,
        exit_z: float = 0.25,
        vol_regime_min: float = 1.0,
    ) -> None:
        super().__init__()
        self.imbalance_z = imbalance_z
        self.burst_z = burst_z
        self.exit_z = exit_z
        self.vol_regime_min = vol_regime_min

    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        imbalance = _first_feature_float(
            event,
            ["trade_count_imbalance_rz", "trade_count_imbalance"],
        )
        burst = _first_feature_float(event, ["burst_intensity_rz", "burst_intensity"])
        if imbalance is None or burst is None:
            return None

        if imbalance >= -self.exit_z:
            return _signal_if_changed(self, event, 0, "imbalance_burst_exhaustion_v1_exit")
        if imbalance > -self.imbalance_z or burst < self.burst_z:
            return None
        if not _vol_filter_passed(event, self.vol_regime_min):
            return None
        return _update_signal_context(
            _signal_if_changed(self, event, 1, "imbalance_burst_exhaustion_v1_long"),
            {
                "entry_reason": "imbalance_burst_exhaustion_v1_long",
                "signal_feature": "trade_count_imbalance_rz",
                "signal_value": imbalance,
                "burst_feature": "burst_intensity_rz",
                "burst_value": burst,
                "vol_regime_min": self.vol_regime_min,
            },
        )

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, Any]:
        return {
            "imbalance_z": self.imbalance_z,
            "burst_z": self.burst_z,
            "exit_z": self.exit_z,
            "vol_regime_min": self.vol_regime_min,
        }


class CumulativeDeltaMeanReversionStrategy(Strategy):
    def __init__(
        self,
        cumulative_delta_threshold: float = 1.0,
        price_extension_threshold: float = 0.3,
        exit_zscore_threshold: float = 0.2,
    ) -> None:
        super().__init__()
        self.cumulative_delta_threshold = cumulative_delta_threshold
        self.price_extension_threshold = price_extension_threshold
        self.exit_zscore_threshold = exit_zscore_threshold

    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        cumulative_delta_z = _first_feature_float(
            event,
            ["cumulative_delta_rz", "rolling_delta_rz"],
        )
        price_extension = _feature_float(event, "micro_range_breakout")
        if cumulative_delta_z is None or price_extension is None:
            return None

        lower_extension = self.price_extension_threshold
        upper_extension = 1.0 - self.price_extension_threshold

        if (
            cumulative_delta_z <= -self.cumulative_delta_threshold
            and price_extension <= lower_extension
        ):
            return _signal_if_changed(
                self,
                event,
                1,
                "cumulative_delta_mean_reversion_long",
            )
        if (
            cumulative_delta_z >= self.cumulative_delta_threshold
            and price_extension >= upper_extension
        ):
            return _signal_if_changed(
                self,
                event,
                -1,
                "cumulative_delta_mean_reversion_short",
            )
        if abs(cumulative_delta_z) <= self.exit_zscore_threshold:
            return _signal_if_changed(self, event, 0, "cumulative_delta_mean_reversion_exit")
        return None

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, Any]:
        return {
            "cumulative_delta_threshold": self.cumulative_delta_threshold,
            "price_extension_threshold": self.price_extension_threshold,
            "exit_zscore_threshold": self.exit_zscore_threshold,
        }


class TradeImbalanceMomentumStrategy(Strategy):
    def __init__(self, imbalance_threshold: float = 0.6, momentum_threshold: float = 0.0) -> None:
        super().__init__()
        self.imbalance_threshold = imbalance_threshold
        self.momentum_threshold = momentum_threshold

    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        imbalance = _feature_float(event, "trade_count_imbalance_rz") or _feature_float(
            event, "trade_count_imbalance"
        )
        momentum = _feature_float(event, "micro_momentum_rz") or _feature_float(
            event, "micro_momentum"
        )
        if imbalance is None or momentum is None:
            return None

        if imbalance > self.imbalance_threshold and momentum > self.momentum_threshold:
            return _signal_if_changed(self, event, 1, "trade_imbalance_momentum_long")
        if imbalance < -self.imbalance_threshold and momentum < -self.momentum_threshold:
            return _signal_if_changed(self, event, -1, "trade_imbalance_momentum_short")
        if abs(imbalance) < self.imbalance_threshold / 2:
            return _signal_if_changed(self, event, 0, "trade_imbalance_exit")
        return None

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, Any]:
        return {
            "imbalance_threshold": self.imbalance_threshold,
            "momentum_threshold": self.momentum_threshold,
        }


class DeltaContinuationStrategy(Strategy):
    def __init__(
        self,
        delta_threshold: float = 0.8,
        imbalance_threshold: float = 0.1,
        exit_delta_threshold: float = 0.2,
        use_quote_delta: bool = False,
    ) -> None:
        super().__init__()
        self.delta_threshold = delta_threshold
        self.imbalance_threshold = imbalance_threshold
        self.exit_delta_threshold = exit_delta_threshold
        self.use_quote_delta = use_quote_delta

    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        delta_keys = (
            ["quote_delta_rz", "quote_delta"]
            if self.use_quote_delta
            else ["delta_rz", "delta"]
        )
        delta_signal = _first_feature_float(event, delta_keys)
        imbalance = _first_feature_float(
            event,
            ["trade_count_imbalance_rz", "trade_count_imbalance"],
        )
        if delta_signal is None or imbalance is None:
            return None

        if (
            delta_signal >= self.delta_threshold
            and imbalance >= self.imbalance_threshold
        ):
            return _signal_if_changed(self, event, 1, "delta_continuation_long")
        if (
            delta_signal <= -self.delta_threshold
            and imbalance <= -self.imbalance_threshold
        ):
            return _signal_if_changed(self, event, -1, "delta_continuation_short")
        if abs(delta_signal) <= self.exit_delta_threshold:
            return _signal_if_changed(self, event, 0, "delta_continuation_exit")
        return None

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, Any]:
        return {
            "delta_threshold": self.delta_threshold,
            "imbalance_threshold": self.imbalance_threshold,
            "exit_delta_threshold": self.exit_delta_threshold,
            "use_quote_delta": self.use_quote_delta,
        }


class BurstBreakoutConfirmationStrategy(Strategy):
    def __init__(self, burst_threshold: float = 1.0, breakout_threshold: float = 1.0) -> None:
        super().__init__()
        self.burst_threshold = burst_threshold
        self.breakout_threshold = breakout_threshold

    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        burst = _feature_float(event, "burst_intensity_rz") or _feature_float(
            event, "burst_intensity"
        )
        breakout = _feature_float(event, "micro_range_breakout")
        if burst is None or breakout is None:
            return None

        if burst > self.burst_threshold and breakout > self.breakout_threshold:
            return _signal_if_changed(self, event, 1, "burst_breakout_long")
        if burst > self.burst_threshold and breakout < 1.0 - self.breakout_threshold:
            return _signal_if_changed(self, event, -1, "burst_breakout_short")
        if burst < self.burst_threshold / 2:
            return _signal_if_changed(self, event, 0, "burst_decay_exit")
        return None

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, Any]:
        return {
            "burst_threshold": self.burst_threshold,
            "breakout_threshold": self.breakout_threshold,
        }


class ExhaustionMeanReversionStrategy(Strategy):
    def __init__(self, delta_threshold: float = 1.5, burst_threshold: float = 1.0) -> None:
        super().__init__()
        self.delta_threshold = delta_threshold
        self.burst_threshold = burst_threshold

    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        rolling_delta = _feature_float(event, "rolling_delta_rz") or _feature_float(
            event, "rolling_delta"
        )
        burst = _feature_float(event, "burst_intensity_rz") or _feature_float(
            event, "burst_intensity"
        )
        breakout = _feature_float(event, "micro_range_breakout")
        if rolling_delta is None or burst is None or breakout is None:
            return None

        if burst > self.burst_threshold and rolling_delta > self.delta_threshold and breakout > 1.0:
            return _signal_if_changed(self, event, -1, "flow_exhaustion_short")
        if (
            burst > self.burst_threshold
            and rolling_delta < -self.delta_threshold
            and breakout < 0.0
        ):
            return _signal_if_changed(self, event, 1, "flow_exhaustion_long")
        if 0.25 <= breakout <= 0.75:
            return _signal_if_changed(self, event, 0, "range_reentry_exit")
        return None

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, Any]:
        return {"delta_threshold": self.delta_threshold, "burst_threshold": self.burst_threshold}
