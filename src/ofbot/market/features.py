from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import log
from statistics import mean
from typing import Any

import numpy as np

from ofbot.market.orderbook import OrderBookState
from ofbot.market.regimes import RegimeClassifier, RegimeLabel
from ofbot.market.trades import AggTradeEvent


EPSILON = 1e-12


@dataclass(slots=True)
class FeatureSnapshot:
    symbol: str
    event_time: datetime
    features: dict[str, float | int | str | None]
    regime: RegimeLabel
    spread_bps: float | None = None
    mid_price: float | None = None
    microprice: float | None = None


@dataclass(slots=True)
class _SymbolFeatureState:
    symbol: str
    horizons_sec: list[int]
    feature_window_sec: int
    large_trade_quantile: float
    zscore_window: int
    trade_events: deque
    mid_events: deque
    spread_events: deque
    trade_rate_history: dict[int, deque]
    last_snapshot_time: datetime | None = None


class FeatureEngine:
    def __init__(
        self,
        *,
        horizons_sec: list[int],
        feature_window_sec: int,
        large_trade_quantile: float = 0.9,
        volatility_window_sec: int = 60,
        zscore_window: int = 200,
    ) -> None:
        self._horizons = sorted(set(horizons_sec))
        self._feature_window_sec = max(1, feature_window_sec)
        self._large_trade_quantile = max(0.5, min(0.999, large_trade_quantile))
        self._volatility_window_sec = max(1, volatility_window_sec)
        self._zscore_window = max(3, zscore_window)
        self._states: dict[str, _SymbolFeatureState] = {}
        self._regime_classifier = RegimeClassifier()
        self._snapshot_buffer: dict[str, deque] = {}

    def state_for(self, symbol: str) -> _SymbolFeatureState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolFeatureState(
                symbol=symbol,
                horizons_sec=self._horizons,
                feature_window_sec=self._feature_window_sec,
                large_trade_quantile=self._large_trade_quantile,
                trade_events=deque(),
                mid_events=deque(),
                spread_events=deque(),
                zscore_window=self._zscore_window,
                trade_rate_history={
                    horizon: deque(maxlen=max(16, self._zscore_window)) for horizon in self._horizons
                },
            )
        return self._states[symbol]

    def on_orderbook(self, event_time: datetime, symbol: str, book: OrderBookState) -> None:
        state = self.state_for(symbol)
        if book.spread_bps is not None:
            state.spread_events.append((event_time, book.spread_bps))
        if book.mid_price is not None:
            state.mid_events.append((event_time, book.mid_price))

    def on_agg_trade(self, event: AggTradeEvent) -> None:
        state = self.state_for(event.symbol)
        state.trade_events.append((event.event_time, event.aggressor_side, event.price, event.quantity))
        self._purge_by_time(event.event_time, state)

    def snapshot(
        self,
        symbol: str,
        *,
        event_time: datetime,
        book: OrderBookState | None,
    ) -> FeatureSnapshot:
        state = self.state_for(symbol)
        self._purge_by_time(event_time, state)

        spread = book.spread_bps if book is not None else None
        mid = book.mid_price if book is not None else None
        micro = book.microprice if book is not None else None
        queue_imbalance = book.queue_imbalance if book is not None else 0.0

        features: dict[str, float | int | str | None] = {
            "symbol": symbol,
            "event_time": event_time.timestamp(),
            "spread_bps": spread,
            "mid_price": mid,
            "microprice": micro,
            "queue_imbalance": queue_imbalance,
        }

        for horizon in self._horizons:
            horizon_features = self._window_features(state=state, now=event_time, horizon_sec=horizon)
            for key, value in horizon_features.items():
                features[f"{key}_{horizon}s"] = value

        realized_vol = self._volatility(state=state, now=event_time, horizon_sec=self._volatility_window_sec)
        features["realized_volatility"] = realized_vol

        flow_score = _as_float(features.get("cvd_base_1s"), default=0.0)
        momentum_score = _as_float(features.get("short_return_bps_1s"), default=0.0)
        regime = self._regime_classifier.observe(
            realized_volatility=realized_vol,
            spread_bps=spread,
            flow_score=float(flow_score or 0.0),
            momentum_bps=float(momentum_score or 0.0),
        )
        features["regime_volatility"] = regime.volatility
        features["regime_spread"] = regime.spread
        features["regime_trend"] = regime.trend
        features["regime_flow"] = regime.flow

        if mid is not None:
            features["microprice_1s"] = _as_float(features.get("microprice_1s"), default=mid)
            features["momentum_bps_1s"] = _as_float(features.get("short_return_bps_1s"), default=0.0)

        features.update(self._zscored(features))
        state.last_snapshot_time = event_time

        return FeatureSnapshot(
            symbol=symbol,
            event_time=event_time,
            features=features,
            regime=regime,
            spread_bps=spread,
            mid_price=mid,
            microprice=micro,
        )

    def _window_features(
        self,
        state: _SymbolFeatureState,
        *,
        now: datetime,
        horizon_sec: int,
    ) -> dict[str, float | None]:
        cutoff = now - timedelta(seconds=horizon_sec)
        rows = [(t, s, p, q) for (t, s, p, q) in state.trade_events if t >= cutoff]

        if not rows:
            return {
                "cvd_base": 0.0,
                "cvd_quote": 0.0,
                "cvd_base_abs": 0.0,
                "trade_count": 0.0,
                "trade_count_imbalance": 0.0,
                "trade_count_intensity": 0.0,
                "signed_trade_intensity": 0.0,
                "large_trade_count": 0.0,
                "large_trade_ratio": 0.0,
                "large_trade_signed_flow": 0.0,
                "burst_intensity": 1.0,
                "distance_vwap_bps": None,
                "short_return_bps": None,
                "microprice_drift_bps": None,
            }

        counts = float(len(rows))
        signed_base_qty = sum(side * qty for _, side, _, qty in rows)
        signed_quote_qty = sum(side * price * qty for _, side, price, qty in rows)
        abs_base_qty = sum(abs(side * qty) for _, side, _, qty in rows)
        abs_quote_qty = sum(abs(price * qty) for _, _, price, qty in rows)
        buy = sum(1 for _, side, _, _ in rows if side > 0)
        sell = counts - buy

        sizes = sorted(abs(qty) for _, _, _, qty in rows)
        threshold = float(np.quantile(sizes, state.large_trade_quantile)) if sizes else 0.0
        is_large = [abs(q) >= threshold if threshold > 0 else False for _, _, _, q in rows]
        large_trade_count = float(sum(1 for large in is_large if large))
        large_trade_signed = sum(side * qty for (_, side, _, qty), large in zip(rows, is_large) if large)
        large_trade_ratio = large_trade_count / counts if counts > EPSILON else 0.0

        abs_qty_sum = max(abs_base_qty, EPSILON)
        vwap = sum(price * qty for _, _, price, qty in rows) / max(abs_base_qty, EPSILON)

        mids = [value for ts, value in state.mid_events if cutoff <= ts <= now]
        first_mid = float(mids[0]) if mids else None
        last_mid = float(mids[-1]) if mids else None

        distance_vwap = None
        short_return = None
        if first_mid is not None and last_mid is not None and first_mid > EPSILON:
            short_return = (log(last_mid / first_mid)) * 10_000.0
            distance_vwap = (last_mid - vwap) / max(vwap, EPSILON) * 10_000.0
        elif first_mid is not None and first_mid > EPSILON:
            short_return = 0.0
            distance_vwap = 0.0

        micro_drift = None
        if len(mids) >= 2 and mids[0] > EPSILON:
            micro_drift = ((mids[-1] - mids[0]) / mids[0]) * 10_000.0

        trade_count = counts
        trade_count_intensity = trade_count / max(float(horizon_sec), EPSILON)
        history = state.trade_rate_history[horizon_sec]
        expected = mean(history) if history else trade_count_intensity
        history.append(trade_count_intensity)
        burst_intensity = trade_count_intensity / max(expected, EPSILON)
        trade_intensity = abs_base_qty / max(trade_count_intensity, EPSILON)

        imbalance = (buy - sell) / max(counts, 1.0)
        return {
            "cvd_base": float(signed_base_qty),
            "cvd_quote": float(signed_quote_qty),
            "cvd_base_abs": abs_base_qty,
            "trade_count": float(trade_count),
            "trade_count_imbalance": float(imbalance),
            "trade_count_intensity": float(trade_count_intensity),
            "signed_trade_intensity": float(signed_base_qty / max(abs_base_qty, EPSILON)),
            "large_trade_count": float(large_trade_count),
            "large_trade_ratio": float(large_trade_ratio),
            "large_trade_signed_flow": float(large_trade_signed),
            "burst_intensity": float(burst_intensity),
            "distance_vwap_bps": distance_vwap,
            "short_return_bps": short_return,
            "microprice_drift_bps": micro_drift,
            "microprice_drift": micro_drift,
            "quote_quote": abs_quote_qty,
        }

    def _volatility(self, state: _SymbolFeatureState, now: datetime, horizon_sec: int) -> float | None:
        cutoff = now - timedelta(seconds=horizon_sec)
        mids = [value for ts, value in state.mid_events if ts >= cutoff]
        if len(mids) < 3:
            return None
        diffs = [
            log(curr / prev)
            for prev, curr in zip(mids[:-1], mids[1:])
            if prev > EPSILON and curr > EPSILON
        ]
        if len(diffs) < 2:
            return None
        return float(np.std(diffs) * 10_000.0)

    def _zscored(self, feature_map: dict[str, float | int | str | None]) -> dict[str, float | None]:
        output: dict[str, float | None] = {}
        for key, value in feature_map.items():
            if not isinstance(value, (int, float)):
                continue
            if key not in self._snapshot_buffer:
                self._snapshot_buffer[key] = deque(maxlen=self._zscore_window)
            history = self._snapshot_buffer[key]
            history.append(float(value))
            if len(history) < max(10, self._zscore_window // 3):
                output[f"{key}_z"] = None
                continue
            arr = np.asarray(list(history), dtype=float)
            mean_value = float(arr.mean())
            std_value = float(arr.std())
            output[f"{key}_z"] = (float(value) - mean_value) / std_value if std_value > EPSILON else 0.0
        return output

    def _purge_by_time(self, now: datetime, state: _SymbolFeatureState) -> None:
        earliest = now - timedelta(seconds=self._feature_window_sec)
        while state.trade_events and state.trade_events[0][0] < earliest:
            state.trade_events.popleft()
        while state.mid_events and state.mid_events[0][0] < earliest:
            state.mid_events.popleft()
        while state.spread_events and state.spread_events[0][0] < earliest:
            state.spread_events.popleft()


def _as_float(value: Any, default: float | None = 0.0) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return default
