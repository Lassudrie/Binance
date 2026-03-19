from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RegimeLabel:
    volatility: str
    spread: str
    trend: str
    flow: str

    def as_dict(self) -> dict[str, str]:
        return {
            "volatility": self.volatility,
            "spread": self.spread,
            "trend": self.trend,
            "flow": self.flow,
        }


class RegimeClassifier:
    """Keep rolling context to map microstructure state to regime labels."""

    def __init__(self, *, window: int = 200) -> None:
        self._window = max(1, window)
        self._volatility = deque(maxlen=self._window)
        self._spread = deque(maxlen=self._window)
        self._flow = deque(maxlen=self._window)
        self._momentum = deque(maxlen=self._window)

    def observe(
        self,
        realized_volatility: float | None,
        spread_bps: float | None,
        flow_score: float,
        momentum_bps: float | None,
    ) -> RegimeLabel:
        if realized_volatility is not None:
            self._volatility.append(float(realized_volatility))
        if spread_bps is not None:
            self._spread.append(float(spread_bps))
        self._flow.append(float(flow_score))
        if momentum_bps is not None:
            self._momentum.append(float(momentum_bps))

        vol_bucket = self._bucket(self._volatility, realized_volatility, [0.0, 1.0])
        spread_bucket = self._bucket(self._spread, spread_bps, [0.0, 50.0])
        trend_bucket = self._trend_bucket(momentum_bps)
        flow_bucket = self._bucket(self._flow, flow_score, [-0.2, 0.2], labels=("low", "neutral", "high"))

        return RegimeLabel(
            volatility=vol_bucket,
            spread=spread_bucket,
            trend=trend_bucket,
            flow=flow_bucket,
        )

    def _bucket(
        self,
        history: deque[float],
        value: float | None,
        thresholds: list[float],
        labels: tuple[str, str, str] = ("low", "high", "high"),
    ) -> str:
        if value is None:
            return labels[1]
        if len(history) < 30:
            return labels[1]
        arr = np.asarray(history, dtype=float)
        q_low = np.quantile(arr, 0.3)
        q_high = np.quantile(arr, 0.7)
        if value <= q_low:
            return "low"
        if value >= q_high:
            return "high"
        return labels[1]

    def _trend_bucket(self, momentum_bps: float | None) -> str:
        if momentum_bps is None:
            return "range"
        if momentum_bps > 2.0:
            return "trend_up"
        if momentum_bps < -2.0:
            return "trend_down"
        return "range"
