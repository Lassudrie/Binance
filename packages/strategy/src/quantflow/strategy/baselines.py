from __future__ import annotations

import random
from typing import Any

from quantflow.data_model.events import BarEvent, PortfolioState, SignalEvent, SignalSide
from quantflow.strategy.base import Strategy


class NoTradeBaselineStrategy(Strategy):
    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        return None

    def reset(self) -> None:
        self.current_target = 0

    def get_params(self) -> dict[str, Any]:
        return {}


class RandomBaselineStrategy(Strategy):
    def __init__(
        self, enter_probability: float = 0.02, flat_probability: float = 0.02, seed: int = 7
    ) -> None:
        super().__init__()
        self.enter_probability = enter_probability
        self.flat_probability = flat_probability
        self.seed = seed
        self._rng = random.Random(seed)

    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        draw = self._rng.random()
        target = self.current_target

        if target == 0:
            if draw < self.enter_probability:
                target = 1
            elif draw > 1.0 - self.enter_probability:
                target = -1
        elif draw < self.flat_probability:
            target = 0

        if target == self.current_target:
            return None

        side = (
            SignalSide.LONG if target > 0 else SignalSide.SHORT if target < 0 else SignalSide.FLAT
        )
        return SignalEvent(
            symbol=event.symbol,
            event_time=event.end_time,
            side=side,
            target_position=target,
            reason="random_baseline",
            context={
                "session": event.features.get("session"),
                "hour_of_day": event.features.get("hour_of_day"),
            },
        )

    def reset(self) -> None:
        self.current_target = 0
        self._rng = random.Random(self.seed)

    def get_params(self) -> dict[str, Any]:
        return {
            "enter_probability": self.enter_probability,
            "flat_probability": self.flat_probability,
            "seed": self.seed,
        }
