from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ofbot.execution.portfolio import PositionSnapshot
from ofbot.market.features import FeatureSnapshot, RegimeLabel


SignalTag = str


@dataclass(slots=True)
class StrategyDecision:
    symbol: str
    side: int
    target_qty: float
    reason: str
    confidence: float = 0.0
    order_type: str = "market"
    max_holding_time_s: int | None = None
    stop_loss_bps: float | None = None
    take_profit_bps: float | None = None
    trailing_stop_bps: float | None = None
    entry_context: dict[str, float | int | str | None] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.entry_context is None:
            self.entry_context = {}


class Strategy(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def decide(
        self,
        snapshot: FeatureSnapshot,
        state: PositionSnapshot,
        *,
        long_only: bool,
    ) -> StrategyDecision | None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self) -> dict[str, float | int | str | bool]:
        raise NotImplementedError
