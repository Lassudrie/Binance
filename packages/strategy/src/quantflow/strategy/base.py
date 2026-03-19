from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from quantflow.data_model.events import BarEvent, PortfolioState, SignalEvent


class Strategy(ABC):
    def __init__(self) -> None:
        self.current_target = 0

    def on_event(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        signal = self.generate_signal(event, state)
        if signal is not None:
            self.current_target = signal.target_position
        return signal

    @abstractmethod
    def generate_signal(self, event: BarEvent, state: PortfolioState) -> SignalEvent | None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        raise NotImplementedError
