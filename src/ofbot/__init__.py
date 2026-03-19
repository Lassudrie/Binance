"""ofbot: paper-only Binance live execution framework."""

from __future__ import annotations

from pathlib import Path


__all__ = ["PROJECT_ROOT"]


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
