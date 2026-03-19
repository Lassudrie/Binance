from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from ofbot.market.trades import MarketEvent


class ParquetRecorder:
    """Append raw events during live operation and persist as Parquet."""

    def __init__(self, raw_dir: Path, session_run_id: str) -> None:
        self.raw_dir = raw_dir
        self.session_run_id = session_run_id
        self._rows: list[dict[str, Any]] = []
        self._sequence = 0

    def record(self, event: MarketEvent, *, received_at: datetime | None = None) -> None:
        self._sequence += 1
        self._rows.append(
            {
                "sequence": self._sequence,
                "run_id": self.session_run_id,
                "event_type": event.event_type,
                "symbol": event.symbol,
                "event_time": event.event_time,
                "received_at": received_at or event.event_time,
                "raw_json": json.dumps(event.raw),
            }
        )

    def flush(self, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not self._rows and target.exists():
            return target
        if self._rows:
            frame = pl.DataFrame(self._rows)
            frame.write_parquet(target)
        else:
            frame = pl.DataFrame(
                {
                    "sequence": pl.Series("sequence", [], dtype=pl.Int64),
                    "run_id": pl.Series("run_id", [], dtype=pl.Utf8),
                    "event_type": pl.Series("event_type", [], dtype=pl.Utf8),
                    "symbol": pl.Series("symbol", [], dtype=pl.Utf8),
                    "event_time": pl.Series("event_time", [], dtype=pl.Datetime("us", "UTC")),
                    "received_at": pl.Series("received_at", [], dtype=pl.Datetime("us", "UTC")),
                    "raw_json": pl.Series("raw_json", [], dtype=pl.Utf8),
                }
                )
            frame.write_parquet(target)
        self._rows.clear()
        return target
