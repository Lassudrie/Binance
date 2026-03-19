from __future__ import annotations

from datetime import UTC, datetime

import duckdb

from ofbot.gateway.recorder import ParquetRecorder
from ofbot.market.trades import parse_book_ticker


def test_recorder_second_empty_flush_does_not_overwrite_existing_parquet(tmp_path) -> None:
    recorder = ParquetRecorder(tmp_path, "run_test")
    event = parse_book_ticker(
        {
            "s": "BTCUSDT",
            "E": 1_700_000_000_000,
            "b": "100.0",
            "B": "2.0",
            "a": "100.1",
            "A": "1.5",
        }
    )
    target = tmp_path / "raw.parquet"

    recorder.record(event, received_at=datetime.now(UTC))
    recorder.flush(target)
    recorder.flush(target)

    conn = duckdb.connect()
    row_count = conn.execute(f"select count(*) from '{target}'").fetchone()[0]
    assert row_count == 1
