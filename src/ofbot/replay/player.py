from __future__ import annotations

from pathlib import Path

import json
import polars as pl

from ofbot.market.trades import (
    MarketEvent,
    parse_agg_trade,
    parse_book_ticker,
    parse_depth,
    parse_depth_snapshot,
    parse_kline,
)


def iter_replay_events(path: str | Path) -> list[MarketEvent]:
    frame = pl.read_parquet(path)
    return iter_replay_events_from_frame(frame)


def iter_replay_events_from_frame(frame: pl.DataFrame) -> list[MarketEvent]:
    if "sequence" in frame.columns:
        frame = frame.sort(["sequence"])
    else:
        frame = frame.sort(["event_time", "symbol"])
    events: list[MarketEvent] = []
    for row in frame.iter_rows(named=True):
        event = _row_to_event(row)
        if event is not None:
            events.append(event)
    return events


def _row_to_event(row: dict[str, object]) -> MarketEvent | None:
    raw_value = row.get("raw_json")
    if isinstance(raw_value, (bytes, bytearray)):
        raw_value = raw_value.decode("utf-8")
    raw = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
    event_type = str(row["event_type"])
    if event_type == "aggTrade":
        return parse_agg_trade(raw)
    if event_type == "bookTicker":
        fallback_event_time = row.get("received_at") or row.get("event_time")
        return parse_book_ticker(raw, fallback_event_time=fallback_event_time)
    if event_type == "depth":
        return parse_depth(raw)
    if event_type == "depth_snapshot":
        return parse_depth_snapshot(raw, fallback_event_time=row["event_time"])
    if event_type == "kline":
        return parse_kline(raw)
    return None
