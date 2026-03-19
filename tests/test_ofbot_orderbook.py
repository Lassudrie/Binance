from __future__ import annotations

from datetime import UTC, datetime

from ofbot.market.orderbook import OrderBookState
from ofbot.market.trades import parse_book_ticker, parse_depth, parse_depth_snapshot


def _event_time_ms(offset_ms: int) -> int:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    return int(base.timestamp() * 1000) + offset_ms


def _book_ticker() -> dict[str, str | int]:
    return {
        "s": "BTCUSDT",
        "E": _event_time_ms(0),
        "b": "100.0",
        "B": "2.0",
        "a": "100.1",
        "A": "1.5",
    }


def test_orderbook_requires_snapshot_then_replays_buffered_depth() -> None:
    book = OrderBookState(symbol="BTCUSDT", require_depth_sync=True, book_ticker_divergence_bps=25.0)
    book.apply_book_ticker(parse_book_ticker(_book_ticker()))

    depth = parse_depth(
        {
            "s": "BTCUSDT",
            "E": _event_time_ms(100),
            "U": 101,
            "u": 102,
            "b": [["100.0", "3.0"]],
            "a": [["100.1", "1.0"]],
        }
    )
    result = book.apply_depth(depth)
    assert result.needs_snapshot is True
    assert book.is_synced is False

    snapshot = parse_depth_snapshot(
        {
            "s": "BTCUSDT",
            "lastUpdateId": 100,
            "bids": [["99.9", "1.0"], ["100.0", "2.0"]],
            "asks": [["100.1", "1.5"], ["100.2", "1.0"]],
        },
        fallback_event_time=depth.event_time,
    )
    replayed = book.apply_depth_snapshot(snapshot)

    assert replayed.applied is True
    assert book.is_synced is True
    assert book.is_tradeable is True
    assert book.bid_price == 100.0
    assert book.bid_qty == 3.0
    assert book.ask_price == 100.1
    assert book.ask_qty == 1.0


def test_orderbook_gap_marks_book_unsynced() -> None:
    book = OrderBookState(symbol="BTCUSDT", require_depth_sync=True, book_ticker_divergence_bps=25.0)
    book.apply_book_ticker(parse_book_ticker(_book_ticker()))
    snapshot = parse_depth_snapshot(
        {
            "s": "BTCUSDT",
            "lastUpdateId": 100,
            "bids": [["100.0", "2.0"]],
            "asks": [["100.1", "1.0"]],
        },
        fallback_event_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    book.apply_depth_snapshot(snapshot)

    gap = parse_depth(
        {
            "s": "BTCUSDT",
            "E": _event_time_ms(100),
            "U": 103,
            "u": 103,
            "b": [["100.0", "2.5"]],
            "a": [["100.1", "1.0"]],
        }
    )
    result = book.apply_depth(gap)

    assert result.needs_snapshot is True
    assert book.is_synced is False
    assert book.last_sync_reason == "depth_gap"


def test_orderbook_detects_divergence_against_book_ticker() -> None:
    book = OrderBookState(symbol="BTCUSDT", require_depth_sync=True, book_ticker_divergence_bps=5.0)
    book.apply_book_ticker(parse_book_ticker(_book_ticker()))

    invalid_snapshot = parse_depth_snapshot(
        {
            "s": "BTCUSDT",
            "lastUpdateId": 100,
            "bids": [["98.0", "1.0"]],
            "asks": [["98.1", "1.0"]],
        },
        fallback_event_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    result = book.apply_depth_snapshot(invalid_snapshot)

    assert result.needs_snapshot is True
    assert book.sync_state == "invalid"
    assert book.is_tradeable is False
