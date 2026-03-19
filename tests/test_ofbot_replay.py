from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl

from ofbot.market.trades import parse_agg_trade, parse_book_ticker, parse_depth, parse_depth_snapshot
from ofbot.replay.eval import run_replay
from tests.ofbot_helpers import build_test_config


def _event_time_ms(base: datetime, offset_seconds: int) -> int:
    return int((base + timedelta(seconds=offset_seconds)).timestamp() * 1000)


def _build_replay_dataset() -> pl.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict] = []
    rows.append(
        {
            "run_id": "integration_run",
            "event_type": "bookTicker",
            "symbol": "BTCUSDT",
            "event_time": base,
            "received_at": base,
            "raw_json": json.dumps(
                parse_book_ticker(
                    {
                        "s": "BTCUSDT",
                        "E": _event_time_ms(base, 0),
                        "b": "100.0",
                        "B": "2.0",
                        "a": "100.2",
                        "A": "1.0",
                    }
                ).raw
            ),
        }
    )

    for i in range(1, 12):
        mid_shift = 0.02 * i
        rows.append(
            {
                "run_id": "integration_run",
                "event_type": "aggTrade",
                "symbol": "BTCUSDT",
                "event_time": base + timedelta(seconds=i),
                "received_at": base + timedelta(seconds=i),
                "raw_json": json.dumps(
                    parse_agg_trade(
                        {
                            "s": "BTCUSDT",
                            "E": _event_time_ms(base, i),
                            "a": 1000 + i,
                            "f": 1000 + i,
                            "l": 1000 + i,
                            "p": f"{100.0 + mid_shift}",
                            "q": "1.0",
                            "m": False,
                        }
                    ).raw
                ),
            }
        )
        if i % 2 == 0:
            rows.append(
                {
                    "run_id": "integration_run",
                    "event_type": "bookTicker",
                    "symbol": "BTCUSDT",
                    "event_time": base + timedelta(seconds=i, milliseconds=100),
                    "received_at": base + timedelta(seconds=i, milliseconds=100),
                    "raw_json": json.dumps(
                        parse_book_ticker(
                            {
                                "s": "BTCUSDT",
                                "E": _event_time_ms(base, i) + 100,
                                "b": f"{100.0 + mid_shift:.6f}",
                                "B": "2.2",
                                "a": f"{100.2 + mid_shift:.6f}",
                                "A": "1.0",
                            }
                        ).raw
                    ),
                }
            )
    return pl.DataFrame(rows)


def _build_depth_replay_dataset() -> pl.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict] = [
        {
            "sequence": 1,
            "run_id": "depth_run",
            "event_type": "bookTicker",
            "symbol": "BTCUSDT",
            "event_time": base,
            "received_at": base,
            "raw_json": json.dumps(
                parse_book_ticker(
                    {
                        "s": "BTCUSDT",
                        "E": _event_time_ms(base, 0),
                        "b": "100.0",
                        "B": "3.0",
                        "a": "100.2",
                        "A": "2.0",
                    }
                ).raw
            ),
        },
        {
            "sequence": 2,
            "run_id": "depth_run",
            "event_type": "depth",
            "symbol": "BTCUSDT",
            "event_time": base + timedelta(milliseconds=50),
            "received_at": base + timedelta(milliseconds=50),
            "raw_json": json.dumps(
                parse_depth(
                    {
                        "s": "BTCUSDT",
                        "E": _event_time_ms(base, 0) + 50,
                        "U": 101,
                        "u": 102,
                        "b": [["100.0", "4.0"]],
                        "a": [["100.2", "2.0"]],
                    }
                ).raw
            ),
        },
        {
            "sequence": 3,
            "run_id": "depth_run",
            "event_type": "depth_snapshot",
            "symbol": "BTCUSDT",
            "event_time": base + timedelta(milliseconds=50),
            "received_at": base + timedelta(milliseconds=50),
            "raw_json": json.dumps(
                parse_depth_snapshot(
                    {
                        "s": "BTCUSDT",
                        "lastUpdateId": 100,
                        "bids": [["99.9", "1.0"], ["100.0", "2.0"]],
                        "asks": [["100.2", "2.5"], ["100.3", "1.0"]],
                    },
                    fallback_event_time=base + timedelta(milliseconds=50),
                ).raw
            ),
        },
    ]
    last_update_id = 102
    prev_bid = 100.0
    prev_ask = 100.2

    for i in range(1, 12):
        mid_shift = 0.03 * i
        next_bid = round(100.0 + mid_shift, 6)
        next_ask = round(100.2 + mid_shift, 6)
        rows.append(
            {
                "sequence": len(rows) + 1,
                "run_id": "depth_run",
                "event_type": "aggTrade",
                "symbol": "BTCUSDT",
                "event_time": base + timedelta(seconds=i),
                "received_at": base + timedelta(seconds=i),
                "raw_json": json.dumps(
                    parse_agg_trade(
                        {
                            "s": "BTCUSDT",
                            "E": _event_time_ms(base, i),
                            "a": 2000 + i,
                            "f": 2000 + i,
                            "l": 2000 + i,
                            "p": f"{100.0 + mid_shift}",
                            "q": "1.0",
                            "m": False,
                        }
                    ).raw
                ),
            }
        )
        rows.append(
            {
                "sequence": len(rows) + 1,
                "run_id": "depth_run",
                "event_type": "bookTicker",
                "symbol": "BTCUSDT",
                "event_time": base + timedelta(seconds=i, milliseconds=100),
                "received_at": base + timedelta(seconds=i, milliseconds=100),
                "raw_json": json.dumps(
                    parse_book_ticker(
                        {
                            "s": "BTCUSDT",
                            "E": _event_time_ms(base, i) + 100,
                            "b": f"{next_bid:.6f}",
                            "B": "3.2",
                            "a": f"{next_ask:.6f}",
                            "A": "2.1",
                        }
                    ).raw
                ),
            }
        )
        rows.append(
            {
                "sequence": len(rows) + 1,
                "run_id": "depth_run",
                "event_type": "depth",
                "symbol": "BTCUSDT",
                "event_time": base + timedelta(seconds=i, milliseconds=150),
                "received_at": base + timedelta(seconds=i, milliseconds=150),
                "raw_json": json.dumps(
                    parse_depth(
                        {
                            "s": "BTCUSDT",
                            "E": _event_time_ms(base, i) + 150,
                            "U": last_update_id + 1,
                            "u": last_update_id + 1,
                            "b": [[f"{prev_bid:.6f}", "0.0"], [f"{next_bid:.6f}", "4.0"]],
                            "a": [[f"{prev_ask:.6f}", "0.0"], [f"{next_ask:.6f}", "2.0"]],
                        }
                    ).raw
                ),
            }
        )
        last_update_id += 1
        prev_bid = next_bid
        prev_ask = next_ask
    return pl.DataFrame(rows)


def test_replay_drives_engine_to_closed_trade(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.features.horizons_sec = [1, 5]
    config.features.zscore_window = 30
    config.execution.default_target_notional_usd = 25.0
    config.execution.min_target_notional_usd = 25.0
    config.execution.max_target_notional_usd = 25.0
    config.strategy_library.continuation.params = {
        "trend_alignment_threshold": 0.0,
        "queue_imbalance_threshold": -1.0,
        "microprice_drift_threshold_bps": -10.0,
        "min_expected_net_edge_bps": -100.0,
        "min_momentum_5s_bps": -100.0,
        "no_trade_z": -100.0,
        "max_holding_time_s": 2,
    }
    config.strategy_library.exhaustion.enabled = False
    config.strategy_library.hybrid.enabled = False
    config.learning.enabled = False

    input_path = tmp_path / "replay_input.parquet"
    _build_replay_dataset().write_parquet(input_path)

    report_path = run_replay(config, input_path)
    assert report_path is not None
    assert Path(report_path).exists()

    conn = duckdb.connect(config.learning.duckdb_path)
    row_count = conn.execute("SELECT COUNT(*) FROM trade_contexts").fetchone()[0]
    assert row_count >= 1


def test_replay_supports_depth_snapshot_events(tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    config.features.horizons_sec = [1, 5]
    config.features.zscore_window = 30
    config.execution.default_target_notional_usd = 25.0
    config.execution.min_target_notional_usd = 25.0
    config.execution.max_target_notional_usd = 25.0
    config.strategy_library.continuation.params = {
        "trend_alignment_threshold": 0.0,
        "queue_imbalance_threshold": -1.0,
        "microprice_drift_threshold_bps": -10.0,
        "min_expected_net_edge_bps": -100.0,
        "min_momentum_5s_bps": -100.0,
        "no_trade_z": -100.0,
        "max_holding_time_s": 2,
    }
    config.strategy_library.exhaustion.enabled = False
    config.strategy_library.hybrid.enabled = False
    config.learning.enabled = False

    input_path = tmp_path / "depth_replay.parquet"
    _build_depth_replay_dataset().write_parquet(input_path)

    report_path = run_replay(config, input_path)
    assert report_path is not None
    assert Path(report_path).exists()

    conn = duckdb.connect(config.learning.duckdb_path)
    trade_count = conn.execute("SELECT COUNT(*) FROM trade_contexts").fetchone()[0]
    fill_count = conn.execute("SELECT COUNT(*) FROM event_journal WHERE action='fill'").fetchone()[0]
    assert trade_count >= 1


def test_replay_bookticker_without_exchange_time_uses_recorded_timestamp(tmp_path) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    input_path = tmp_path / "bookticker_fallback.parquet"
    pl.DataFrame(
        [
            {
                "sequence": 1,
                "run_id": "fallback_run",
                "event_type": "bookTicker",
                "symbol": "BTCUSDT",
                "event_time": base,
                "received_at": base,
                "raw_json": json.dumps(
                    {
                        "s": "BTCUSDT",
                        "b": "100.0",
                        "B": "2.0",
                        "a": "100.2",
                        "A": "1.0",
                    }
                ),
            }
        ]
    ).write_parquet(input_path)

    from ofbot.replay.player import iter_replay_events

    events = iter_replay_events(input_path)

    assert len(events) == 1
    assert events[0].event_type == "bookTicker"
    assert events[0].event_time == base
