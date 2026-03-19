from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def trade_contexts_to_df(store) -> pd.DataFrame:
    return pd.DataFrame(store.run_query_df("SELECT * FROM trade_contexts"))


def event_journal_to_df(store) -> pd.DataFrame:
    return pd.DataFrame(store.run_query_df("SELECT * FROM event_journal"))


def write_run_artifacts(store, run_id: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades = store._conn.execute(
        "SELECT * FROM trade_contexts WHERE run_id=?", [run_id]
    ).fetch_df()
    events = store._conn.execute(
        "SELECT * FROM event_journal WHERE run_id=?", [run_id]
    ).fetch_df()
    trades_csv = output_dir / "trades.csv"
    events_csv = output_dir / "events.csv"
    if not trades.empty:
        trades.to_csv(trades_csv, index=False)
    if not events.empty:
        events.to_csv(events_csv, index=False)
    return {"trades_csv": str(trades_csv), "events_csv": str(events_csv)}


def write_daily_artifacts(store, date_value: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades = store._conn.execute(
        "SELECT * FROM trade_contexts WHERE CAST(entry_time AS DATE) = CAST(? AS DATE)",
        [date_value],
    ).fetch_df()
    events = store._conn.execute(
        "SELECT * FROM event_journal WHERE CAST(event_time AS DATE) = CAST(? AS DATE)",
        [date_value],
    ).fetch_df()
    trades_csv = output_dir / "trades.csv"
    events_csv = output_dir / "events.csv"
    if not trades.empty:
        trades.to_csv(trades_csv, index=False)
    if not events.empty:
        events.to_csv(events_csv, index=False)
    return {"trades_csv": str(trades_csv), "events_csv": str(events_csv)}
