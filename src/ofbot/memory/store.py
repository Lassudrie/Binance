from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.path))
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS event_journal (
                id BIGINT PRIMARY KEY,
                run_id VARCHAR,
                event_time TIMESTAMP,
                symbol VARCHAR,
                action VARCHAR,
                strategy VARCHAR,
                params JSON,
                regime VARCHAR,
                order_intent VARCHAR,
                feature_snapshot JSON,
                fill_qty DOUBLE,
                fill_price DOUBLE,
                spread_bps DOUBLE,
                slippage_bps DOUBLE,
                realized_pnl DOUBLE,
                mae_bps DOUBLE,
                mfe_bps DOUBLE,
                holding_seconds DOUBLE,
                vol_bucket VARCHAR,
                error_tags VARCHAR,
                reason VARCHAR
            )
            """
        )
        self._ensure_columns(
            "event_journal",
            {
                "fee": "DOUBLE",
                "notional": "DOUBLE",
                "is_maker": "BOOLEAN",
                "order_id": "VARCHAR",
                "fill_seq": "INTEGER",
                "fill_role": "VARCHAR",
            },
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_contexts (
                id BIGINT PRIMARY KEY,
                run_id VARCHAR,
                symbol VARCHAR,
                strategy VARCHAR,
                regime VARCHAR,
                context JSON,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                direction INT,
                qty DOUBLE,
                entry_price DOUBLE,
                exit_price DOUBLE,
                realized_pnl DOUBLE,
                gross_pnl DOUBLE,
                fees DOUBLE,
                mae_bps DOUBLE,
                mfe_bps DOUBLE,
                spread_entry_bps DOUBLE,
                spread_exit_bps DOUBLE,
                holding_seconds DOUBLE,
                error_tags VARCHAR
            )
            """
        )
        self._ensure_columns(
            "trade_contexts",
            {
                "entry_fees": "DOUBLE",
                "exit_fees": "DOUBLE",
                "entry_fill_count": "INTEGER",
                "exit_fill_count": "INTEGER",
                "entry_notional": "DOUBLE",
                "exit_notional": "DOUBLE",
                "closed_qty": "DOUBLE",
            },
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS policy_performance (
                id BIGINT PRIMARY KEY,
                run_id VARCHAR,
                strategy VARCHAR,
                symbol VARCHAR,
                regime VARCHAR,
                sample_count INTEGER,
                mean_pnl DOUBLE,
                variance_pnl DOUBLE,
                last_updated TIMESTAMP
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS policy_state (
                id BIGINT PRIMARY KEY,
                run_id VARCHAR,
                symbol VARCHAR,
                regime VARCHAR,
                incumbent_strategy VARCHAR,
                challenger_strategy VARCHAR,
                incumbent_mean_pnl DOUBLE,
                challenger_mean_pnl DOUBLE,
                promotion_reason VARCHAR,
                updated_at TIMESTAMP
            )
            """
        )

    def _ensure_columns(self, table: str, columns: dict[str, str]) -> None:
        existing = {
            str(row[1])
            for row in self._conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        }
        for column_name, column_type in columns.items():
            if column_name in existing:
                continue
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_name} {column_type}")

    def next_id(self, table: str) -> int:
        value = self._conn.execute(f"SELECT COALESCE(MAX(id),0)+1 FROM {table}").fetchone()[0]
        return int(value)

    def insert_event_journal(
        self,
        *,
        run_id: str,
        event_time: datetime,
        symbol: str,
        action: str,
        strategy: str,
        params: dict[str, Any] | None = None,
        regime: str | None = None,
        order_intent: str | None = None,
        feature_snapshot: dict[str, Any] | None = None,
        fill_qty: float | None = None,
        fill_price: float | None = None,
        spread_bps: float | None = None,
        slippage_bps: float | None = None,
        realized_pnl: float | None = None,
        mae_bps: float | None = None,
        mfe_bps: float | None = None,
        holding_seconds: float | None = None,
        vol_bucket: str | None = None,
        error_tags: str | None = None,
        reason: str | None = None,
        fee: float | None = None,
        notional: float | None = None,
        is_maker: bool | None = None,
        order_id: str | None = None,
        fill_seq: int | None = None,
        fill_role: str | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO event_journal (
                id,
                run_id,
                event_time,
                symbol,
                action,
                strategy,
                params,
                regime,
                order_intent,
                feature_snapshot,
                fill_qty,
                fill_price,
                spread_bps,
                slippage_bps,
                realized_pnl,
                mae_bps,
                mfe_bps,
                holding_seconds,
                vol_bucket,
                error_tags,
                reason,
                fee,
                notional,
                is_maker,
                order_id,
                fill_seq,
                fill_role
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                self.next_id("event_journal"),
                run_id,
                _to_naive_datetime(event_time),
                symbol,
                action,
                strategy,
                _dump_json(params or {}),
                regime,
                order_intent,
                _dump_json(feature_snapshot or {}),
                fill_qty,
                fill_price,
                spread_bps,
                slippage_bps,
                realized_pnl,
                mae_bps,
                mfe_bps,
                holding_seconds,
                vol_bucket,
                error_tags,
                reason,
                fee,
                notional,
                is_maker,
                order_id,
                fill_seq,
                fill_role,
            ],
        )

    def insert_trade_context(
        self,
        *,
        run_id: str,
        symbol: str,
        strategy: str,
        regime: str,
        context: dict[str, Any] | None,
        entry_time: datetime,
        exit_time: datetime,
        direction: int,
        qty: float,
        entry_price: float,
        exit_price: float,
        realized_pnl: float,
        gross_pnl: float,
        fees: float,
        mae_bps: float,
        mfe_bps: float,
        spread_entry_bps: float | None,
        spread_exit_bps: float | None,
        holding_seconds: float,
        error_tags: str | None = None,
        entry_fees: float | None = None,
        exit_fees: float | None = None,
        entry_fill_count: int | None = None,
        exit_fill_count: int | None = None,
        entry_notional: float | None = None,
        exit_notional: float | None = None,
        closed_qty: float | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO trade_contexts (
                id,
                run_id,
                symbol,
                strategy,
                regime,
                context,
                entry_time,
                exit_time,
                direction,
                qty,
                entry_price,
                exit_price,
                realized_pnl,
                gross_pnl,
                fees,
                mae_bps,
                mfe_bps,
                spread_entry_bps,
                spread_exit_bps,
                holding_seconds,
                error_tags,
                entry_fees,
                exit_fees,
                entry_fill_count,
                exit_fill_count,
                entry_notional,
                exit_notional,
                closed_qty
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                self.next_id("trade_contexts"),
                run_id,
                symbol,
                strategy,
                regime,
                _dump_json(context or {}),
                _to_naive_datetime(entry_time),
                _to_naive_datetime(exit_time),
                direction,
                qty,
                entry_price,
                exit_price,
                realized_pnl,
                gross_pnl,
                fees,
                mae_bps,
                mfe_bps,
                spread_entry_bps,
                spread_exit_bps,
                holding_seconds,
                error_tags,
                entry_fees,
                exit_fees,
                entry_fill_count,
                exit_fill_count,
                entry_notional,
                exit_notional,
                closed_qty,
            ],
        )

    def read_performance(
        self,
        run_id: str,
        strategy: str,
        symbol: str | None = None,
        regime: str | None = None,
    ) -> list[tuple[str, str, int, float, datetime, datetime]]:
        conditions = ["run_id = ?", "strategy = ?"]
        params: list[Any] = [run_id, strategy]
        if symbol is not None:
            conditions.append("symbol = ?")
            params.append(symbol)
        if regime is not None:
            conditions.append("regime = ?")
            params.append(regime)
        q = f"""
            SELECT run_id, symbol, direction, realized_pnl, entry_time, exit_time
            FROM trade_contexts
            WHERE {' AND '.join(conditions)}
            ORDER BY exit_time ASC
        """
        return self._conn.execute(q, params).fetchall()

    def read_all_trades(self, run_id: str) -> list[tuple]:
        return self._conn.execute(
            """
            SELECT *
            FROM trade_contexts
            WHERE run_id = ?
            ORDER BY exit_time ASC
            """,
            [run_id],
        ).fetchall()

    def read_trades_by_date(self, date_value: str) -> list[tuple]:
        return self._conn.execute(
            """
            SELECT *
            FROM trade_contexts
            WHERE CAST(entry_time AS DATE) = CAST(? AS DATE)
            ORDER BY exit_time ASC
            """,
            [date_value],
        ).fetchall()

    def update_policy_stats(
        self,
        run_id: str,
        strategy: str,
        symbol: str,
        regime: str,
        sample_count: int,
        mean: float,
        variance: float,
    ) -> None:
        existing = self._conn.execute(
            "SELECT COUNT(*) FROM policy_performance WHERE run_id=? AND strategy=? AND symbol=? AND regime=?",
            [run_id, strategy, symbol, regime],
        ).fetchone()[0]
        if existing:
            self._conn.execute(
                """
                UPDATE policy_performance
                SET sample_count=?, mean_pnl=?, variance_pnl=?, last_updated=?
                WHERE run_id=? AND strategy=? AND symbol=? AND regime=?
                """,
                [sample_count, mean, variance, _to_naive_datetime(datetime.now(UTC)), run_id, strategy, symbol, regime],
            )
            return
        self._conn.execute(
            """
            INSERT INTO policy_performance
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            [
                self.next_id("policy_performance"),
                run_id,
                strategy,
                symbol,
                regime,
                sample_count,
                mean,
                variance,
                _to_naive_datetime(datetime.now(UTC)),
            ],
        )

    def read_policy_stats(self, run_id: str, symbol: str) -> list[tuple[str, str, int, float, float]]:
        return self._conn.execute(
            """
            SELECT strategy, regime, sample_count, mean_pnl, variance_pnl
            FROM policy_performance
            WHERE run_id = ? AND symbol = ?
            ORDER BY mean_pnl DESC
            """,
            [run_id, symbol],
        ).fetchall()

    def upsert_policy_state(
        self,
        *,
        run_id: str,
        symbol: str,
        regime: str,
        incumbent: str,
        challenger: str,
        incumbent_mean: float,
        challenger_mean: float,
        reason: str,
    ) -> None:
        existing = self._conn.execute(
            "SELECT COUNT(*) FROM policy_state WHERE run_id=? AND symbol=? AND regime=?",
            [run_id, symbol, regime],
        ).fetchone()[0]
        if existing:
            self._conn.execute(
                """
                UPDATE policy_state
                SET incumbent_strategy=?, challenger_strategy=?, incumbent_mean_pnl=?, challenger_mean_pnl=?, promotion_reason=?, updated_at=?
                WHERE run_id=? AND symbol=? AND regime=?
                """,
                [
                    incumbent,
                    challenger,
                    incumbent_mean,
                    challenger_mean,
                    reason,
                    _to_naive_datetime(datetime.now(UTC)),
                    run_id,
                    symbol,
                    regime,
                ],
            )
            return
        self._conn.execute(
            """
            INSERT INTO policy_state
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            [
                self.next_id("policy_state"),
                run_id,
                symbol,
                regime,
                incumbent,
                challenger,
                incumbent_mean,
                challenger_mean,
                reason,
                _to_naive_datetime(datetime.now(UTC)),
            ],
        )

    def get_policy_state(self, run_id: str, symbol: str) -> dict[str, tuple[str, str, float]]:
        rows = self._conn.execute(
            """
            SELECT regime, incumbent_strategy, challenger_strategy, incumbent_mean_pnl
            FROM policy_state
            WHERE run_id = ? AND symbol = ?
            """,
            [run_id, symbol],
        ).fetchall()
        out: dict[str, tuple[str, str, float]] = {}
        for regime, incumbent, challenger, mean_pnl in rows:
            out[str(regime)] = (str(incumbent), str(challenger), float(mean_pnl))
        return out

    def run_query_df(self, query: str) -> Any:
        return self._conn.execute(query).fetch_df()


def _dump_json(payload: dict[str, Any] | None) -> str:
    import json

    return json.dumps(payload or {}, ensure_ascii=False)


def _to_naive_datetime(value: datetime) -> datetime:
    return value.replace(tzinfo=None) if value.tzinfo is not None else value
