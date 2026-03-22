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
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_reviews (
                id BIGINT PRIMARY KEY,
                run_id VARCHAR UNIQUE,
                verdict VARCHAR,
                config_snapshot_path VARCHAR,
                config_snapshot_hash VARCHAR,
                raw_input_path VARCHAR,
                report_dir VARCHAR,
                summary JSON,
                notes JSON,
                updated_at TIMESTAMP
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lessons_learned (
                id BIGINT PRIMARY KEY,
                run_id VARCHAR,
                lesson_key VARCHAR,
                scope VARCHAR,
                lesson_type VARCHAR,
                strategy VARCHAR,
                regime VARCHAR,
                evidence JSON,
                recommended_action JSON,
                created_at TIMESTAMP
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS manual_notes (
                id BIGINT PRIMARY KEY,
                run_id VARCHAR,
                note_scope VARCHAR,
                trade_context_id BIGINT,
                tag VARCHAR,
                note_text VARCHAR,
                created_at TIMESTAMP
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_candidates (
                id BIGINT PRIMARY KEY,
                candidate_id VARCHAR UNIQUE,
                source_run_id VARCHAR,
                strategy VARCHAR,
                archetype VARCHAR,
                status VARCHAR,
                base_config_hash VARCHAR,
                overlay_yaml_path VARCHAR,
                params JSON,
                rationale JSON,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candidate_validations (
                id BIGINT PRIMARY KEY,
                candidate_id VARCHAR,
                validation_run_id VARCHAR,
                baseline_net_pnl DOUBLE,
                candidate_net_pnl DOUBLE,
                baseline_trade_count INTEGER,
                candidate_trade_count INTEGER,
                baseline_max_drawdown DOUBLE,
                candidate_max_drawdown DOUBLE,
                verdict VARCHAR,
                summary JSON,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS offline_research_candidates (
                id BIGINT PRIMARY KEY,
                candidate_id VARCHAR UNIQUE,
                strategy_name VARCHAR,
                bar_size VARCHAR,
                execution_mode VARCHAR,
                status VARCHAR,
                params JSON,
                dataset_window VARCHAR,
                config_hash VARCHAR,
                report_dir VARCHAR,
                summary JSON,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candidate_rollouts (
                id BIGINT PRIMARY KEY,
                candidate_id VARCHAR,
                run_id VARCHAR,
                rollout_stage VARCHAR,
                net_pnl DOUBLE,
                trade_count INTEGER,
                critical_risk_count INTEGER,
                infrastructure_risk_count INTEGER,
                active_config_path VARCHAR,
                summary JSON,
                created_at TIMESTAMP,
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

    def upsert_run_review(
        self,
        *,
        run_id: str,
        verdict: str,
        config_snapshot_path: str | None,
        config_snapshot_hash: str | None,
        raw_input_path: str | None,
        report_dir: str | None,
        summary: dict[str, Any] | None,
        notes: dict[str, Any] | None = None,
    ) -> None:
        existing = self._conn.execute(
            "SELECT COUNT(*) FROM run_reviews WHERE run_id=?",
            [run_id],
        ).fetchone()[0]
        values = [
            verdict,
            config_snapshot_path,
            config_snapshot_hash,
            raw_input_path,
            report_dir,
            _dump_json(summary or {}),
            _dump_json(notes or {}),
            _to_naive_datetime(datetime.now(UTC)),
            run_id,
        ]
        if existing:
            self._conn.execute(
                """
                UPDATE run_reviews
                SET verdict=?, config_snapshot_path=?, config_snapshot_hash=?, raw_input_path=?, report_dir=?, summary=?, notes=?, updated_at=?
                WHERE run_id=?
                """,
                values,
            )
            return
        self._conn.execute(
            """
            INSERT INTO run_reviews
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            [
                self.next_id("run_reviews"),
                run_id,
                verdict,
                config_snapshot_path,
                config_snapshot_hash,
                raw_input_path,
                report_dir,
                _dump_json(summary or {}),
                _dump_json(notes or {}),
                _to_naive_datetime(datetime.now(UTC)),
            ],
        )

    def delete_lessons_for_run(self, run_id: str) -> None:
        self._conn.execute("DELETE FROM lessons_learned WHERE run_id=?", [run_id])

    def insert_lesson_learned(
        self,
        *,
        run_id: str,
        lesson_key: str,
        scope: str,
        lesson_type: str,
        strategy: str | None,
        regime: str | None,
        evidence: dict[str, Any] | None,
        recommended_action: dict[str, Any] | None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO lessons_learned
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            [
                self.next_id("lessons_learned"),
                run_id,
                lesson_key,
                scope,
                lesson_type,
                strategy,
                regime,
                _dump_json(evidence or {}),
                _dump_json(recommended_action or {}),
                _to_naive_datetime(datetime.now(UTC)),
            ],
        )

    def insert_manual_note(
        self,
        *,
        run_id: str,
        note_scope: str,
        tag: str,
        note_text: str,
        trade_context_id: int | None = None,
    ) -> int:
        note_id = self.next_id("manual_notes")
        self._conn.execute(
            """
            INSERT INTO manual_notes
            VALUES (?,?,?,?,?,?,?)
            """,
            [
                note_id,
                run_id,
                note_scope,
                trade_context_id,
                tag,
                note_text,
                _to_naive_datetime(datetime.now(UTC)),
            ],
        )
        return note_id

    def get_trade_context(self, trade_context_id: int) -> tuple[Any, ...] | None:
        row = self._conn.execute(
            "SELECT * FROM trade_contexts WHERE id=?",
            [trade_context_id],
        ).fetchone()
        return tuple(row) if row is not None else None

    def upsert_learning_candidate(
        self,
        *,
        candidate_id: str,
        source_run_id: str,
        strategy: str,
        archetype: str,
        status: str,
        base_config_hash: str | None,
        overlay_yaml_path: str,
        params: dict[str, Any] | None,
        rationale: dict[str, Any] | None,
    ) -> None:
        existing = self._conn.execute(
            "SELECT COUNT(*) FROM learning_candidates WHERE candidate_id=?",
            [candidate_id],
        ).fetchone()[0]
        now = _to_naive_datetime(datetime.now(UTC))
        if existing:
            self._conn.execute(
                """
                UPDATE learning_candidates
                SET source_run_id=?, strategy=?, archetype=?, status=?, base_config_hash=?, overlay_yaml_path=?, params=?, rationale=?, updated_at=?
                WHERE candidate_id=?
                """,
                [
                    source_run_id,
                    strategy,
                    archetype,
                    status,
                    base_config_hash,
                    overlay_yaml_path,
                    _dump_json(params or {}),
                    _dump_json(rationale or {}),
                    now,
                    candidate_id,
                ],
            )
            return
        self._conn.execute(
            """
            INSERT INTO learning_candidates
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                self.next_id("learning_candidates"),
                candidate_id,
                source_run_id,
                strategy,
                archetype,
                status,
                base_config_hash,
                overlay_yaml_path,
                _dump_json(params or {}),
                _dump_json(rationale or {}),
                now,
                now,
            ],
        )

    def update_learning_candidate_status(self, candidate_id: str, status: str) -> None:
        self._conn.execute(
            """
            UPDATE learning_candidates
            SET status=?, updated_at=?
            WHERE candidate_id=?
            """,
            [status, _to_naive_datetime(datetime.now(UTC)), candidate_id],
        )

    def upsert_candidate_validation(
        self,
        *,
        candidate_id: str,
        validation_run_id: str,
        baseline_net_pnl: float,
        candidate_net_pnl: float,
        baseline_trade_count: int,
        candidate_trade_count: int,
        baseline_max_drawdown: float,
        candidate_max_drawdown: float,
        verdict: str,
        summary: dict[str, Any] | None,
    ) -> None:
        existing = self._conn.execute(
            """
            SELECT COUNT(*) FROM candidate_validations
            WHERE candidate_id=? AND validation_run_id=?
            """,
            [candidate_id, validation_run_id],
        ).fetchone()[0]
        now = _to_naive_datetime(datetime.now(UTC))
        values = [
            baseline_net_pnl,
            candidate_net_pnl,
            baseline_trade_count,
            candidate_trade_count,
            baseline_max_drawdown,
            candidate_max_drawdown,
            verdict,
            _dump_json(summary or {}),
            now,
            candidate_id,
            validation_run_id,
        ]
        if existing:
            self._conn.execute(
                """
                UPDATE candidate_validations
                SET baseline_net_pnl=?, candidate_net_pnl=?, baseline_trade_count=?, candidate_trade_count=?, baseline_max_drawdown=?, candidate_max_drawdown=?, verdict=?, summary=?, updated_at=?
                WHERE candidate_id=? AND validation_run_id=?
                """,
                values,
            )
            return
        self._conn.execute(
            """
            INSERT INTO candidate_validations
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                self.next_id("candidate_validations"),
                candidate_id,
                validation_run_id,
                baseline_net_pnl,
                candidate_net_pnl,
                baseline_trade_count,
                candidate_trade_count,
                baseline_max_drawdown,
                candidate_max_drawdown,
                verdict,
                _dump_json(summary or {}),
                now,
                now,
            ],
        )

    def upsert_offline_research_candidate(
        self,
        *,
        candidate_id: str,
        strategy_name: str,
        bar_size: str,
        execution_mode: str,
        status: str,
        params: dict[str, Any] | None,
        dataset_window: str,
        config_hash: str | None,
        report_dir: str | None,
        summary: dict[str, Any] | None,
    ) -> None:
        existing = self._conn.execute(
            "SELECT COUNT(*) FROM offline_research_candidates WHERE candidate_id=?",
            [candidate_id],
        ).fetchone()[0]
        now = _to_naive_datetime(datetime.now(UTC))
        values = [
            strategy_name,
            bar_size,
            execution_mode,
            status,
            _dump_json(params or {}),
            dataset_window,
            config_hash,
            report_dir,
            _dump_json(summary or {}),
            now,
            candidate_id,
        ]
        if existing:
            self._conn.execute(
                """
                UPDATE offline_research_candidates
                SET strategy_name=?, bar_size=?, execution_mode=?, status=?, params=?, dataset_window=?, config_hash=?, report_dir=?, summary=?, updated_at=?
                WHERE candidate_id=?
                """,
                values,
            )
            return
        self._conn.execute(
            """
            INSERT INTO offline_research_candidates
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                self.next_id("offline_research_candidates"),
                candidate_id,
                strategy_name,
                bar_size,
                execution_mode,
                status,
                _dump_json(params or {}),
                dataset_window,
                config_hash,
                report_dir,
                _dump_json(summary or {}),
                now,
                now,
            ],
        )

    def list_offline_research_candidates(self, *, limit: int = 20) -> list[dict[str, Any]]:
        frame = self.run_query_df(
            """
            SELECT *
            FROM offline_research_candidates
            ORDER BY updated_at DESC, candidate_id ASC
            LIMIT ?
            """,
            [limit],
        )
        return [normalize_record(row) for row in frame.to_dict(orient="records")]

    def upsert_candidate_rollout(
        self,
        *,
        candidate_id: str,
        run_id: str,
        rollout_stage: str,
        net_pnl: float,
        trade_count: int,
        critical_risk_count: int,
        infrastructure_risk_count: int,
        active_config_path: str | None,
        summary: dict[str, Any] | None,
    ) -> None:
        existing = self._conn.execute(
            """
            SELECT COUNT(*) FROM candidate_rollouts
            WHERE candidate_id=? AND run_id=?
            """,
            [candidate_id, run_id],
        ).fetchone()[0]
        now = _to_naive_datetime(datetime.now(UTC))
        values = [
            rollout_stage,
            net_pnl,
            trade_count,
            critical_risk_count,
            infrastructure_risk_count,
            active_config_path,
            _dump_json(summary or {}),
            now,
            candidate_id,
            run_id,
        ]
        if existing:
            self._conn.execute(
                """
                UPDATE candidate_rollouts
                SET rollout_stage=?, net_pnl=?, trade_count=?, critical_risk_count=?, infrastructure_risk_count=?, active_config_path=?, summary=?, updated_at=?
                WHERE candidate_id=? AND run_id=?
                """,
                values,
            )
            return
        self._conn.execute(
            """
            INSERT INTO candidate_rollouts
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                self.next_id("candidate_rollouts"),
                candidate_id,
                run_id,
                rollout_stage,
                net_pnl,
                trade_count,
                critical_risk_count,
                infrastructure_risk_count,
                active_config_path,
                _dump_json(summary or {}),
                now,
                now,
            ],
        )

    def get_candidate_rollouts(self, candidate_id: str) -> list[dict[str, Any]]:
        frame = self.run_query_df(
            """
            SELECT *
            FROM candidate_rollouts
            WHERE candidate_id=?
            ORDER BY created_at ASC, run_id ASC
            """,
            [candidate_id],
        )
        return [normalize_record(row) for row in frame.to_dict(orient="records")]

    def run_query_df(self, query: str, params: list[Any] | None = None) -> Any:
        return self._conn.execute(query, params or []).fetch_df()


def _dump_json(payload: dict[str, Any] | None) -> str:
    import json

    return json.dumps(payload or {}, ensure_ascii=False)


def normalize_record(payload: dict[str, Any]) -> dict[str, Any]:
    import json

    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    normalized[key] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    pass
        normalized[key] = value
    return normalized


def _to_naive_datetime(value: datetime) -> datetime:
    return value.replace(tzinfo=None) if value.tzinfo is not None else value
