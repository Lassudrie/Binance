from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from ofbot.execution.portfolio import TradeRecord
from ofbot.memory.store import MemoryStore
from ofbot.market.features import FeatureSnapshot

COMPACT_FEATURE_KEYS = (
    "queue_imbalance",
    "cvd_base_1s",
    "cvd_base_1s_z",
    "microprice_drift_bps_1s",
    "momentum_bps_1s",
    "momentum_bps_5s",
    "realized_volatility",
    "regime_volatility",
    "regime_spread",
    "regime_trend",
    "regime_flow",
)


@dataclass(slots=True)
class RunContext:
    run_id: str
    symbol: str
    strategy: str


def build_run_id() -> str:
    now = datetime.now(tz=UTC)
    return f"run_{now.strftime('%Y%m%d_%H%M%S')}"


class JournalWriter:
    def __init__(self, store: MemoryStore, run_id: str | None = None) -> None:
        self.store = store
        self.run_id = run_id or build_run_id()
        self._fill_seq = 0

    def append_signal(
        self,
        *,
        snapshot: FeatureSnapshot,
        strategy_id: str,
        order_intent: str,
        signal_strength: float | None = None,
        rationale: str | None = None,
        params: dict[str, float | int | str | bool | None] | None = None,
    ) -> None:
        payload = {"strength": signal_strength}
        if params:
            payload.update(params)
        self.store.insert_event_journal(
            run_id=self.run_id,
            event_time=snapshot.event_time,
            symbol=snapshot.symbol,
            action="signal",
            strategy=strategy_id,
            params=payload,
            regime=str(snapshot.regime),
            order_intent=order_intent,
            feature_snapshot=_compact_feature_snapshot(snapshot),
            reason=rationale,
            vol_bucket=str(snapshot.features.get("regime_volatility", "")) if snapshot.features else None,
        )

    def append_fill(
        self,
        *,
        snapshot: FeatureSnapshot,
        strategy_id: str,
        fill_qty: float,
        fill_price: float,
        spread_bps: float | None,
        slippage_bps: float,
        fee: float,
        notional: float,
        is_maker: bool,
        order_id: str,
        fill_role: str,
    ) -> None:
        self._fill_seq += 1
        self.store.insert_event_journal(
            run_id=self.run_id,
            event_time=snapshot.event_time,
            symbol=snapshot.symbol,
            action="fill",
            strategy=strategy_id,
            params=None,
            regime=str(snapshot.regime),
            order_intent="fill",
            feature_snapshot=_compact_feature_snapshot(snapshot),
            fill_qty=fill_qty,
            fill_price=fill_price,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            fee=fee,
            notional=notional,
            is_maker=is_maker,
            order_id=order_id,
            fill_seq=self._fill_seq,
            fill_role=fill_role,
        )

    def append_trade_context(self, trade: TradeRecord, *, symbol: str, strategy_id: str, run_id: str | None = None) -> None:
        active_run = run_id or self.run_id
        self.store.insert_trade_context(
            run_id=active_run,
            symbol=symbol,
            strategy=strategy_id,
            regime=trade.regime or "unlabeled",
            context={"source": "paper_broker"},
            entry_time=trade.entry_time,
            exit_time=trade.exit_time,
            direction=trade.side,
            qty=trade.qty,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            realized_pnl=trade.realized_pnl,
            gross_pnl=trade.gross_pnl,
            fees=trade.fees,
            mae_bps=trade.mae_bps,
            mfe_bps=trade.mfe_bps,
            spread_entry_bps=trade.spread_entry_bps,
            spread_exit_bps=trade.spread_exit_bps,
            holding_seconds=trade.holding_seconds,
            entry_fees=trade.entry_fees,
            exit_fees=trade.exit_fees,
            entry_fill_count=trade.entry_fill_count,
            exit_fill_count=trade.exit_fill_count,
            entry_notional=trade.entry_notional,
            exit_notional=trade.exit_notional,
            closed_qty=trade.closed_qty,
        )


def _compact_feature_snapshot(snapshot: FeatureSnapshot) -> dict[str, float | int | str | None]:
    compact: dict[str, float | int | str | None] = {
        "feature_count": len(snapshot.features),
    }
    if snapshot.spread_bps is not None:
        compact["spread_bps"] = snapshot.spread_bps
    if snapshot.mid_price is not None:
        compact["mid_price"] = snapshot.mid_price
    if snapshot.microprice is not None:
        compact["microprice"] = snapshot.microprice
    for key in COMPACT_FEATURE_KEYS:
        value = snapshot.features.get(key)
        if value is None:
            continue
        compact[key] = value
    return compact
