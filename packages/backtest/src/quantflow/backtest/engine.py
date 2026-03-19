from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl
from quantflow.backtest.brackets import BracketManager
from quantflow.backtest.metrics import compute_summary_metrics
from quantflow.backtest.portfolio import Portfolio, TradeRecord
from quantflow.backtest.risk import RiskManager
from quantflow.backtest.transforms import SignalToOrderTranslator
from quantflow.core.config import AppConfig
from quantflow.data_model.events import BarEvent, FillEvent, PortfolioState, SignalEvent
from quantflow.execution.simulator import ExecutionSimulator
from quantflow.strategy.base import Strategy

CORE_BAR_COLUMNS = {
    "symbol",
    "bar_start",
    "bar_end",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
}


@dataclass(slots=True)
class BacktestResult:
    strategy_name: str
    metrics: dict[str, Any]
    equity_curve: list[dict[str, float | str]]
    trades: list[TradeRecord]
    fills: list[FillEvent]


def _normalize_feature_value(value: Any) -> float | int | str | None:
    if value is None:
        return None
    if isinstance(value, (float, int, str)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def bars_to_events(features: pl.DataFrame) -> list[BarEvent]:
    events: list[BarEvent] = []
    ordered = features.sort(["bar_end", "symbol"])
    for row in ordered.iter_rows(named=True):
        feature_map = {
            key: _normalize_feature_value(value)
            for key, value in row.items()
            if key not in CORE_BAR_COLUMNS
        }
        events.append(
            BarEvent(
                symbol=str(row["symbol"]),
                start_time=row["bar_start"],
                end_time=row["bar_end"],
                open_price=float(row["open"]),
                high_price=float(row["high"]),
                low_price=float(row["low"]),
                close_price=float(row["close"]),
                volume=float(row["volume"]),
                trade_count=int(row["trade_count"]),
                features=feature_map,
            )
        )
    return events


def _infer_bar_seconds(events: list[BarEvent]) -> float:
    unique_times = sorted({event.end_time for event in events})
    if len(unique_times) < 2:
        return 1.0
    return max((unique_times[1] - unique_times[0]).total_seconds(), 1.0)


def _portfolio_target(state: PortfolioState, symbol: str) -> int:
    position = state.positions.get(symbol)
    if position is None or abs(position.quantity) < 1e-12:
        return 0
    return 1 if position.quantity > 0.0 else -1


def _strategy_for_symbol(
    strategies: dict[str, Strategy],
    template: Strategy,
    symbol: str,
) -> Strategy:
    strategy = strategies.get(symbol)
    if strategy is None:
        strategy = deepcopy(template)
        strategy.reset()
        strategies[symbol] = strategy
    return strategy


def _group_event_batches(events: list[BarEvent]) -> list[list[BarEvent]]:
    batches: list[list[BarEvent]] = []
    current_batch: list[BarEvent] = []
    current_time = None
    for event in events:
        if current_time is None or event.end_time == current_time:
            current_batch.append(event)
            current_time = event.end_time
            continue
        batches.append(current_batch)
        current_batch = [event]
        current_time = event.end_time
    if current_batch:
        batches.append(current_batch)
    return batches


def _open_position_count(state: PortfolioState) -> int:
    return sum(1 for position in state.positions.values() if abs(position.quantity) > 1e-12)


def _daily_realized_pnl(trades: list[TradeRecord], trading_date: date) -> float:
    return sum(trade.net_pnl for trade in trades if trade.exit_time.date() == trading_date)


def _atr_ratio(signal: SignalEvent) -> float:
    value = signal.context.get("atr_ratio")
    if isinstance(value, (int, float)):
        return float(value)
    return float("-inf")


def _filter_entry_signals(
    signals: list[tuple[BarEvent, SignalEvent]],
    state: PortfolioState,
    config: AppConfig,
    *,
    trading_date: date,
    day_start_equity: float | None,
    closed_trades: list[TradeRecord],
) -> list[tuple[BarEvent, SignalEvent]]:
    if not signals:
        return []

    open_positions = _open_position_count(state)
    available_slots = max(config.risk.max_open_positions - open_positions, 0)
    realized_today = _daily_realized_pnl(closed_trades, trading_date)
    daily_limit_amount = (
        day_start_equity * config.risk.risk_per_trade_fraction * config.risk.daily_loss_limit_r
        if day_start_equity is not None
        and config.risk.risk_per_trade_fraction > 0.0
        and config.risk.daily_loss_limit_r > 0.0
        else None
    )
    entries_blocked = daily_limit_amount is not None and realized_today <= -daily_limit_amount

    passthrough: list[tuple[BarEvent, SignalEvent]] = []
    entry_candidates: list[tuple[BarEvent, SignalEvent]] = []

    for event, signal in signals:
        current_qty = (
            state.positions.get(signal.symbol).quantity
            if signal.symbol in state.positions
            else 0.0
        )
        is_new_entry = abs(current_qty) < 1e-12 and signal.target_position != 0
        if not is_new_entry:
            passthrough.append((event, signal))
            continue
        if entries_blocked:
            continue
        entry_candidates.append((event, signal))

    if available_slots <= 0:
        return passthrough

    entry_candidates.sort(key=lambda item: (-_atr_ratio(item[1]), item[1].symbol))
    return passthrough + entry_candidates[:available_slots]


def run_backtest(features: pl.DataFrame, strategy: Strategy, config: AppConfig) -> BacktestResult:
    strategy.reset()
    events = bars_to_events(features)
    if not events:
        raise ValueError("Feature dataset produced no events")

    execution = ExecutionSimulator(config.execution)
    translator = SignalToOrderTranslator(
        position_size=config.strategy.position_size,
        allow_short=config.portfolio.allow_short,
        risk_per_trade_fraction=config.risk.risk_per_trade_fraction,
        use_limit_orders=config.execution.default_order_type == "limit",
        limit_offset_bps=config.execution.limit_offset_bps,
    )
    portfolio = Portfolio(initial_cash=config.portfolio.initial_cash)
    risk_manager = RiskManager(config.risk)
    bracket_manager = BracketManager()
    execution.reset()
    risk_manager.reset()
    bracket_manager.reset()

    symbol_strategies: dict[str, Strategy] = {}
    symbol_indices: dict[str, int] = {}
    last_marks: dict[str, float] = {}
    final_state: PortfolioState | None = None
    active_day: date | None = None
    day_start_equity: float | None = None

    for batch in _group_event_batches(events):
        for event in batch:
            current_index = symbol_indices.get(event.symbol, 0)
            fills = execution.process_bar(event, current_index)
            for fill in fills:
                portfolio.apply_fill(fill)
                bracket_manager.sync_after_fill(
                    fill,
                    portfolio.positions.get(fill.symbol),
                    current_index,
                )

        for event in batch:
            current_index = symbol_indices.get(event.symbol, 0)
            portfolio.update_trade_excursions(event.symbol, event.high_price, event.low_price)
            bracket_fill = bracket_manager.evaluate(
                event,
                portfolio.positions.get(event.symbol),
                current_index,
                execution,
            )
            if bracket_fill is None:
                continue
            portfolio.apply_fill(bracket_fill)
            bracket_manager.sync_after_fill(
                bracket_fill,
                portfolio.positions.get(bracket_fill.symbol),
                current_index,
            )

        for event in batch:
            last_marks[event.symbol] = event.close_price

        batch_time = batch[0].end_time
        final_state = portfolio.record_equity(batch_time, last_marks)
        trading_date = batch_time.date()
        if active_day != trading_date:
            active_day = trading_date
            day_start_equity = final_state.equity

        candidate_signals: list[tuple[BarEvent, SignalEvent]] = []
        for event in batch:
            current_index = symbol_indices.get(event.symbol, 0)
            symbol_strategy = _strategy_for_symbol(symbol_strategies, strategy, event.symbol)
            symbol_strategy.current_target = _portfolio_target(final_state, event.symbol)

            signal = risk_manager.evaluate(event, final_state, current_index)
            if signal is None:
                signal = symbol_strategy.on_event(event, final_state)
                signal = risk_manager.filter_signal(signal, final_state, current_index)
                if signal is None:
                    symbol_strategy.current_target = _portfolio_target(final_state, event.symbol)
            if signal is not None:
                candidate_signals.append((event, signal))

        accepted_signals = _filter_entry_signals(
            candidate_signals,
            final_state,
            config,
            trading_date=trading_date,
            day_start_equity=day_start_equity,
            closed_trades=portfolio.closed_trades,
        )
        for event, signal in accepted_signals:
            current_index = symbol_indices.get(event.symbol, 0)
            orders = translator.from_signal(signal, final_state, event.close_price)
            orders = [
                order for order in orders if not execution.has_pending_equivalent_order(order)
            ]
            execution.submit_orders(orders, current_index)

        for event in batch:
            symbol_indices[event.symbol] = symbol_indices.get(event.symbol, 0) + 1

    if final_state is None:
        raise ValueError("Backtest did not produce a final portfolio state")

    metrics = compute_summary_metrics(
        equity_curve=portfolio.equity_curve,
        trades=portfolio.closed_trades,
        fills=portfolio.fills,
        initial_cash=config.portfolio.initial_cash,
        bar_seconds=_infer_bar_seconds(events),
    )
    metrics["final_cash"] = final_state.cash

    return BacktestResult(
        strategy_name=strategy.__class__.__name__,
        metrics=metrics,
        equity_curve=portfolio.equity_curve,
        trades=portfolio.closed_trades,
        fills=portfolio.fills,
    )
