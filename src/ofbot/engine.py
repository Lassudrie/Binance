from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ofbot.config import AppConfig
from ofbot.execution.costs import estimate_market_order_cost
from ofbot.execution.paper_broker import PaperBroker
from ofbot.execution.portfolio import PaperPortfolio, PositionSnapshot, TradeRecord
from ofbot.execution.risk import RiskManager
from ofbot.gateway.binance_public_rest import BinancePublicRestClient
from ofbot.gateway.recorder import ParquetRecorder
from ofbot.logging import get_logger
from ofbot.market.features import FeatureEngine, FeatureSnapshot
from ofbot.market.orderbook import OrderBookState
from ofbot.market.regimes import RegimeLabel
from ofbot.market.trades import (
    AggTradeEvent,
    BookTickerEvent,
    DepthEvent,
    DepthSnapshotEvent,
    KlineEvent,
    MarketEvent,
)
from ofbot.memory import reports as report_mod
from ofbot.memory.bandit import ThompsonPolicySelector
from ofbot.memory.journal import JournalWriter
from ofbot.memory.store import MemoryStore
from ofbot.strategy import build_strategy
from ofbot.strategy.base import Strategy
from ofbot.strategy.base import StrategyDecision
from ofbot.utils.manual_orders import ManualOrderRequest, iter_manual_order_requests

EPSILON = 1e-12


class PaperEngine:
    """Event-driven paper trading engine for live and replay flows."""

    def __init__(
        self,
        config: AppConfig,
        *,
        run_id: str | None = None,
        depth_snapshot_client: BinancePublicRestClient | None = None,
        runtime_clock: Literal["wall", "event"] = "wall",
    ) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self._depth_snapshot_client = depth_snapshot_client

        self.store = MemoryStore(config.learning.duckdb_path)
        self.journal = JournalWriter(self.store, run_id=run_id)

        self.broker = PaperBroker(config.execution)
        self.portfolio = PaperPortfolio(initial_cash=config.execution.initial_cash)
        self.risk = RiskManager(
            stale_data_s=config.risk.stale_data_s,
            broken_ws_s=config.risk.broken_ws_s,
            abnormal_spread_bps=config.risk.abnormal_spread_bps,
            abnormal_volatility_bps=config.risk.abnormal_volatility_bps,
            max_position_size=config.risk.max_position_size,
            max_notional=config.risk.max_notional,
            max_concurrent_exposure=config.risk.max_concurrent_exposure,
            cooldown_after_loss_s=config.risk.cooldown_after_loss_s,
            daily_loss_limit=config.risk.daily_loss_limit,
            max_holding_time_s=config.risk.max_holding_time_s,
            catastrophic_stop_loss_bps=config.risk.catastrophic_stop_loss_bps,
            min_hold_before_soft_exit_s=config.risk.min_hold_before_soft_exit_s,
            stop_loss_bps=config.risk.stop_loss_bps,
            take_profit_bps=config.risk.take_profit_bps,
            trailing_stop_bps=config.risk.trailing_stop_bps,
        )
        self.feature_engine = FeatureEngine(
            horizons_sec=config.features.horizons_sec,
            feature_window_sec=config.features.feature_window_sec,
            large_trade_quantile=config.features.large_trade_quantile,
            volatility_window_sec=config.features.volatility_window_sec,
            zscore_window=config.features.zscore_window,
        )

        self.order_books: dict[str, OrderBookState] = {
            symbol: OrderBookState(
                symbol=symbol,
                require_depth_sync=depth_snapshot_client is not None,
                book_ticker_divergence_bps=config.gateway.book_ticker_divergence_bps,
            )
            for symbol in config.symbol_set
        }

        self.long_only = not config.execution.allow_short
        self.strategies: dict[str, Strategy] = {}
        self.candidate_names: list[str] = []
        for variant in [
            config.strategy_library.continuation,
            config.strategy_library.exhaustion,
            config.strategy_library.hybrid,
        ]:
            if not variant.enabled:
                continue
            self.strategies[variant.name] = build_strategy(variant, long_only=self.long_only)
            self.candidate_names.append(variant.name)

        self.default_strategy = config.strategy_library.default
        if self.default_strategy not in self.strategies:
            self.default_strategy = self.candidate_names[0] if self.candidate_names else "continuation"
            if self.default_strategy not in self.strategies:
                self.strategies[self.default_strategy] = build_strategy(config.strategy_library.continuation, long_only=self.long_only)

        self.selector = ThompsonPolicySelector(
            store=self.store,
            run_id=self.journal.run_id,
            symbol="multi",
            min_samples_total=config.learning.min_samples_total,
            min_samples_per_regime=config.learning.min_samples_per_regime,
            min_improvement_bps=config.learning.min_improvement_bps,
            rolling_window=config.learning.rolling_window,
            confidence=config.learning.confidence,
            max_drawdown_bps=config.learning.max_drawdown_bps,
        )
        self.incumbent_by_regime: dict[str, str] = {
            regime: state[0] for regime, state in self.store.get_policy_state(self.journal.run_id, "multi").items()
        }
        if not self.incumbent_by_regime and self.default_strategy in self.strategies:
            self.incumbent_by_regime = {}

        self._last_ws_event_time: dict[str, datetime] = {}
        self._recorder: ParquetRecorder | None = (
            ParquetRecorder(config.paths.raw_dir, self.journal.run_id)
            if self.config.paper_local_record_raw and self.config.mode == "paper_local"
            else None
        )
        self._raw_flushed = False
        self._closed = False
        self._final_report_path: Path | None = None
        self._live_counters: Counter[str] = Counter()
        self._event_type_counts: Counter[str] = Counter()
        self._risk_gate_counts: Counter[str] = Counter()
        self._submit_counts_by_strategy: Counter[str] = Counter()
        self._submit_counts_by_symbol: Counter[str] = Counter()
        self._telemetry_started_at = datetime.now(UTC)
        self._telemetry_last_emitted_at = self._telemetry_started_at
        self._telemetry_last_counters: Counter[str] = Counter()
        self._telemetry_interval_s = max(5, int(config.live_telemetry_interval_s))
        self._telemetry_top_k = max(1, int(config.live_telemetry_top_k))
        self._telemetry_seq = 0
        self._last_submit: dict[str, Any] | None = None
        self._last_fill: dict[str, Any] | None = None
        self._last_trade: dict[str, Any] | None = None
        self._manual_control_memory_dir = self.config.paths.memory_dir
        self._runtime_clock = runtime_clock

        self.last_event_time = datetime.now(UTC)

    @property
    def run_id(self) -> str:
        return self.journal.run_id

    def process_events(self, events: Iterable[MarketEvent]) -> None:
        for event in events:
            self.process_event(event)

    def process_event(self, event: MarketEvent) -> None:
        if self._closed:
            return

        symbol = event.symbol
        if symbol not in self.order_books:
            return

        self._live_counters["processed_events"] += 1
        self._event_type_counts[event.event_type] += 1
        try:
            processing_time = self._runtime_now(event.event_time)
            self.last_event_time = event.event_time
            self._last_ws_event_time[symbol] = processing_time
            self._record_raw_event(event)

            book = self.order_books[symbol]
            if isinstance(event, AggTradeEvent):
                self.feature_engine.on_agg_trade(event)
            elif isinstance(event, BookTickerEvent):
                book.apply_book_ticker(event)
            elif isinstance(event, DepthSnapshotEvent):
                sync_result = book.apply_depth_snapshot(event)
                if sync_result.needs_snapshot and self._depth_snapshot_client is not None:
                    self._sync_depth_book(symbol=symbol, event_time=event.event_time, trigger_reason=sync_result.reason)
            elif isinstance(event, DepthEvent):
                sync_result = book.apply_depth(event)
                if sync_result.needs_snapshot:
                    if not self._sync_depth_book(
                        symbol=symbol,
                        event_time=event.event_time,
                        trigger_reason=sync_result.reason,
                    ):
                        return
                elif not sync_result.applied:
                    return
            elif isinstance(event, KlineEvent):
                pass

            if not book.is_tradeable or book.mid_price is None or book.mid_price <= 0:
                self._process_pending_orders_with_l1(
                    event=event,
                    book=book,
                    processing_time=processing_time,
                )
                return

            self.feature_engine.on_orderbook(event.event_time, symbol=symbol, book=book)
            snapshot = self.feature_engine.snapshot(symbol, event_time=event.event_time, book=book)
            self.portfolio.mark(symbol, book.mid_price, event.event_time)
            state = self.portfolio.snapshot(symbol)
            self._drain_manual_order_requests(
                snapshot,
                state,
                processing_time=processing_time,
            )
            state = self.portfolio.snapshot(symbol)

            if abs(state.net_position) > 0:
                exit_side = -1 if state.net_position > 0 else 1
                exit_cost = estimate_market_order_cost(
                    self.config.execution,
                    book=book,
                    side=exit_side,
                    requested_qty=abs(state.net_position),
                )
                risk_exit = self.risk.evaluate_position_exit(
                    symbol=symbol,
                    snapshot=snapshot,
                    state=state,
                    now=processing_time,
                    estimated_exit_cost_bps=exit_cost.one_way_cost_bps,
                )
                if risk_exit is not None:
                    self._submit_decision(
                        risk_exit,
                        snapshot,
                        state,
                        active_strategy="risk",
                        decision_time=processing_time,
                    )
                    if risk_exit.side == 0:
                        self._process_fills(snapshot, processing_time=processing_time)
                        return

            if abs(state.net_position) == 0 or decision_input_allowed(snapshot):
                selected_name, decision = self._select_and_decide(
                    snapshot,
                    state,
                    now=processing_time,
                )
                if decision is not None:
                    self._submit_decision(
                        decision,
                        snapshot,
                        state,
                        active_strategy=selected_name,
                        decision_time=processing_time,
                    )

            self._process_fills(snapshot, processing_time=processing_time)
        finally:
            self.maybe_emit_live_telemetry()

    def _process_pending_orders_with_l1(
        self,
        *,
        event: MarketEvent,
        book: OrderBookState,
        processing_time: datetime,
    ) -> None:
        if self.broker.pending_order_count <= 0:
            return
        if book.mid_price is None or book.mid_price <= 0:
            return
        if book.bid_price is None or book.ask_price is None:
            return

        # During temporary depth desyncs, keep market orders executable off fresh L1
        # instead of letting them expire without a fill attempt.
        self.feature_engine.on_orderbook(event.event_time, symbol=event.symbol, book=book)
        snapshot = self.feature_engine.snapshot(event.symbol, event_time=event.event_time, book=book)
        self.portfolio.mark(event.symbol, book.mid_price, event.event_time)
        self._process_fills(snapshot, processing_time=processing_time)

    def _drain_manual_order_requests(
        self,
        snapshot: FeatureSnapshot,
        state: PositionSnapshot,
        *,
        processing_time: datetime,
    ) -> None:
        requests = iter_manual_order_requests(self._manual_control_memory_dir, symbol=snapshot.symbol)
        if not requests:
            return

        for path, request in requests:
            try:
                self._apply_manual_order_request(
                    snapshot=snapshot,
                    state=state,
                    request=request,
                    processing_time=processing_time,
                )
                state = self.portfolio.snapshot(snapshot.symbol)
            except Exception as exc:
                self.logger.error(
                    "manual_order_request_failed symbol=%s request_id=%s error=%s",
                    snapshot.symbol,
                    request.request_id,
                    exc,
                )
            finally:
                path.unlink(missing_ok=True)

    def _apply_manual_order_request(
        self,
        *,
        snapshot: FeatureSnapshot,
        state: PositionSnapshot,
        request: ManualOrderRequest,
        processing_time: datetime,
    ) -> None:
        decision = self._manual_decision_from_request(snapshot=snapshot, state=state, request=request)
        if decision is None:
            return
        projected_qty = self._normalize_target_qty(decision)
        if not request.force and self._requires_entry_gate(current_qty=state.net_position, projected_qty=projected_qty):
            reason = self._risk_pre_entry_context(
                snapshot,
                state,
                projected_qty=projected_qty,
                now=processing_time,
            )
            if reason is not None:
                self._append_risk_gate(
                    snapshot,
                    "manual",
                    reason,
                    params=self._signal_params(decision),
                )
                return
        self._submit_decision(
            decision,
            snapshot,
            state,
            active_strategy="manual",
            decision_time=processing_time,
        )

    def _manual_decision_from_request(
        self,
        *,
        snapshot: FeatureSnapshot,
        state: PositionSnapshot,
        request: ManualOrderRequest,
    ) -> StrategyDecision | None:
        context: dict[str, float | int | str | None] = {
            "source": "manual_control",
            "manual_request_id": request.request_id,
            "manual_force": int(request.force),
            "manual_qty": request.qty,
            "manual_notional_usd": request.notional_usd,
        }
        if request.side == 0:
            return StrategyDecision(
                symbol=snapshot.symbol,
                side=0,
                target_qty=0.0,
                reason=request.reason,
                confidence=1.0,
                order_type=request.order_type,
                entry_context=context,
            )

        mid_price = float(snapshot.mid_price or 0.0)
        if mid_price <= EPSILON:
            self.logger.warning(
                "manual_order_request_skipped symbol=%s request_id=%s reason=no_mid_price",
                snapshot.symbol,
                request.request_id,
            )
            return None

        delta_qty = float(request.qty or 0.0)
        if delta_qty <= EPSILON and request.notional_usd is not None:
            delta_qty = float(request.notional_usd) / mid_price
        if delta_qty <= EPSILON:
            self.logger.warning(
                "manual_order_request_skipped symbol=%s request_id=%s reason=no_size",
                snapshot.symbol,
                request.request_id,
            )
            return None

        signed_delta = delta_qty if request.side > 0 else -delta_qty
        target_qty = state.net_position + signed_delta
        if self.long_only:
            target_qty = max(0.0, target_qty)
        decision_side = 1 if target_qty > state.net_position + EPSILON else -1 if target_qty < state.net_position - EPSILON else 0
        if decision_side == 0:
            return None
        return StrategyDecision(
            symbol=snapshot.symbol,
            side=decision_side,
            target_qty=abs(target_qty) if target_qty >= 0 else -abs(target_qty),
            reason=request.reason,
            confidence=1.0,
            order_type=request.order_type,
            max_holding_time_s=self.config.risk.max_holding_time_s,
            stop_loss_bps=self.config.risk.stop_loss_bps,
            take_profit_bps=self.config.risk.take_profit_bps,
            trailing_stop_bps=self.config.risk.trailing_stop_bps,
            entry_context=context,
        )

    def _select_and_decide(
        self,
        snapshot: FeatureSnapshot,
        state: PositionSnapshot,
        *,
        now: datetime,
    ) -> tuple[str, Any | None]:
        strategy_name = self.default_strategy
        strategy = self.strategies[strategy_name]
        if self.config.learning.enabled and len(self.candidate_names) >= 2:
            strategy_name, strategy = self._select_strategy(snapshot.regime)

        if abs(state.net_position) <= EPSILON:
            reason = self._risk_pre_entry_context(snapshot, state, projected_qty=0.0, now=now)
            if reason is not None:
                self._append_risk_gate(snapshot, strategy_name, reason)
                return strategy_name, None

        decision = strategy.decide(snapshot, state, long_only=self.long_only)
        if decision is None:
            return strategy_name, None

        projected_qty = self._normalize_target_qty(decision)
        if self._requires_entry_gate(current_qty=state.net_position, projected_qty=projected_qty):
            edge_context, edge_reason = self._prepare_entry_context(
                snapshot=snapshot,
                decision=decision,
                strategy_name=strategy_name,
            )
            gate_params = {
                **dict(getattr(decision, "entry_context", {}) or {}),
                **edge_context,
            }
            if edge_reason is not None:
                self._append_risk_gate(snapshot, strategy_name, edge_reason, params=gate_params)
                return strategy_name, None

            decision.entry_context = gate_params
            decision.target_qty = abs(float(edge_context.get("resolved_target_qty", 0.0)))
            projected_qty = self._normalize_target_qty(decision)
            reason = self._risk_pre_entry_context(snapshot, state, projected_qty=projected_qty, now=now)
            if reason is not None:
                self._append_risk_gate(snapshot, strategy_name, reason, params=gate_params)
                return strategy_name, None

        return strategy_name, decision

    def _risk_pre_entry_context(
        self,
        snapshot: FeatureSnapshot,
        state: PositionSnapshot,
        *,
        projected_qty: float,
        now: datetime,
    ) -> str | None:
        ws_healthy = self._is_ws_healthy(snapshot.symbol, snapshot.event_time)
        projected_notional = abs(snapshot.mid_price or 0.0) * abs(projected_qty)
        allowed, reason = self.risk.pre_entry_gate(
            symbol=snapshot.symbol,
            snapshot=snapshot,
            portfolio_snapshot=state,
            now=now,
            event_timestamp=snapshot.event_time,
            websocket_healthy=ws_healthy,
            open_positions=len(self.portfolio.open_symbols),
            total_equity=self.portfolio.equity,
            projected_position_qty=projected_qty,
            projected_notional=projected_notional,
        )
        if not allowed:
            return reason or "risk_block"
        return None

    def _prepare_entry_context(
        self,
        *,
        snapshot: FeatureSnapshot,
        decision: Any,
        strategy_name: str,
    ) -> tuple[dict[str, float | int | str | bool | None], str | None]:
        side = 1 if getattr(decision, "side", 0) > 0 else -1
        strategy = self.strategies.get(strategy_name)
        params = strategy.get_parameters() if strategy is not None else {}
        min_expected_net_edge_bps = float(params.get("min_expected_net_edge_bps", 0.0) or 0.0)
        mid_price = float(snapshot.mid_price or 0.0)
        if mid_price <= EPSILON:
            return {"expected_net_edge_bps": None}, "risk_expected_net_edge"

        seed_notional = self.config.execution.default_target_notional_usd
        seed_qty = max(seed_notional / mid_price, EPSILON)
        seed_cost_estimate = estimate_market_order_cost(
            self.config.execution,
            book=self.order_books.get(snapshot.symbol),
            side=side,
            requested_qty=seed_qty,
        )
        move_components = self._expected_move_proxy_components(
            snapshot,
            side=side,
            strategy_params=params,
        )
        expected_move_proxy_bps = move_components["expected_move_proxy_bps"]
        seed_expected_net_edge_bps = expected_move_proxy_bps - seed_cost_estimate.roundtrip_cost_bps
        confidence = float(getattr(decision, "confidence", 0.0) or 0.0)
        confidence_multiplier = clamp(0.5 + confidence, 0.75, 1.25)
        edge_floor = max(min_expected_net_edge_bps, 1.0)
        edge_multiplier = clamp(seed_expected_net_edge_bps / edge_floor, 0.5, 1.5)
        desired_notional_usd = clamp(
            self.config.execution.default_target_notional_usd * confidence_multiplier * edge_multiplier,
            self.config.execution.min_target_notional_usd,
            self.config.execution.max_target_notional_usd,
        )
        resolved_target_qty = desired_notional_usd / mid_price
        cost_estimate = estimate_market_order_cost(
            self.config.execution,
            book=self.order_books.get(snapshot.symbol),
            side=side,
            requested_qty=resolved_target_qty,
        )
        expected_net_edge_bps = expected_move_proxy_bps - cost_estimate.roundtrip_cost_bps
        entry_context: dict[str, float | int | str | bool | None] = {
            "expected_move_proxy_bps": expected_move_proxy_bps,
            "roundtrip_cost_est_bps": cost_estimate.roundtrip_cost_bps,
            "estimated_entry_cost_bps": cost_estimate.one_way_cost_bps,
            "expected_net_edge_bps": expected_net_edge_bps,
            "participation_rate": cost_estimate.participation_rate,
            "top_of_book_qty": cost_estimate.top_of_book_qty,
            "confidence_multiplier": confidence_multiplier,
            "edge_multiplier": edge_multiplier,
            "desired_notional_usd": desired_notional_usd,
            "resolved_target_qty": resolved_target_qty,
            "min_expected_net_edge_bps": min_expected_net_edge_bps,
        }
        entry_context.update(move_components)
        if expected_net_edge_bps < min_expected_net_edge_bps:
            return entry_context, "risk_expected_net_edge"
        return entry_context, None

    def _expected_move_proxy_components(
        self,
        snapshot: FeatureSnapshot,
        *,
        side: int,
        strategy_params: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        params = strategy_params or {}
        fast_horizon = max(1, int(params.get("edge_fast_horizon_s", 15) or 15))
        slow_horizon = max(fast_horizon, int(params.get("edge_slow_horizon_s", 30) or 30))
        fast_weight, slow_weight = _normalized_pair(
            float(params.get("edge_fast_weight", 0.6) or 0.6),
            float(params.get("edge_slow_weight", 0.4) or 0.4),
        )
        cvd_bonus_bps = max(0.0, float(params.get("edge_cvd_bonus_bps", 1.5) or 0.0))
        queue_bonus_bps = max(0.0, float(params.get("edge_queue_bonus_bps", 4.0) or 0.0))
        bonus_cap_bps = max(0.0, float(params.get("edge_bonus_cap_bps", 12.0) or 0.0))
        no_trade_z = float(params.get("no_trade_z", params.get("notrade_z", 0.0)) or 0.0)
        queue_threshold = float(params.get("queue_imbalance_threshold", 0.0) or 0.0)

        fast_micro = self._directional_feature_bps(
            snapshot,
            side=side,
            primary=f"microprice_drift_bps_{fast_horizon}s",
            fallback="microprice_drift_bps",
        )
        fast_momentum = self._directional_feature_bps(
            snapshot,
            side=side,
            primary=f"momentum_bps_{fast_horizon}s",
            fallback=f"short_return_bps_{fast_horizon}s",
        )
        slow_micro = self._directional_feature_bps(
            snapshot,
            side=side,
            primary=f"microprice_drift_bps_{slow_horizon}s",
            fallback="microprice_drift_bps",
        )
        slow_momentum = self._directional_feature_bps(
            snapshot,
            side=side,
            primary=f"momentum_bps_{slow_horizon}s",
            fallback=f"short_return_bps_{slow_horizon}s",
        )
        fast_component = max(fast_micro, fast_momentum)
        slow_component = max(slow_micro, slow_momentum)
        trend_component = fast_weight * fast_component + slow_weight * slow_component

        cvd_z = float(snapshot.features.get("cvd_base_1s_z") or 0.0)
        queue_imbalance = float(snapshot.features.get("queue_imbalance") or 0.0)
        if side < 0:
            cvd_z = -cvd_z
            queue_imbalance = -queue_imbalance
        cvd_bonus = cvd_bonus_bps * max(0.0, cvd_z - no_trade_z)
        queue_bonus = queue_bonus_bps * max(0.0, queue_imbalance - queue_threshold)
        conviction_bonus = min(cvd_bonus + queue_bonus, bonus_cap_bps) if bonus_cap_bps > 0.0 else cvd_bonus + queue_bonus
        expected_move_proxy_bps = trend_component + conviction_bonus

        return {
            "expected_move_proxy_bps": expected_move_proxy_bps,
            "expected_trend_component_bps": trend_component,
            "expected_conviction_bonus_bps": conviction_bonus,
            "expected_fast_component_bps": fast_component,
            "expected_slow_component_bps": slow_component,
            "expected_cvd_bonus_bps": cvd_bonus,
            "expected_queue_bonus_bps": queue_bonus,
        }

    def _directional_feature_bps(
        self,
        snapshot: FeatureSnapshot,
        *,
        side: int,
        primary: str,
        fallback: str | None = None,
    ) -> float:
        raw = snapshot.features.get(primary)
        if raw is None and fallback is not None:
            raw = snapshot.features.get(fallback)
        value = float(raw or 0.0)
        if side < 0:
            value = -value
        return max(0.0, value)

    def _process_fills(self, snapshot: FeatureSnapshot, *, processing_time: datetime) -> None:
        fills = self.broker.process(
            symbol=snapshot.symbol,
            now=processing_time,
            book=self.order_books[snapshot.symbol],
        )
        for fill in fills:
            self._live_counters["fills"] += 1
            self._last_fill = {
                "symbol": fill.symbol,
                "side": fill.side,
                "qty": round(fill.fill_qty, 8),
                "price": round(fill.fill_price, 8),
                "fee": round(fill.fee, 8),
                "slippage_bps": round(fill.slippage_bps, 4),
                "fill_role": self._infer_fill_role(fill.symbol, fill.side),
                "event_time": fill.event_time.isoformat(),
            }
            fill_role = self._infer_fill_role(fill.symbol, fill.side)
            self.journal.append_fill(
                snapshot=snapshot,
                strategy_id=fill.strategy_id or "unknown",
                fill_qty=fill.fill_qty,
                fill_price=fill.fill_price,
                spread_bps=fill.spread_bps,
                slippage_bps=fill.slippage_bps,
                fee=fill.fee,
                notional=fill.notional,
                is_maker=fill.is_maker,
                order_id=fill.order_id,
                fill_role=fill_role,
            )
            for trade in self.portfolio.apply_fill(fill):
                self._on_closed_trade(trade)

    def _on_closed_trade(self, trade: TradeRecord) -> None:
        self._live_counters["trades_closed"] += 1
        self._last_trade = {
            "symbol": trade.symbol,
            "strategy": trade.strategy or "unknown",
            "realized_pnl": round(trade.realized_pnl, 8),
            "gross_pnl": round(trade.gross_pnl, 8),
            "fees": round(trade.fees, 8),
            "holding_seconds": round(trade.holding_seconds, 3),
            "entry_fill_count": trade.entry_fill_count,
            "exit_fill_count": trade.exit_fill_count,
        }
        self.journal.append_trade_context(
            trade=trade,
            symbol=trade.symbol,
            strategy_id=trade.strategy or "unknown",
        )
        self.risk.register_realized_pnl(trade.symbol, trade.realized_pnl, trade.exit_time)
        self.risk.clear_position_extremes(trade.symbol)
        self.logger.info(
            "trade_closed",
            extra={
                "symbol": trade.symbol,
                "strategy": trade.strategy,
                "realized_pnl": trade.realized_pnl,
                "holding_seconds": trade.holding_seconds,
            },
        )

    def _submit_decision(
        self,
        decision: Any,
        snapshot: FeatureSnapshot,
        state: PositionSnapshot,
        *,
        active_strategy: str,
        decision_time: datetime,
    ) -> None:
        strategy_id = active_strategy
        reference = snapshot.mid_price or 0.0

        context = dict(getattr(decision, "entry_context", {}) or {})
        context["strategy"] = strategy_id
        context["regime"] = self._regime_key(snapshot.regime)
        context["decision_reason"] = decision.reason
        decision.entry_context = context

        orders = self.broker.queue_from_decision(
            decision=decision,
            symbol=snapshot.symbol,
            current_qty=state.net_position,
            reference_price=reference,
            event_time=decision_time,
            strategy_id=strategy_id,
        )
        for order in orders:
            self.broker.submit(order)
            self._live_counters["submits"] += 1
            self._submit_counts_by_strategy[strategy_id] += 1
            self._submit_counts_by_symbol[snapshot.symbol] += 1
            self._last_submit = {
                "symbol": snapshot.symbol,
                "strategy": strategy_id,
                "reason": decision.reason,
                "target_qty": round(abs(float(order.target_qty)), 8),
                "order_type": order.order_type,
                "expected_net_edge_bps": _rounded_or_none(context.get("expected_net_edge_bps")),
                "desired_notional_usd": _rounded_or_none(context.get("desired_notional_usd")),
                "event_time": snapshot.event_time.isoformat(),
            }
            self.journal.append_signal(
                snapshot=snapshot,
                strategy_id=strategy_id,
                order_intent="submit",
                signal_strength=getattr(decision, "confidence", None),
                rationale=decision.reason,
                params=self._signal_params(decision),
            )

        if not orders:
            self.journal.append_signal(
                snapshot=snapshot,
                strategy_id=strategy_id,
                order_intent="no_op",
                signal_strength=getattr(decision, "confidence", None),
                rationale=decision.reason,
                params=self._signal_params(decision),
            )

    def _select_strategy(self, regime: RegimeLabel) -> tuple[str, Strategy]:
        regime_key = self._regime_key(regime)
        strategy_name, _reason = self.selector.select(
            default_strategy=self.default_strategy,
            regime=regime_key,
            candidate_strategies=self.candidate_names,
            incumbent_by_regime=self.incumbent_by_regime,
            allow_promotion=self.config.learning.enable_promotion,
        )
        self.incumbent_by_regime[regime_key] = strategy_name
        return strategy_name, self.strategies[strategy_name]

    def _regime_key(self, regime: RegimeLabel) -> str:
        return f"{regime.volatility}|{regime.spread}|{regime.trend}|{regime.flow}"

    def _is_ws_healthy(self, symbol: str, now: datetime) -> bool:
        last = self._last_ws_event_time.get(symbol)
        if last is None:
            return False
        return (now - last).total_seconds() <= self.config.risk.broken_ws_s

    def _record_raw_event(self, event: MarketEvent) -> None:
        if self._recorder is not None:
            self._recorder.record(event, received_at=event.event_time)

    def _sync_depth_book(self, *, symbol: str, event_time: datetime, trigger_reason: str | None) -> bool:
        if self._depth_snapshot_client is None:
            return False

        book = self.order_books[symbol]
        reason = trigger_reason or "depth_snapshot_required"
        for attempt in range(1, 4):
            try:
                snapshot = self._depth_snapshot_client.get_depth_snapshot(symbol, event_time=event_time)
            except Exception as exc:
                self.logger.warning(
                    "depth_snapshot_fetch_failed symbol=%s attempt=%d reason=%s error=%s",
                    symbol,
                    attempt,
                    reason,
                    exc,
                )
                continue

            self._record_raw_event(snapshot)
            sync_result = book.apply_depth_snapshot(snapshot)
            self.logger.info(
                "depth_snapshot_applied symbol=%s attempt=%d reason=%s synced=%s last_update_id=%s",
                symbol,
                attempt,
                reason,
                book.is_synced,
                book.last_update_id,
            )
            if book.is_synced and not sync_result.needs_snapshot:
                return True
            reason = sync_result.reason or "depth_snapshot_retry_required"
        return book.is_synced

    def _append_risk_gate(
        self,
        snapshot: FeatureSnapshot,
        strategy_name: str,
        reason: str,
        *,
        params: dict[str, float | int | str | bool | None] | None = None,
    ) -> None:
        self._live_counters["risk_gates"] += 1
        self._risk_gate_counts[reason] += 1
        self.journal.append_signal(
            snapshot=snapshot,
            strategy_id=strategy_name,
            order_intent=reason,
            signal_strength=0.0,
            rationale="risk_gate",
            params=params,
        )

    def _normalize_target_qty(self, decision: Any) -> float:
        desired_qty = float(getattr(decision, "target_qty", 0.0))
        if getattr(decision, "side", 0) > 0:
            return abs(desired_qty)
        if getattr(decision, "side", 0) < 0:
            return -abs(desired_qty)
        return 0.0

    def _requires_entry_gate(self, *, current_qty: float, projected_qty: float) -> bool:
        if abs(projected_qty) <= EPSILON:
            return False
        if abs(current_qty) <= EPSILON:
            return True
        if current_qty * projected_qty < 0:
            return True
        return abs(projected_qty) > abs(current_qty) + EPSILON

    def _signal_params(self, decision: Any) -> dict[str, float | int | str | bool | None]:
        params: dict[str, float | int | str | bool | None] = {
            "resolved_target_qty": abs(float(getattr(decision, "target_qty", 0.0))),
        }
        for key, value in dict(getattr(decision, "entry_context", {}) or {}).items():
            if value is None or isinstance(value, (float, int, str, bool)):
                params[key] = value
        return params

    def _infer_fill_role(self, symbol: str, side: int) -> str:
        current_qty = self.portfolio.positions.get(symbol, 0.0)
        if abs(current_qty) <= EPSILON:
            return "entry"
        if (current_qty > 0 and side > 0) or (current_qty < 0 and side < 0):
            return "entry"
        return "exit"

    def flush_reports(self, output_root: Path | None = None) -> Path | None:
        if output_root is None:
            return None
        return report_mod.build_run_report(
            store=self.store,
            run_id=self.run_id,
            output_root=output_root,
        )

    def finalize(self) -> Path | None:
        if self._final_report_path is not None:
            return self._final_report_path

        self.emit_live_telemetry(force=True)
        self._flush_raw()
        reports_dir = self.config.paths.reports_dir / self.run_id
        self._final_report_path = self.flush_reports(reports_dir)
        self.close()
        return self._final_report_path

    def close(self) -> None:
        if self._closed:
            return
        self._flush_raw()
        if self._depth_snapshot_client is not None:
            self._depth_snapshot_client.close()
            self._depth_snapshot_client = None
        self.store.close()
        self._closed = True

    def _flush_raw(self) -> None:
        if self._raw_flushed or self._recorder is None:
            return
        self._recorder.flush(self.config.paths.raw_dir / f"raw_{self.journal.run_id}.parquet")
        self._raw_flushed = True

    def maybe_emit_live_telemetry(self) -> dict[str, Any] | None:
        if self._closed:
            return None
        now = datetime.now(UTC)
        if (now - self._telemetry_last_emitted_at).total_seconds() < self._telemetry_interval_s:
            return None
        return self.emit_live_telemetry(force=True, emitted_at=now)

    def emit_live_telemetry(
        self,
        *,
        force: bool = False,
        emitted_at: datetime | None = None,
    ) -> dict[str, Any] | None:
        if self._closed:
            return None
        now = emitted_at or datetime.now(UTC)
        if not force and (now - self._telemetry_last_emitted_at).total_seconds() < self._telemetry_interval_s:
            return None

        snapshot = self.get_live_telemetry_snapshot(emitted_at=now)
        self._telemetry_seq += 1
        self._telemetry_last_emitted_at = now
        self._telemetry_last_counters = Counter(self._live_counters)

        self.logger.info(
            "live_telemetry run_id=%s seq=%d uptime_s=%d processed=%d delta_processed=%d submits=%d delta_submits=%d fills=%d delta_fills=%d trades=%d delta_trades=%d open_positions=%d pending_orders=%d equity=%.2f cash=%.2f realized=%.2f unrealized=%.2f fees=%.2f event_types=%s top_risk_gates=%s strategies=%s broker_stages=%s broker_reasons=%s open=%s pending=%s last_submit=%s last_fill=%s last_trade=%s last_order=%s",
            self.run_id,
            snapshot["seq"],
            snapshot["uptime_s"],
            snapshot["processed_events"],
            snapshot["delta_processed_events"],
            snapshot["submits"],
            snapshot["delta_submits"],
            snapshot["fills"],
            snapshot["delta_fills"],
            snapshot["trades_closed"],
            snapshot["delta_trades_closed"],
            snapshot["open_position_count"],
            snapshot["pending_order_count"],
            snapshot["equity"],
            snapshot["cash"],
            snapshot["realized_pnl"],
            snapshot["unrealized_pnl"],
            snapshot["fees_paid"],
            self._format_pairs(snapshot["event_type_counts"]),
            self._format_pairs(snapshot["risk_gate_counts"]),
            self._format_pairs(snapshot["submit_counts_by_strategy"]),
            self._format_pairs(snapshot["broker_stage_counts"]),
            self._format_pairs(snapshot["broker_reason_counts"]),
            self._format_open_positions(snapshot["open_positions"]),
            self._format_pending(snapshot["pending_orders_by_symbol"]),
            self._format_last_event(snapshot["last_submit"]),
            self._format_last_event(snapshot["last_fill"]),
            self._format_last_event(snapshot["last_trade"]),
            self._format_last_event(snapshot["last_order_lifecycle"]),
        )
        return snapshot

    def get_live_telemetry_snapshot(self, *, emitted_at: datetime | None = None) -> dict[str, Any]:
        now = emitted_at or datetime.now(UTC)
        realized_pnl = sum(self.portfolio.realized_pnl.values())
        unrealized_pnl = sum(self.portfolio.unrealized_pnl.values())
        fees_paid = sum(self.portfolio.fees_paid.values())
        open_positions = []
        for symbol in sorted(self.portfolio.open_symbols):
            state = self.portfolio.snapshot(symbol)
            open_positions.append(
                {
                    "symbol": symbol,
                    "qty": round(state.net_position, 8),
                    "avg_entry_price": round(state.avg_entry_price, 8),
                    "last_price": _rounded_or_none(state.last_price),
                    "unrealized_pnl": round(state.unrealized_pnl, 8),
                    "mae_bps": round(state.mae_bps, 4),
                    "mfe_bps": round(state.mfe_bps, 4),
                }
            )

        current = Counter(self._live_counters)
        previous = self._telemetry_last_counters
        event_type_counts = self._top_counts(self._event_type_counts, top_k=4)
        risk_gate_counts = self._top_counts(self._risk_gate_counts, top_k=self._telemetry_top_k)
        submit_counts_by_strategy = self._top_counts(self._submit_counts_by_strategy, top_k=self._telemetry_top_k)
        broker_stage_counts = self._top_counts(Counter(self.broker.lifecycle_stage_counts), top_k=4)
        broker_reason_counts = self._top_counts(Counter(self.broker.lifecycle_reason_counts), top_k=6)
        return {
            "run_id": self.run_id,
            "seq": self._telemetry_seq + 1,
            "emitted_at": now.isoformat(),
            "uptime_s": int((now - self._telemetry_started_at).total_seconds()),
            "processed_events": int(current.get("processed_events", 0)),
            "delta_processed_events": int(current.get("processed_events", 0) - previous.get("processed_events", 0)),
            "submits": int(current.get("submits", 0)),
            "delta_submits": int(current.get("submits", 0) - previous.get("submits", 0)),
            "fills": int(current.get("fills", 0)),
            "delta_fills": int(current.get("fills", 0) - previous.get("fills", 0)),
            "trades_closed": int(current.get("trades_closed", 0)),
            "delta_trades_closed": int(current.get("trades_closed", 0) - previous.get("trades_closed", 0)),
            "risk_gates": int(current.get("risk_gates", 0)),
            "delta_risk_gates": int(current.get("risk_gates", 0) - previous.get("risk_gates", 0)),
            "cash": round(self.portfolio.cash, 8),
            "equity": round(self.portfolio.equity, 8),
            "gross_notional": round(self.portfolio.gross_notional, 8),
            "net_exposure": round(self.portfolio.net_exposure, 8),
            "realized_pnl": round(realized_pnl, 8),
            "unrealized_pnl": round(unrealized_pnl, 8),
            "fees_paid": round(fees_paid, 8),
            "open_position_count": len(open_positions),
            "open_positions": open_positions,
            "pending_order_count": self.broker.pending_order_count,
            "pending_orders_by_symbol": dict(self.broker.pending_orders_by_symbol),
            "book_states": {symbol: book.sync_state for symbol, book in self.order_books.items()},
            "event_type_counts": event_type_counts,
            "risk_gate_counts": risk_gate_counts,
            "submit_counts_by_strategy": submit_counts_by_strategy,
            "submit_counts_by_symbol": self._top_counts(self._submit_counts_by_symbol, top_k=self._telemetry_top_k),
            "broker_stage_counts": broker_stage_counts,
            "broker_reason_counts": broker_reason_counts,
            "last_submit": dict(self._last_submit) if self._last_submit is not None else None,
            "last_fill": dict(self._last_fill) if self._last_fill is not None else None,
            "last_trade": dict(self._last_trade) if self._last_trade is not None else None,
            "last_order_lifecycle": self.broker.last_lifecycle_event,
        }

    def _top_counts(self, counts: Counter[str], *, top_k: int) -> list[tuple[str, int]]:
        items = counts.most_common(top_k)
        return [(name, int(value)) for name, value in items]

    def _format_pairs(self, items: list[tuple[str, int]]) -> str:
        if not items:
            return "-"
        return ",".join(f"{name}:{value}" for name, value in items)

    def _format_open_positions(self, positions: list[dict[str, Any]]) -> str:
        if not positions:
            return "-"
        return ";".join(
            f"{item['symbol']}:{item['qty']}@{item['avg_entry_price']} uPnL={item['unrealized_pnl']}"
            for item in positions
        )

    def _format_pending(self, pending: dict[str, int]) -> str:
        if not pending:
            return "-"
        return ",".join(f"{symbol}:{count}" for symbol, count in sorted(pending.items()))

    def _format_last_event(self, payload: dict[str, Any] | None) -> str:
        if not payload:
            return "-"
        ordered = []
        for key in sorted(payload):
            ordered.append(f"{key}={payload[key]}")
        return "|".join(ordered)

    def _runtime_now(self, market_time: datetime) -> datetime:
        if self._runtime_clock == "event":
            return market_time
        return datetime.now(UTC)


def decision_input_allowed(snapshot: FeatureSnapshot) -> bool:
    # A conservative entry rule: only consider decisions when fresh spread/vol data exists.
    if snapshot.spread_bps is None:
        return False
    return True


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalized_pair(first: float, second: float) -> tuple[float, float]:
    total = max(first + second, EPSILON)
    return first / total, second / total


def _rounded_or_none(value: Any, *, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)
