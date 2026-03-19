from __future__ import annotations

from dataclasses import dataclass

from ofbot.config import ExecutionConfig
from ofbot.market.orderbook import OrderBookState

EPSILON = 1e-12


@dataclass(slots=True)
class MarketCostEstimate:
    requested_qty: float
    top_of_book_qty: float
    visible_depth_qty: float
    participation_rate: float
    spread_bps: float
    fee_bps: float
    slippage_bps: float
    one_way_cost_bps: float
    roundtrip_cost_bps: float
    fill_price: float | None = None
    used_synthetic_liquidity: bool = False


def simulate_market_order_fill(
    execution: ExecutionConfig,
    *,
    book: OrderBookState | None,
    side: int,
    requested_qty: float,
) -> MarketCostEstimate:
    safe_qty = max(float(requested_qty), 0.0)
    spread_bps = execution.base_spread_bps
    fee_bps = float(execution.taker_fee_bps)
    if safe_qty <= EPSILON:
        return MarketCostEstimate(
            requested_qty=0.0,
            top_of_book_qty=0.0,
            visible_depth_qty=0.0,
            participation_rate=0.0,
            spread_bps=spread_bps,
            fee_bps=fee_bps,
            slippage_bps=0.0,
            one_way_cost_bps=fee_bps,
            roundtrip_cost_bps=2.0 * fee_bps,
            fill_price=None,
            used_synthetic_liquidity=False,
        )

    if (
        book is None
        or book.bid_price is None
        or book.ask_price is None
        or book.bid_price <= 0.0
        or book.ask_price <= 0.0
        or book.mid_price is None
        or book.mid_price <= 0.0
    ):
        slippage_bps = (
            spread_bps / 2.0
            + execution.fallback_half_spread_bps
            + execution.impact_bps_per_unit_participation
        )
        one_way_cost_bps = fee_bps + slippage_bps
        return MarketCostEstimate(
            requested_qty=safe_qty,
            top_of_book_qty=0.0,
            visible_depth_qty=0.0,
            participation_rate=1.0,
            spread_bps=spread_bps,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            one_way_cost_bps=one_way_cost_bps,
            roundtrip_cost_bps=2.0 * one_way_cost_bps,
            fill_price=None,
            used_synthetic_liquidity=True,
        )

    spread_bps = float(book.spread_bps) if book.spread_bps is not None else spread_bps
    top_of_book_qty = float(book.ask_qty or 0.0) if side > 0 else float(book.bid_qty or 0.0)
    levels = book.market_levels(side, max_depth_bps=execution.market_sweep_depth_bps)
    visible_depth_qty = sum(max(float(qty), 0.0) for _, qty in levels)
    fill_price, used_synthetic_liquidity = _sweep_fill_price(
        execution,
        book=book,
        side=side,
        requested_qty=safe_qty,
        levels=levels,
        top_of_book_qty=top_of_book_qty,
        visible_depth_qty=visible_depth_qty,
    )
    if fill_price is None:
        one_way_cost_bps = fee_bps
        return MarketCostEstimate(
            requested_qty=safe_qty,
            top_of_book_qty=top_of_book_qty,
            visible_depth_qty=visible_depth_qty,
            participation_rate=1.0,
            spread_bps=spread_bps,
            fee_bps=fee_bps,
            slippage_bps=0.0,
            one_way_cost_bps=one_way_cost_bps,
            roundtrip_cost_bps=2.0 * one_way_cost_bps,
            fill_price=None,
            used_synthetic_liquidity=used_synthetic_liquidity,
        )

    participation_base = visible_depth_qty if visible_depth_qty > EPSILON else top_of_book_qty
    if participation_base <= EPSILON:
        participation_rate = 1.0
    else:
        participation_rate = min(1.0, safe_qty / participation_base)

    mid_price = float(book.mid_price)
    if side > 0:
        slippage_bps = max(0.0, (fill_price - mid_price) / mid_price * 10_000.0)
    else:
        slippage_bps = max(0.0, (mid_price - fill_price) / mid_price * 10_000.0)
    one_way_cost_bps = fee_bps + slippage_bps
    return MarketCostEstimate(
        requested_qty=safe_qty,
        top_of_book_qty=max(top_of_book_qty, 0.0),
        visible_depth_qty=visible_depth_qty,
        participation_rate=participation_rate,
        spread_bps=spread_bps,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        one_way_cost_bps=one_way_cost_bps,
        roundtrip_cost_bps=2.0 * one_way_cost_bps,
        fill_price=fill_price,
        used_synthetic_liquidity=used_synthetic_liquidity,
    )


def estimate_market_order_cost(
    execution: ExecutionConfig,
    *,
    book: OrderBookState | None,
    side: int,
    requested_qty: float,
) -> MarketCostEstimate:
    return simulate_market_order_fill(
        execution,
        book=book,
        side=side,
        requested_qty=requested_qty,
    )


def estimate_limit_order_cost(
    execution: ExecutionConfig,
    *,
    book: OrderBookState | None,
    side: int,
    requested_qty: float,
) -> MarketCostEstimate:
    safe_qty = max(float(requested_qty), 0.0)
    spread_bps = execution.base_spread_bps
    if book is not None and book.spread_bps is not None:
        spread_bps = float(book.spread_bps)
    fee_bps = float(execution.maker_fee_bps)
    slippage_bps = float(execution.fallback_half_spread_bps)
    one_way_cost_bps = fee_bps + slippage_bps
    top_of_book_qty = 0.0
    if book is not None:
        top_of_book_qty = float(book.ask_qty or 0.0) if side > 0 else float(book.bid_qty or 0.0)
    participation_rate = min(1.0, safe_qty / max(top_of_book_qty, safe_qty, EPSILON)) if safe_qty > EPSILON else 0.0
    return MarketCostEstimate(
        requested_qty=safe_qty,
        top_of_book_qty=max(top_of_book_qty, 0.0),
        visible_depth_qty=max(top_of_book_qty, 0.0),
        participation_rate=participation_rate,
        spread_bps=spread_bps,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        one_way_cost_bps=one_way_cost_bps,
        roundtrip_cost_bps=2.0 * one_way_cost_bps,
        fill_price=None,
        used_synthetic_liquidity=False,
    )


def estimate_order_cost(
    execution: ExecutionConfig,
    *,
    book: OrderBookState | None,
    side: int,
    requested_qty: float,
    order_type: str,
) -> MarketCostEstimate:
    resolved_order_type = order_type.lower()
    if resolved_order_type == "limit" and execution.allow_limit_orders:
        return estimate_limit_order_cost(
            execution,
            book=book,
            side=side,
            requested_qty=requested_qty,
        )
    return estimate_market_order_cost(
        execution,
        book=book,
        side=side,
        requested_qty=requested_qty,
    )


def _sweep_fill_price(
    execution: ExecutionConfig,
    *,
    book: OrderBookState,
    side: int,
    requested_qty: float,
    levels: list[tuple[float, float]],
    top_of_book_qty: float,
    visible_depth_qty: float,
) -> tuple[float | None, bool]:
    remaining_qty = requested_qty
    gross_notional = 0.0
    last_price = book.ask_price if side > 0 else book.bid_price

    for price, qty in levels:
        available_qty = max(float(qty), 0.0)
        if available_qty <= EPSILON:
            continue
        take_qty = min(remaining_qty, available_qty)
        gross_notional += take_qty * float(price)
        remaining_qty -= take_qty
        last_price = float(price)
        if remaining_qty <= EPSILON:
            break

    used_synthetic_liquidity = remaining_qty > EPSILON
    if remaining_qty > EPSILON and last_price is not None:
        reference_depth = max(visible_depth_qty, top_of_book_qty, requested_qty, EPSILON)
        sweep_multiple = max(1.0, requested_qty / reference_depth)
        synthetic_impact_bps = execution.fallback_half_spread_bps + (
            execution.impact_bps_per_unit_participation * sweep_multiple
        )
        synthetic_price = last_price * (
            1.0 + synthetic_impact_bps / 10_000.0 if side > 0 else 1.0 - synthetic_impact_bps / 10_000.0
        )
        gross_notional += remaining_qty * synthetic_price
        remaining_qty = 0.0

    filled_qty = requested_qty - remaining_qty
    if filled_qty <= EPSILON:
        return None, used_synthetic_liquidity
    return gross_notional / filled_qty, used_synthetic_liquidity
