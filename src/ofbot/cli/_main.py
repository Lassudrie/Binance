from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

from ofbot.config import load_config
from ofbot.engine import PaperEngine
from ofbot.gateway.binance_public_rest import BinancePublicRestClient
from ofbot.gateway.binance_public_ws import BinancePublicWebSocket
from ofbot.logging import configure_logging
from ofbot.memory.reports import build_daily_report
from ofbot.memory.store import MemoryStore
from ofbot.replay.eval import run_replay
from ofbot.utils.manual_orders import write_manual_order_request

DEFAULT_LOCAL_CONFIG = "config/local.paper.yaml"
DEFAULT_REPORT_DATE = "today"

console = Console()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ofbot")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    subs = parser.add_subparsers(dest="command", required=True)

    live = subs.add_parser("live", help="run live paper trading loop")
    live.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    live.add_argument("--symbol", action="append", help="override configured symbols")
    live.add_argument("--max-events", type=int, default=0, help="stop after N raw events (0=run forever)")
    live.add_argument("--max-seconds", type=int, default=0, help="stop after N seconds (0=run forever)")

    replay = subs.add_parser("replay", help="run replay from recorded parquet")
    replay.add_argument("--input", required=True, help="path to raw replay parquet")
    replay.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)

    report = subs.add_parser("report", help="generate report from learning memory")
    report.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    report.add_argument(
        "--date",
        default=DEFAULT_REPORT_DATE,
        help="YYYY-MM-DD or 'today' (default: today)",
    )

    paper_order = subs.add_parser("paper-order", help="enqueue a manual paper-only order for a live session")
    paper_order.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    paper_order.add_argument("--symbol", required=True)
    paper_order.add_argument("--side", choices=["buy", "sell", "flat"], required=True)
    paper_order.add_argument("--qty", type=float, default=0.0, help="delta quantity to trade")
    paper_order.add_argument("--notional-usd", type=float, default=0.0, help="delta notional in USD")
    paper_order.add_argument("--order-type", choices=["market", "limit"], default="market")
    paper_order.add_argument("--reason", default="manual_paper_order")
    paper_order.add_argument("--force", action="store_true", help="bypass pre-entry risk for paper-only testing")
    return parser


async def _run_live(config_path: str, max_events: int, max_seconds: int, symbols: list[str] | None) -> None:
    config = load_config(config_path)
    if symbols:
        config.symbols = symbols
        config.paths = config.paths
    if config.mode == "paper_testnet" and config.enable_testnet_signed_endpoints:
        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
        if not api_key or not api_secret:
            config.enable_testnet_signed_endpoints = False
    depth_snapshot_client = BinancePublicRestClient(
        base_url=config.gateway.public_rest_base,
        timeout_seconds=config.gateway.request_timeout_s,
        depth_snapshot_limit=config.gateway.depth_snapshot_limit,
    )
    client = PaperEngine(config, depth_snapshot_client=depth_snapshot_client)
    ws = BinancePublicWebSocket(symbols=config.symbol_set, config=config.gateway)

    processed = 0
    console.rule("ofbot live run")
    console.print(f"run_id={client.run_id} symbols={','.join(config.symbol_set)}")

    async def _consume_events() -> None:
        nonlocal processed
        async for event in ws.events():
            client.process_event(event.event)
            processed += 1
            if max_events > 0 and processed >= max_events:
                break

    try:
        if max_seconds > 0:
            async with asyncio.timeout(max_seconds):
                await _consume_events()
        else:
            await _consume_events()
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/yellow]")
    except TimeoutError:
        console.print(f"[yellow]max_seconds reached ({max_seconds}s)[/yellow]")
    finally:
        report_path = client.finalize()
        if report_path is not None:
            console.print(f"report: {report_path}")
        client.close()


def _run_replay(config_path: str, input_path: str) -> None:
    config = load_config(config_path)
    report_path = run_replay(config, Path(input_path))
    if report_path is not None:
        console.rule("ofbot replay report")
        console.print(f"report: {report_path}")


def _run_report(config_path: str, date_value: str) -> None:
    config = load_config(config_path)
    store = MemoryStore(config.learning.duckdb_path)
    try:
        report_path = build_daily_report(store=store, date_value=date_value, output_root=config.paths.reports_dir)
    finally:
        store.close()
    console.rule("ofbot daily report")
    console.print(f"report: {report_path}")


def _run_paper_order(
    config_path: str,
    *,
    symbol: str,
    side: str,
    qty: float,
    notional_usd: float,
    order_type: str,
    reason: str,
    force: bool,
) -> None:
    config = load_config(config_path)
    if config.mode != "paper_local":
        raise SystemExit("manual paper orders are only supported in paper_local mode")

    qty_value = float(qty) if qty > 0 else None
    notional_value = float(notional_usd) if notional_usd > 0 else None
    if side != "flat" and qty_value is None and notional_value is None:
        raise SystemExit("provide --qty or --notional-usd for buy/sell manual orders")

    side_value = {"buy": 1, "sell": -1, "flat": 0}[side]
    request_path = write_manual_order_request(
        config.paths.memory_dir,
        symbol=symbol,
        side=side_value,
        order_type=order_type,
        qty=qty_value,
        notional_usd=notional_value,
        reason=reason,
        force=force,
    )
    console.rule("ofbot paper order queued")
    console.print(f"request: {request_path}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging(verbose=args.verbose)
    if args.command == "live":
        asyncio.run(_run_live(args.config, args.max_events, args.max_seconds, args.symbol))
    elif args.command == "replay":
        _run_replay(args.config, args.input)
    elif args.command == "report":
        date_value = args.date
        if date_value == "today":
            date_value = datetime.now(UTC).date().isoformat()
        _run_report(args.config, date_value)
    elif args.command == "paper-order":
        _run_paper_order(
            args.config,
            symbol=args.symbol,
            side=args.side,
            qty=args.qty,
            notional_usd=args.notional_usd,
            order_type=args.order_type,
            reason=args.reason,
            force=args.force,
        )


if __name__ == "__main__":
    main()
